import base64
import re
import uuid

from langchain_core.messages import (HumanMessage, AIMessage, SystemMessage,)
from app.models import Message, Conversation, Document, Image

from app.llm import llm, vision_llm
from app.prompts import SYSTEM_PROMPT
from app.services.document_service import extract_text
from app.services.image_gen_service import generate_image


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks the vision model emits."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _build_human_message(text: str, images):
    """Build a HumanMessage — plain text, or multimodal if images are present."""

    if not images:
        return HumanMessage(content=text)

    content = [{"type": "text", "text": text}]

    for image in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{image.mime_type};base64,{image.data_base64}"},
        })

    return HumanMessage(content=content)


def chat_service(message: str, db, conversation_id: str, images=None):
    """Stream an assistant reply for `message` and persist both sides to the DB.

    Conversation memory is loaded fresh from the database on every call, so
    there is no in-memory/global history to keep in sync across requests.

    `images` (optional) are base64-encoded attachments for THIS turn only —
    see app.models.ImageAttachment. If this turn or any earlier turn in the
    conversation included an image, the request is routed to the
    vision-capable model instead of the regular text model.
    """

    images = images or []

    is_first_message = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .count()
    ) == 0

    history = load_conversation_history(
        db,
        conversation_id
    )

    history.append(
        _build_human_message(message, images)
    )

    is_vision_turn = bool(images) or any(
        isinstance(entry, HumanMessage) and isinstance(entry.content, list)
        for entry in history
    )

    model = vision_llm if is_vision_turn else llm

    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=message
    )

    db.add(user_message)
    db.flush()  # assigns user_message.id without ending the transaction

    for image in images:
        db.add(Image(
            conversation_id=conversation_id,
            message_id=user_message.id,
            source="uploaded",
            filename=image.filename,
            mime_type=image.mime_type,
            data_base64=image.data_base64,
        ))

    db.commit()

    full_response = ""
    thinking_buffer = ""
    past_thinking = not is_vision_turn  # only the vision model emits <think> blocks

    try:
        for chunk in model.stream(history):

            if chunk.content:

                full_response += chunk.content

                if past_thinking:
                    yield chunk.content
                    continue

                thinking_buffer += chunk.content

                if "</think>" in thinking_buffer:
                    past_thinking = True
                    remainder = thinking_buffer.split("</think>", 1)[1]
                    if remainder:
                        yield remainder

        # Safety net: if the model never emitted a closing </think> tag (not
        # every response reasons out loud), don't silently swallow the whole
        # reply — yield whatever was buffered instead of nothing.
        if not past_thinking and thinking_buffer:
            yield thinking_buffer

    except Exception as exc:

        # Persist whatever we managed to stream before the failure, so the
        # conversation history stays consistent, then surface the error.
        error_note = f"\n\n_[Error: response interrupted — {exc}]_"
        full_response += error_note
        yield error_note

    finally:

        clean_response = _strip_think_tags(full_response) if is_vision_turn else full_response

        if clean_response.strip():

            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=clean_response
            )

            db.add(assistant_message)
            db.commit()

        # Title the conversation after the reply is in, so generating it
        # never delays the first streamed token of the visible answer.
        if is_first_message:
            maybe_set_conversation_title(db, conversation_id, message)

def load_conversation_history(db, conversation_id):

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id)
        .all()
    )

    history = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

    documents = get_conversation_documents(db, conversation_id)

    if documents:

        documents_text = "\n\n---\n\n".join(
            f"Document: {doc.filename}\n\n{doc.content}" for doc in documents
        )

        history.append(
            SystemMessage(
                content=(
                    "The user has shared the following document(s) in this "
                    "conversation. Use them to answer questions when "
                    "relevant, and say so if the answer isn't in them:\n\n"
                    f"{documents_text}"
                )
            )
        )

    for message in messages:

        if message.role == "user":

            uploaded_images = (
                db.query(Image)
                .filter(Image.message_id == message.id, Image.source == "uploaded")
                .all()
            )

            history.append(
                _build_human_message(message.content, uploaded_images)
            )

        elif message.role == "assistant":

            history.append(
                AIMessage(content=message.content)
            )

    return history

def create_conversation(db, title="New Chat"):

    conversation_id = str(uuid.uuid4())

    conversation = Conversation(
        id=conversation_id,
        title=title
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


def list_conversations(db):
    """Return all conversations, most recently created first."""

    return (
        db.query(Conversation)
        .order_by(Conversation.created_at.desc())
        .all()
    )


def get_conversation_messages(db, conversation_id):
    """Return the raw message rows for a conversation, oldest first."""

    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id)
        .all()
    )


def get_message_images(db, message_id):
    """Return all images (uploaded or generated) attached to one message."""

    return (
        db.query(Image)
        .filter(Image.message_id == message_id)
        .order_by(Image.id)
        .all()
    )


def generate_and_save_image(db, conversation_id: str, prompt: str):
    """Generate an image from a prompt and record it as a chat turn.

    Saves a user "message" (the prompt) and an assistant "message" (a short
    caption) so the exchange shows up naturally in conversation history,
    with the generated image attached to the assistant's turn.
    """

    is_first_message = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .count()
    ) == 0

    image_bytes = generate_image(prompt)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=f"🎨 Generate an image: {prompt}",
    )
    db.add(user_message)
    db.flush()

    assistant_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=f'Here\'s the image I generated for: "{prompt}"',
    )
    db.add(assistant_message)
    db.flush()

    image = Image(
        conversation_id=conversation_id,
        message_id=assistant_message.id,
        source="generated",
        filename=f"generated_{assistant_message.id}.jpg",
        mime_type="image/jpeg",
        data_base64=image_base64,
        prompt=prompt,
    )
    db.add(image)
    db.commit()
    db.refresh(assistant_message)
    db.refresh(image)

    if is_first_message:
        maybe_set_conversation_title(db, conversation_id, prompt)

    return assistant_message, image


def add_document_to_conversation(db, conversation_id, filename: str, file_bytes: bytes):
    """Extract text from an uploaded file and attach it to a conversation.

    Raises document_service.UnsupportedDocumentType for unrecognized file
    types; the caller is expected to turn that into a 400 response.
    """

    content = extract_text(filename, file_bytes)

    document = Document(
        conversation_id=conversation_id,
        filename=filename,
        content=content,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


def get_conversation_documents(db, conversation_id):
    """Return all documents attached to a conversation, oldest first."""

    return (
        db.query(Document)
        .filter(Document.conversation_id == conversation_id)
        .order_by(Document.id)
        .all()
    )


def delete_conversation(db, conversation_id):
    """Delete a conversation and all of its messages, documents, and images."""

    db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).delete()

    db.query(Document).filter(
        Document.conversation_id == conversation_id
    ).delete()

    db.query(Image).filter(
        Image.conversation_id == conversation_id
    ).delete()

    db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).delete()

    db.commit()


TITLE_MAX_LENGTH = 40


def build_conversation_title(message: str) -> str:
    """Turn a raw first message into a clean, display-ready title.

    Collapses whitespace/newlines, capitalizes the first letter, and
    truncates long messages at a word boundary with an ellipsis instead
    of showing the message verbatim.
    """

    text = " ".join(message.split())

    if not text:
        return "New Chat"

    text = text[0].upper() + text[1:]

    if len(text) <= TITLE_MAX_LENGTH:
        return text.rstrip(" .,!?")

    truncated = text[:TITLE_MAX_LENGTH]

    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]

    return truncated.rstrip(" .,!?") + "..."


TITLE_PROMPT = (
    "Summarize the user's message as a short chat title of 2 to 5 words. "
    "Respond with the title only — no quotes, no punctuation at the end, "
    "no explanation."
)


def generate_smart_title(message: str):
    """Ask the LLM to summarize the topic into a short title.

    Returns the cleaned title, or None if the call fails or the model
    returns something unusable — callers should fall back to
    `build_conversation_title` in that case.
    """

    try:
        response = llm.invoke([
            SystemMessage(content=TITLE_PROMPT),
            HumanMessage(content=message),
        ])

        title = (response.content or "").strip().strip('"').strip("'")
        title = " ".join(title.split())

        if not title:
            return None

        # Guard against the model ignoring the length instruction.
        if len(title) > TITLE_MAX_LENGTH:
            title = build_conversation_title(title)

        return title[0].upper() + title[1:]

    except Exception:
        # Title generation is a nice-to-have — never let it break the chat.
        return None


def maybe_set_conversation_title(db, conversation_id, first_message: str):
    """Auto-title a conversation from its first user message, once.

    Tries an LLM-generated summary first (e.g. "What is RAG?" ->
    "Understanding RAG"); falls back to a cleaned/truncated version of
    the raw message if that call fails for any reason.
    """

    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )

    if conversation and conversation.title == "New Chat":

        title = generate_smart_title(first_message) or build_conversation_title(first_message)

        conversation.title = title

        db.commit()