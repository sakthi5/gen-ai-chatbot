from langchain_core.messages import (HumanMessage, AIMessage, SystemMessage,)
from app.models import Message, Conversation

from app.llm import llm
from app.prompts import SYSTEM_PROMPT
import uuid


def chat_service(message: str, db, conversation_id: str):
    """Stream an assistant reply for `message` and persist both sides to the DB.

    Conversation memory is loaded fresh from the database on every call, so
    there is no in-memory/global history to keep in sync across requests.
    """

    history = load_conversation_history(
        db,
        conversation_id
    )

    is_first_message = len(history) == 1  # only the system prompt so far

    history.append(
        HumanMessage(content=message)
    )

    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=message
    )

    db.add(user_message)
    db.commit()

    full_response = ""

    try:
        for chunk in llm.stream(history):

            if chunk.content:

                full_response += chunk.content

                yield chunk.content

    except Exception as exc:

        # Persist whatever we managed to stream before the failure, so the
        # conversation history stays consistent, then surface the error.
        error_note = f"\n\n_[Error: response interrupted — {exc}]_"
        full_response += error_note
        yield error_note

    finally:

        if full_response.strip():

            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response
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

    for message in messages:

        if message.role == "user":

            history.append(
                HumanMessage(content=message.content)
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


def delete_conversation(db, conversation_id):
    """Delete a conversation and all of its messages."""

    db.query(Message).filter(
        Message.conversation_id == conversation_id
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