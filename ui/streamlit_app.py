import base64
import os
import streamlit as st
import requests

DOCUMENT_EXTENSIONS = {"pdf", "txt", "docx"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Images are sent inline as base64 in the /chat JSON body (not a separate
# multipart upload like documents), so keep them small enough that the
# request stays fast — base64 inflates size by roughly a third.
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB

# Overridable via env var in case 8020 is also taken on your machine, e.g.:
#   API_URL=http://127.0.0.1:8030 streamlit run ui/streamlit_app.py
API_URL = os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8020")

st.set_page_config(
    page_title="Gen AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# ChatGPT-like styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Wide enough for tables/code in replies without cramming; still
           capped so lines of plain text don't stretch too wide to read.
           padding-top: 60px matches Streamlit's own header height. The
           sticky title below is pinned at top:60px too — without this,
           the title's *reserved* flow space (based on where it would
           naturally sit, ~8px from the top) didn't match where it
           actually rendered once stuck (60px), and whatever followed it
           (e.g. the "Attached:" caption) would render into that gap and
           peek out from behind/above the title while scrolling. */
        .stMainBlockContainer { max-width: 1200px; padding-top: 60px; }

        section[data-testid="stSidebar"] { min-width: 280px; }

        /* --- Sidebar: shrink the oversized header/collapse-arrow area --- */
        [data-testid="stSidebarHeader"] {
            min-height: 0 !important;
            height: auto !important;
            padding: 6px 8px !important;
        }
        [data-testid="stSidebarCollapseButton"] button {
            width: 22px !important;
            height: 22px !important;
            min-width: 22px !important;
            min-height: 22px !important;
            padding: 0 !important;
        }
        [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
            font-size: 16px !important;
        }

        /* --- Sidebar: tighten the gap before each conversation's delete icon --- */
        [data-testid="stSidebarUserContent"] [data-testid="stHorizontalBlock"] {
            gap: 6px !important;
            align-items: center !important;
        }
        [data-testid="stSidebarUserContent"] [data-testid="stHorizontalBlock"]
            [data-testid="stColumn"]:last-child button {
            padding: 4px 5px !important;
            width: 100% !important;
        }

        /* --- Main content uses the full width once the sidebar is collapsed --- */
        [data-testid="stSidebar"][aria-expanded="false"] + div .stMainBlockContainer {
            max-width: 100% !important;
        }

        /* --- Sticky, centered page title ---
               A sticky element can only stay pinned while scrolling within
               the bounds of its own DIRECT PARENT's box. Streamlit wraps
               the title in its own tiny stElementContainer (barely taller
               than the title itself), so making the inner heading sticky
               only "stuck" for a few px before scrolling away with that
               wrapper. Making the *wrapper* sticky instead works, because
               the wrapper's own parent (stVerticalBlock) spans the entire
               conversation, giving it room to stay stuck the whole time.
               top is offset by Streamlit's own header height (60px, fixed
               at the very top with a much higher z-index) — at top:0 our
               title was rendering directly underneath that header and
               getting completely hidden behind it. */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stElementContainer"]:has(h1) {
            position: sticky;
            top: 60px;
            z-index: 999;
            padding: 0.5rem 0;
            margin: 0 0 0.5rem 0;
            background: #0e1117;
        }
        [data-testid="stMainBlockContainer"] > div > [data-testid="stElementContainer"]:has(h1) h1 {
            text-align: center;
            font-size: 1.5rem !important;
            margin: 0 !important;
        }
        @media (prefers-color-scheme: light) {
            [data-testid="stMainBlockContainer"] > div > [data-testid="stElementContainer"]:has(h1) {
                background: #ffffff;
            }
        }

        /* --- Chat bubbles: user on the right, assistant on the left --- */
        [data-testid="stChatMessage"] {
            background: transparent !important;
            padding: 4px 0 !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
            justify-content: flex-start;
        }
        [data-testid="stChatMessageContent"] {
            flex: 0 1 auto !important;
            /* Streamlit centers this by default with large auto margins —
               reset that so our flex justify-content controls alignment. */
            margin: 0 !important;
            max-width: 100%;
            padding: 10px 14px;
            border-radius: 16px;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] {
            background: #6c5ce7;
            color: #ffffff;
            border-radius: 16px 16px 4px 16px;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
            [data-testid="stChatMessageContent"] {
            background: #262730;
            color: #e6e6e6;
            border-radius: 16px 16px 16px 4px;
        }

        /* --- Chat input: move the attach icon after the send button, with a
               custom hover tooltip describing what/how to attach ---
               `order` has to be set on the actual flex-item wrapper divs
               (one level above the testid elements), not the testid
               elements themselves, which aren't the direct flex children. */
        [data-testid="stChatInput"] div:has(> [data-testid="stChatInputSubmitButton"]) {
            order: 1;
            align-self: center;
        }
        [data-testid="stChatInput"] div:has(> [data-testid="stChatInputFileUploadButton"]) {
            order: 2;
            align-self: center;
        }
        [data-testid="stChatInputFileUploadButton"] {
            position: relative;
        }
        
        /* Suppress Streamlit's own native "Upload a file" tooltip so only
           our custom one (below) shows — otherwise the two overlap. This
           app doesn't use `help=` tooltips anywhere else, so hiding the
           tooltip layer globally is safe. */
        [data-testid="stTooltipContent"] {
            display: none !important;
        }
        [data-testid="stChatInputFileUploadButton"]:hover::after {
            content: "Attach a document or image\A No file chosen\A PDF, TXT, DOCX, PNG, JPG, WEBP";
            white-space: pre-line;
            position: absolute;
            bottom: 100%;
            right: 0;
            margin-bottom: 8px;
            background: #262730;
            color: #ffffff;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            line-height: 1.5;
            width: 210px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            pointer-events: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "export_bytes" not in st.session_state:
    st.session_state.export_bytes = None

if "export_ready_for" not in st.session_state:
    # (conversation_id, format) the currently-held export_bytes are for —
    # so switching chats or formats doesn't offer a stale download.
    st.session_state.export_ready_for = None


def start_new_chat():
    st.session_state.messages = []
    st.session_state.conversation_id = None


def get_attached_documents(conversation_id: str):
    """List documents already attached to a conversation, if any."""
    if conversation_id is None:
        return []
    try:
        response = requests.get(
            f"{API_URL}/conversations/{conversation_id}/documents", timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return []


def load_conversation(conversation_id: str):
    """Fetch a past conversation's messages from the backend and load them."""
    try:
        response = requests.get(
            f"{API_URL}/conversations/{conversation_id}/messages",
            timeout=10,
        )
        response.raise_for_status()

        st.session_state.messages = [
            {"role": m["role"], "content": m["content"], "images": m.get("images", [])}
            for m in response.json()
        ]
        st.session_state.conversation_id = conversation_id

    except requests.exceptions.RequestException as e:
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.error(f"Couldn't load that conversation: {e}")


def attach_documents(files):
    """Upload each file to the active conversation (creating one if needed).

    Returns the list of filenames successfully attached.
    """
    if not files:
        return []

    if st.session_state.conversation_id is None:
        try:
            create_response = requests.post(f"{API_URL}/conversations", timeout=10)
            create_response.raise_for_status()
            st.session_state.conversation_id = create_response.json()["id"]
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't start a conversation: {e}")
            return []

    attached_names = []

    for file in files:
        try:
            with st.spinner(f"Reading {file.name}..."):
                upload_response = requests.post(
                    f"{API_URL}/conversations/{st.session_state.conversation_id}/documents",
                    files={"file": (file.name, file.getvalue())},
                    timeout=30,
                )
                upload_response.raise_for_status()
            attached_names.append(file.name)

        except requests.exceptions.RequestException as e:
            detail = None
            if e.response is not None:
                try:
                    detail = e.response.json().get("detail")
                except ValueError:
                    pass
            st.error(f"Couldn't attach {file.name}: {detail or e}")

    return attached_names


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _mime_type_for(extension: str) -> str:
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(extension, "application/octet-stream")


def encode_images_for_chat(files):
    """Base64-encode image attachments for inline use in the /chat request.

    Returns (image_payloads, rejected_filenames) — files over the size
    limit are skipped rather than silently truncated or sent anyway.
    """

    payloads = []
    rejected = []

    for file in files:
        if file.size > MAX_IMAGE_BYTES:
            rejected.append(file.name)
            continue

        payloads.append({
            "filename": file.name,
            "mime_type": _mime_type_for(_extension(file.name)),
            "data_base64": base64.b64encode(file.getvalue()).decode("utf-8"),
        })

    return payloads, rejected


def render_message_images(images, key_prefix: str):
    """Render any images attached to a chat message, with a download button."""

    for i, image in enumerate(images):
        image_bytes = base64.b64decode(image["data_base64"])
        st.image(image_bytes)
        st.download_button(
            "⬇️ Download image",
            data=image_bytes,
            file_name=image.get("filename", "image.jpg"),
            mime=image.get("mime_type", "image/jpeg"),
            key=f"dl_img_{key_prefix}_{i}",
        )


# ---------------------------------------------------------------------------
# Sidebar — conversation list
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Gen AI Chatbot")

    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()
    st.caption("Conversations")

    try:
        conversations_response = requests.get(f"{API_URL}/conversations", timeout=10)
        conversations_response.raise_for_status()
        conversations = conversations_response.json()
    except requests.exceptions.RequestException:
        conversations = None
        st.warning("Backend not reachable. Is the FastAPI server running?")

    if conversations is not None:
        if not conversations:
            st.caption("No conversations yet.")

        for convo in conversations:
            is_active = convo["id"] == st.session_state.conversation_id
            col1, col2 = st.columns([5, 1])

            with col1:
                label = ("📝 " if is_active else "💬 ") + convo["title"]
                if st.button(label, key=f"open_{convo['id']}", use_container_width=True):
                    load_conversation(convo["id"])
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"delete_{convo['id']}"):
                    try:
                        requests.delete(f"{API_URL}/conversations/{convo['id']}", timeout=10)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Couldn't delete: {e}")
                    else:
                        if is_active:
                            start_new_chat()
                        st.rerun()

    st.divider()

    with st.expander("⬇️ Export conversation"):

        if st.session_state.conversation_id is None:
            st.caption("Start a conversation first.")

        else:
            export_format = st.radio(
                "Format", ["DOCX", "PDF"], horizontal=True, key="export_format_choice"
            )
            fmt = export_format.lower()

            if st.button("Prepare download", use_container_width=True, key="prepare_export_button"):
                try:
                    with st.spinner(f"Building {export_format}..."):
                        export_response = requests.get(
                            f"{API_URL}/conversations/{st.session_state.conversation_id}/export",
                            params={"format": fmt},
                            timeout=30,
                        )
                        export_response.raise_for_status()

                    st.session_state.export_bytes = export_response.content
                    st.session_state.export_ready_for = (st.session_state.conversation_id, fmt)

                except requests.exceptions.RequestException as e:
                    st.error(f"Couldn't export: {e}")

            if st.session_state.export_ready_for == (st.session_state.conversation_id, fmt):
                mime = (
                    "application/pdf" if fmt == "pdf"
                    else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                st.download_button(
                    "⬇️ Download",
                    data=st.session_state.export_bytes,
                    file_name=f"conversation.{fmt}",
                    mime=mime,
                    use_container_width=True,
                    key="download_export_button",
                )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🤖 Gen AI Chatbot")

attached = get_attached_documents(st.session_state.conversation_id)
if attached:
    st.caption("📎 Attached: " + ", ".join(d["filename"] for d in attached))

if not st.session_state.messages:
    st.caption(
        "Ask me anything to get started, attach a document/image, "
        "or just ask me to generate one (e.g. \"draw a red robot waving\")."
    )

for msg_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_message_images(message.get("images", []), key_prefix=f"hist_{msg_index}")

prompt = st.chat_input(
    "Enter your question:",
    accept_file=True,
    file_type=sorted(DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS),
    max_upload_size=200,
)

if prompt:

    user_text = (prompt.text or "").strip()

    document_files = [f for f in prompt.files if _extension(f.name) in DOCUMENT_EXTENSIONS]
    image_files = [f for f in prompt.files if _extension(f.name) in IMAGE_EXTENSIONS]

    attached_names = attach_documents(document_files)
    image_payloads, rejected_images = encode_images_for_chat(image_files)

    for name in rejected_images:
        st.error(f"{name} is over the 8MB limit for images — try a smaller file.")

    has_files = attached_names or image_payloads

    # Nothing to send to the model (just an attachment, no question) — show
    # the attachment and stop here instead of calling /chat with nothing.
    if not user_text and has_files and not image_payloads:
        st.rerun()

    elif user_text or has_files:

        message_to_send = user_text or "Please read and explain the attached document/image."

        display_text = message_to_send
        if attached_names:
            display_text = "📎 " + ", ".join(attached_names) + "\n\n" + message_to_send

        st.session_state.messages.append({
            "role": "user",
            "content": display_text,
            "images": image_payloads,
        })

        with st.chat_message("user"):
            st.markdown(display_text)
            render_message_images(image_payloads, key_prefix="pending")

        with st.chat_message("assistant"):

            placeholder = st.empty()
            full_response = ""
            image_was_generated = False

            try:
                with st.spinner("Thinking..."):
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": message_to_send,
                            "conversation_id": st.session_state.conversation_id,
                            "images": image_payloads,
                        },
                        stream=True,
                        timeout=60,
                    )
                    response.raise_for_status()

                new_conversation_id = response.headers.get("X-Conversation-ID")
                if new_conversation_id:
                    st.session_state.conversation_id = new_conversation_id

                image_was_generated = response.headers.get("X-Image-Generated") == "true"

                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

            except requests.exceptions.RequestException as e:
                full_response = ""
                placeholder.error(
                    f"Something went wrong talking to the backend: {e}\n\n"
                    "Make sure the FastAPI server is running (`uvicorn app.api:app --reload --port 8020`)."
                )

        if image_was_generated:
            # The assistant's reply came with a generated image attached —
            # reload from the backend instead of manually appending, so the
            # image (which /chat's plain-text stream can't carry) shows up
            # immediately rather than only after the conversation is
            # reopened.
            load_conversation(st.session_state.conversation_id)
            st.rerun()

        elif full_response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "images": [],
            })
            st.rerun()
