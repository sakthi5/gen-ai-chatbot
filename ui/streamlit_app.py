import os
import streamlit as st
import requests

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
        .stMainBlockContainer { max-width: 800px; padding-top: 1rem; }
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
            padding: 4px 0 !important;
            width: 100% !important;
        }

        /* --- Main content uses the full width once the sidebar is collapsed --- */
        [data-testid="stSidebar"][aria-expanded="false"] + div .stMainBlockContainer {
            max-width: 100% !important;
        }

        /* --- Sticky, centered page title --- */
        [data-testid="stMainBlockContainer"] [data-testid="stHeading"]:first-of-type {
            position: sticky;
            top: 0;
            z-index: 999;
            text-align: center;
            padding: 0.6rem 0;
            margin: -1rem -1rem 0.5rem -1rem;
            background: #0e1117;
        }
        @media (prefers-color-scheme: light) {
            [data-testid="stMainBlockContainer"] [data-testid="stHeading"]:first-of-type {
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
            max-width: 75%;
            padding: 10px 14px;
            border-radius: 16px;
        }
        [data-testid="stChatMessageContent"] p {
            margin-bottom: 0;
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
            content: "Attach a document (PDF, TXT, or DOCX)\A No file chosen\A 200MB per file • PDF, TXT, DOCX";
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
            {"role": m["role"], "content": m["content"]}
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

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🤖 Gen AI Chatbot")

attached = get_attached_documents(st.session_state.conversation_id)
if attached:
    st.caption("📎 Attached: " + ", ".join(d["filename"] for d in attached))

if not st.session_state.messages:
    st.caption("Ask me anything to get started, or attach a document below.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Enter your question:",
    accept_file=True,
    file_type=["pdf", "txt", "docx"],
    max_upload_size=200,
)

if prompt:

    user_text = (prompt.text or "").strip()
    attached_names = attach_documents(prompt.files)

    # Nothing to send to the model (just an attachment, no question) — show
    # the attachment and stop here instead of calling /chat with nothing.
    if not user_text and attached_names:
        st.rerun()

    elif user_text or attached_names:

        message_to_send = user_text or "Please read and explain the attached document."

        display_text = message_to_send
        if attached_names:
            display_text = "📎 " + ", ".join(attached_names) + "\n\n" + message_to_send

        st.session_state.messages.append({"role": "user", "content": display_text})

        with st.chat_message("user"):
            st.markdown(display_text)

        with st.chat_message("assistant"):

            placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("Thinking..."):
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": message_to_send,
                            "conversation_id": st.session_state.conversation_id,
                        },
                        stream=True,
                        timeout=60,
                    )
                    response.raise_for_status()

                new_conversation_id = response.headers.get("X-Conversation-ID")
                if new_conversation_id:
                    st.session_state.conversation_id = new_conversation_id

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

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()
