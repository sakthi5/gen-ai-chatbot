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
# Minimal ChatGPT-like styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { max-width: 800px; padding-top: 2rem; }
        section[data-testid="stSidebar"] { min-width: 280px; }
        .stChatMessage { padding: 0.75rem 0; }
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

if not st.session_state.messages:
    st.caption("Ask me anything to get started.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter your question:")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": user_input,
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
                "Make sure the FastAPI server is running (`uvicorn app.api:app --reload`)."
            )

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
