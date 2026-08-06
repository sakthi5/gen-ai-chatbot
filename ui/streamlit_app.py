import streamlit as st
import requests

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Gen AI Chatbot")

if st.button("🗑️ Clear Chat"):

    requests.post("http://127.0.0.1:8000/clear")

    st.session_state.messages = []

    st.rerun()
    
# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_input = st.chat_input("Enter your question:")

# Send button
if user_input:

  try:

    with st.spinner("Thinking..."):

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={
                "message": user_input
            }
        )

        result = response.json()

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["reply"]
    })

    st.rerun()

  except Exception as e:

    st.error("Unable to connect to the AI server.")