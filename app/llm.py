from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY

# LangChain's wrapper around the Groq API. It handles auth, request
# formatting, and response parsing so we don't hand-roll HTTP calls.
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0.7,  # 0 = deterministic, 1+ = more creative; 0.7 is a balanced default
    streaming=True,
)

# A separate, vision-capable model for messages that include an image.
# gpt-oss-20b above is text-only, so any turn with an image attached gets
# routed to this model instead (see chat_service.history_has_image).
vision_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="qwen/qwen3.6-27b",
    temperature=0.3,
    streaming=True,
)
