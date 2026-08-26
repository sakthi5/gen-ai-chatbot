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
