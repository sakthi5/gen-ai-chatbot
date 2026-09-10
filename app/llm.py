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
# routed to this model instead (see chat_service.is_vision_turn).
#
# reasoning_effort="none" turns off this model's default "think out loud
# in <think>...</think> before answering" behavior. Two problems, one
# fix: (1) that reasoning could run long enough on its own to exceed the
# account's output-tokens-per-minute limit before an answer was even
# reached — Groq was rejecting some requests outright with a 429
# ("Request too large ... on output tokens per minute"); (2) a "detailed
# description" style prompt could exhaust a fixed max_tokens budget
# mid-reasoning, before ever producing a real answer. This model is used
# here for simple vision Q&A, not math/coding, so we don't need the
# reasoning step at all. max_tokens is still capped as a sane ceiling.
vision_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="qwen/qwen3.6-27b",
    temperature=0.3,
    streaming=True,
    max_tokens=700,
    reasoning_effort="none",
)
