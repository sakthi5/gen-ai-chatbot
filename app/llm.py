from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY

# Create the Groq LLM instance
# Create the LLM client
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

# ChatGroq
# This is the LangChain wrapper that connects to Groq.
# Instead of manually making HTTP requests, we use this class.
# This controls how creative the model is.

# ChatGroq is the LLM client.
# It acts as a bridge between your code and the AI model.
# Instead of writing complicated HTTP requests yourself, you simply use the client

# Without ChatGroq

# import requests

# response = requests.post(
#     "https://api.groq.com/...",
#     headers={...},
#     json={...}
# )

# You'd have to:

# Create HTTP requests
# Add headers
# Add authentication
# Parse JSON
# Handle errors

# That's a lot of work.

# With ChatGroq

# llm = ChatGroq(...)
# "Create a Groq client object."

# This object knows:

# Which API to call
# Which model to use
# Which API key to use
# How to send prompts
# How to receive responses

# 0.0 → Very deterministic
# 0.3 → Focused
# 0.7 → Balanced ✅
# 1.0+ → More creative  