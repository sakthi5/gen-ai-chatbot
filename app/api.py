from fastapi import FastAPI
from app.models import ChatRequest
from app.services.chat_service import (
    chat_service,
    clear_chat_service,
)

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Welcome to Gen AI Chatbot API"
    }

@app.post("/chat")
def chat(request: ChatRequest):

    reply = chat_service(request.message)

    return {
        "reply": reply
    }

@app.post("/clear")
def clear_chat():

    clear_chat_service()

    return {
        "message": "Conversation cleared."
    }
