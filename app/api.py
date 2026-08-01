from fastapi import FastAPI
from pydantic import BaseModel
from app.llm import llm

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "Welcome to Gen AI Chatbot API"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    response = llm.invoke(request.message)

    return {
        "reply": response.content
    }