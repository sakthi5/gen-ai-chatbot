from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.models import ChatRequest, Conversation
from app.services.chat_service import (
    chat_service,
    create_conversation,
    list_conversations,
    get_conversation_messages,
    delete_conversation,
    add_document_to_conversation,
    get_conversation_documents,
)
from app.services.document_service import UnsupportedDocumentType
from app.database import engine, Base, get_db

app = FastAPI(title="Gen AI Chatbot API")

# Allow the Streamlit frontend (or any local dev client) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-ID"],
)

Base.metadata.create_all(bind=engine)


@app.get("/")
def home():
    return {
        "message": "Welcome to Gen AI Chatbot API"
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
    db=Depends(get_db)
):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    conversation_id = request.conversation_id

    if conversation_id is None:

        conversation = create_conversation(db)

        conversation_id = conversation.id

    else:

        exists = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )

        if exists is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    return StreamingResponse(
        chat_service(
            request.message,
            db,
            conversation_id
        ),
        media_type="text/plain",
        headers={
            "X-Conversation-ID": conversation_id
        }
    )


@app.post("/conversations")
def start_conversation(db=Depends(get_db)):
    """Create a new, empty conversation (e.g. to attach a document before chatting)."""

    conversation = create_conversation(db)

    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
    }


@app.get("/conversations")
def get_conversations(db=Depends(get_db)):

    conversations = list_conversations(db)

    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at,
        }
        for c in conversations
    ]


@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str, db=Depends(get_db)):

    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = get_conversation_messages(db, conversation_id)

    return [
        {
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at,
        }
        for m in messages
    ]


@app.post("/conversations/{conversation_id}/documents")
async def upload_document(
    conversation_id: str,
    file: UploadFile = File(...),
    db=Depends(get_db),
):
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        document = add_document_to_conversation(
            db, conversation_id, file.filename, file_bytes
        )
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "id": document.id,
        "filename": document.filename,
        "characters_extracted": len(document.content),
    }


@app.get("/conversations/{conversation_id}/documents")
def list_documents(conversation_id: str, db=Depends(get_db)):
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    documents = get_conversation_documents(db, conversation_id)

    return [
        {
            "id": d.id,
            "filename": d.filename,
            "characters": len(d.content),
        }
        for d in documents
    ]


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str, db=Depends(get_db)):

    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    delete_conversation(db, conversation_id)

    return {"message": "Conversation deleted."}
