import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import app
from app.database import Base, get_db


# Use an isolated in-memory SQLite database for tests instead of chatbot.db.
# StaticPool keeps every connection pointing at the same in-memory DB instead
# of each `connect()` call getting its own throwaway database.
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def fake_llm_stream(monkeypatch):
    """Avoid real Groq API calls in tests by stubbing the LLM entirely."""

    class FakeChunk:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def stream(self, _messages):
            yield FakeChunk("Hello ")
            yield FakeChunk("world!")

    monkeypatch.setattr("app.services.chat_service.llm", FakeLLM())


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Gen AI Chatbot API"}


def test_chat_creates_conversation_and_streams_reply():
    response = client.post("/chat", json={"message": "Hi there"})

    assert response.status_code == 200
    assert response.text == "Hello world!"

    conversation_id = response.headers.get("X-Conversation-ID")
    assert conversation_id


def test_chat_rejects_empty_message():
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400


def test_chat_with_unknown_conversation_id_returns_404():
    response = client.post(
        "/chat",
        json={"message": "Hi", "conversation_id": "does-not-exist"},
    )
    assert response.status_code == 404


def test_conversation_history_round_trip():
    # First message creates a conversation.
    first = client.post("/chat", json={"message": "Remember this"})
    conversation_id = first.headers["X-Conversation-ID"]

    # Second message continues the same conversation.
    client.post(
        "/chat",
        json={"message": "Follow up", "conversation_id": conversation_id},
    )

    messages = client.get(f"/conversations/{conversation_id}/messages").json()

    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_list_and_delete_conversation():
    created = client.post("/chat", json={"message": "Temp chat"})
    conversation_id = created.headers["X-Conversation-ID"]

    listing = client.get("/conversations").json()
    assert any(c["id"] == conversation_id for c in listing)

    delete_response = client.delete(f"/conversations/{conversation_id}")
    assert delete_response.status_code == 200

    messages_after_delete = client.get(f"/conversations/{conversation_id}/messages")
    assert messages_after_delete.status_code == 404
