import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import app
from app.database import Base, get_db
from app.services.chat_service import build_conversation_title, generate_smart_title


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

        def invoke(self, _messages):
            # Stands in for the LLM-generated title call.
            return FakeChunk("Fake Smart Title")

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


def test_build_conversation_title_short_message():
    assert build_conversation_title("what is my name") == "What is my name"


def test_build_conversation_title_truncates_long_message():
    title = build_conversation_title(
        "Explain FastAPI and its features in a lot more detail than usual"
    )
    assert title.endswith("...")
    assert len(title) <= 43  # 40 chars + "..."
    assert not title.endswith(" ...")  # no dangling space before the ellipsis


def test_build_conversation_title_collapses_whitespace():
    assert build_conversation_title("  hi\nam   sakthi  ") == "Hi am sakthi"


def test_chat_sets_title_from_llm_summary():
    """With a working LLM, the title comes from the summarizer, not the raw message."""

    response = client.post("/chat", json={"message": "What is RAG?"})
    conversation_id = response.headers["X-Conversation-ID"]

    conversations = client.get("/conversations").json()
    convo = next(c for c in conversations if c["id"] == conversation_id)

    assert convo["title"] == "Fake Smart Title"


def test_generate_smart_title_falls_back_on_llm_failure(monkeypatch):
    """If the summarizer call fails, we fall back to the simple truncation."""

    class BrokenLLM:
        def invoke(self, _messages):
            raise RuntimeError("Groq is down")

    monkeypatch.setattr("app.services.chat_service.llm", BrokenLLM())

    assert generate_smart_title("What is RAG?") is None


def test_list_and_delete_conversation():
    created = client.post("/chat", json={"message": "Temp chat"})
    conversation_id = created.headers["X-Conversation-ID"]

    listing = client.get("/conversations").json()
    assert any(c["id"] == conversation_id for c in listing)

    delete_response = client.delete(f"/conversations/{conversation_id}")
    assert delete_response.status_code == 200

    messages_after_delete = client.get(f"/conversations/{conversation_id}/messages")
    assert messages_after_delete.status_code == 404
