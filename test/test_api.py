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


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, chunks=("Hello ", "world!")):
        self.last_stream_messages = None
        self.chunks = chunks

    def stream(self, messages):
        self.last_stream_messages = messages
        for chunk in self.chunks:
            yield FakeChunk(chunk)

    def invoke(self, _messages):
        # Stands in for the LLM-generated title call.
        return FakeChunk("Fake Smart Title")


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """Avoid real Groq API calls in tests by stubbing both LLMs entirely."""

    fake = FakeLLM(("Hello ", "world!"))
    fake_vision = FakeLLM(("It shows ", "a red robot."))
    monkeypatch.setattr("app.services.chat_service.llm", fake)
    monkeypatch.setattr("app.services.chat_service.vision_llm", fake_vision)
    fake.vision = fake_vision
    return fake


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


def test_chat_shows_friendly_message_on_rate_limit_error(monkeypatch):
    class RateLimitedLLM:
        def stream(self, _messages):
            raise RuntimeError(
                "Error code: 413 - {'error': {'message': 'Request too large "
                "... on tokens per minute (TPM)', 'code': 'rate_limit_exceeded'}}"
            )
            yield  # pragma: no cover - unreachable, makes this a generator

    monkeypatch.setattr("app.services.chat_service.llm", RateLimitedLLM())

    response = client.post("/chat", json={"message": "Explain this document"})

    assert response.status_code == 200
    assert "too large for the current API plan" in response.text
    assert "rate_limit_exceeded" not in response.text  # no raw error dumped


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


def test_create_empty_conversation():
    response = client.post("/conversations")
    assert response.status_code == 200
    assert response.json()["title"] == "New Chat"


def test_upload_document_and_list_it():
    conversation_id = client.post("/conversations").json()["id"]

    upload = client.post(
        f"/conversations/{conversation_id}/documents",
        files={"file": ("notes.txt", b"The secret word is banana.", "text/plain")},
    )

    assert upload.status_code == 200
    body = upload.json()
    assert body["filename"] == "notes.txt"
    assert body["characters_extracted"] == len("The secret word is banana.")

    documents = client.get(f"/conversations/{conversation_id}/documents").json()
    assert len(documents) == 1
    assert documents[0]["filename"] == "notes.txt"


def test_uploading_the_same_document_twice_does_not_duplicate_it():
    """A user re-attaching the same file (e.g. after nothing seemed to
    happen the first time) shouldn't end up with the same document twice
    in context."""

    conversation_id = client.post("/conversations").json()["id"]
    file_args = {"file": ("notes.txt", b"The secret word is banana.", "text/plain")}

    first = client.post(f"/conversations/{conversation_id}/documents", files=file_args)
    second = client.post(f"/conversations/{conversation_id}/documents", files=file_args)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]

    documents = client.get(f"/conversations/{conversation_id}/documents").json()
    assert len(documents) == 1


def test_upload_unsupported_document_type_returns_400():
    conversation_id = client.post("/conversations").json()["id"]

    upload = client.post(
        f"/conversations/{conversation_id}/documents",
        files={"file": ("archive.zip", b"whatever", "application/zip")},
    )

    assert upload.status_code == 400


def test_upload_document_to_unknown_conversation_returns_404():
    upload = client.post(
        "/conversations/does-not-exist/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert upload.status_code == 404


def test_chat_includes_attached_document_in_llm_context(fake_llm):
    conversation_id = client.post("/conversations").json()["id"]

    client.post(
        f"/conversations/{conversation_id}/documents",
        files={"file": ("notes.txt", b"The secret word is banana.", "text/plain")},
    )

    client.post(
        "/chat",
        json={"message": "What's the secret word?", "conversation_id": conversation_id},
    )

    sent_contents = [m.content for m in fake_llm.last_stream_messages]
    assert any("The secret word is banana." in c for c in sent_contents)


# --- Vision (image understanding) ---------------------------------------

TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_chat_with_image_routes_to_vision_model(fake_llm):
    response = client.post(
        "/chat",
        json={
            "message": "What color is this?",
            "images": [
                {
                    "filename": "test.png",
                    "mime_type": "image/png",
                    "data_base64": TINY_PNG_BASE64,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.text == "It shows a red robot."  # from fake_llm.vision, not fake_llm

    # The regular text model should never have been called for this turn.
    assert fake_llm.last_stream_messages is None


def test_chat_with_image_strips_think_tags_and_persists_clean_text(monkeypatch):
    thinking_llm = FakeLLM(("<think>hmm let me look</think>", "It's a cat."))
    monkeypatch.setattr("app.services.chat_service.vision_llm", thinking_llm)

    response = client.post(
        "/chat",
        json={
            "message": "What is this?",
            "images": [{
                "filename": "test.png",
                "mime_type": "image/png",
                "data_base64": TINY_PNG_BASE64,
            }],
        },
    )
    conversation_id = response.headers["X-Conversation-ID"]

    # The <think> block shouldn't reach the client...
    assert "<think>" not in response.text
    assert response.text.strip() == "It's a cat."

    # ...nor get saved to the DB.
    messages = client.get(f"/conversations/{conversation_id}/messages").json()
    assistant_reply = next(m for m in messages if m["role"] == "assistant")
    assert "<think>" not in assistant_reply["content"]


def test_uploaded_image_is_replayed_in_later_turns(fake_llm):
    first = client.post(
        "/chat",
        json={
            "message": "What is in this image?",
            "images": [{
                "filename": "test.png",
                "mime_type": "image/png",
                "data_base64": TINY_PNG_BASE64,
            }],
        },
    )
    conversation_id = first.headers["X-Conversation-ID"]

    # A follow-up with no new image should still route to the vision model
    # and still carry the original image in history.
    client.post(
        "/chat",
        json={"message": "Are you sure?", "conversation_id": conversation_id},
    )

    sent = fake_llm.vision.last_stream_messages
    human_messages_with_images = [
        m for m in sent
        if isinstance(m.content, list)
        and any(block.get("type") == "image_url" for block in m.content)
    ]
    assert len(human_messages_with_images) == 1


# --- Image generation -----------------------------------------------------

def test_generate_image_creates_conversation_and_saves_image(monkeypatch):
    monkeypatch.setattr(
        "app.api.generate_and_save_image",
        lambda db, conversation_id, prompt: _fake_generate_and_save_image(db, conversation_id, prompt),
    )

    response = client.post("/generate-image", json={"prompt": "a red robot"})

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"]
    assert body["image"]["mime_type"] == "image/jpeg"
    assert body["image"]["data_base64"] == "ZmFrZS1pbWFnZS1ieXRlcw=="


def _fake_generate_and_save_image(db, conversation_id, prompt):
    from app.models import Message, Image

    assistant_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=f'Here\'s the image I generated for: "{prompt}"',
    )
    db.add(assistant_message)
    db.flush()

    image = Image(
        conversation_id=conversation_id,
        message_id=assistant_message.id,
        source="generated",
        filename="fake.jpg",
        mime_type="image/jpeg",
        data_base64="ZmFrZS1pbWFnZS1ieXRlcw==",  # base64("fake-image-bytes")
        prompt=prompt,
    )
    db.add(image)
    db.commit()
    db.refresh(assistant_message)
    db.refresh(image)
    return assistant_message, image


def test_generate_image_rejects_empty_prompt():
    response = client.post("/generate-image", json={"prompt": "   "})
    assert response.status_code == 400


def test_generate_image_returns_502_on_generation_failure(monkeypatch):
    from app.services.image_gen_service import ImageGenerationError

    def broken(db, conversation_id, prompt):
        raise ImageGenerationError("upstream is down")

    monkeypatch.setattr("app.api.generate_and_save_image", broken)

    response = client.post("/generate-image", json={"prompt": "anything"})
    assert response.status_code == 502


# --- Natural-language image requests via /chat -----------------------

def test_chat_detects_image_request_and_generates_instead_of_replying(monkeypatch):
    monkeypatch.setattr(
        "app.api.generate_and_save_image",
        lambda db, conversation_id, prompt: _fake_generate_and_save_image(
            db, conversation_id, prompt
        ),
    )

    response = client.post("/chat", json={"message": "can you generate a dog image?"})

    assert response.status_code == 200
    assert response.headers.get("X-Image-Generated") == "true"
    assert "dog image" in response.text  # the canned confirmation echoes the prompt

    conversation_id = response.headers["X-Conversation-ID"]
    messages = client.get(f"/conversations/{conversation_id}/messages").json()
    assistant_reply = next(m for m in messages if m["role"] == "assistant")
    assert len(assistant_reply["images"]) == 1


def test_chat_with_image_request_and_attached_image_prefers_vision(fake_llm):
    """Attaching an image is a vision question, not a generation request,
    even if the wording also happens to match the image-request heuristic."""

    response = client.post(
        "/chat",
        json={
            "message": "can you describe this image?",
            "images": [{
                "filename": "test.png",
                "mime_type": "image/png",
                "data_base64": TINY_PNG_BASE64,
            }],
        },
    )

    assert response.headers.get("X-Image-Generated") is None
    assert response.text == "It shows a red robot."  # from the vision fake, not image-gen


def test_chat_falls_back_to_normal_reply_when_image_generation_fails(monkeypatch, fake_llm):
    from app.services.image_gen_service import ImageGenerationError

    def broken(db, conversation_id, prompt):
        raise ImageGenerationError("upstream is down")

    monkeypatch.setattr("app.api.generate_and_save_image", broken)

    response = client.post("/chat", json={"message": "please draw a sunset"})

    assert response.status_code == 200
    assert response.headers.get("X-Image-Generated") is None
    assert response.text == "Hello world!"  # normal fake_llm reply, not an error


def test_chat_with_ordinary_message_does_not_trigger_image_generation(fake_llm):
    response = client.post("/chat", json={"message": "What is RAG?"})

    assert response.headers.get("X-Image-Generated") is None
    assert response.text == "Hello world!"


# --- Document export --------------------------------------------------

def test_export_conversation_as_docx():
    conversation_id = client.post("/chat", json={"message": "Hello there"}).headers[
        "X-Conversation-ID"
    ]

    response = client.get(f"/conversations/{conversation_id}/export?format=docx")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert len(response.content) > 0


def test_export_conversation_as_pdf():
    conversation_id = client.post("/chat", json={"message": "Hello there"}).headers[
        "X-Conversation-ID"
    ]

    response = client.get(f"/conversations/{conversation_id}/export?format=pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")


def test_export_conversation_rejects_bad_format():
    conversation_id = client.post("/chat", json={"message": "Hello there"}).headers[
        "X-Conversation-ID"
    ]

    response = client.get(f"/conversations/{conversation_id}/export?format=exe")
    assert response.status_code == 400


def test_export_unknown_conversation_returns_404():
    response = client.get("/conversations/does-not-exist/export?format=docx")
    assert response.status_code == 404
