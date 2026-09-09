import pytest

from app.services.image_gen_service import (
    generate_image,
    looks_like_image_request,
    ImageGenerationError,
)


@pytest.mark.parametrize("message,expected", [
    ("can you generate a dog image?", True),
    ("generate an image of a sunset", True),
    ("create a picture of a robot", True),
    ("please make a logo for my startup", True),
    ("paint an illustration of a forest", True),
    ("draw me a cat", True),
    ("please draw a sunset over mountains", True),
    ("paint a picture of a lighthouse", True),
    ("can you generate a summary of this document?", False),
    ("what is RAG?", False),
    ("generate a random number between 1 and 10", False),
    ("what did you draw from this conclusion?", False),
])
def test_looks_like_image_request(message, expected):
    assert looks_like_image_request(message) is expected


def test_generate_image_returns_bytes(monkeypatch):
    class FakeResponse:
        content = b"fake-jpeg-bytes"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "app.services.image_gen_service.requests.get",
        lambda *a, **k: FakeResponse(),
    )

    result = generate_image("a red robot")
    assert result == b"fake-jpeg-bytes"


def test_generate_image_rejects_empty_prompt():
    with pytest.raises(ImageGenerationError):
        generate_image("   ")


def test_generate_image_wraps_network_errors(monkeypatch):
    import requests

    def broken(*a, **k):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr("app.services.image_gen_service.requests.get", broken)

    with pytest.raises(ImageGenerationError):
        generate_image("a red robot")


def test_generate_image_rejects_empty_response(monkeypatch):
    class EmptyResponse:
        content = b""

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "app.services.image_gen_service.requests.get",
        lambda *a, **k: EmptyResponse(),
    )

    with pytest.raises(ImageGenerationError):
        generate_image("a red robot")
