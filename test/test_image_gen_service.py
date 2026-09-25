import pytest

from app.services.image_gen_service import (
    generate_image,
    looks_like_image_request,
    extract_image_prompt,
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


@pytest.mark.parametrize("message,expected", [
    # These two are the actual bug: Pollinations reliably 500s on prompts
    # that still contain the request's own meta-words ("generate", "image").
    ("generate a cute cat image", "a cute cat"),
    ("generate a red robot image", "a red robot"),
    ("can you generate a dog image?", "a dog"),
    ("draw me a cat", "a cat"),
    ("please draw a sunset over mountains", "a sunset over mountains"),
    ("paint a picture of a lighthouse", "a lighthouse"),
    ("create a picture of a robot", "a robot"),
    ("design an icon of a mountain", "a mountain"),
    # A subject that happens to contain an image-noun word as itself
    # (not "X of Y") should be left alone rather than losing "logo".
    ("please make a logo for my startup", "a logo for my startup"),
    # Already-clean prompts (e.g. from a direct API call) pass through
    # unchanged.
    ("a red robot", "a red robot"),
])
def test_extract_image_prompt(message, expected):
    assert extract_image_prompt(message) == expected


def test_extract_image_prompt_never_returns_empty_string_for_sparse_input():
    # generate_image() itself rejects an empty prompt outright, so
    # extraction must never hand it one for realistic sparse input (an
    # actually-empty message can't reach here — looks_like_image_request
    # never matches empty text in the first place).
    for message in ["draw", "generate", "draw me"]:
        assert extract_image_prompt(message) != ""


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


def test_generate_image_omits_auth_header_without_a_key(monkeypatch):
    """No POLLINATIONS_API_KEY set — the request should be fully anonymous
    (no Authorization header), same as before this was added."""

    monkeypatch.setattr("app.services.image_gen_service.POLLINATIONS_API_KEY", None)

    captured = {}

    class FakeResponse:
        content = b"fake-jpeg-bytes"

        def raise_for_status(self):
            pass

    def fake_get(url, params, headers, timeout):
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("app.services.image_gen_service.requests.get", fake_get)

    generate_image("a red robot")
    assert "Authorization" not in captured["headers"]


def test_generate_image_sends_auth_header_when_key_is_set(monkeypatch):
    monkeypatch.setattr(
        "app.services.image_gen_service.POLLINATIONS_API_KEY", "test-key-123"
    )

    captured = {}

    class FakeResponse:
        content = b"fake-jpeg-bytes"

        def raise_for_status(self):
            pass

    def fake_get(url, params, headers, timeout):
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("app.services.image_gen_service.requests.get", fake_get)

    generate_image("a red robot")
    assert captured["headers"]["Authorization"] == "Bearer test-key-123"


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
