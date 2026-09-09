import pytest

from app.services.image_gen_service import generate_image, ImageGenerationError


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
