import urllib.parse

import requests

# Pollinations.ai — free, keyless text-to-image API. A plain GET request
# with the prompt URL-encoded into the path returns image bytes directly.
# No API key or account needed for this project's usage level; see
# https://github.com/pollinations/pollinations/blob/main/APIDOCS.md
IMAGE_GEN_URL = "https://image.pollinations.ai/prompt/{prompt}"

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 768


class ImageGenerationError(RuntimeError):
    """Raised when the image generation API fails or returns something unusable."""


def generate_image(prompt: str) -> bytes:
    """Generate an image from a text prompt and return the raw image bytes.

    Raises ImageGenerationError on any failure (network issue, bad status,
    empty response) so the caller can turn it into a clean error response
    instead of a raw exception leaking through.
    """

    if not prompt or not prompt.strip():
        raise ImageGenerationError("Prompt cannot be empty.")

    encoded_prompt = urllib.parse.quote(prompt.strip())
    url = IMAGE_GEN_URL.format(prompt=encoded_prompt)

    try:
        response = requests.get(
            url,
            params={
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "nologo": "true",
            },
            headers={"User-Agent": "gen-ai-chatbot/1.0"},
            timeout=60,
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as exc:
        raise ImageGenerationError(f"Image generation request failed: {exc}") from exc

    image_bytes = response.content

    if not image_bytes:
        raise ImageGenerationError("Image generation returned an empty response.")

    return image_bytes
