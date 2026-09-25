import re
import urllib.parse

import requests

from app.config import POLLINATIONS_API_KEY

# Pollinations.ai — free, keyless text-to-image API. A plain GET request
# with the prompt URL-encoded into the path returns image bytes directly.
# No API key or account needed for this project's usage level; see
# https://github.com/pollinations/pollinations/blob/main/APIDOCS.md
IMAGE_GEN_URL = "https://image.pollinations.ai/prompt/{prompt}"

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 768


# Simple keyword heuristic so typing "can you generate a dog image?" into
# the normal chat box works too, not just the dedicated "Generate an image"
# panel — requires an action verb AND an image-ish noun so plain questions
# like "generate a summary" don't misfire. It's not real intent
# understanding, so an informational question like "how do diffusion
# models generate images" will still (rarely) trigger generation instead
# of being answered — an accepted tradeoff for a fast, free heuristic
# instead of spending an extra LLM call just to classify every message.
IMAGE_REQUEST_PATTERN = re.compile(
    r"\b(generate|create|draw|paint|make|produce|design)\b"
    r".{0,40}?"
    r"\b(image|picture|photo|drawing|painting|illustration|artwork|logo|icon)s?\b",
    re.IGNORECASE,
)

# "draw"/"paint" alone (no separate "picture"/"image" noun needed) are
# unambiguous enough as commands — "draw me a cat", "paint a sunset" — to
# match without the stricter two-word pairing above.
DRAW_OR_PAINT_COMMAND_PATTERN = re.compile(
    r"^\s*(?:can you |could you |please )?(draw|paint)\b",
    re.IGNORECASE,
)


def looks_like_image_request(message: str) -> bool:
    """Heuristic: does this message look like a request to generate an image?"""
    return bool(
        IMAGE_REQUEST_PATTERN.search(message)
        or DRAW_OR_PAINT_COMMAND_PATTERN.search(message)
    )


# Pollinations reliably 500s on prompts that still contain the *meta* words
# from the request itself ("generate", "image", etc.) rather than just the
# subject — confirmed directly: "generate a cute cat image" and "generate a
# red robot image" failed consistently, while the bare subjects ("a cute
# cat", "a red robot") succeeded every time. This isn't random flakiness —
# it's the raw chat message being sent as the literal prompt. Strip the
# request phrasing down to just the subject before ever calling the API.
_LEADING_POLITE_WRAPPER = re.compile(
    r"^(can you|could you|please|would you)\s+", re.IGNORECASE
)
_LEADING_ACTION_VERB = re.compile(
    r"^(generate|create|draw|paint|make|produce|design)\s+(me\s+)?",
    re.IGNORECASE,
)
_TRAILING_IMAGE_NOUN = re.compile(
    r"\s+(image|picture|photo|drawing|painting|illustration|artwork)s?[?.!]*$",
    re.IGNORECASE,
)
_LEADING_IMAGE_NOUN_OF = re.compile(
    r"^(an?\s+)?(image|picture|photo|drawing|painting|illustration|artwork|logo|icon)s?"
    r"\s+(of|showing|depicting)\s+",
    re.IGNORECASE,
)


def extract_image_prompt(message: str) -> str:
    """Strip generation-request phrasing down to just the image's subject.

    e.g. "generate a cute cat image" -> "a cute cat"
         "can you draw me a sunset?" -> "a sunset"
         "paint a picture of a lighthouse" -> "a lighthouse"

    Falls back to the original message if stripping would leave nothing
    (better to send something than an empty prompt).
    """

    text = message.strip()
    text = _LEADING_POLITE_WRAPPER.sub("", text)
    text = _LEADING_ACTION_VERB.sub("", text)
    text = _TRAILING_IMAGE_NOUN.sub("", text)
    text = _LEADING_IMAGE_NOUN_OF.sub("", text)
    text = text.strip().rstrip("?.!").strip()

    return text or message


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

    headers = {"User-Agent": "gen-ai-chatbot/1.0"}

    # nologo=true only actually removes the watermark for a registered
    # account — anonymous/keyless requests get it stamped on regardless of
    # this param. Free key: https://enter.pollinations.ai. Also generally
    # gets a higher rate limit than fully anonymous requests.
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    try:
        response = requests.get(
            url,
            params={
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "nologo": "true",
            },
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as exc:
        raise ImageGenerationError(f"Image generation request failed: {exc}") from exc

    image_bytes = response.content

    if not image_bytes:
        raise ImageGenerationError("Image generation returned an empty response.")

    return image_bytes
