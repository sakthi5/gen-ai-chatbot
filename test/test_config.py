from app.config import GROQ_API_KEY


def test_groq_api_key_is_loaded():
    """The Groq API key should be loaded from the environment/.env file."""

    assert GROQ_API_KEY, "GROQ_API_KEY is missing — check your .env file."
    assert isinstance(GROQ_API_KEY, str)
