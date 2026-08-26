from app.llm import llm


def test_llm_is_configured_for_streaming():
    """Sanity-check the LLM client config without making a real API call."""

    assert llm.streaming is True
    assert llm.model_name == "openai/gpt-oss-20b"
