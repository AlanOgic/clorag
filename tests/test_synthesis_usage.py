"""Tests for synthesis usage capture (streaming variant)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from clorag.web.search.synthesis import SynthesisResult, synthesize_answer_stream


async def test_streaming_returns_usage_via_result_sink() -> None:
    """After the text stream completes, result_sink callback fires with usage."""
    fake_usage = MagicMock(
        input_tokens=1500,
        output_tokens=400,
        cache_read_input_tokens=1200,
        cache_creation_input_tokens=0,
    )
    fake_final_message = MagicMock(usage=fake_usage)

    async def fake_text_stream():
        for chunk in ["Hello", " world"]:
            yield chunk

    fake_stream = MagicMock()
    fake_stream.text_stream = fake_text_stream()
    fake_stream.get_final_message = AsyncMock(return_value=fake_final_message)

    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
    fake_ctx.__aexit__ = AsyncMock(return_value=None)

    fake_client = MagicMock()
    fake_client.messages.stream = MagicMock(return_value=fake_ctx)

    captured: list[SynthesisResult] = []

    with patch("clorag.web.search.synthesis.get_anthropic", return_value=fake_client):
        chunks = []
        async for item in synthesize_answer_stream(
            query="test query",
            chunks=[{"content": "some context", "source": "doc"}],
            result_sink=captured.append,
        ):
            chunks.append(item)

    assert "".join(chunks) == "Hello world"
    assert len(captured) == 1
    result = captured[0]
    assert result.input_tokens == 1500
    assert result.output_tokens == 400
    assert result.cache_read_tokens == 1200
    assert result.cache_creation_tokens == 0
    assert result.model  # whatever the model setting resolves to


async def test_streaming_works_without_result_sink() -> None:
    """Backward compat — caller doesn't have to provide result_sink."""
    fake_usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    fake_final_message = MagicMock(usage=fake_usage)

    async def fake_text_stream():
        yield "hi"

    fake_stream = MagicMock()
    fake_stream.text_stream = fake_text_stream()
    fake_stream.get_final_message = AsyncMock(return_value=fake_final_message)

    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=fake_stream)
    fake_ctx.__aexit__ = AsyncMock(return_value=None)

    fake_client = MagicMock()
    fake_client.messages.stream = MagicMock(return_value=fake_ctx)

    with patch("clorag.web.search.synthesis.get_anthropic", return_value=fake_client):
        chunks = []
        async for item in synthesize_answer_stream(
            query="q",
            chunks=[{"content": "some context", "source": "doc"}],
        ):
            chunks.append(item)

    assert chunks == ["hi"]
    fake_stream.get_final_message.assert_not_called()
