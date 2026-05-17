"""Tests for OpenAI-compatible API endpoint."""

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from clorag.web.routers.openai_compat import router
from clorag.web.search.synthesis import SynthesisResult


def _make_synth_result(
    text: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> SynthesisResult:
    """Create a SynthesisResult with configurable usage for tests."""
    return SynthesisResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        model="claude-sonnet-test",
    )


def _mock_settings(api_key: str | None = "test-key-123") -> MagicMock:
    """Create mock settings with the given API key."""
    settings = MagicMock()
    if api_key is not None:
        settings.openai_compat_api_key = MagicMock(
            get_secret_value=MagicMock(return_value=api_key)
        )
    else:
        settings.openai_compat_api_key = None
    return settings


def _create_test_app() -> FastAPI:
    """Create a minimal FastAPI app with the OpenAI compat router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_create_test_app())


VALID_HEADERS = {"Authorization": "Bearer test-key-123"}
SETTINGS_PATCH = "clorag.web.routers.openai_compat.get_settings"
SEARCH_PATCH = "clorag.web.routers.openai_compat.perform_search"
SYNTH_PATCH = "clorag.web.routers.openai_compat.synthesize_answer"
STREAM_PATCH = "clorag.web.routers.openai_compat.synthesize_answer_stream"


class TestAuth:
    """Test Bearer token authentication."""

    def test_missing_auth_header(self, client: TestClient) -> None:
        with patch(SETTINGS_PATCH, return_value=_mock_settings()):
            resp = client.post("/v1/chat/completions", json={
                "model": "clorag",
                "messages": [{"role": "user", "content": "hello"}],
            })
        assert resp.status_code == 401
        assert "authorization" in resp.json()["error"]["message"].lower()

    def test_wrong_api_key(self, client: TestClient) -> None:
        with patch(SETTINGS_PATCH, return_value=_mock_settings("correct-key")):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "clorag", "messages": [{"role": "user", "content": "hello"}]},
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert resp.status_code == 401

    def test_api_key_not_configured(self, client: TestClient) -> None:
        with patch(SETTINGS_PATCH, return_value=_mock_settings(None)):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "clorag", "messages": [{"role": "user", "content": "hello"}]},
                headers={"Authorization": "Bearer anything"},
            )
        assert resp.status_code == 503


class TestModelsEndpoint:
    """Test GET /v1/models."""

    def test_list_models(self, client: TestClient) -> None:
        with patch(SETTINGS_PATCH, return_value=_mock_settings()):
            resp = client.get("/v1/models", headers=VALID_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "clorag"
        assert body["data"][0]["object"] == "model"
        assert body["data"][0]["owned_by"] == "cyanview"

    def test_models_requires_auth(self, client: TestClient) -> None:
        with patch(SETTINGS_PATCH, return_value=_mock_settings()):
            resp = client.get("/v1/models")
        assert resp.status_code == 401


class TestChatCompletions:
    """Test POST /v1/chat/completions (non-streaming)."""

    def _mock_search_and_synth(self) -> ExitStack:
        """Context manager that patches search + synthesis."""
        mock_chunks = [
            {"text": "RIO config guide", "source_type": "documentation",
             "url": "https://support.cyanview.com/rio", "title": "RIO Guide", "score": 0.9}
        ]
        mock_results = [MagicMock(score=0.9, source="documentation", metadata={})]

        search_mock = AsyncMock(return_value=(mock_results, mock_chunks, None, True))
        synth_mock = AsyncMock(return_value=_make_synth_result("Here is how to configure RIO."))

        stack = ExitStack()
        stack.enter_context(patch(SETTINGS_PATCH, return_value=_mock_settings()))
        stack.enter_context(patch(SEARCH_PATCH, search_mock))
        stack.enter_context(patch(SYNTH_PATCH, synth_mock))
        return stack

    def test_valid_request_returns_openai_format(self, client: TestClient) -> None:
        with self._mock_search_and_synth():
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "clorag", "messages": [{"role": "user", "content": "How to configure RIO?"}]},
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200
        body = resp.json()
        # Required root fields per OpenAI SDK
        assert body["id"].startswith("chatcmpl-")
        assert body["object"] == "chat.completion"
        assert isinstance(body["created"], int)
        assert body["model"] == "clorag"
        # Choices
        assert len(body["choices"]) == 1
        choice = body["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert "configure RIO" in choice["message"]["content"]
        assert choice["finish_reason"] == "stop"
        # Usage present
        assert "usage" in body
        assert "prompt_tokens" in body["usage"]

    def test_sources_appended_to_response(self, client: TestClient) -> None:
        with self._mock_search_and_synth():
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "clorag", "messages": [{"role": "user", "content": "RIO config"}]},
                headers=VALID_HEADERS,
            )
        content = resp.json()["choices"][0]["message"]["content"]
        assert "Sources:" in content
        assert "RIO Guide" in content

    def test_empty_messages_returns_400(self, client: TestClient) -> None:
        with patch(SETTINGS_PATCH, return_value=_mock_settings()):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "clorag", "messages": []},
                headers=VALID_HEADERS,
            )
        assert resp.status_code == 400

    def test_no_user_message_returns_400(self, client: TestClient) -> None:
        with patch(SETTINGS_PATCH, return_value=_mock_settings()):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "clorag", "messages": [{"role": "system", "content": "You are helpful"}]},
                headers=VALID_HEADERS,
            )
        assert resp.status_code == 400

    def test_conversation_history_forwarded(self, client: TestClient) -> None:
        mock_chunks = [
            {"text": "info", "source_type": "documentation",
             "url": "https://example.com", "title": "Doc", "score": 0.8}
        ]
        synth_mock = AsyncMock(return_value=_make_synth_result("Follow-up answer."))

        with (
            patch(SETTINGS_PATCH, return_value=_mock_settings()),
            patch(SEARCH_PATCH, new_callable=AsyncMock, return_value=([], mock_chunks, None, True)),
            patch(SYNTH_PATCH, synth_mock),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "clorag",
                    "messages": [
                        {"role": "user", "content": "What is RIO?"},
                        {"role": "assistant", "content": "RIO is a hardware device."},
                        {"role": "user", "content": "How to configure it?"},
                    ],
                },
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200
        # Verify conversation history was passed to synthesize_answer
        synth_mock.assert_called_once()
        call_args = synth_mock.call_args
        # synthesize_answer(query, chunks, conversation_history, graph_context)
        conv_history = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("conversation_history")
        assert conv_history is not None
        assert len(conv_history) == 2  # 1 prior user + 1 prior assistant

    def test_system_messages_ignored_in_history(self, client: TestClient) -> None:
        """System messages should not appear in conversation history."""
        mock_chunks = [
            {"text": "info", "source_type": "documentation",
             "url": "https://example.com", "title": "Doc", "score": 0.8}
        ]
        synth_mock = AsyncMock(return_value=_make_synth_result("Answer."))

        with (
            patch(SETTINGS_PATCH, return_value=_mock_settings()),
            patch(SEARCH_PATCH, new_callable=AsyncMock, return_value=([], mock_chunks, None, True)),
            patch(SYNTH_PATCH, synth_mock),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "clorag",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": "Hello"},
                    ],
                },
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200
        synth_mock.assert_called_once()
        # conversation_history should be None (no prior user/assistant pairs)
        conv_history = synth_mock.call_args[0][2]
        assert conv_history is None

    def test_extra_fields_accepted(self, client: TestClient) -> None:
        """Extra OpenAI fields (temperature, top_p, etc.) are accepted without error."""
        with self._mock_search_and_synth():
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "clorag",
                    "messages": [{"role": "user", "content": "test"}],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500,
                    "max_completion_tokens": 500,
                    "frequency_penalty": 0.5,
                    "presence_penalty": 0.5,
                    "n": 1,
                    "user": "test-user",
                },
                headers=VALID_HEADERS,
            )
        assert resp.status_code == 200


class TestChatCompletionsStreaming:
    """Test POST /v1/chat/completions with stream=true."""

    def test_streaming_response_format(self, client: TestClient) -> None:
        mock_chunks = [
            {"text": "info", "source_type": "documentation",
             "url": "https://example.com", "title": "Doc", "score": 0.8}
        ]

        async def mock_stream(*args, **kwargs):
            yield "Hello "
            yield "world"

        with (
            patch(SETTINGS_PATCH, return_value=_mock_settings()),
            patch(SEARCH_PATCH, new_callable=AsyncMock, return_value=([], mock_chunks, None, True)),
            patch(STREAM_PATCH, side_effect=mock_stream),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "clorag",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                },
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        # Parse SSE events
        import json
        lines = resp.text.strip().split("\n\n")
        events = []
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line.removeprefix("data: ")))

        # First event: role chunk
        assert events[0]["object"] == "chat.completion.chunk"
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"

        # Content chunks present
        content_events = [e for e in events if e["choices"][0]["delta"].get("content")]
        assert len(content_events) > 0

        # Last event: finish_reason stop
        last_event = events[-1]
        assert last_event["choices"][0]["finish_reason"] == "stop"

        # Stream ends with [DONE]
        assert resp.text.strip().endswith("data: [DONE]")


class TestUsageReporting:
    """Test real token usage in non-streaming and streaming responses."""

    def test_non_streaming_usage_real_values(self, client: TestClient) -> None:
        """Non-streaming response should report real token counts."""
        mock_chunks = [
            {"text": "RIO config guide", "source_type": "documentation",
             "url": "https://support.cyanview.com/rio", "title": "RIO Guide", "score": 0.9}
        ]
        # input=800, cache_read=0, cache_creation=600 → prompt_tokens=1400
        # output=250 → completion_tokens=250, total=1650
        synth_mock = AsyncMock(return_value=_make_synth_result(
            "Here is how to configure RIO.",
            input_tokens=800,
            output_tokens=250,
            cache_read_tokens=0,
            cache_creation_tokens=600,
        ))

        with (
            patch(SETTINGS_PATCH, return_value=_mock_settings()),
            patch(SEARCH_PATCH, new_callable=AsyncMock,
                  return_value=([], mock_chunks, None, True)),
            patch(SYNTH_PATCH, synth_mock),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "clorag", "messages": [{"role": "user", "content": "How to configure RIO?"}]},
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] == 1400   # 800 + 0 + 600
        assert usage["completion_tokens"] == 250
        assert usage["total_tokens"] == 1650

    def test_streaming_include_usage_emits_usage_chunk(self, client: TestClient) -> None:
        """Streaming with include_usage=true should emit a final usage chunk before [DONE]."""
        mock_chunks = [
            {"text": "info", "source_type": "documentation",
             "url": "https://example.com", "title": "Doc", "score": 0.8}
        ]

        # Synthesize result with known token counts
        usage_result = _make_synth_result(
            "",
            input_tokens=500,
            output_tokens=120,
            cache_read_tokens=200,
            cache_creation_tokens=0,
        )

        async def mock_stream(*args, result_sink=None, **kwargs):
            yield "Hello "
            yield "world"
            if result_sink is not None:
                result_sink(usage_result)

        with (
            patch(SETTINGS_PATCH, return_value=_mock_settings()),
            patch(SEARCH_PATCH, new_callable=AsyncMock, return_value=([], mock_chunks, None, True)),
            patch(STREAM_PATCH, side_effect=mock_stream),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "clorag",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200

        import json as _json
        raw_lines = resp.text.strip().split("\n\n")

        # Last line must be [DONE]
        assert raw_lines[-1] == "data: [DONE]"

        # Penultimate data line should be the usage chunk (empty choices)
        usage_line = raw_lines[-2]
        assert usage_line.startswith("data: ")
        usage_chunk = _json.loads(usage_line.removeprefix("data: "))
        assert usage_chunk["choices"] == []
        assert "usage" in usage_chunk
        # prompt_tokens = input(500) + cache_read(200) + cache_creation(0) = 700
        assert usage_chunk["usage"]["prompt_tokens"] == 700
        assert usage_chunk["usage"]["completion_tokens"] == 120
        assert usage_chunk["usage"]["total_tokens"] == 820

    def test_streaming_without_include_usage_no_extra_chunk(self, client: TestClient) -> None:
        """Streaming without include_usage should NOT emit an extra usage chunk."""
        mock_chunks = [
            {"text": "info", "source_type": "documentation",
             "url": "https://example.com", "title": "Doc", "score": 0.8}
        ]

        async def mock_stream(*args, result_sink=None, **kwargs):
            yield "Hello"
            # result_sink should be None here — but call it anyway to confirm no chunk emitted
            if result_sink is not None:
                result_sink(_make_synth_result("", input_tokens=100, output_tokens=50))

        with (
            patch(SETTINGS_PATCH, return_value=_mock_settings()),
            patch(SEARCH_PATCH, new_callable=AsyncMock, return_value=([], mock_chunks, None, True)),
            patch(STREAM_PATCH, side_effect=mock_stream),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "clorag",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                },
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200

        import json as _json
        raw_lines = resp.text.strip().split("\n\n")

        # Last line is [DONE]
        assert raw_lines[-1] == "data: [DONE]"

        # Penultimate chunk is the finish_reason=stop chunk (has choices), not a usage chunk
        penultimate = _json.loads(raw_lines[-2].removeprefix("data: "))
        assert len(penultimate["choices"]) > 0
        assert "usage" not in penultimate
