from __future__ import annotations

import io
import json
import unittest
from urllib import error
from unittest.mock import patch

from catan_bench.llm import LLMRequestTooLargeError, OpenAICompatibleChatClient


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class OpenAICompatibleChatClientTests(unittest.TestCase):
    def test_complete_uses_openai_reasoning_field_by_default(self) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(api_key_env="OPENAI_API_KEY")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="fake-model",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_enabled=False,
                )

        self.assertEqual(captured_body["reasoning"], {"enabled": False})
        self.assertNotIn("include_reasoning", captured_body)

    def test_complete_uses_groq_include_reasoning_field(self) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
        )

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="fake-model",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_enabled=False,
                )

        self.assertEqual(captured_body["include_reasoning"], False)
        self.assertNotIn("reasoning", captured_body)

    def test_complete_retries_retryable_http_errors(self) -> None:
        attempts: list[str] = []

        def fake_urlopen(req, timeout):
            attempts.append("called")
            if len(attempts) == 1:
                raise error.HTTPError(
                    req.full_url,
                    429,
                    "Too Many Requests",
                    hdrs=None,
                    fp=io.BytesIO(b'{"error":"rate_limited"}'),
                )
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_key_env="OPENAI_API_KEY",
            max_attempts=3,
            retry_backoff_seconds=0.01,
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                with patch("catan_bench.llm.time.sleep") as sleep_mock:
                    response = client.complete(
                        model="fake-model",
                        messages=[{"role": "user", "content": "{}"}],
                        temperature=0.1,
                    )

        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleep_mock.call_count, 1)
        self.assertIn("choices", response)

    def test_complete_does_not_retry_non_retryable_http_errors(self) -> None:
        attempts: list[str] = []

        def fake_urlopen(req, timeout):
            attempts.append("called")
            raise error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"bad_request"}'),
            )

        client = OpenAICompatibleChatClient(
            api_key_env="OPENAI_API_KEY",
            max_attempts=3,
            retry_backoff_seconds=0.01,
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                with patch("catan_bench.llm.time.sleep") as sleep_mock:
                    with self.assertRaises(RuntimeError):
                        client.complete(
                            model="fake-model",
                            messages=[{"role": "user", "content": "{}"}],
                            temperature=0.1,
                        )

        self.assertEqual(len(attempts), 1)
        self.assertEqual(sleep_mock.call_count, 0)

    def test_complete_raises_specific_error_for_payload_too_large(self) -> None:
        def fake_urlopen(req, timeout):
            raise error.HTTPError(
                req.full_url,
                413,
                "Payload Too Large",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"request_too_large"}'),
            )

        client = OpenAICompatibleChatClient(
            api_key_env="OPENAI_API_KEY",
            max_attempts=3,
            retry_backoff_seconds=0.01,
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                with self.assertRaises(LLMRequestTooLargeError):
                    client.complete(
                        model="fake-model",
                        messages=[{"role": "user", "content": "{}"}],
                        temperature=0.1,
                    )

    def test_complete_retries_without_json_mode_when_provider_rejects_it(self) -> None:
        captured_bodies: list[dict[str, object]] = []

        def fake_urlopen(req, timeout):
            captured_bodies.append(json.loads(req.data.decode("utf-8")))
            if len(captured_bodies) == 1:
                raise error.HTTPError(
                    req.full_url,
                    400,
                    "Bad Request",
                    hdrs=None,
                    fp=io.BytesIO(
                        json.dumps(
                            {
                                "error": {
                                    "message": (
                                        "Failed to validate JSON. Please adjust your prompt."
                                    ),
                                    "code": "json_validate_failed",
                                }
                            }
                        ).encode("utf-8")
                    ),
                )
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(api_key_env="OPENAI_API_KEY")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                response = client.complete(
                    model="fake-model",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                )

        self.assertEqual(len(captured_bodies), 2)
        self.assertIn("response_format", captured_bodies[0])
        self.assertNotIn("response_format", captured_bodies[1])
        self.assertIn("choices", response)
