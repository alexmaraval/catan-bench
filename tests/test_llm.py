from __future__ import annotations

import io
import json
import unittest
from urllib import error
from unittest.mock import patch

from catan_bench.llm import OpenAICompatibleChatClient


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
