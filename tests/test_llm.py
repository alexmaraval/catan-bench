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

    def test_complete_uses_reasoning_effort_none_for_supported_google_models(
        self,
    ) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env="GEMINI_API_KEY",
        )

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="gemini-2.5-flash",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_enabled=False,
                )

        self.assertEqual(captured_body["reasoning_effort"], "none")
        self.assertNotIn("reasoning", captured_body)
        self.assertNotIn("include_reasoning", captured_body)

    def test_complete_omits_reasoning_field_for_unsupported_google_models(self) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env="GEMINI_API_KEY",
        )

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="gemini-2.0-flash",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_enabled=False,
                )

        self.assertNotIn("reasoning", captured_body)
        self.assertNotIn("reasoning_effort", captured_body)
        self.assertNotIn("include_reasoning", captured_body)

    def test_complete_omits_reasoning_field_for_together_when_disabled(self) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://api.together.ai/v1",
            api_key_env="TOGETHER_API_KEY",
        )

        with patch.dict("os.environ", {"TOGETHER_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="moonshotai/Kimi-K2.5",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_enabled=False,
                )

        self.assertNotIn("reasoning", captured_body)
        self.assertNotIn("reasoning_effort", captured_body)
        self.assertNotIn("include_reasoning", captured_body)

    def test_complete_uses_reasoning_effort_none_for_openrouter_when_disabled(
        self,
    ) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="openai/gpt-5-mini",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_enabled=False,
                )

        self.assertEqual(captured_body["reasoning"], {"effort": "none"})
        self.assertNotIn("reasoning_effort", captured_body)
        self.assertNotIn("include_reasoning", captured_body)

    def test_complete_enables_reasoning_for_openrouter_when_requested(self) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="openai/gpt-5-mini",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_enabled=True,
                )

        self.assertEqual(captured_body["reasoning"], {"enabled": True})
        self.assertNotIn("reasoning_effort", captured_body)
        self.assertNotIn("include_reasoning", captured_body)

    def test_complete_uses_explicit_reasoning_effort_for_openrouter(self) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_effort="low",
                )

        self.assertEqual(captured_body["reasoning"], {"effort": "low"})
        self.assertNotIn("reasoning_effort", captured_body)
        self.assertNotIn("include_reasoning", captured_body)

    def test_complete_includes_openrouter_provider_preferences(self) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            request_provider={
                "require_parameters": True,
                "ignore": ["deepinfra"],
                "sort": "throughput",
            },
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="stepfun/step-3.5-flash",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                )

        self.assertEqual(
            captured_body["provider"],
            {
                "require_parameters": True,
                "ignore": ["deepinfra"],
                "sort": "throughput",
            },
        )

    def test_complete_omits_response_format_when_disabled(self) -> None:
        captured_body: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            json_response_format=False,
            request_provider={"require_parameters": True},
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                client.complete(
                    model="stepfun/step-3.5-flash",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                )

        self.assertNotIn("response_format", captured_body)
        self.assertEqual(captured_body["provider"], {"require_parameters": True})

    def test_complete_rejects_provider_preferences_for_non_openrouter_api(self) -> None:
        client = OpenAICompatibleChatClient(
            api_key_env="OPENAI_API_KEY",
            request_provider={"ignore": ["deepinfra"]},
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "only supported when `api_base` targets OpenRouter"
            ):
                client.complete(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                )

    def test_complete_rejects_reasoning_enabled_and_effort_together(self) -> None:
        client = OpenAICompatibleChatClient(api_key_env="OPENAI_API_KEY")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "either `reasoning_enabled` or `reasoning_effort`"
            ):
                client.complete(
                    model="fake-model",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                    reasoning_enabled=False,
                    reasoning_effort="low",
                )

    def test_complete_rejects_openai_website_api_base_with_helpful_error(self) -> None:
        client = OpenAICompatibleChatClient(
            api_base="https://www.openai.com/completions/v1",
            api_key_env="OPENAI_API_KEY",
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "Invalid OpenAI `api_base`.*https://api.openai.com/v1",
            ):
                client.complete(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                )

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

    def test_complete_retries_without_json_mode_on_405_with_nested_provider_error(
        self,
    ) -> None:
        captured_bodies: list[dict[str, object]] = []

        def fake_urlopen(req, timeout):
            captured_bodies.append(json.loads(req.data.decode("utf-8")))
            if len(captured_bodies) == 1:
                raise error.HTTPError(
                    req.full_url,
                    405,
                    "Method Not Allowed",
                    hdrs=None,
                    fp=io.BytesIO(
                        json.dumps(
                            {
                                "error": {
                                    "message": "Provider returned error",
                                    "code": 405,
                                    "metadata": {
                                        "raw": json.dumps(
                                            {
                                                "detail": (
                                                    "json_object response format is not "
                                                    "supported for model: "
                                                    "stepfun-ai/Step-3.5-Flash"
                                                )
                                            }
                                        )
                                    },
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
                    model="stepfun/step-3.5-flash",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                )

        self.assertEqual(len(captured_bodies), 2)
        self.assertIn("response_format", captured_bodies[0])
        self.assertNotIn("response_format", captured_bodies[1])
        self.assertIn("choices", response)

    def test_complete_retries_without_json_mode_on_404_missing_parameter_endpoint(
        self,
    ) -> None:
        captured_bodies: list[dict[str, object]] = []

        def fake_urlopen(req, timeout):
            captured_bodies.append(json.loads(req.data.decode("utf-8")))
            if len(captured_bodies) == 1:
                raise error.HTTPError(
                    req.full_url,
                    404,
                    "Not Found",
                    hdrs=None,
                    fp=io.BytesIO(
                        json.dumps(
                            {
                                "error": {
                                    "message": (
                                        "No endpoints found that can handle the "
                                        "requested parameters."
                                    ),
                                    "code": 404,
                                }
                            }
                        ).encode("utf-8")
                    ),
                )
            return _FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

        client = OpenAICompatibleChatClient(
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            request_provider={"require_parameters": True},
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("catan_bench.llm.request.urlopen", side_effect=fake_urlopen):
                response = client.complete(
                    model="stepfun/step-3.5-flash",
                    messages=[{"role": "user", "content": "{}"}],
                    temperature=0.1,
                )

        self.assertEqual(len(captured_bodies), 2)
        self.assertIn("response_format", captured_bodies[0])
        self.assertNotIn("response_format", captured_bodies[1])
        self.assertEqual(captured_bodies[1]["provider"], {"require_parameters": True})
        self.assertIn("choices", response)
