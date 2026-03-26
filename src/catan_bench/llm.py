from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse
from urllib import error, request


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float | None = None,
        reasoning_enabled: bool | None = None,
    ) -> dict[str, object]:
        ...


class LLMRequestTooLargeError(RuntimeError):
    """Raised when the provider rejects a request because the prompt is too large."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleChatClient:
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float | None = None,
        reasoning_enabled: bool | None = None,
    ) -> dict[str, object]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                "Missing API key for LLM player. "
                f"Set environment variable {self.api_key_env!r} before running, "
                "or change `api_key_env` in configs/openai-players.toml to point at the "
                "variable you want to use. Example: "
                f"export {self.api_key_env}=<your_api_key>"
            )

        endpoint = self.api_base.rstrip("/") + "/chat/completions"
        use_json_response_format = True
        attempt_index = 0
        while attempt_index < self.max_attempts:
            req = self._build_request(
                endpoint=endpoint,
                api_key=api_key,
                body=self._request_body(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    reasoning_enabled=reasoning_enabled,
                    use_json_response_format=use_json_response_format,
                ),
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:  # pragma: no cover - network dependency.
                details = exc.read().decode("utf-8", errors="replace")
                if self._should_retry_without_response_format(
                    exc,
                    details=details,
                    use_json_response_format=use_json_response_format,
                ):
                    use_json_response_format = False
                    continue
                if self._is_request_too_large_error(exc):
                    raise LLMRequestTooLargeError(
                        f"LLM request exceeded the provider payload limit: {details}"
                    ) from exc
                attempt_index += 1
                if self._should_retry_http_error(exc) and attempt_index < self.max_attempts:
                    time.sleep(self._retry_delay(exc, attempt_index))
                    continue
                raise RuntimeError(
                    f"LLM request failed with HTTP {exc.code}: {details}"
                ) from exc
            except error.URLError as exc:  # pragma: no cover - network dependency.
                attempt_index += 1
                if attempt_index < self.max_attempts:
                    time.sleep(self.retry_backoff_seconds * (2**attempt_index))
                    continue
                raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        raise RuntimeError("LLM request failed after exhausting retry attempts.")

    def _request_body(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float | None,
        reasoning_enabled: bool | None,
        use_json_response_format: bool,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if use_json_response_format:
            body["response_format"] = {"type": "json_object"}
        if top_p is not None:
            body["top_p"] = top_p
        if reasoning_enabled is not None:
            body.update(
                self._reasoning_request_fields(
                    model=model,
                    reasoning_enabled=reasoning_enabled,
                )
            )
        return body

    @staticmethod
    def _build_request(
        *,
        endpoint: str,
        api_key: str,
        body: dict[str, object],
    ) -> request.Request:
        return request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "catan-bench/0.4.0",
            },
            method="POST",
        )

    def _reasoning_request_fields(
        self,
        *,
        model: str,
        reasoning_enabled: bool,
    ) -> dict[str, object]:
        provider = self._provider_name()
        if provider == "groq":
            return {"include_reasoning": reasoning_enabled}
        if provider == "google":
            if reasoning_enabled is False and self._google_supports_reasoning_effort_none(model):
                return {"reasoning_effort": "none"}
            return {}
        if provider == "openrouter":
            if reasoning_enabled is False:
                return {"reasoning": {"effort": "none"}}
            return {"reasoning": {"enabled": True}}
        if provider == "together":
            return {}
        return {"reasoning": {"enabled": reasoning_enabled}}

    @staticmethod
    def _google_supports_reasoning_effort_none(model: str) -> bool:
        normalized = model.strip().lower()
        return normalized.startswith("gemini-2.5-") and "pro" not in normalized and not normalized.startswith(
            "gemini-3"
        )

    def _provider_name(self) -> str:
        hostname = urlparse(self.api_base).hostname or ""
        hostname = hostname.lower()
        if hostname.endswith("groq.com"):
            return "groq"
        if hostname.endswith("googleapis.com"):
            return "google"
        if hostname.endswith("openrouter.ai"):
            return "openrouter"
        if hostname.endswith("together.ai"):
            return "together"
        return "default"

    def _retry_delay(self, exc: error.HTTPError, attempt_index: int) -> float:
        """Return seconds to sleep before the next attempt, respecting Retry-After."""
        retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
        if retry_after is not None:
            try:
                return float(retry_after) + 0.5
            except ValueError:
                pass
        return self.retry_backoff_seconds * (2**attempt_index)

    @staticmethod
    def _should_retry_http_error(exc: error.HTTPError) -> bool:
        return exc.code in {408, 429, 500, 502, 503, 504}

    @staticmethod
    def _is_request_too_large_error(exc: error.HTTPError) -> bool:
        return exc.code == 413

    @staticmethod
    def _should_retry_without_response_format(
        exc: error.HTTPError,
        *,
        details: str,
        use_json_response_format: bool,
    ) -> bool:
        if exc.code != 400 or not use_json_response_format:
            return False
        error_payload: dict[str, object] = {}
        try:
            parsed = json.loads(details)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            error_payload = dict(parsed["error"])
        code = str(error_payload.get("code", "")).lower()
        message = str(error_payload.get("message", "")).lower()
        if code == "json_validate_failed":
            return True
        return (
            "failed to validate json" in message
            or (
                "response_format" in message
                and ("unsupported" in message or "not supported" in message)
            )
        )
