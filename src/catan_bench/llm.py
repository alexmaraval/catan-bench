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
        reasoning_effort: str | None = None,
    ) -> dict[str, object]: ...


class LLMRequestTooLargeError(RuntimeError):
    """Raised when the provider rejects a request because the prompt is too large."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleChatClient:
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    json_response_format: bool = True
    request_provider: dict[str, object] | None = None

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float | None = None,
        reasoning_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, object]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                "Missing API key for LLM player. "
                f"Set environment variable {self.api_key_env!r} before running, "
                "or change `api_key_env` in your players config to point at the "
                "variable you want to use. Example: "
                f"export {self.api_key_env}=<your_api_key>"
            )

        endpoint = self._chat_completions_endpoint()
        use_json_response_format = self.json_response_format
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
                    reasoning_effort=reasoning_effort,
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
                if (
                    self._should_retry_http_error(exc)
                    and attempt_index < self.max_attempts
                ):
                    time.sleep(self._retry_delay(exc, attempt_index))
                    continue
                raise RuntimeError(
                    f"LLM request failed for model {model!r} with HTTP {exc.code}: {details}"
                ) from exc
            except error.URLError as exc:  # pragma: no cover - network dependency.
                attempt_index += 1
                if attempt_index < self.max_attempts:
                    time.sleep(self.retry_backoff_seconds * (2**attempt_index))
                    continue
                raise RuntimeError(
                    f"LLM request failed for model {model!r}: {exc.reason}"
                ) from exc

        raise RuntimeError("LLM request failed after exhausting retry attempts.")

    def _chat_completions_endpoint(self) -> str:
        parsed = urlparse(self.api_base)
        hostname = (parsed.hostname or "").lower()
        if hostname in {"openai.com", "www.openai.com"}:
            raise RuntimeError(
                "Invalid OpenAI `api_base`: the players config points at the website "
                f"{self.api_base!r} instead of the API. Use "
                "'https://api.openai.com/v1'."
            )
        return self.api_base.rstrip("/") + "/chat/completions"

    def _request_body(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float | None,
        reasoning_enabled: bool | None,
        reasoning_effort: str | None,
        use_json_response_format: bool,
    ) -> dict[str, object]:
        provider_name = self._provider_name()
        body: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.request_provider is not None:
            if provider_name != "openrouter":
                raise ValueError(
                    "`provider` request preferences are only supported when "
                    "`api_base` targets OpenRouter."
                )
            body["provider"] = dict(self.request_provider)
        if use_json_response_format:
            body["response_format"] = {"type": "json_object"}
        if top_p is not None:
            body["top_p"] = top_p
        if reasoning_enabled is not None and reasoning_effort is not None:
            raise ValueError(
                "Pass either `reasoning_enabled` or `reasoning_effort`, not both."
            )
        if reasoning_effort is not None:
            body.update(self._reasoning_effort_request_fields(reasoning_effort))
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
                "User-Agent": "catan-bench/1.2.0",
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
            if (
                reasoning_enabled is False
                and self._google_supports_reasoning_effort_none(model)
            ):
                return {"reasoning_effort": "none"}
            return {}
        if provider == "openrouter":
            if reasoning_enabled is False:
                return {"reasoning": {"effort": "none"}}
            return {"reasoning": {"enabled": True}}
        if provider == "together":
            return {}
        return {"reasoning": {"enabled": reasoning_enabled}}

    def _reasoning_effort_request_fields(
        self, reasoning_effort: str
    ) -> dict[str, object]:
        provider = self._provider_name()
        if provider == "openrouter":
            return {"reasoning": {"effort": reasoning_effort}}
        return {"reasoning_effort": reasoning_effort}

    @staticmethod
    def _google_supports_reasoning_effort_none(model: str) -> bool:
        normalized = model.strip().lower()
        return (
            normalized.startswith("gemini-2.5-")
            and "pro" not in normalized
            and not normalized.startswith("gemini-3")
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
        retry_after = (
            exc.headers.get("Retry-After") if exc.headers is not None else None
        )
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
        if exc.code not in {400, 404, 405} or not use_json_response_format:
            return False
        error_payload: dict[str, object] = {}
        try:
            parsed = json.loads(details)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            error_payload = dict(parsed["error"])
        code = str(error_payload.get("code", "")).lower()
        message_candidates = list(
            OpenAICompatibleChatClient._iter_error_text_candidates(
                parsed if parsed is not None else details
            )
        )
        if code == "json_validate_failed":
            return True
        return any(
            "failed to validate json" in candidate
            or (
                "no endpoints found" in candidate
                and "requested parameters" in candidate
            )
            or (
                ("response_format" in candidate or "response format" in candidate)
                and ("unsupported" in candidate or "not supported" in candidate)
            )
            for candidate in message_candidates
        )

    @staticmethod
    def _iter_error_text_candidates(payload: object) -> tuple[str, ...]:
        candidates: list[str] = []

        def collect(value: object) -> None:
            if isinstance(value, str):
                normalized = value.lower()
                candidates.append(normalized)
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    return
                collect(decoded)
                return
            if isinstance(value, dict):
                for nested in value.values():
                    collect(nested)
                return
            if isinstance(value, list):
                for nested in value:
                    collect(nested)

        collect(payload)
        return tuple(candidates)
