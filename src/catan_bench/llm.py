from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
    ) -> dict[str, object]:
        ...


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
    ) -> dict[str, object]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                "Missing API key for LLM player. "
                f"Set environment variable {self.api_key_env!r} before running, "
                "or change `api_key_env` in configs/players.toml to point at the "
                "variable you want to use. Example: "
                f"export {self.api_key_env}=<your_api_key>"
            )

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        endpoint = self.api_base.rstrip("/") + "/chat/completions"
        req = request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt_index in range(self.max_attempts):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:  # pragma: no cover - network dependency.
                details = exc.read().decode("utf-8", errors="replace")
                if self._should_retry_http_error(exc) and attempt_index + 1 < self.max_attempts:
                    time.sleep(self._retry_delay(exc, attempt_index))
                    continue
                raise RuntimeError(
                    f"LLM request failed with HTTP {exc.code}: {details}"
                ) from exc
            except error.URLError as exc:  # pragma: no cover - network dependency.
                if attempt_index + 1 < self.max_attempts:
                    time.sleep(self.retry_backoff_seconds * (2**attempt_index))
                    continue
                raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        raise RuntimeError("LLM request failed after exhausting retry attempts.")

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
