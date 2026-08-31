"""Minimal Anthropic Messages API client.

Same reasoning as alpaca.py: one dependency (requests), everything visible.

The one non-obvious thing in here is `ask_json`. Getting reliable structured
output from a model is the single most common place hackathon agents fall over
live. Three defences, cheapest first:
  1. Prefill the assistant turn with '{' so the model cannot open with prose.
  2. Extract the first balanced JSON object if it wraps output in a fence anyway.
  3. Return a typed failure instead of raising, so one bad member does not take
     the whole committee down mid-demo.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.environ.get("QUORUM_MODEL", "claude-sonnet-4-5")


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1200,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise LLMError(
                "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and add your key "
                "from https://console.anthropic.com"
            )
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._s = requests.Session()
        self._s.headers.update(
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        )
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def ask(self, system: str, user: str, prefill: str | None = None, temperature: float = 0.3) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "temperature": temperature,
        }
        last_err = ""
        for attempt in range(3):
            try:
                r = self._s.post(API_URL, json=body, timeout=self.timeout)
                if r.status_code in (429, 529):
                    time.sleep(2 ** attempt + 1)
                    last_err = f"{r.status_code}: {r.text[:200]}"
                    continue
                if r.status_code >= 400:
                    raise LLMError(f"{r.status_code}: {r.text[:300]}")
                payload = r.json()
                self.calls += 1
                usage = payload.get("usage", {})
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                text = "".join(
                    b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"
                )
                return (prefill or "") + text
            except requests.RequestException as exc:
                last_err = str(exc)[:200]
                time.sleep(2 ** attempt)
        raise LLMError(f"Failed after 3 attempts: {last_err}")

    def ask_json(self, system: str, user: str, temperature: float = 0.3) -> dict[str, Any]:
        raw = self.ask(system, user, prefill="{", temperature=temperature)
        parsed = extract_json(raw)
        if parsed is None:
            return {"_parse_error": True, "_raw": raw[:500]}
        return parsed


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of arbitrary model output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
