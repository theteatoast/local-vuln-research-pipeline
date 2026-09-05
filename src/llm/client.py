"""OpenAI-compatible API client for llama-server.

Handles connection, retries, JSON parsing with repair, Qwen thinking tokens,
and self-consistency.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from src.config import config
from src.utils.logger import get_logger

logger = get_logger()

QWEN_THINKING_START = "<|think_start|>"
QWEN_THINKING_END = "<|think_end|>"
QWEN_RESPONSE_START = "<|response_start|>"


class LLMClient:
    def __init__(self):
        port = config.get("server.port", 8080)
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
        self.max_retries = 3
        self.retry_delay = 2.0
        self._model_name = "local-model"

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        extra: dict[str, Any] = {
            "top_p": 0.95,
            "extra_body": {"top_k": 20, "presence_penalty": 0.0},
        }
        if json_mode:
            extra["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                content = resp.choices[0].message.content or ""
                content = self._strip_thinking(content)
                return content.strip()
            except (APIConnectionError, RateLimitError) as e:
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"LLM API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    raise ConnectionError(f"LLM connection failed after {self.max_retries} attempts: {e}")
            except APIError as e:
                logger.error(f"LLM API error: {e}")
                raise

        return ""

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.6,
        max_tokens: int = 2048,
    ) -> dict[str, Any] | list[Any]:
        raw = self.chat(system, user, temperature, max_tokens, json_mode=True)
        parsed = self._parse_json(raw)
        if parsed is None:
            raw = self.chat(system, user, temperature, max_tokens)
            parsed = self._parse_json(raw)
        if parsed is None:
            return {}
        return parsed

    def self_consistent(
        self,
        system: str,
        user: str,
        runs: int = 3,
        temperature: float = 0.3,
        agreement_threshold: float = 0.66,
    ) -> dict[str, Any] | None:
        results = []
        for _ in range(runs):
            result = self.chat_json(system, user, temperature)
            if result:
                results.append(result)

        if not results:
            return None

        if len(results) == 1:
            return results[0]

        agreement_count = sum(
            1 for r in results
            if self._results_agree(r, results[0])
        )
        if agreement_count / len(results) >= agreement_threshold:
            return results[0]

        return None

    def _strip_thinking(self, text: str) -> str:
        text = re.sub(
            r"<\|think_start\|>.*?<\|think_end\|>",
            "", text, flags=re.DOTALL,
        )
        if QWEN_RESPONSE_START in text:
            text = text.split(QWEN_RESPONSE_START, 1)[-1]
        return text.strip()

    def _parse_json(self, text: str) -> dict | list | None:
        if not text:
            return None

        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for i, c in enumerate(text):
            if c in "{[":
                depth = 0
                for j in range(i, len(text)):
                    if text[j] in "{[":
                        depth += 1
                    elif text[j] in "}]":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[i:j + 1])
                            except json.JSONDecodeError:
                                break
                break

        return None

    def _results_agree(self, a: dict, b: dict) -> bool:
        # Path/memory validation results agree on their decision field (verdict).
        # Fall back to vulnerability_class for prompts that emit that instead.
        a_verdict = str(a.get("verdict", "")).lower()
        b_verdict = str(b.get("verdict", "")).lower()
        if a_verdict or b_verdict:
            return a_verdict == b_verdict
        a_class = str(a.get("vulnerability_class", "") or a.get("class", ""))
        b_class = str(b.get("vulnerability_class", "") or b.get("class", ""))
        return a_class == b_class

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
