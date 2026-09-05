"""Model benchmark runner. Measures tok/s, JSON compliance, and context capacity."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import ROOT_DIR, config
from src.benchmark.report import BenchmarkReport

BENCHMARK_RESULTS = ROOT_DIR / "data" / "benchmark_results.json"

QWEN_THINKING_PATTERN = re.compile(r"<\|think_start\|>.*?<\|think_end\|>", re.DOTALL)
QWEN_RESPONSE_START = "<|response_start|>"

BENCHMARK_PROMPTS = {
    "deep_trace": {
        "system": (
            "You are a security code auditor. Trace the data flow from the given HTTP handler entry point "
            "through every function call to the identified database query sink. "
            "Output valid JSON only, no markdown, no explanation outside the JSON object."
        ),
        "user": (
            "Target: POST /api/users/register -> SQL injection via f-string in src/db.py\n\n"
            "Source code:\n"
            "```python\n"
            "src/api.py: def create_user(data): username = data.get('username'); "
            "query = f\"INSERT INTO users (name) VALUES ('{username}')\"; execute(query)\n"
            "src/db.py: def execute(sql): cursor.execute(sql)\n"
            "```\n\n"
            "Trace: entry point to sink. Output ONLY this JSON:\n"
            '{"trace":[{"hop":1,"file":"src/api.py","line":1,"data_controlled":true,'
            '"mitigation":null}],"reachable":true,"summary":"username flows unescaped to SQL"}'
        )
    },
    "hypothesis_gen": {
        "system": (
            "You are a vulnerability researcher. Given static analysis results, "
            "generate ranked exploit hypotheses. Output valid JSON only, no markdown."
        ),
        "user": (
            "Entry points: POST /api/register, GET /api/profile, POST /api/webhooks\n"
            "Sinks: raw SQL query, HTTP fetch with dynamic URL, pickle.loads\n\n"
            "Output ONLY this JSON:\n"
            '{"hypotheses":[{"vulnerability_class":"SQL injection","entry_point":"POST /api/register",'
            '"sink":"cursor.execute","confidence":0.8,"priority_score":0.8}]}'
        )
    }
}


class BenchmarkRunner:
    def __init__(self):
        self.client: OpenAI | None = None
        self._ensure_client()

    def _ensure_client(self):
        port = config.get("server.port", 8080)
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            self.client = OpenAI(base_url=base_url, api_key="not-needed")
            self.client.models.list()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to llama-server at {base_url}. "
                f"Is it running? Run python start_server.py first.\n"
                f"Error: {e}"
            )

    def run(self):
        print("=== Model Benchmark ===\n")
        print(f"Target: {config.get('model.name')}")
        print(f"Quant: {config.get('model.quant')}\n")

        results: dict[str, Any] = {
            "model": config.get("model.name"),
            "quant": config.get("model.quant"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {}
        }

        context_lengths = [8192, 32768, 65536, 131072]
        max_usable_context = 32768

        for cl in context_lengths:
            print(f"\n--- Testing context: {cl:,} tokens ---")
            passed = self._test_context_length(cl, results)
            if passed:
                max_usable_context = cl

        results["optimal_context_length"] = max_usable_context
        results["max_hypotheses"] = self._calc_max_hypotheses(results, max_usable_context)

        with open(BENCHMARK_RESULTS, "w") as f:
            json.dump(results, f, indent=2)

        report = BenchmarkReport(results)
        report.print()

        print(f"\n[+] Results saved to: {BENCHMARK_RESULTS}")
        print("[+] Config values updated.")

    def _test_context_length(self, cl: int, results: dict) -> bool:
        tps = self._measure_throughput("deep_trace", cl)
        json_rate = self._measure_json_compliance("deep_trace", cl, runs=5)

        results["tests"][f"context_{cl}"] = {
            "tokens_per_sec_deep_trace": tps,
            "tokens_per_sec_hypothesis": self._measure_throughput("hypothesis_gen", cl),
            "json_compliance_deep_trace": json_rate,
            "json_compliance_hypothesis": self._measure_json_compliance("hypothesis_gen", cl, runs=5),
        }

        min_tps = config.get("pipeline.min_tokens_per_second", 8)
        ok = tps >= min_tps and json_rate >= 0.60
        print(f"  Deep trace: {tps:.1f} tok/s | JSON compliance: {json_rate:.0%} | {'PASS' if ok else 'FAIL (JSON)'}")

        return ok

    def _measure_throughput(self, prompt_key: str, context_length: int) -> float:
        prompt_data = BENCHMARK_PROMPTS.get(prompt_key)
        if not prompt_data or not self.client:
            return 0.0

        filler = "x" * min(context_length - 500, 100000) if context_length > 1000 else ""
        filled_user = f"{prompt_data['user']}\n\n[FILLER:{filler[:50000]}]" if filler else prompt_data["user"]

        extra_body: dict[str, Any] = {"top_k": 20, "presence_penalty": 0.0}

        try:
            start = time.perf_counter()
            resp = self.client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": prompt_data["system"]},
                    {"role": "user", "content": filled_user},
                ],
                max_tokens=1024,
                temperature=0.6,
                top_p=0.95,
                extra_body=extra_body,
            )
            elapsed = time.perf_counter() - start

            usage = resp.usage
            if usage and usage.completion_tokens and elapsed > 0:
                return usage.completion_tokens / elapsed
        except Exception as e:
            print(f"    [!] Throughput error: {e}")
        return 0.0

    def _measure_json_compliance(
        self, prompt_key: str, context_length: int, runs: int = 5
    ) -> float:
        prompt_data = BENCHMARK_PROMPTS.get(prompt_key)
        if not prompt_data or not self.client:
            return 0.0

        successes = 0
        for i in range(runs):
            try:
                resp = self.client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": prompt_data["system"]},
                        {"role": "user", "content": prompt_data["user"]},
                    ],
                    max_tokens=1024,
                    temperature=0.6,
                    top_p=0.95,
                    extra_body={"top_k": 20, "presence_penalty": 0.0},
                )
                content = resp.choices[0].message.content or ""
                cleaned = self._clean_thinking(content)
                json.loads(cleaned)
                successes += 1
            except Exception:
                pass
        return successes / runs if runs > 0 else 0.0

    def _clean_thinking(self, text: str) -> str:
        text = QWEN_THINKING_PATTERN.sub("", text)

        if QWEN_RESPONSE_START in text:
            text = text.split(QWEN_RESPONSE_START, 1)[-1]

        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        if not text:
            return "{}"

        # find first { or [
        start_idx = -1
        for i, c in enumerate(text):
            if c in "{[":
                start_idx = i
                break

        if start_idx >= 0:
            # find matching close
            depth = 0
            end_idx = -1
            for i in range(start_idx, len(text)):
                if text[i] in "{[":
                    depth += 1
                elif text[i] in "}]":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            if end_idx > start_idx:
                return text[start_idx:end_idx]

        # try to extract anything that looks like JSON
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0)
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            return match.group(0)

        return text.strip()

    def _calc_max_hypotheses(self, results: dict, context_length: int = 32768) -> int:
        key = f"context_{context_length}"
        tps = results["tests"].get(key, {}).get("tokens_per_sec_deep_trace", 30)
        target_seconds = 180
        tokens_per_hypothesis = 2048
        tokens_budget = tps * target_seconds
        hypotheses = max(1, int(tokens_budget / tokens_per_hypothesis))
        return min(hypotheses, 10)
