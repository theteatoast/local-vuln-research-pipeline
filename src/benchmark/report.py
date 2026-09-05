"""Benchmark report formatting and recommendations."""
from __future__ import annotations
from typing import Any


class BenchmarkReport:
    def __init__(self, results: dict[str, Any]):
        self.r = results

    def print(self):
        print("\n" + "=" * 60)
        print("  BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Model:        {self.r.get('model', '?')}")
        print(f"  Quant:        {self.r.get('quant', '?')}")
        print(f"  Date:         {self.r.get('date', '?')}")
        print("-" * 60)

        tests = self.r.get("tests", {})

        print("\n  Context Length Performance:")
        print(f"  {'Context':<12} {'Trace tok/s':<14} {'JSON OK':<10} {'Verdict'}")
        for key, data in sorted(tests.items()):
            if key.startswith("context_"):
                cl = int(key.split("_")[1])
                tps = data.get("tokens_per_sec_deep_trace", 0)
                json_rate = data.get("json_compliance_deep_trace", 0)
                verdict = "PASS" if tps >= 8 and json_rate >= 0.80 else "FAIL"
                print(f"  {cl:<12,} {tps:<14.1f} {json_rate:<10.0%} {verdict}")

        optimal_cl = self.r.get("optimal_context_length", 32768)
        print(f"\n  Optimal context: {optimal_cl:,} tokens")

        max_h = self.r.get("max_hypotheses", 5)
        print(f"\n  Max hypotheses per run: {max_h}")

        print("\n  JSON Compliance Summary:")
        ctx_optimal = f"context_{optimal_cl}"
        if ctx_optimal in tests:
            dt_json = tests[ctx_optimal].get("json_compliance_deep_trace", 0)
            hg_json = tests[ctx_optimal].get("json_compliance_hypothesis", 0)
            print(f"    Deep trace:    {dt_json:.0%}")
            print(f"    Hypothesis gen:{hg_json:.0%}")

            if dt_json < 0.85 or hg_json < 0.85:
                print("\n  [!] WARNING: JSON compliance below 85%")
                print("  [!] Output repair + retry logic will be enabled in pipeline")

        print("\n  Recommendations:")
        if optimal_cl < 65536:
            print(f"    - Context limited to {optimal_cl:,} tokens. Large repos will use aggressive chunking.")
        print(f"    - self-consistency (3 runs) enabled for confidence <= 0.7")
        print("=" * 60)
