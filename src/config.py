"""Configuration management with {AUTO} placeholder resolution."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
BENCHMARK_CACHE = ROOT_DIR / "data" / "benchmark_results.json"
CALIBRATION_CACHE = ROOT_DIR / "data" / "calibration.json"


class Config:
    def __init__(self):
        self._data: dict[str, Any] = {}
        self.reload()

    def reload(self):
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                self._data = yaml.safe_load(f) or {}
        self._apply_overrides()

    def _apply_overrides(self):
        bench = self._load_json(BENCHMARK_CACHE)
        calib = self._load_json(CALIBRATION_CACHE)
        overrides: dict[str, Any] = {}

        if bench:
            overrides["pipeline_max_hypotheses"] = bench.get("max_hypotheses", 5)
            overrides["server_context_length"] = bench.get("optimal_context_length", 32768)

        if calib:
            overrides["thresholds_hypothesis_confidence_cutoff"] = calib.get(
                "confidence_cutoff", 0.6
            )

        for key, value in overrides.items():
            self._set_nested(key.replace("_", "/"), value)

    def _set_nested(self, path: str, value: Any):
        keys = path.split("/")
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, None)
            else:
                return default
            if val is None:
                return default
        if val == "AUTO":
            return default
        return val

    @property
    def model_path(self) -> Path:
        return ROOT_DIR / self.get("model.file", "models/model.gguf")

    @property
    def context_length(self) -> int:
        return self.get("server.context_length", 32768)

    @property
    def max_hypotheses(self) -> int:
        return self.get("pipeline.max_hypotheses", 5)

    def update(self, path: str, value: Any):
        keys = path.split(".")
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(self._data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None


config = Config()
