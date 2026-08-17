"""Pricing: token -> USD conversion.

IMPORTANT: prices change. The table below is a *configurable default*,
not a claim. Override with --price-config pointing to a JSON file of the
same shape, and verify against the official price list before any
customer-facing report (https://docs.claude.com -> pricing).

Cache reads are billed at a fraction of input price; cache writes at a
premium. Re-sent context waste is therefore valued at the *blended*
prompt rate the customer actually paid that turn — this keeps the
receipt honest against the "but it was cached" objection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# USD per 1M tokens. PLACEHOLDER DEFAULTS — verify before customer use.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "opus":    {"input": 5.00, "output": 25.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    "sonnet":  {"input": 3.00, "output": 15.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    "haiku":   {"input": 1.00, "output": 5.00,  "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    "default": {"input": 3.00, "output": 15.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
}

TOP_TIER_SUBSTRINGS = ("opus",)  # used by the model-tier flag rule


@dataclass
class PriceBook:
    prices: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: str | None = None) -> "PriceBook":
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                return cls(prices=json.load(fh))
        return cls(prices=DEFAULT_PRICES)

    def _entry(self, model: str | None) -> dict[str, float]:
        m = (model or "").lower()
        for key, entry in self.prices.items():
            if key != "default" and key in m:
                return entry
        return self.prices["default"]

    def output_rate(self, model: str | None) -> float:
        """USD per token (not per MTok)."""
        return self._entry(model)["output"] / 1_000_000

    def input_rate(self, model: str | None) -> float:
        return self._entry(model)["input"] / 1_000_000

    def blended_prompt_rate(self, model: str | None, usage: dict | None) -> float:
        """USD per prompt token actually paid this turn, given cache mix.

        Falls back to plain input rate when usage is missing.
        """
        e = self._entry(model)
        base = e["input"] / 1_000_000
        if not usage:
            return base
        inp = int(usage.get("input_tokens", 0))
        cr = int(usage.get("cache_read_input_tokens", 0))
        cw = int(usage.get("cache_creation_input_tokens", 0))
        total = inp + cr + cw
        if total <= 0:
            return base
        usd = base * inp + base * e["cache_read_mult"] * cr + base * e["cache_write_mult"] * cw
        return usd / total

    def turn_cost_usd(self, model: str | None, usage: dict | None) -> float:
        if not usage:
            return 0.0
        e = self._entry(model)
        base_in = e["input"] / 1_000_000
        return (base_in * int(usage.get("input_tokens", 0))
                + base_in * e["cache_read_mult"] * int(usage.get("cache_read_input_tokens", 0))
                + base_in * e["cache_write_mult"] * int(usage.get("cache_creation_input_tokens", 0))
                + (e["output"] / 1_000_000) * int(usage.get("output_tokens", 0)))


def is_top_tier(model: str | None) -> bool:
    m = (model or "").lower()
    return any(s in m for s in TOP_TIER_SUBSTRINGS)
