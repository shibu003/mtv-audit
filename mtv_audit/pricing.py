"""Pricing: token -> USD conversion.

IMPORTANT: prices change. The table below is a *configurable default*,
not a claim. Override with --price-config pointing to a JSON file of the
same shape, and verify against the official price list before any
customer-facing report (https://docs.claude.com -> pricing).

Cache reads are billed at a fraction of input price; cache writes at a
premium. Re-sent context waste is therefore valued at the *blended*
prompt rate the customer actually paid that turn — this keeps the
receipt honest against the "but it was cached" objection.

KNOWN LIMITATION: `cache_write_mult` is the 5-minute-TTL multiplier
(1.25x). A 1-hour-TTL write costs 2x, and the session log's
`cache_creation_input_tokens` does not say which TTL was used — so a
session that caches at 1h is under-counted on the write leg. Fixing it
needs a TTL signal the transcript doesn't carry; do not paper over it
with an average.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# USD per 1M tokens, per the published price list (checked 2026-08-17 against
# platform.claude.com/docs/en/about-claude/pricing). Still a *configurable*
# default rather than a claim — re-check before any customer-facing report.
#
# ORDER MATTERS. _entry() returns the first key that is a substring of the
# model id, so specific keys must precede the general ones they extend:
# "sonnet-5" before "sonnet", or every Sonnet 5 turn prices at the 4.6 rate.
# Keep that invariant when editing this table or a --price-config file.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # Fable/Mythos 5 — above Opus. Without these they fell through to
    # "default" and priced at 3/15, a 3.3x under-count of the real 10/50.
    "fable":    {"input": 10.00, "output": 50.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    "mythos":   {"input": 10.00, "output": 50.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    # Retired, but historical transcripts still contain them, and they cost
    # 3x what the current Opus does — "opus" alone would under-count them.
    "opus-4-1": {"input": 15.00, "output": 75.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    "opus":     {"input": 5.00,  "output": 25.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    # Sonnet 5 is 2/10 — the launch introductory price became the standard
    # price; the scheduled rise to 3/15 was cancelled. Sonnet 4.6 and earlier
    # remain 3/15, so this needs its own key ahead of the general one.
    "sonnet-5": {"input": 2.00,  "output": 10.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    "sonnet":   {"input": 3.00,  "output": 15.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    "haiku":    {"input": 1.00,  "output": 5.00,  "cache_read_mult": 0.10, "cache_write_mult": 1.25},
    "default":  {"input": 3.00,  "output": 15.00, "cache_read_mult": 0.10, "cache_write_mult": 1.25},
}

# Used by the model-tier flag rule. Fable/Mythos sit above Opus on both
# capability and price, so leaving them out meant the most expensive model
# available was the one tier the rule could never flag.
TOP_TIER_SUBSTRINGS = ("opus", "fable", "mythos")


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
