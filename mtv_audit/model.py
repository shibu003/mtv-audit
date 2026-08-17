"""Core data model for MTV audit.

A Session is a sequence of Turns; each Turn holds content Blocks.
Token counts at block level are *estimates* (chars/4) and are later
scaled against reported API usage per turn (see attribution.turn_scale).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

CHARS_PER_TOKEN = 4  # coarse v0 estimator; documented in report methodology

FAILURE_PATTERNS = [
    re.compile(p, re.MULTILINE)
    for p in (
        r"Traceback \(most recent call last\)",
        r"\bFAILED\b",
        r"\b\d+ failed\b",
        r"\bAssertionError\b",
        r"\bexit code [1-9]\d*\b",
        r"^E\s{2,}",
        r"\bError:\s",
    )
]

SUCCESS_PATTERNS = [
    re.compile(p, re.MULTILINE)
    for p in (
        r"\b\d+ passed\b(?!.*\bfailed\b)",
        r"\ball tests pass(ed)?\b",
        r"^OK$",
    )
]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def has_failure_marker(text: str) -> bool:
    return any(p.search(text or "") for p in FAILURE_PATTERNS)


def has_success_marker(text: str) -> bool:
    t = text or ""
    if has_failure_marker(t):
        return False
    return any(p.search(t) for p in SUCCESS_PATTERNS)


@dataclass
class Block:
    kind: str                      # text | thinking | tool_use | tool_result
    text: str                      # flattened textual content
    block_id: str                  # stable id (tool_use id, or synthetic)
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    is_failure: bool = False
    is_success: bool = False
    est_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.est_tokens:
            self.est_tokens = estimate_tokens(self.text)
        if self.kind == "tool_result":
            self.is_failure = has_failure_marker(self.text)
            self.is_success = has_success_marker(self.text)


@dataclass
class Turn:
    index: int
    role: str                      # user | assistant
    blocks: list[Block] = field(default_factory=list)
    model: Optional[str] = None
    usage: Optional[dict[str, Any]] = None   # raw API usage dict if present
    timestamp: Optional[str] = None
    is_compact_summary: bool = False         # True for a compaction free-window boundary turn

    # --- usage helpers -------------------------------------------------
    def reported_prompt_tokens(self) -> int:
        """input + cache_read + cache_creation (what the API processed as prompt)."""
        if not self.usage:
            return 0
        u = self.usage
        return int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) + int(
            u.get("cache_creation_input_tokens", 0)
        )

    def reported_output_tokens(self) -> int:
        return int(self.usage.get("output_tokens", 0)) if self.usage else 0

    def est_output_tokens(self) -> int:
        return sum(b.est_tokens for b in self.blocks)

    # --- content helpers ------------------------------------------------
    def tool_uses(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "tool_use"]

    def thinking_tokens(self) -> int:
        return sum(b.est_tokens for b in self.blocks if b.kind == "thinking")

    def text_tokens(self) -> int:
        return sum(b.est_tokens for b in self.blocks if b.kind == "text")


@dataclass
class Session:
    turns: list[Turn] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def total_reported(self) -> dict[str, int]:
        tot = {"input_tokens": 0, "cache_read_input_tokens": 0,
               "cache_creation_input_tokens": 0, "output_tokens": 0}
        for t in self.assistant_turns():
            if t.usage:
                for k in tot:
                    tot[k] += int(t.usage.get(k, 0))
        tot["prompt_total"] = (tot["input_tokens"] + tot["cache_read_input_tokens"]
                               + tot["cache_creation_input_tokens"])
        tot["grand_total"] = tot["prompt_total"] + tot["output_tokens"]
        return tot
