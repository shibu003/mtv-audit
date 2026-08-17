"""Synthesis of Claude Code-shaped session records.

Used to build the bundled fixture and test mini-sessions. Deterministic:
usage numbers are derived from cumulative estimated context with a fixed
1.1 overhead factor (stands in for system prompt / tool schemas).
"""
from __future__ import annotations

import json
from typing import Any

OVERHEAD = 1.1


class SessionBuilder:
    def __init__(self, session_id: str = "fixture-session-0001") -> None:
        self.session_id = session_id
        self.records: list[dict[str, Any]] = []
        self._cum_est = 0
        self._uuid = 0
        self._tool_seq = 0

    # ---- low-level -----------------------------------------------------
    def _next_uuid(self) -> str:
        self._uuid += 1
        return f"uuid-{self._uuid:04d}"

    def next_tool_id(self) -> str:
        self._tool_seq += 1
        return f"toolu_{self._tool_seq:04d}"

    def _est(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _record(self, rtype: str, message: dict[str, Any]) -> dict[str, Any]:
        rec = {
            "type": rtype,
            "uuid": self._next_uuid(),
            "sessionId": self.session_id,
            "timestamp": f"2026-06-11T09:{len(self.records):02d}:00Z",
            "message": message,
        }
        self.records.append(rec)
        return rec

    def _content_est(self, content: list[dict]) -> int:
        total = 0
        for b in content:
            if b.get("type") == "text":
                total += self._est(b.get("text", ""))
            elif b.get("type") == "thinking":
                total += self._est(b.get("thinking", ""))
            elif b.get("type") == "tool_use":
                total += self._est(json.dumps(b.get("input", {})))
            elif b.get("type") == "tool_result":
                c = b.get("content")
                if isinstance(c, str):
                    total += self._est(c)
                elif isinstance(c, list):
                    total += sum(self._est(i.get("text", "")) for i in c if isinstance(i, dict))
        return total

    # ---- public builders -------------------------------------------------
    def user_text(self, text: str) -> None:
        content = [{"type": "text", "text": text}]
        self._record("user", {"role": "user", "content": content})
        self._cum_est += self._content_est(content)

    def tool_result(self, tool_use_id: str, text: str) -> None:
        content = [{"type": "tool_result", "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": text}]}]
        self._record("user", {"role": "user", "content": content})
        self._cum_est += self._content_est(content)

    def assistant(self, model: str, blocks: list[dict],
                  cache_hit: bool = True) -> None:
        prompt = int(self._cum_est * OVERHEAD)
        if cache_hit and self.records:
            cache_read = int(prompt * 0.85)
            cache_creation = prompt - cache_read
            input_tokens = 0
        else:
            cache_read = 0
            cache_creation = prompt
            input_tokens = 0
        out = self._content_est(blocks)
        usage = {
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
            "output_tokens": out,
        }
        self._record("assistant", {"role": "assistant", "model": model,
                                   "content": blocks, "usage": usage})
        self._cum_est += self._content_est(blocks)

    # ---- block helpers ---------------------------------------------------
    @staticmethod
    def thinking(text: str) -> dict:
        return {"type": "thinking", "thinking": text}

    @staticmethod
    def text(text: str) -> dict:
        return {"type": "text", "text": text}

    def tool_use(self, name: str, tool_input: dict, tool_id: str | None = None) -> dict:
        return {"type": "tool_use", "id": tool_id or self.next_tool_id(),
                "name": name, "input": tool_input}

    # ---- output ------------------------------------------------------------
    def write_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
