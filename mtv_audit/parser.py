"""Parser: Claude Code session JSONL -> Session.

Designed to be *tolerant*: real Claude Code logs vary across versions.
We only rely on the stable core shape:

    {"type": "user"|"assistant", "message": {"role": ..., "content": ...,
     "model"?: ..., "usage"?: ...}, "timestamp"?: ...}

- content may be a plain string or a list of typed blocks
  (text / thinking / tool_use / tool_result).
- tool_result content may itself be a string or a list of {type: text}.
- Lines of other types (summary, system, progress, ...) are skipped.
- Malformed JSON lines are skipped and counted in meta["skipped_lines"].
"""
from __future__ import annotations

import json
from typing import Any

from .model import Block, Session, Turn


def _flatten_tool_result_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _blocks_from_content(content: Any, turn_index: int) -> list[Block]:
    blocks: list[Block] = []
    if isinstance(content, str):
        if content.strip():
            blocks.append(Block(kind="text", text=content, block_id=f"t{turn_index}.text0"))
        return blocks
    if not isinstance(content, list):
        return blocks
    for i, raw in enumerate(content):
        if not isinstance(raw, dict):
            continue
        btype = raw.get("type")
        bid = raw.get("id") or raw.get("tool_use_id") or f"t{turn_index}.b{i}"
        if btype == "text":
            blocks.append(Block(kind="text", text=str(raw.get("text", "")), block_id=f"{bid}.text"))
        elif btype == "thinking":
            blocks.append(Block(kind="thinking", text=str(raw.get("thinking", "")),
                                block_id=f"{bid}.thinking"))
        elif btype == "tool_use":
            tool_input = raw.get("input") if isinstance(raw.get("input"), dict) else {}
            blocks.append(Block(kind="tool_use",
                                text=json.dumps(tool_input, ensure_ascii=False),
                                block_id=str(bid),
                                tool_name=str(raw.get("name", "")),
                                tool_input=tool_input))
        elif btype == "tool_result":
            blocks.append(Block(kind="tool_result",
                                text=_flatten_tool_result_content(raw.get("content")),
                                block_id=f"{bid}.result"))
        # other block types (image, ...) ignored in v0
    return blocks


def load_session(path: str) -> Session:
    session = Session(source_path=path)
    skipped = 0
    idx = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            rtype = rec.get("type")
            if rtype not in ("user", "assistant"):
                skipped += 1
                continue
            msg = rec.get("message") or {}
            role = msg.get("role") or rtype
            blocks = _blocks_from_content(msg.get("content"), idx)
            turn = Turn(
                index=idx,
                role=role,
                blocks=blocks,
                model=msg.get("model"),
                usage=msg.get("usage") if isinstance(msg.get("usage"), dict) else None,
                timestamp=rec.get("timestamp"),
                is_compact_summary=bool(rec.get("isCompactSummary")),
            )
            session.turns.append(turn)
            idx += 1
    session.meta["skipped_lines"] = skipped
    session.meta["session_id"] = _first_session_id(path)
    return session


def _first_session_id(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("sessionId"):
                    return str(rec["sessionId"])
    except OSError:
        pass
    return "unknown-session"
