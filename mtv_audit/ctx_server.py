"""Minimal `ctx_read` / `ctx_search` MCP server for the D-α probe (arm S).

Zero dependencies: a hand-rolled JSON-RPC 2.0 stdio server speaking just enough
MCP (initialize / tools/list / tools/call) for Claude Code to page a stubbed
block back in. This is the *probe* shim (§B.3) — NOT the T9 production server.
It is read-only over a probe manifest and logs every call so the true fault
rate can be reconstructed even if a transcript is incomplete.

Register (founder, arm S only):
    claude mcp add svm-probe -- python3 -m mtv_audit.ctx_server \
        --manifest reports/dalpha_manifest_real.json \
        --log reports/dalpha_calls_real.jsonl

Design law 1 (schema < 400 tok) is asserted by tests; the two tool schemas
below are deliberately terse.
"""
from __future__ import annotations

import argparse
import json
import sys

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "ctx_read",
        "description": "Return the full body of a paged-out context block by its handle. "
                       "Use when a [SVM:paged h=...] stub is not enough and you need the "
                       "actual content. Optional line range like \"1-40\".",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "the h=... handle from the stub"},
                "lines": {"type": "string", "description": "optional line range, e.g. 1-40"},
            },
            "required": ["handle"],
        },
    },
    {
        "name": "ctx_search",
        "description": "Full-text search the paged context store; returns matching handles "
                       "with a short snippet. Use to locate a block when you don't have its handle.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


class Store:
    def __init__(self, manifest_path: str) -> None:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.by_handle = {b["handle"]: b for b in data.get("blocks", [])}

    def read(self, handle: str, lines: str | None) -> tuple[str, bool]:
        b = self.by_handle.get(handle)
        if not b:
            return f"no block with handle {handle!r}", True
        text = b.get("full_text", "")
        if lines:
            try:
                lo, hi = (int(x) for x in lines.split("-"))
                text = "\n".join(text.splitlines()[lo - 1: hi])
            except (ValueError, TypeError):
                pass
        return text, False

    def search(self, query: str, limit: int = 8) -> tuple[str, bool]:
        q = (query or "").lower()
        hits = []
        for h, b in self.by_handle.items():
            ft = b.get("full_text", "")
            if q and q in ft.lower():
                snippet = " ".join(ft.split())[:120]
                hits.append(f"h={h} type={b.get('type')} size={b.get('size_tok')}tok :: {snippet}")
        if not hits:
            return f"no matches for {query!r}", False
        return "\n".join(hits[:limit]), False


def schema_token_estimate() -> int:
    """Rough billed-token size of the tool schemas (chars/4) for the law-1 check."""
    blob = json.dumps(TOOLS, ensure_ascii=False)
    return max(1, len(blob) // 4)


def handle_request(req: dict, store: Store, log) -> dict | None:
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "svm-probe", "version": "0.1.0"},
        }}
    if method in ("notifications/initialized", "initialized"):
        return None   # notification, no response
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if log is not None:
            log.write(json.dumps({"tool": name, "args": args}, ensure_ascii=False) + "\n")
            log.flush()
        if name == "ctx_read":
            text, is_err = store.read(args.get("handle", ""), args.get("lines"))
        elif name == "ctx_search":
            text, is_err = store.search(args.get("query", ""))
        else:
            text, is_err = f"unknown tool {name!r}", True
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": text}], "isError": is_err,
        }}
    # unknown method
    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve(manifest_path: str, log_path: str | None) -> int:
    store = Store(manifest_path)
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = handle_request(req, store, log)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
    finally:
        if log:
            log.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mtv-audit-ctx-server",
                                 description="ctx_read/ctx_search MCP shim for the D-α probe")
    ap.add_argument("--manifest", required=True, help="probe manifest JSON (with full_text)")
    ap.add_argument("--log", default=None, help="append every tool call here (fault ground truth)")
    args = ap.parse_args(argv)
    return serve(args.manifest, args.log)


if __name__ == "__main__":
    raise SystemExit(main())
