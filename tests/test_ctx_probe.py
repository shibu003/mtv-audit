"""ctx_read MCP shim (§B.3) + probe scoring driver (§B.7)."""
import json

from conftest import GOAL, parse_built

from mtv_audit import ctx_server
from mtv_audit.probe import ProbeBlock, score_probe
from mtv_audit.svm import run_svm_audit
from mtv_audit.synth import SessionBuilder

MODEL = "claude-opus-4-8"
NEWS = ("Quarterly newsletter ficus parking garage renovation potluck stapler "
        "thermostat elevator carpet beverage vending lobby placard. " * 60)
DISTINCT = ("zorptangle quibblefrotz mandelbrick snorfwidget plimthacket "
            "grebulon flandermaus wozzlepic")


def _manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"blocks": [
        {"handle": "HX", "type": "file_read", "kind": "tool_result",
         "size_tok": 500, "full_text": "the body of HX\nsecond line\nthird line"},
    ]}), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------
# ctx_server (design law 1 + JSON-RPC behaviour)
# --------------------------------------------------------------------------

def test_schema_under_400_tokens():
    assert ctx_server.schema_token_estimate() < 400      # design law 1


def test_ctx_server_initialize_list_call(tmp_path):
    store = ctx_server.Store(_manifest(tmp_path))
    init = ctx_server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, store, None)
    assert init["result"]["protocolVersion"] == ctx_server.PROTOCOL_VERSION
    # notification -> no response
    assert ctx_server.handle_request(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}, store, None) is None
    lst = ctx_server.handle_request(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, store, None)
    assert {t["name"] for t in lst["result"]["tools"]} == {"ctx_read", "ctx_search"}


def test_ctx_read_hit_and_miss(tmp_path):
    store = ctx_server.Store(_manifest(tmp_path))
    hit = ctx_server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "ctx_read", "arguments": {"handle": "HX"}}}, store, None)
    assert "body of HX" in hit["result"]["content"][0]["text"]
    assert hit["result"]["isError"] is False
    miss = ctx_server.handle_request({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "ctx_read", "arguments": {"handle": "NOPE"}}}, store, None)
    assert miss["result"]["isError"] is True


def test_ctx_read_line_range(tmp_path):
    store = ctx_server.Store(_manifest(tmp_path))
    r = ctx_server.handle_request({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "ctx_read", "arguments": {"handle": "HX", "lines": "2-2"}}},
        store, None)
    assert r["result"]["content"][0]["text"].strip() == "second line"


def test_ctx_search_finds_handle(tmp_path):
    store = ctx_server.Store(_manifest(tmp_path))
    r = ctx_server.handle_request({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
        "params": {"name": "ctx_search", "arguments": {"query": "body of HX"}}}, store, None)
    assert "h=HX" in r["result"]["content"][0]["text"]


# --------------------------------------------------------------------------
# scoring driver (B.7)
# --------------------------------------------------------------------------

def _passive_audit(tmp_path):
    b = SessionBuilder(); b.user_text(GOAL)
    t1 = b.tool_use("Read", {"file_path": "NEWS.md"}); b.assistant(MODEL, [t1])
    b.tool_result(t1["id"], NEWS)
    for i in range(16):
        b.assistant(MODEL, [b.text(f"Step {i}: edit src/payments.py refund ledger.")])
    return run_svm_audit(parse_built(b, tmp_path))


def _armS(tmp_path, name, *, ctx_read: bool):
    b = SessionBuilder(); b.user_text("do task T1")
    if ctx_read:
        b.assistant(MODEL, [b.tool_use("ctx_read", {"handle": "HX"})])
        b.tool_result("toolu_0001", DISTINCT)
    else:
        b.assistant(MODEL, [b.text("done without paging anything in")])
    p = tmp_path / name; b.write_jsonl(str(p)); return str(p)


def _armV(tmp_path, name):
    b = SessionBuilder(); b.user_text("do task T1")
    b.assistant(MODEL, [b.text("done with the full block in context")])
    p = tmp_path / name; b.write_jsonl(str(p)); return str(p)


def test_score_probe_fhat_budget_and_b5(tmp_path):
    blocks = [ProbeBlock(handle="HX", block_id="b.result", type="file_read",
                         kind="tool_result", size_tok=500, billed=1650.0, birth_turn=2,
                         lifespan_turns=10, causal_refs=0, passive_rent_usd=0.3,
                         full_text=DISTINCT)]
    results = {
        "tasks": {"T1": ["HX"]},
        "runs": [
            {"task": "T1", "arm": "S", "variant": "2line", "rep": 1,
             "transcript": _armS(tmp_path, "s1.jsonl", ctx_read=True), "success": True},
            {"task": "T1", "arm": "S", "variant": "2line", "rep": 2,
             "transcript": _armS(tmp_path, "s2.jsonl", ctx_read=False), "success": True},
            {"task": "T1", "arm": "V", "variant": "-", "rep": 1,
             "transcript": _armV(tmp_path, "v1.jsonl"), "success": True},
        ],
    }
    score = score_probe(results, blocks, _passive_audit(tmp_path))
    assert abs(score["fhat_by_type"]["file_read"] - 0.5) < 1e-9   # 1 fault / 2 trials
    assert score["trials_by_type"]["file_read"] == 2
    assert score["worst_events"] == 0
    assert score["within_budget"] is True
    assert score["b5_stub_form"]["2line"]["trials"] == 2
    assert score["b5_stub_form"]["2line"]["fault_rate"] == 0.5
    assert score["gate"] in ("GREEN", "AMBER", "RED")
