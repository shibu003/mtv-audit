"""D-α probe harness (addendum-A §B) — offline ($0) logic only."""
import json

from conftest import GOAL, parse_built

from mtv_audit.model import Session
from mtv_audit.probe import (
    ProbeBlock, classify_block, make_stub, recompute_profitable, scan_faults,
    select_probe_blocks,
)
from mtv_audit.svm import run_svm_audit
from mtv_audit.synth import SessionBuilder

MODEL = "claude-opus-4-8"
NEWS = ("Quarterly newsletter ficus parking garage renovation potluck stapler "
        "thermostat elevator carpet beverage vending lobby placard. " * 60)


def _passive_audit(tmp_path):
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Read", {"file_path": "NEWS.md"})
    b.assistant(MODEL, [t1])
    b.tool_result(t1["id"], NEWS)
    for i in range(16):
        b.assistant(MODEL, [b.text(f"Step {i}: edit src/payments.py refund ledger.")])
    return run_svm_audit(parse_built(b, tmp_path))


# --------------------------------------------------------------------------
# B.1 classification
# --------------------------------------------------------------------------

def test_classify_recognizes_system_summary_and_code():
    s = Session()   # no origin tools -> exercises the content fallback
    assert classify_block(s, "x.result", "tool_result",
                          "Base directory for this skill: /x") == "skill_system"
    assert classify_block(s, "x.text", "text", "<task-notification> done") == "skill_system"
    assert classify_block(s, "x.result", "tool_result",
                          "それでは、findings を整理します。") == "summary"
    assert classify_block(s, "x.result", "tool_result",
                          "1\timport React from 'react'") == "file_read"
    assert classify_block(s, "x.result", "tool_result",
                          "=== handleAct definition ===\nfoo:bar") == "search"


# --------------------------------------------------------------------------
# B.2 deterministic stubs
# --------------------------------------------------------------------------

def test_stub_deterministic_and_compact():
    pb = ProbeBlock(handle="abcd1234", block_id="t.result", type="file_read",
                    kind="tool_result", size_tok=3000, billed=9900.0, birth_turn=31,
                    lifespan_turns=300, causal_refs=2, passive_rent_usd=1.0,
                    full_text="line one of the body\nlots more lines\n" * 50)
    a, b = make_stub(pb, 2), make_stub(pb, 2)
    assert a == b                              # deterministic
    assert "ctx_read(\"abcd1234\")" in a
    assert "abcd1234" in a and "born=t31" in a
    assert len(a) < 400                        # design law 1: stub is small
    zero = make_stub(pb, 0)
    assert "要約" not in zero and "ctx_read(\"abcd1234\")" in zero  # address-only


def test_select_covers_required_strata(tmp_path):
    # synthetic has one big block; just assert selection runs and stubs attach
    audit = _passive_audit(tmp_path)
    blocks = select_probe_blocks(audit, target=5)
    assert blocks
    assert all(pb.stub_2line and pb.stub_0line for pb in blocks)
    assert all(pb.passive_rent_usd >= 0 for pb in blocks)


# --------------------------------------------------------------------------
# B.4 fault scan
# --------------------------------------------------------------------------

def test_scan_faults_counts_ctx_read_and_flags_hallucination(tmp_path):
    pb = ProbeBlock(handle="HANDLE99", block_id="b.result", type="file_read",
                    kind="tool_result", size_tok=500, billed=1650.0, birth_turn=2,
                    lifespan_turns=10, causal_refs=0, passive_rent_usd=0.3,
                    full_text="zorptangle quibblefrotz mandelbrick snorfwidget "
                              "plimthacket grebulon flandermaus wozzlepic")
    # case A: model pages it back -> a fault, no hallucination
    bb = SessionBuilder(); bb.user_text("do the task")
    tu = bb.tool_use("ctx_read", {"handle": "HANDLE99"})
    bb.assistant(MODEL, [tu])
    pathA = tmp_path / "armS_A.jsonl"; bb.write_jsonl(str(pathA))
    sA = scan_faults(str(pathA), [pb])["HANDLE99"]
    assert sA.ctx_read_count == 1 and not sA.hallucinated

    # case B: model reproduces distinctive body tokens WITHOUT ctx_read -> worst event
    bc = SessionBuilder(); bc.user_text("do the task")
    bc.assistant(MODEL, [bc.text("zorptangle quibblefrotz mandelbrick snorfwidget "
                                 "plimthacket grebulon flandermaus wozzlepic appear")])
    pathB = tmp_path / "armS_B.jsonl"; bc.write_jsonl(str(pathB))
    sB = scan_faults(str(pathB), [pb])["HANDLE99"]
    assert sB.ctx_read_count == 0 and sB.hallucinated


# --------------------------------------------------------------------------
# B.6 profitable-pool recompute + gate
# --------------------------------------------------------------------------

def test_recompute_gate_green_when_faults_rare(tmp_path):
    audit = _passive_audit(tmp_path)
    v = recompute_profitable(audit, [], {"file_read": 0.0, "other": 0.0},
                             worst_events=0, healthy_recall=1.0)
    assert v.profitable_pct >= 0.60 and v.gate == "GREEN"


def test_recompute_gate_red_on_hallucination(tmp_path):
    audit = _passive_audit(tmp_path)
    v = recompute_profitable(audit, [], {"file_read": 0.0, "other": 0.0},
                             worst_events=1, healthy_recall=1.0)
    assert v.gate == "RED"          # any hallucination forces RED regardless of %


def test_recompute_gate_red_when_faults_frequent(tmp_path):
    audit = _passive_audit(tmp_path)
    # huge fault rate -> stubbing never profitable -> pool collapses
    v = recompute_profitable(audit, [], {"file_read": 50.0, "other": 50.0},
                             rprime=200.0, worst_events=0)
    assert v.profitable_pct < 0.30 and v.gate == "RED"
