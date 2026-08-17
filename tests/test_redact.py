"""Redaction is the thing standing between a session log and a shared receipt.

Two levels of check:
  1. the scrubber catches each credential shape it claims to catch
  2. nothing planted in a session survives into the rendered receipt

(2) is the one that matters, and it is only meaningful if the planted secret
really reaches the receipt's excerpt column when redaction is off. The
`test_excerpt_column_is_actually_exercised` case pins that down — without it,
(2) could pass simply because nothing was ever excerpted.
"""
from __future__ import annotations

from conftest import GOAL, TRACE, parse_built

from mtv_audit.attribution import audit_all_dials
from mtv_audit.redact import scrub, scrub_path
from mtv_audit.report import render_receipt
from mtv_audit.synth import SessionBuilder

MODEL = "claude-sonnet-4-6"

# Every one of these is a synthetic value in a real credential's shape.
SECRETS = [
    "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "sk-proj-BBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "AIzaSyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "ghp_DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
    "xoxb-1111111111-2222222222-abcdefghijkl",
    "AKIAEEEEEEEEEEEEEEEE",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r",
    "postgres://admin:hunter2@db.internal:5432/prod",
    "alice.tanaka@acme-client.co.jp",
    "/Users/shibuyaryouyuu/clients/acme/secret-project",
]


# --------------------------------------------------------------------------
# 1. the scrubber itself


def test_scrub_removes_each_secret_shape():
    for raw in SECRETS:
        out = scrub(raw)
        assert out != raw, f"scrub left {raw!r} untouched"
        assert "redacted" in out, f"scrub produced no marker for {raw!r}"


def test_scrub_keeps_ordinary_text():
    text = "retrying the refund test after a 500 from the payments stub"
    assert scrub(text) == text


def test_scrub_path_keeps_only_the_filename():
    out = scrub_path("/Users/shibuyaryouyuu/clients/acme/session-01.jsonl")
    assert out == "session-01.jsonl"
    assert "acme" not in out and "shibuyaryouyuu" not in out


# --------------------------------------------------------------------------
# 2. end to end: log -> receipt


def _session_leaking(secret: str, tmp_path):
    """A retry loop whose re-sent failure trace carries `secret`.

    Shaped like tests/test_retry.py: a trace from a failed tool call stays in
    context across later turns, which is what the retry rule attributes and
    what lands in the Top-10 excerpt column.
    """
    # The secret goes at the FRONT: an excerpt is the first ~110 characters,
    # so a secret buried at the end would never reach the receipt and this
    # whole file would pass without testing anything.
    tainted = f"Authorization: {secret}\n" + TRACE
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(MODEL, [t1])           # turn 1
    b.tool_result(t1["id"], tainted)   # turn 2 (failure, carries the secret)
    t2 = b.tool_use("Edit", {"file_path": "src/payments.py",
                             "old_string": "a", "new_string": "b"})
    b.assistant(MODEL, [t2])           # turn 3 — trace re-sent
    b.tool_result(t2["id"], "updated")
    t3 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(MODEL, [t3])           # turn 5 — trace re-sent again
    b.tool_result(t3["id"], "1 passed in 0.2s")
    return parse_built(b, tmp_path)


def _receipt(session, book, provenance="redacted-real") -> str:
    return render_receipt(session, audit_all_dials(session, book), book,
                          detail_dial="balanced", sessions_per_month=100,
                          replay=None, data_provenance=provenance)


def test_excerpt_column_is_actually_exercised(tmp_path, book):
    """Guard against this file passing vacuously.

    The end-to-end cases below only prove something if the session really
    produces excerpts. Assert that directly, on the raw ledger, before
    redaction has a chance to act.
    """
    session = _session_leaking(SECRETS[0], tmp_path)
    items = audit_all_dials(session, book)["balanced"].top_items(10)
    assert items, "no ledger entries — the end-to-end cases would be vacuous"
    assert any(it["excerpt"] for it in items), "no excerpt text to redact"


def test_no_secret_reaches_the_receipt(tmp_path, book):
    for secret in SECRETS:
        session = _session_leaking(secret, tmp_path)
        receipt = _receipt(session, book)
        assert secret not in receipt, f"{secret!r} survived into the receipt"


def test_source_path_is_not_in_the_receipt(tmp_path, book):
    session = _session_leaking(SECRETS[0], tmp_path)
    receipt = _receipt(session, book)
    assert str(tmp_path) not in receipt, "the containing directory leaked"
    assert "session.jsonl" in receipt, "the filename should still identify the run"
