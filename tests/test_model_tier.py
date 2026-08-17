"""model rule: top-tier model on trivial steps — FLAG ONLY, never summed."""
import pytest
from conftest import GOAL, parse_built

from mtv_audit.attribution import (DIALS, RECOVERABLE_CHANNELS,
                                   attribute_model_flags, run_audit)
from mtv_audit.synth import SessionBuilder

OPUS = "claude-opus-4-5-20251101"
SONNET = "claude-sonnet-4-6"


def _trivial_session(tmp_path, model: str):
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Read", {"file_path": ".editorconfig"})
    b.assistant(model, [b.thinking("quick config check"), t1])
    b.tool_result(t1["id"], "root = true")
    return parse_built(b, tmp_path)


def test_top_tier_on_trivial_step_is_flagged(tmp_path, book):
    session = _trivial_session(tmp_path, OPUS)
    entries = attribute_model_flags(session, DIALS["balanced"], book)
    assert len(entries) == 1
    e = entries[0]
    assert e.channel == "model"
    assert e.flagged is True
    assert e.usd > 0
    assert "opus" in e.note


def test_lower_tier_on_trivial_step_not_flagged(tmp_path, book):
    session = _trivial_session(tmp_path, SONNET)
    assert attribute_model_flags(session, DIALS["balanced"], book) == []


def test_flag_excluded_from_recoverable_sum(tmp_path, book):
    session = _trivial_session(tmp_path, OPUS)
    ledger = run_audit(session, "balanced", book)
    model_usd = sum(e.usd for e in ledger.entries if e.channel == "model")
    assert model_usd > 0
    rec = ledger.recoverable_usd()
    assert rec == pytest.approx(
        sum(e.usd for e in ledger.entries if e.channel in RECOVERABLE_CHANNELS)
    )
    # the flag is reported but does not inflate the recoverable claim
    assert rec + model_usd == pytest.approx(sum(e.usd for e in ledger.entries))
