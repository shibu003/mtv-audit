"""retry rule: contaminated retries."""
from conftest import GOAL, TRACE, parse_built

from mtv_audit.attribution import DIALS, attribute_clean, attribute_retry
from mtv_audit.synth import SessionBuilder

MODEL = "claude-sonnet-4-6"


def _failing_session(tmp_path):
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(MODEL, [t1])           # turn 1
    b.tool_result(t1["id"], TRACE)     # turn 2 (failure)
    t2 = b.tool_use("Edit", {"file_path": "src/payments.py",
                             "old_string": "a", "new_string": "b"})
    b.assistant(MODEL, [t2])           # turn 3
    b.tool_result(t2["id"], "updated")
    t3 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(MODEL, [t3])           # turn 5
    b.tool_result(t3["id"], "1 passed in 0.2s")
    return parse_built(b, tmp_path)


def test_failure_trace_counted_each_subsequent_turn(tmp_path, book):
    session = _failing_session(tmp_path)
    entries = attribute_retry(session, DIALS["balanced"], book, claimed=set())
    assert {e.turn_index for e in entries} == {3, 5}
    assert all(e.channel == "retry" and e.usd > 0 for e in entries)


def test_nothing_counted_before_failure(tmp_path, book):
    session = _failing_session(tmp_path)
    entries = attribute_retry(session, DIALS["balanced"], book, claimed=set())
    assert not any(e.turn_index <= 2 for e in entries)


def test_precedence_retry_claims_before_clean(tmp_path, book):
    """The trace mentions the goal files, so clean would see it as relevant —
    but it must be claimed by retry and never double-counted by clean."""
    session = _failing_session(tmp_path)
    claimed: set = set()
    retry_entries = attribute_retry(session, DIALS["saver"], book, claimed)
    clean_entries = attribute_clean(session, DIALS["saver"], book, claimed)
    trace_ids = {e.block_id for e in retry_entries}
    assert trace_ids
    assert not any(e.block_id in trace_ids for e in clean_entries)
    keys = [(e.turn_index, e.block_id) for e in retry_entries + clean_entries]
    assert len(keys) == len(set(keys))  # no (turn, block) counted twice


def test_optimizer_grace_skips_first_retry_turn(tmp_path, book):
    session = _failing_session(tmp_path)
    entries = attribute_retry(session, DIALS["optimizer"], book, claimed=set())
    assert {e.turn_index for e in entries} == {5}  # grace=1 tolerates turn 3
