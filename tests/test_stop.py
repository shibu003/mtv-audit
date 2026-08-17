"""stop rule: tokens after the job was done (no circuit breaker)."""
from conftest import GOAL, TRACE, parse_built

from mtv_audit.attribution import DIALS, attribute_stop, find_stop_boundary
from mtv_audit.synth import SessionBuilder

MODEL = "claude-sonnet-4-6"
PASSED = ("============================= test session starts ===================\n"
          "tests/test_payments.py::test_refund PASSED\n"
          "=========================== 2 passed in 0.18s ===========================\n")


def _base(b: SessionBuilder) -> None:
    """t0 user goal, t1 Edit (state change), t2 result, t3 pytest, t4 result."""
    b.user_text(GOAL)
    e = b.tool_use("Edit", {"file_path": "src/payments.py",
                            "old_string": "a", "new_string": "b"})
    b.assistant(MODEL, [e])
    b.tool_result(e["id"], "The file src/payments.py has been updated.")
    t = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(MODEL, [t])


def test_post_success_turns_counted(tmp_path, book):
    b = SessionBuilder()
    _base(b)
    b.tool_result("toolu_0002", PASSED)            # t4: success boundary
    r = b.tool_use("Read", {"file_path": ".editorconfig"})
    b.assistant(MODEL, [b.text("Double-checking the config just in case."), r])  # t5
    b.tool_result(r["id"], "root = true")          # t6
    b.assistant(MODEL, [b.text("Everything still looks good.")])                 # t7
    session = parse_built(b, tmp_path)

    assert find_stop_boundary(session) == 4
    entries = attribute_stop(session, DIALS["balanced"], book, set(), {})
    assert [e.turn_index for e in entries] == [5, 7]
    assert all(e.usd > 0 for e in entries)
    assert all("turn 4" in e.note for e in entries)


def test_session_ending_at_success_has_zero_stop(tmp_path, book):
    b = SessionBuilder()
    _base(b)
    b.tool_result("toolu_0002", PASSED)            # t4: success, then silence
    session = parse_built(b, tmp_path)

    assert find_stop_boundary(session) == 4
    assert attribute_stop(session, DIALS["balanced"], book, set(), {}) == []


def test_abandonment_counts_from_last_state_change(tmp_path, book):
    b = SessionBuilder()
    _base(b)
    b.tool_result("toolu_0002", TRACE)             # t4: still failing
    b.assistant(MODEL, [b.text("Stopping here; will pick this up later.")])      # t5
    session = parse_built(b, tmp_path)

    # never reached success after the last state change at t1
    assert find_stop_boundary(session) == 1
    entries = attribute_stop(session, DIALS["balanced"], book, set(), {})
    assert [e.turn_index for e in entries] == [3, 5]
    assert all(e.usd > 0 for e in entries)
