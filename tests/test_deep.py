"""deep rule: overthinking on ex-post trivial steps."""
from conftest import GOAL, parse_built

from mtv_audit.attribution import DIALS, attribute_deep
from mtv_audit.synth import SessionBuilder

MODEL = "claude-sonnet-4-6"
LONG_THINK = "Deliberating elaborately about a trivial config read. " * 40  # ~540 tok


def _session(tmp_path, *, thinking: str, tool: str, tool_input: dict,
             result: str = "root = true"):
    b = SessionBuilder()
    b.user_text(GOAL)
    blocks = []
    if thinking:
        blocks.append(b.thinking(thinking))
    t1 = b.tool_use(tool, tool_input)
    blocks.append(t1)
    b.assistant(MODEL, blocks)
    b.tool_result(t1["id"], result)
    return parse_built(b, tmp_path)


def test_overthinking_on_trivial_read_counted(tmp_path, book):
    session = _session(tmp_path, thinking=LONG_THINK, tool="Read",
                       tool_input={"file_path": ".editorconfig"})
    claimed_out: dict = {}
    entries = attribute_deep(session, DIALS["balanced"], book, claimed_out)
    assert len(entries) == 1
    e = entries[0]
    think_tokens = session.turns[1].thinking_tokens()
    assert e.tokens_est == think_tokens - DIALS["balanced"].deep_thinking_allowance
    assert claimed_out[1] == e.tokens_scaled
    assert e.usd > 0


def test_short_thinking_not_counted(tmp_path, book):
    session = _session(tmp_path, thinking="brief check of the config path",
                       tool="Read", tool_input={"file_path": ".editorconfig"})
    assert attribute_deep(session, DIALS["balanced"], book, {}) == []


def test_non_trivial_step_not_counted(tmp_path, book):
    big_edit = {"file_path": "src/payments.py",
                "old_string": "x" * 400, "new_string": "y" * 400}
    session = _session(tmp_path, thinking=LONG_THINK, tool="Edit",
                       tool_input=big_edit, result="updated")
    assert attribute_deep(session, DIALS["balanced"], book, {}) == []


def test_failed_step_is_not_expost_trivial(tmp_path, book):
    session = _session(tmp_path, thinking=LONG_THINK, tool="Read",
                       tool_input={"file_path": ".editorconfig"},
                       result="Error: file not found\n1 failed")
    assert attribute_deep(session, DIALS["balanced"], book, {}) == []
