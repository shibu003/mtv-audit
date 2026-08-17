"""comm rule: full-state rebroadcast to subagents."""
from conftest import GOAL, RELEVANT_BIG, parse_built

from mtv_audit.attribution import DIALS, attribute_comm
from mtv_audit.synth import SessionBuilder

MODEL = "claude-sonnet-4-6"


def _session_with_task(tmp_path, prompt: str):
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Read", {"file_path": "src/payments.py"})
    b.assistant(MODEL, [t1])
    b.tool_result(t1["id"], RELEVANT_BIG)
    t_task = b.tool_use("Task", {"description": "review", "prompt": prompt})
    b.assistant(MODEL, [t_task])
    b.tool_result(t_task["id"], "ok")
    return parse_built(b, tmp_path)


def test_rebroadcast_payload_counted(tmp_path, book):
    session = _session_with_task(tmp_path, "Please review this code:\n" + RELEVANT_BIG)
    entries = attribute_comm(session, DIALS["balanced"], book, claimed=set())
    assert len(entries) == 1
    e = entries[0]
    assert e.channel == "comm" and e.usd > 0
    assert "overlap" in e.note


def test_novel_payload_not_counted(tmp_path, book):
    novel = ("Investigate intermittent websocket disconnects on the realtime "
             "gateway under elevated packet reordering with jitter histograms "
             "and kernel buffer tuning experiments. " * 6)
    session = _session_with_task(tmp_path, novel)
    entries = attribute_comm(session, DIALS["balanced"], book, claimed=set())
    assert entries == []


def test_non_subagent_tools_ignored(tmp_path, book):
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Bash", {"command": "echo " + RELEVANT_BIG[:200]})
    b.assistant(MODEL, [t1])
    b.tool_result(t1["id"], "ok")
    session = parse_built(b, tmp_path)
    entries = attribute_comm(session, DIALS["balanced"], book, claimed=set())
    assert entries == []
