"""clean rule: stale context re-reads."""
from conftest import GOAL, IRRELEVANT_BIG, RELEVANT_BIG, parse_built

from mtv_audit.attribution import DIALS, attribute_clean
from mtv_audit.synth import SessionBuilder

MODEL = "claude-sonnet-4-6"


def _base_session(tmp_path, payload: str):
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Read", {"file_path": "README.md"} if payload is IRRELEVANT_BIG
                    else {"file_path": "src/payments.py"})
    b.assistant(MODEL, [t1])
    b.tool_result(t1["id"], payload)
    # two more assistant turns that keep re-sending the block
    t2 = b.tool_use("Read", {"file_path": "src/payments.py"})
    b.assistant(MODEL, [t2])
    b.tool_result(t2["id"], "def refund(order, amount): return order.captured - amount")
    b.assistant(MODEL, [b.text("Working on src/payments.py refund now.")])
    return parse_built(b, tmp_path)


def test_irrelevant_block_counted_each_resend(tmp_path, book):
    session = _base_session(tmp_path, IRRELEVANT_BIG)
    entries = attribute_clean(session, DIALS["balanced"], book, claimed=set())
    stale = [e for e in entries if "README" not in e.block_id and IRRELEVANT_BIG[:30] in (e.excerpt + IRRELEVANT_BIG[:30])]
    targets = [e for e in entries if e.excerpt.startswith("Quarterly newsletter")]
    # the big irrelevant block is re-sent into assistant turns 3 and 5
    assert len(targets) == 2
    assert {e.turn_index for e in targets} == {3, 5}
    assert all(e.usd > 0 and e.tokens_scaled > 0 for e in targets)


def test_relevant_block_not_counted(tmp_path, book):
    session = _base_session(tmp_path, RELEVANT_BIG)
    entries = attribute_clean(session, DIALS["balanced"], book, claimed=set())
    assert not any(e.excerpt.startswith("def refund") for e in entries)


def test_small_blocks_ignored(tmp_path, book):
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Read", {"file_path": "notes.txt"})
    b.assistant(MODEL, [t1])
    b.tool_result(t1["id"], "tiny unrelated note about ficus")  # << min_block_tokens
    b.assistant(MODEL, [b.text("Continuing with src/payments.py.")])
    session = parse_built(b, tmp_path)
    entries = attribute_clean(session, DIALS["balanced"], book, claimed=set())
    assert entries == []


def test_claimed_blocks_skipped(tmp_path, book):
    session = _base_session(tmp_path, IRRELEVANT_BIG)
    stale_block_id = next(
        bl.block_id for t in session.turns for bl in t.blocks
        if bl.kind == "tool_result" and bl.text.startswith("Quarterly newsletter")
    )
    claimed = {(3, stale_block_id), (5, stale_block_id)}
    entries = attribute_clean(session, DIALS["balanced"], book, claimed=claimed)
    assert not any(e.block_id == stale_block_id for e in entries)
