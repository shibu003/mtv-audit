"""D-γ (H23) analysis tests: boundary handling, per-segment α, HELD discipline."""
from mtv_audit.dgamma import analyze_dgamma, _segment_for
from mtv_audit.model import Block, Session, Turn
from mtv_audit.svm import SegmentFit, calibrate


def _assist(idx, text, prompt, out=20, ts=None, compact=False, kind="tool_result"):
    return Turn(index=idx, role="assistant",
                blocks=[Block(kind=kind, text=text, block_id=f"b{idx}.{kind}")],
                usage={"input_tokens": prompt, "output_tokens": out},
                timestamp=ts, is_compact_summary=compact)


def _compact_turn(idx, summary):
    return Turn(index=idx, role="user",
                blocks=[Block(kind="text", text=summary, block_id=f"c{idx}.text")],
                is_compact_summary=True)


def test_no_boundary_returns_none():
    s = Session(turns=[_assist(i, f"payload alpha beta {i} " * 20, 1000 + i * 500)
                       for i in range(6)])
    assert analyze_dgamma(s) is None


def test_boundary_detected_and_segments_split():
    # 4 pre-boundary assistant turns, a compaction turn, then 4 post-boundary.
    shared = "the buddy life photo pipeline gemini sharp compositor r2 storage fact upsert "
    turns = []
    for i in range(4):
        turns.append(_assist(i, shared * 30, 2000 + i * 800))
    turns.append(_compact_turn(4, "summary: " + shared * 5))
    for j in range(5, 9):
        # post-boundary turns echo the shared vocabulary in assistant text
        # (refs are detected on assistant text + tool_use, not tool_result)
        turns.append(_assist(j, shared * 30, 1500 + (j - 5) * 300, kind="text"))
    s = Session(turns=turns)
    res = analyze_dgamma(s)
    assert res is not None
    assert res.boundary_turn == 4
    # calibration split into two segments at the boundary
    assert len(res.segments) == 2
    # disagreement = pre-B blocks referenced after B; the shared vocab guarantees some
    assert len(res.disagree) >= 1
    assert all(d.birth_turn < 4 for d in res.disagree)


def test_per_segment_alpha_billing():
    segs = [SegmentFit(0, 10, 100.0, 2.5, 0.99), SegmentFit(50, 10, 100.0, 0.8, 0.95)]
    assert _segment_for(3, segs).scale == 2.5
    assert _segment_for(70, segs).scale == 0.8
    assert _segment_for(50, segs).scale == 0.8   # boundary turn belongs to its own segment


def test_held_when_single_segment_disagreement_and_alpha_spread():
    # construct a result via the real path; with one boundary all pre-B
    # disagreement sits in segment 0, and if α spread >= 2x the verdict is HELD.
    shared = "alpha beta gamma delta epsilon zeta payload content body text "
    turns = [_assist(i, shared * 40, 3000 + i * 1200) for i in range(5)]
    turns.append(_compact_turn(5, "sum " + shared * 3))
    for j in range(6, 10):
        turns.append(_assist(j, shared * 40, 1200 + (j - 6) * 200, kind="text"))
    res = analyze_dgamma(Session(turns=turns))
    if res and len(res.seg_distribution) <= 1:
        alphas = [s.scale for s in res.segments]
        if max(alphas) / min(alphas) >= 2.0:
            assert res.verdict == "HELD"
