"""D-γ (H23): free-window eviction delta vs *default* compaction.

Default compaction at a free window B summarizes ALL pre-B resident context
(lossy, irreversible). A value-judged eviction keeps high-n̂ blocks verbatim
and drops low-value ones. H23 scores ONLY the disagreement — blocks where the
two policies differ — weighted by per-segment billed mass. Absolute
"evict vs no-evict" is rejected: it double-counts what compaction already
saves (the D-α / B.3 optimism). The delta is the gain compaction leaves on
the table, i.e. what an own layer would add *on top of* the platform.

Disagreement direction (one boundary): default DROPS every pre-B block
(summarized). SVM KEEPS verbatim the blocks it predicts referenced after B.
  disagreement = { pre-B pageable blocks referenced after B }
  gross_mass   = Σ billed_seg(b)                         (H23 literal win formula)
  net_delta    = Σ [ reread_avoided(b) − extra_rent(b) ] (economically real)

  reread_avoided: default dropped b, so a post-B use must re-fetch the body
    (D-α observed: disk re-read) → billed·P_WRITE + CALL_OVERHEAD.
  extra_rent: SVM keeps b verbatim from B to its first post-B use, paying a
    cached read each assistant turn → billed·P_READ·residency_after.

Per-segment α is ENFORCED (H23 ① finding): each block is billed with the α of
the segment it was born in. α shifts across the boundary (Shibubu 2.524→0.816),
so a single global α skews the delta. The health report surfaces whether the
delta survives the α choice — if the disagreement mass is priced entirely at an
unexplained high α, the verdict is HELD (D-α discipline: separate real effect
from measurement artifact before any GREEN/RED).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Session
from .svm import (
    CALL_OVERHEAD, P_READ, P_WRITE, TAU_REF,
    Calibration, build_block_lives, calibrate,
)


@dataclass
class DisagreeBlock:
    block_id: str
    kind: str
    tool_name: str | None
    est_tokens: int
    segment_start: int            # start_turn of the block's calibration segment
    seg_alpha: float              # per-segment α used to bill it
    billed_seg: float             # seg_alpha · est  (per-segment billed mass)
    billed_global: float          # global primary α · est  (for back-out)
    birth_turn: int
    refs_after: list[int]         # post-boundary causal references
    first_ref_after: int
    residency_after: int          # assistant turns kept resident B -> first_ref_after
    reread_avoided: float         # billed cost default pays to re-fetch the body
    extra_rent: float             # billed cost SVM pays to keep it verbatim
    net: float                    # reread_avoided − extra_rent
    excerpt: str


@dataclass
class DGammaResult:
    boundary_turn: int
    segments: list                # SegmentFit list (per-segment O/α/R²)
    total_billed_seg: float       # denominator for θ_δ (all pageable, per-seg α)
    gross_mass_seg: float         # Σ billed_seg over disagreement (H23 literal)
    gross_mass_global: float      # same set, global α  (α back-out comparison)
    net_delta_seg: float
    net_delta_global: float
    disagree: list                # list[DisagreeBlock], sorted by billed_seg desc
    seg_distribution: dict        # segment_start -> count of disagreement blocks
    theta_delta: float            # threshold (fraction of total billed)
    # verdict fields
    delta_ratio_seg: float        # net_delta_seg / total_billed_seg
    survives_alpha: bool          # sign stable across per-seg vs global α
    verdict: str                  # "GREEN" | "RED" | "HELD"
    note: str = ""


def _segment_for(turn_index: int, segments: list) -> object:
    """The calibration segment a turn belongs to (last segment starting <= t)."""
    chosen = segments[0]
    for s in segments:
        if s.start_turn <= turn_index:
            chosen = s
        else:
            break
    return chosen


def analyze_dgamma(session: Session, cal: Calibration | None = None,
                   ref_metric: str = "block", theta_delta: float = 0.10,
                   top_k: int = 12) -> DGammaResult | None:
    """Compute the H23 delta on one free-window session. Returns None if the
    session has no compaction boundary (D-γ is N/A — like T1)."""
    cal = cal or calibrate(session)
    boundaries = [t.index for t in session.turns if t.is_compact_summary]
    if not boundaries:
        return None
    B = boundaries[0]                          # first free window
    segments = cal.segments
    global_alpha = cal.scale
    assist_after = [t.index for t in session.assistant_turns() if t.index > B]

    lives = build_block_lives(session, cal, ref_metric=ref_metric)

    # denominator: total pageable billed mass, per-segment α
    total_billed_seg = 0.0
    for bl in lives:
        seg = _segment_for(bl.birth_turn, segments)
        total_billed_seg += seg.scale * bl.est_tokens

    disagree: list[DisagreeBlock] = []
    for bl in lives:
        if bl.birth_turn >= B:                 # only pre-boundary blocks can disagree
            continue
        refs_after = [r for r in bl.ref_turns if r > B]
        if not refs_after:                     # SVM drops too -> agreement, no delta
            continue
        seg = _segment_for(bl.birth_turn, segments)
        seg_alpha = seg.scale
        billed_seg = seg_alpha * bl.est_tokens
        billed_global = global_alpha * bl.est_tokens
        first_ref = min(refs_after)
        residency_after = sum(1 for i in assist_after if i <= first_ref)
        reread_avoided = billed_seg * P_WRITE + CALL_OVERHEAD
        extra_rent = billed_seg * P_READ * residency_after
        net = reread_avoided - extra_rent
        disagree.append(DisagreeBlock(
            block_id=bl.block_id, kind=bl.kind, tool_name=bl.tool_name,
            est_tokens=bl.est_tokens, segment_start=seg.start_turn,
            seg_alpha=seg_alpha, billed_seg=billed_seg, billed_global=billed_global,
            birth_turn=bl.birth_turn, refs_after=refs_after, first_ref_after=first_ref,
            residency_after=residency_after, reread_avoided=reread_avoided,
            extra_rent=extra_rent, net=net, excerpt=bl.excerpt))

    disagree.sort(key=lambda d: d.billed_seg, reverse=True)

    gross_mass_seg = sum(d.billed_seg for d in disagree)
    gross_mass_global = sum(d.billed_global for d in disagree)
    net_delta_seg = sum(d.net for d in disagree)
    # net under global α: re-bill reread/rent with global α
    net_delta_global = sum(
        (d.billed_global * P_WRITE + CALL_OVERHEAD)
        - (d.billed_global * P_READ * d.residency_after)
        for d in disagree)

    seg_dist: dict[int, int] = {}
    for d in disagree:
        seg_dist[d.segment_start] = seg_dist.get(d.segment_start, 0) + 1

    delta_ratio_seg = net_delta_seg / total_billed_seg if total_billed_seg else 0.0
    survives_alpha = (net_delta_seg > 0) == (net_delta_global > 0)

    # Verdict (暫定). HOLD if the delta does not survive the α choice, OR if the
    # disagreement mass is confined to a single segment whose α is an unexplained
    # outlier vs the other segment (can't tell real value from unit price yet).
    alphas = sorted({round(s.scale, 3) for s in segments})
    alpha_spread = (max(alphas) / min(alphas)) if len(alphas) > 1 and min(alphas) > 0 else 1.0
    single_seg_disagree = len(seg_dist) <= 1
    held = (not survives_alpha) or (single_seg_disagree and alpha_spread >= 2.0)

    if held:
        verdict = "HELD"
        note = (f"α spread across segments = {alpha_spread:.1f}x and disagreement "
                f"is confined to {len(seg_dist)} segment(s); cannot yet separate "
                f"real value from unit-price artifact. Explain the α divergence "
                f"before GREEN/RED (D-α discipline).")
    elif delta_ratio_seg >= theta_delta:
        verdict = "GREEN"
        note = f"net delta {delta_ratio_seg:.1%} ≥ θ_δ={theta_delta:.0%} (暫定)"
    else:
        verdict = "RED"
        note = f"net delta {delta_ratio_seg:.1%} < θ_δ={theta_delta:.0%} (暫定)"

    return DGammaResult(
        boundary_turn=B, segments=segments, total_billed_seg=total_billed_seg,
        gross_mass_seg=gross_mass_seg, gross_mass_global=gross_mass_global,
        net_delta_seg=net_delta_seg, net_delta_global=net_delta_global,
        disagree=disagree, seg_distribution=seg_dist, theta_delta=theta_delta,
        delta_ratio_seg=delta_ratio_seg, survives_alpha=survives_alpha,
        verdict=verdict, note=note)


def render_dgamma(res: DGammaResult, top_k: int = 12) -> str:
    if res is None:
        return "D-γ N/A: no compaction free window in this session."
    L: list[str] = []
    p = L.append
    p("# D-γ (H23) — free-window eviction delta vs default compaction\n")
    p(f"- boundary turn: {res.boundary_turn}")
    p("- calibration segments (per-segment α, H23 ① health):")
    for s in res.segments:
        p(f"    start_turn={s.start_turn:5d}  n={s.n:5d}  α={s.scale:.3f}  R²={s.r2:.4f}")
    p(f"- total pageable billed mass (per-seg α): {res.total_billed_seg:,.0f}")
    p(f"- disagreement blocks (pre-B, referenced after B): {len(res.disagree)}")
    p(f"- segment distribution of disagreement: {res.seg_distribution}")
    p("")
    p("## H23 delta")
    p(f"- gross disagreement mass  (literal win) : per-seg α {res.gross_mass_seg:,.0f}"
      f"  |  global α {res.gross_mass_global:,.0f}")
    p(f"- net delta (reread_avoided − extra_rent): per-seg α {res.net_delta_seg:,.0f}"
      f"  |  global α {res.net_delta_global:,.0f}")
    p(f"- net delta ratio (per-seg / total billed): {res.delta_ratio_seg:.2%}"
      f"   (θ_δ = {res.theta_delta:.0%}, 暫定)")
    p(f"- survives α back-out (sign stable): {res.survives_alpha}")
    p("")
    p(f"## VERDICT: {res.verdict} (暫定)")
    p(f"  {res.note}")
    p("")
    p(f"## qualitative sanity — top {top_k} disagreement blocks by billed mass")
    for d in res.disagree[:top_k]:
        tn = d.tool_name or d.kind
        p(f"- [{tn}] est={d.est_tokens} billed_seg={d.billed_seg:,.0f} "
          f"refs_after={len(d.refs_after)} resid_after={d.residency_after} "
          f"net={d.net:,.0f}")
        p(f"    «{d.excerpt}»")
    return "\n".join(L)


def _main(argv: list[str] | None = None) -> int:
    import sys
    from .parser import load_session
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m mtv_audit.dgamma <session.jsonl>")
        return 2
    session = load_session(args[0])
    res = analyze_dgamma(session)
    print(render_dgamma(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
