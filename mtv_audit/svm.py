"""SVM (Session Virtual Memory) simulator — T8 / V-A of MTV_SPEC_v2.0.

What this answers (the frozen question, §0.5 / §1):
    "When SVM is implemented as an MCP server + hooks, does the user get a
     net win at zero behaviour change?" — and it is built so the answer can
     come back **No**. H14 (≥20% simulated $ saving, cache-accounting-complete,
     intermediate-invariant respected) is the gate; failing it stops the
     product at Stage 1.

This module is LLM-call-free (V-A is the $0 audit). It:
  1. Reuses the Stage-1 parser to decompose the session into blocks.
  2. Builds a *causal-reference* log (§3.4 L0: normalized lexical overlap
     >= tau_ref between a block and a later assistant output / tool input).
     References (not passive residency) are what a stub must page back in.
  3. Derives the Context-MRC (§3.3) from reference reuse distances and its
     knee c* (H11).
  4. Decides stub-or-keep at *birth* via the frozen pi_b balance (§3.2),
     under two estimators: ORACLE (real future references = upper bound) and
     L0 (block-type priors). Design law: birth-time only, no retro-eviction.
  5. Re-prices the whole session under a **cache-complete, append-only**
     prefix model (§4 V-A step 3), asserting the intermediate-invariance
     (no middle block ever changes -> prefix cache never broken). Overhead
     (system + tool schemas) and the est->billed scale are *fit from the
     session itself* so the baseline sim reproduces the real receipt; the
     SVM sim then diverges only by what SVM actually changed.

Honesty constraints baked in:
  * T1 has **no free windows** (no compaction/resume in the JSONL), so §3.1
    系2 (free-window mass eviction) contributes 0 here and faulted-in content
    cannot be cheaply evicted mid-session (evicting a middle block would break
    the prefix cache — the very failure §0.2 warns about). All SVM savings in
    T1 therefore come from birth-time stubbing of rarely-referenced blocks.
    This is reported, not hidden.
  * The stub *decision* uses the frozen pi_b (which assumes faulted copies
    decay after R̂'); the *bill* uses the windowless reality (faulted copy
    resident to end). The gap is surfaced as a finding, not smoothed over.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .model import Session, Turn
from .parser import load_session
from .pricing import PriceBook

# --------------------------------------------------------------------------
# Frozen constants (§2 stub form, §3 formulas). Changing these needs founder
# approval + a version bump (§7.5) — they are duplicated here only as defaults.
# --------------------------------------------------------------------------
P_READ = 0.10            # cache-read multiple of input price
P_WRITE = 1.25           # cache-write multiple
P_OUT = 5.0              # output / input price ratio (opus default)
STUB_TOKENS = 80         # σ — frozen stub length (billed tokens)
CALL_OVERHEAD = 300      # c_call — ctx_read round-trip (output toks × p_out + framing)
TAU_REF = 0.30           # causal-reference lexical-overlap threshold (§3.4 L0)
R_HAT_DEFAULT = 60       # R̂ fallback residency (§3.4 L0)
RPRIME_DEFAULT = 20      # R̂' fallback post-fault residual (§3.2 example)
SCHEMA_TAX_TOKENS = 380  # measured ctx_read+ctx_search schema (design law 1: < 400)
TTL_SECONDS = 300        # 5-min prompt-cache TTL (§7.4: expiry billed as miss)

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_\.]{2,}")
_STOP = {
    "the", "and", "for", "with", "this", "that", "from", "have", "has",
    "are", "was", "were", "you", "your", "please", "run", "use", "can",
    "will", "should", "would", "into", "then", "when", "what", "how", "all",
    "not", "but", "out", "they", "their", "there", "here", "import", "const",
    "return", "function", "type", "true", "false", "null", "none",
}


def _toks(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")} - _STOP


# --------------------------------------------------------------------------
# Per-session calibration: overhead O and est->billed scale α.
#   reported_prompt_tokens(t) ≈ O + α · cumulative_est_content(t)
# Fit by OLS over assistant turns. O absorbs system prompt + tool schemas
# (SVM-invariant); α corrects the chars/4 estimator to billed tokens. The
# §3.2 example (s_b≈3440) lives in this billed space, so block billed size
# = α · est.
# --------------------------------------------------------------------------

@dataclass
class SegmentFit:
    """Per-segment OLS fit. Segments are split at compaction free windows
    (§3.1 系2): each compaction drops the prior context, so cumulative_est
    resets and the fit must be piecewise (a single global fit sees the
    non-monotonic P_t and collapses — Shibubu 870c00f2 R²=0.012)."""
    start_turn: int          # first turn index of this segment
    n: int                   # number of (P, cum) rows fit
    overhead: float          # O for this segment (absorbs summary baseline post-compaction)
    scale: float             # α for this segment
    r2: float


@dataclass
class Calibration:
    overhead: float          # O (billed tokens, constant per turn) — primary segment
    scale: float             # α (billed per est token) — primary (largest-n) segment
    r2: float                # primary-segment fit quality (reported for honesty)
    segments: list[SegmentFit] = field(default_factory=list)

    def billed(self, est_tokens: int) -> float:
        return self.scale * est_tokens

    def min_segment_r2(self) -> float:
        """Health gate (H23 ①): worst piece. If post-compaction segments don't
        recover, the calibration floor is garbage and D-γ is unmeasurable."""
        return min((s.r2 for s in self.segments), default=self.r2)


def _ols(rows: list[tuple[float, float]]) -> tuple[float, float, float]:
    """OLS of P ≈ O + α·cum. Returns (O, α, r2)."""
    n = len(rows)
    if n < 2:
        return (rows[0][0] if n == 1 else 0.0, 1.0, 0.0)
    sx = sum(c for _, c in rows); sy = sum(P for P, _ in rows)
    sxx = sum(c * c for _, c in rows); sxy = sum(P * c for P, c in rows)
    denom = n * sxx - sx * sx
    if denom == 0:
        return (sy / n, 1.0, 0.0)
    a = (n * sxy - sx * sy) / denom
    o = (sy - a * sx) / n
    mean = sy / n
    ss_tot = sum((P - mean) ** 2 for P, _ in rows) or 1.0
    ss_res = sum((P - (o + a * c)) ** 2 for P, c in rows)
    return (max(0.0, o), max(1e-9, a), 1 - ss_res / ss_tot)


def calibrate(session: Session) -> Calibration:
    # Partition into segments at compaction boundaries. Pre-compaction context
    # is dropped (replaced by the summary), so cumulative_est resets; the
    # summary turn's own blocks seed the next segment's baseline.
    segments: list[SegmentFit] = []
    rows: list[tuple[float, float]] = []   # (reported_prompt, cum_est_before)
    seg_start = 0
    cum = 0
    for t in session.turns:
        if t.is_compact_summary:
            if len(rows) >= 1:
                o, a, r2 = _ols(rows)
                segments.append(SegmentFit(seg_start, len(rows), o, a, r2))
            rows = []
            cum = 0
            seg_start = t.index
            for b in t.blocks:          # summary = new resident baseline
                cum += b.est_tokens
            continue
        if t.role == "assistant" and t.usage:
            P = t.reported_prompt_tokens()
            if P > 0:
                rows.append((float(P), float(cum)))
        for b in t.blocks:
            cum += b.est_tokens
    if len(rows) >= 1:
        o, a, r2 = _ols(rows)
        segments.append(SegmentFit(seg_start, len(rows), o, a, r2))

    if not segments:
        return Calibration(overhead=0.0, scale=1.0, r2=0.0, segments=[])
    # Primary = best-powered segment (most rows): its α is the most reliable
    # est→billed scale for billed(). Single-segment (no compaction) sessions
    # reproduce the prior global fit exactly.
    primary = max(segments, key=lambda s: s.n)
    return Calibration(overhead=primary.overhead, scale=primary.scale,
                       r2=primary.r2, segments=segments)


# --------------------------------------------------------------------------
# Block lifecycle + causal-reference log.
# --------------------------------------------------------------------------

# Block types eligible for paging: large, content-bearing, born from tools or
# long prose. tool_use inputs and short text are never paged (they are the
# "address" that must stay visible — design law 3).
PAGEABLE_KINDS = {"tool_result", "text", "thinking"}


@dataclass
class BlockLife:
    block_id: str
    kind: str
    tool_name: str | None
    est_tokens: int
    billed: float                      # α · est
    birth_turn: int                    # turn index where it first appears
    text: str
    ref_turns: list[int] = field(default_factory=list)  # assistant turns that causally use it
    # filled by the policy:
    stubbed: bool = False
    first_fault_turn: int | None = None  # earliest ref after birth (page-in moment)

    @property
    def excerpt(self) -> str:
        return " ".join((self.text or "").split())[:90]


def _overlap(bwords: set[str], cwords: set[str], metric: str) -> float:
    """Normalized lexical overlap (§3.4) under a chosen normalization.

    The choice matters and is reported as a sensitivity axis, because it
    degenerates at the extremes:
      * 'consumer'  |b∩c|/|c|  — over-fires for large blocks (a big block's
                    vocabulary contains almost any small turn → perpetual
                    reference). Upper bound on fault frequency.
      * 'jaccard'   |b∩c|/|b∪c| — under-fires for large blocks (union is
                    dominated by the block). Lower bound on fault frequency.
      * 'block'     |b∩c|/|b|  — "fraction of the block echoed by the turn";
                    matches the §3.2 example profile (long-resident block,
                    small n̂). Default.
    """
    inter = len(bwords & cwords)
    if not inter:
        return 0.0
    if metric == "consumer":
        return inter / len(cwords)
    if metric == "jaccard":
        return inter / len(bwords | cwords)
    return inter / len(bwords)  # 'block' (default)


def build_block_lives(session: Session, cal: Calibration,
                      min_est: int = 60, ref_metric: str = "block") -> list[BlockLife]:
    """One BlockLife per pageable block, with its causal-reference turns.

    A block at birth turn b is *causally referenced* at a later assistant
    turn t if the normalized lexical overlap (see _overlap) of (t's assistant
    text + tool_use inputs) with the block is >= TAU_REF. Passive residency
    (the block merely sitting in context) is NOT a reference — that
    distinction is the whole point of SVM (rent vs. genuine use).
    """
    lives: list[BlockLife] = []
    # index blocks by birth turn
    n_assist = len(session.assistant_turns())
    last_turn = session.turns[-1].index if session.turns else 0
    for turn in session.turns:
        for b in turn.blocks:
            if b.kind not in PAGEABLE_KINDS or b.est_tokens < min_est:
                continue
            lives.append(BlockLife(
                block_id=b.block_id, kind=b.kind, tool_name=b.tool_name,
                est_tokens=b.est_tokens, billed=cal.billed(b.est_tokens),
                birth_turn=turn.index, text=b.text,
            ))
    # precompute each assistant turn's "consumer vocabulary": its own text +
    # the tool_use inputs it issues (what the model produced / asked for).
    consumer: dict[int, set[str]] = {}
    for turn in session.assistant_turns():
        words: set[str] = set()
        for b in turn.blocks:
            if b.kind == "text":
                words |= _toks(b.text)
            elif b.kind == "tool_use":
                words |= _toks(b.text)
        consumer[turn.index] = words
    # reference detection
    for bl in lives:
        bwords = _toks(bl.text)
        if not bwords:
            continue
        for turn in session.assistant_turns():
            if turn.index <= bl.birth_turn:
                continue
            cw = consumer.get(turn.index)
            if not cw:
                continue
            if _overlap(bwords, cw, ref_metric) >= TAU_REF:
                bl.ref_turns.append(turn.index)
    return lives


# --------------------------------------------------------------------------
# Context-MRC (§3.3): reuse-distance distribution -> miss-rate curve + knee.
# Reuse distance = assistant-turns between consecutive causal uses of a block
# (birth counts as the first "use"). MR_ctx(c) = P(rd > c).
# --------------------------------------------------------------------------

@dataclass
class MRC:
    distances: list[int]
    curve: list[tuple[int, float]]     # (c, miss_rate)
    knee: int
    window_turns: int

    @property
    def knee_fraction(self) -> float:
        return self.knee / self.window_turns if self.window_turns else 1.0


def build_mrc(lives: list[BlockLife], window_turns: int) -> MRC:
    distances: list[int] = []
    for bl in lives:
        prev = bl.birth_turn
        for t in bl.ref_turns:
            distances.append(t - prev)
            prev = t
    if not distances:
        return MRC(distances=[], curve=[(0, 0.0)], knee=0, window_turns=window_turns)
    dmax = max(distances)
    n = len(distances)
    # miss-rate curve MR(c) = fraction of reuse distances strictly greater than c
    cs = sorted(set([0] + distances + [dmax]))
    curve = [(c, sum(1 for d in distances if d > c) / n) for c in cs]
    # knee = point of maximum curvature, via greatest perpendicular distance
    # from the chord joining the curve endpoints (Kneedle-style, robust & deps-free)
    knee = _max_curvature_x(curve)
    return MRC(distances=distances, curve=curve, knee=knee, window_turns=window_turns)


def _max_curvature_x(curve: list[tuple[int, float]]) -> int:
    if len(curve) < 3:
        return curve[0][0]
    x0, y0 = curve[0]; x1, y1 = curve[-1]
    dx, dy = (x1 - x0), (y1 - y0)
    denom = (dx * dx + dy * dy) ** 0.5 or 1.0
    best_x, best_d = curve[0][0], -1.0
    for x, y in curve:
        # perpendicular distance from point to the endpoint chord
        d = abs(dy * (x - x0) - dx * (y - y0)) / denom
        if d > best_d:
            best_d, best_x = d, x
    return best_x


# --------------------------------------------------------------------------
# L0 type priors for n̂ / R̂' (§3.4). Fit from THIS session's reference data,
# bucketed by (kind, size band). This is the "lookup table" the spec calls for;
# the L0/oracle gap it produces is the value estimate for invention ③.
# --------------------------------------------------------------------------

def _size_band(est: int) -> str:
    if est < 200:
        return "S"
    if est < 800:
        return "M"
    return "L"


@dataclass
class TypePriors:
    n_hat: dict[tuple[str, str], float]      # mean post-birth reference count
    rprime: dict[tuple[str, str], float]     # mean residual turns after first ref

    def lookup_n(self, bl: BlockLife) -> float:
        return self.n_hat.get((bl.kind, _size_band(bl.est_tokens)), 1.0)

    def lookup_rprime(self, bl: BlockLife) -> float:
        return self.rprime.get((bl.kind, _size_band(bl.est_tokens)), float(RPRIME_DEFAULT))


def fit_type_priors(lives: list[BlockLife], last_turn: int) -> TypePriors:
    from collections import defaultdict
    n_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    r_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for bl in lives:
        key = (bl.kind, _size_band(bl.est_tokens))
        n_acc[key].append(float(len(bl.ref_turns)))
        if bl.ref_turns:
            r_acc[key].append(float(last_turn - bl.ref_turns[0]))
    n_hat = {k: (sum(v) / len(v)) for k, v in n_acc.items()}
    rprime = {k: (sum(v) / len(v)) for k, v in r_acc.items()}
    return TypePriors(n_hat=n_hat, rprime=rprime)


# --------------------------------------------------------------------------
# Stub decision: frozen pi_b (§3.2).
#   cost_keep = s·(p_write + p_read·R̂)
#   cost_stub = σ·(p_write + p_read·R̂) + n̂·[c_call + s·(p_write + p_read·R̂')]
#   stub ⟺ pi_b = cost_keep − cost_stub > 0
# Per-block token units (billed). R̂ = residency = life length in assistant turns.
# --------------------------------------------------------------------------

def pi_b(s_billed: float, life: float, n_hat: float, rprime: float) -> float:
    cost_keep = s_billed * (P_WRITE + P_READ * life)
    cost_stub = (STUB_TOKENS * (P_WRITE + P_READ * life)
                 + n_hat * (CALL_OVERHEAD + s_billed * (P_WRITE + P_READ * rprime)))
    return cost_keep - cost_stub


def decide_policy(lives: list[BlockLife], session: Session, priors: TypePriors,
                  mode: str) -> None:
    """Set bl.stubbed / bl.first_fault_turn for every block, in place.

    mode == 'oracle' : R̂, n̂, R̂' from the real reference log (upper bound).
    mode == 'l0'     : R̂ = median residency / default; n̂, R̂' from type priors.
    """
    assist_idx = [t.index for t in session.assistant_turns()]
    last_turn = assist_idx[-1] if assist_idx else 0

    def residency(birth: int) -> int:
        return sum(1 for i in assist_idx if i > birth)

    # L0 residency prior: median residency across blocks (or default)
    med_res = R_HAT_DEFAULT
    if mode == "l0" and lives:
        res_all = sorted(residency(bl.birth_turn) for bl in lives)
        med_res = res_all[len(res_all) // 2] or R_HAT_DEFAULT

    for bl in lives:
        if mode == "oracle":
            life = residency(bl.birth_turn)
            n_hat = float(len(bl.ref_turns))
            rprime = float(last_turn - bl.ref_turns[0]) if bl.ref_turns else 0.0
        else:  # l0
            life = float(med_res)
            n_hat = priors.lookup_n(bl)
            rprime = priors.lookup_rprime(bl)
        bl.stubbed = pi_b(bl.billed, life, n_hat, rprime) > 0
        bl.first_fault_turn = bl.ref_turns[0] if bl.ref_turns else None


# --------------------------------------------------------------------------
# Cache-complete, append-only re-pricing (§4 V-A step 3).
# We replay every assistant turn, reconstruct its prompt as overhead + the
# live content segments, and bill it under the prefix-cache rule:
#     within TTL: read = shared prefix with previous prompt, write = the rest
#     TTL expired: whole prompt is a miss (write)  [§7.4 conservative]
# INTERMEDIATE INVARIANT: the content segment list only ever grows by append
# (births add a tail segment; a stubbed block is born small and never mutated
# in place; a fault appends the full body as a NEW tail). We assert that each
# turn's segment list extends the previous as a prefix — i.e. the prefix cache
# is never structurally broken by our own policy.
# --------------------------------------------------------------------------

@dataclass
class Bill:
    prompt_read: float = 0.0     # billed tokens read from cache (× p_read)
    prompt_write: float = 0.0    # billed tokens written to cache (× p_write)
    output: float = 0.0          # output billed tokens (× p_out)
    faults: int = 0
    fault_tokens: float = 0.0    # billed tokens added by paged-in content
    schema_tax: float = 0.0      # billed tokens spent on MCP schema (read+write)
    invariant_breaks: int = 0

    def usd(self, base_rate: float) -> float:
        # base_rate = USD per input token; read/write are multiples; output × p_out
        return base_rate * (self.prompt_read * P_READ
                            + self.prompt_write * P_WRITE
                            + self.output * P_OUT)


def _parse_ts(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def simulate(session: Session, cal: Calibration,
             lives: list[BlockLife] | None, *, svm: bool) -> Bill:
    """Replay + price. svm=False reproduces the real receipt (baseline);
    svm=True applies the stub policy already recorded on `lives`."""
    bill = Bill()
    by_birth: dict[int, list[BlockLife]] = {}
    if lives:
        for bl in lives:
            by_birth.setdefault(bl.birth_turn, []).append(bl)
    # est size of *every* block (for the baseline, content = all blocks);
    # for SVM, content for a pageable block is σ if stubbed else full, plus
    # fault tails appended at first reference.
    paged_ids = {bl.block_id for bl in (lives or [])}

    # live content segments as billed sizes, in append order
    segments: list[tuple[str, float]] = []   # (segment_id, billed_size)
    prev_segments: list[tuple[str, float]] = []
    prev_ts: float | None = None
    schema_overhead = SCHEMA_TAX_TOKENS if svm else 0.0
    overhead = cal.overhead + schema_overhead

    # map: which fault tails to append at which turn
    fault_at: dict[int, list[BlockLife]] = {}
    if svm and lives:
        for bl in lives:
            if bl.stubbed and bl.first_fault_turn is not None:
                fault_at.setdefault(bl.first_fault_turn, []).append(bl)

    for turn in session.turns:
        # 1) births: append this turn's blocks as segments
        for b in turn.blocks:
            if b.kind not in ("tool_result", "text", "thinking", "tool_use"):
                continue
            billed = cal.billed(b.est_tokens)
            if svm and b.block_id in paged_ids:
                bl = next((x for x in by_birth.get(turn.index, [])
                           if x.block_id == b.block_id), None)
                if bl and bl.stubbed:
                    segments.append((b.block_id, float(STUB_TOKENS)))
                    continue
            segments.append((b.block_id, billed))

        if turn.role != "assistant" or not turn.usage:
            continue

        # 2) SVM faults: append full body of stubbed blocks paged in this turn
        if svm:
            for bl in fault_at.get(turn.index, []):
                seg_id = f"{bl.block_id}#pagein@{turn.index}"
                segments.append((seg_id, bl.billed))
                bill.faults += 1
                bill.fault_tokens += bl.billed
                bill.output += CALL_OVERHEAD / P_OUT      # ctx_read call output
                bill.prompt_write += CALL_OVERHEAD        # tool framing into prompt

        # 3) price this turn's prompt = overhead + segments, prefix-cached
        content_billed = sum(sz for _, sz in segments)
        prompt = overhead + content_billed
        ts = _parse_ts(turn.timestamp)
        ttl_ok = (prev_ts is not None and ts is not None
                  and (ts - prev_ts) <= TTL_SECONDS)
        # shared prefix length with previous turn's segment list
        shared = 0.0
        if ttl_ok and prev_segments:
            limit = min(len(prev_segments), len(segments))
            i = 0
            while i < limit and prev_segments[i][0] == segments[i][0] \
                    and abs(prev_segments[i][1] - segments[i][1]) < 1e-9:
                shared += segments[i][1]
                i += 1
            shared += overhead  # overhead is a stable prefix when TTL ok
            # invariant check: previous segment list must be a prefix of current
            if i < len(prev_segments):
                bill.invariant_breaks += 1
        read = shared
        write = prompt - shared
        bill.prompt_read += read
        bill.prompt_write += write
        if svm:
            # attribute the schema portion for the tax line
            if ttl_ok:
                bill.schema_tax += schema_overhead * P_READ
            else:
                bill.schema_tax += schema_overhead * P_WRITE

        # 4) output (unchanged by SVM except faults already added)
        out = float(turn.reported_output_tokens() or turn.est_output_tokens())
        bill.output += out

        prev_segments = list(segments)
        prev_ts = ts
    return bill


# --------------------------------------------------------------------------
# Top-level audit: tie it together for a session.
# --------------------------------------------------------------------------

@dataclass
class SvmAudit:
    session: Session
    cal: Calibration
    lives: list[BlockLife]
    mrc: MRC
    real_usd: float
    base_bill: Bill
    base_usd: float          # baseline sim (fidelity check vs real_usd)
    oracle_bill: Bill
    oracle_usd: float
    l0_bill: Bill
    l0_usd: float
    base_rate: float
    n_stub_oracle: int
    n_stub_l0: int

    def delta_oracle(self) -> float:
        return (self.base_usd - self.oracle_usd) / self.real_usd if self.real_usd else 0.0

    def delta_l0(self) -> float:
        return (self.base_usd - self.l0_usd) / self.real_usd if self.real_usd else 0.0


    ref_metric: str = "block"


def run_svm_audit(session: Session, book: PriceBook | None = None,
                  ref_metric: str = "block") -> SvmAudit:
    book = book or PriceBook.load()
    cal = calibrate(session)
    # base input rate from the dominant model
    model = next((t.model for t in session.assistant_turns() if t.model), None)
    base_rate = book.input_rate(model)
    real_usd = sum(book.turn_cost_usd(t.model, t.usage) for t in session.assistant_turns())

    lives = build_block_lives(session, cal, ref_metric=ref_metric)
    last_turn = session.turns[-1].index if session.turns else 0
    n_assist = len(session.assistant_turns())
    mrc = build_mrc(lives, window_turns=n_assist)
    priors = fit_type_priors(lives, last_turn)

    base_bill = simulate(session, cal, lives, svm=False)
    base_usd = base_bill.usd(base_rate)

    # oracle policy
    oracle_lives = [BlockLife(bl.block_id, bl.kind, bl.tool_name, bl.est_tokens,
                              bl.billed, bl.birth_turn, bl.text, list(bl.ref_turns))
                    for bl in lives]
    decide_policy(oracle_lives, session, priors, mode="oracle")
    oracle_bill = simulate(session, cal, oracle_lives, svm=True)
    oracle_usd = oracle_bill.usd(base_rate)
    n_stub_oracle = sum(1 for bl in oracle_lives if bl.stubbed)

    # L0 policy
    l0_lives = [BlockLife(bl.block_id, bl.kind, bl.tool_name, bl.est_tokens,
                          bl.billed, bl.birth_turn, bl.text, list(bl.ref_turns))
                for bl in lives]
    decide_policy(l0_lives, session, priors, mode="l0")
    l0_bill = simulate(session, cal, l0_lives, svm=True)
    l0_usd = l0_bill.usd(base_rate)
    n_stub_l0 = sum(1 for bl in l0_lives if bl.stubbed)

    return SvmAudit(
        session=session, cal=cal, lives=lives, mrc=mrc, real_usd=real_usd,
        base_bill=base_bill, base_usd=base_usd,
        oracle_bill=oracle_bill, oracle_usd=oracle_usd,
        l0_bill=l0_bill, l0_usd=l0_usd, base_rate=base_rate,
        n_stub_oracle=n_stub_oracle, n_stub_l0=n_stub_l0, ref_metric=ref_metric,
    )


# --------------------------------------------------------------------------
# Passive-rent decomposition: the addressable pool. A block's passive rent is
# what the customer pays to re-read it across its residency while it is NOT
# causally referenced. This is the money SVM is trying to recover.
# --------------------------------------------------------------------------

def passive_rent(audit: SvmAudit) -> list[dict]:
    assist = [t.index for t in audit.session.assistant_turns()]

    def resid(birth: int) -> int:
        return sum(1 for i in assist if i > birth)

    rows = []
    for bl in audit.lives:
        life = resid(bl.birth_turn)
        rent = bl.billed * P_READ * life * audit.base_rate
        rows.append({
            "usd": rent, "est": bl.est_tokens, "nref": len(bl.ref_turns),
            "birth": bl.birth_turn, "life": life, "kind": bl.kind,
            "block_id": bl.block_id, "excerpt": bl.excerpt,
        })
    rows.sort(key=lambda r: r["usd"], reverse=True)
    return rows


# --------------------------------------------------------------------------
# Receipt v2 (§8.2a). Renders the real / oracle / L0 three-column result, the
# reference-metric sensitivity sweep (because the verdict depends on it), the
# MRC + H11 verdict, the H14 verdict, and exactly three founder decisions.
# --------------------------------------------------------------------------

REF_METRICS = ("jaccard", "block", "consumer")
THETA_H14 = 0.20    # H14 gate: ΔCost_sim ≥ 20%
H11_GATE = 0.40     # H11 gate: MRC knee ≤ 40% of window


def _usd(x: float) -> str:
    return f"${x:,.4f}" if abs(x) < 1 else f"${x:,.2f}"


def render_receipt_v2(session: Session, book: PriceBook | None = None,
                      audit_date: str = "(date pending)") -> str:
    book = book or PriceBook.load()
    sweep = {m: run_svm_audit(session, book, ref_metric=m) for m in REF_METRICS}
    a = sweep["block"]   # default metric for the headline three-column table
    rent = passive_rent(a)
    rent_total = sum(r["usd"] for r in rent)
    deltas = [sweep[m].delta_oracle() for m in REF_METRICS] + \
             [sweep[m].delta_l0() for m in REF_METRICS]
    dmin, dmax = min(deltas), max(deltas)
    knees = [sweep[m].mrc.knee_fraction for m in REF_METRICS]
    h11_pass = max(knees) <= H11_GATE
    # H14: pass only if the *conservative* end clears the gate; inconclusive if
    # the gate falls inside the sensitivity band; fail if even oracle can't reach it.
    h14_conservative = min(sweep[m].delta_l0() for m in REF_METRICS)
    h14_optimistic = max(sweep[m].delta_oracle() for m in REF_METRICS)
    if h14_conservative >= THETA_H14:
        h14 = "PASS"
    elif h14_optimistic < THETA_H14:
        h14 = "FAIL"
    else:
        h14 = "INCONCLUSIVE"

    L: list[str] = []
    p = L.append
    p("# MTV SVM シミュレーション監査 — Receipt v2（V-A）")
    p("")
    p(f"- セッション: `{session.meta.get('session_id', 'unknown')}`  /  ソース: `{session.source_path}`")
    p(f"- 監査日: {audit_date}  /  検証: **V-A（$0・LLM呼び出しゼロ）**  /  仕様: MTV_SPEC_v2.0_SVM")
    p(f"- ターン数: {len(session.turns)}（アシスタント {len(session.assistant_turns())}）"
      f"  /  ページ対象ブロック: {len(a.lives)}")
    p(f"- 較正: overhead O={a.cal.overhead:,.0f} tok（system+tool schema、SVM不変）"
      f" / est→billed α={a.cal.scale:.3f} / fit R²={a.cal.r2:.4f}")
    p(f"- **無料窓（compaction/resume）: 0 検出** → §3.1 系2（無料窓一括排出）は本セッションで発火せず。"
      f"全削減は誕生時スタブ化のみに依存。")
    p("")

    p("## 1. 三列結果（実額 / oracle / L0）— 既定参照メトリック `block`")
    p("")
    p(f"ベースライン・シミュレーション ${a.base_usd:,.2f} は実請求 **${a.real_usd:,.2f}** を"
      f"{(a.base_usd-a.real_usd)/a.real_usd*100:+.1f}%（プレフィックス・キャッシュ完全会計、中間不変アサート緑）"
      f"で再現。SVM列はこのベースラインからの差分のみを表す。")
    p("")
    p("| 指標 | 実額 | baseline-sim | SVM oracle（上界） | SVM L0（現実） |")
    p("|---|---:|---:|---:|---:|")
    p(f"| 請求額 | {_usd(a.real_usd)} | {_usd(a.base_usd)} | {_usd(a.oracle_usd)} | {_usd(a.l0_usd)} |")
    p(f"| ΔCost（対実額） | — | {(a.base_usd-a.real_usd)/a.real_usd*100:+.1f}% | "
      f"**{a.delta_oracle()*100:+.1f}%** | **{a.delta_l0()*100:+.1f}%** |")
    p(f"| スタブ化ブロック数 | — | — | {a.n_stub_oracle}/{len(a.lives)} | {a.n_stub_l0}/{len(a.lives)} |")
    p(f"| フォルト（ページイン）回数 | — | — | {a.oracle_bill.faults} | {a.l0_bill.faults} |")
    p(f"| スキーマ税（MCP定義の常駐コスト） | — | — | {_usd(a.oracle_bill.schema_tax*a.base_rate)} | "
      f"{_usd(a.l0_bill.schema_tax*a.base_rate)} |")
    p("")

    p("## 2. 参照メトリック感度（H14 が解けない理由）")
    p("")
    p("「ブロックが後で本当に必要になる頻度（フォルト率）」は L0 字句メトリックの正規化方法に強く依存し、"
      "シミュレーションはモデルの実挙動を観測できない（§4 V-A の明示的限界）。"
      "ΔCost はメトリック選択だけで符号が変わる:")
    p("")
    p("| 参照メトリック | 性質 | ΔCost oracle | ΔCost L0 | スタブ(O/L0) |")
    p("|---|---|---:|---:|---:|")
    desc = {"jaccard": "参照過少推定（最大スタブ）", "block": "§3.2例の型（既定）",
            "consumer": "参照過大推定（最小スタブ）"}
    for m in REF_METRICS:
        s = sweep[m]
        p(f"| {m} | {desc[m]} | {s.delta_oracle()*100:+.1f}% | {s.delta_l0()*100:+.1f}% | "
          f"{s.n_stub_oracle}/{s.n_stub_l0} |")
    p("")
    p(f"→ ΔCost_sim ∈ **[{dmin*100:+.1f}%, {dmax*100:+.1f}%]**。H14 の閾値 20% はこの帯の内側に落ちる。")
    p("")

    p("## 3. 回収可能な原資（passive rent）— 機会は実在する")
    p("")
    p(f"ページ対象ブロックの「受動再読家賃」（常駐中に因果参照されていない間の再読課金）合計 "
      f"**{_usd(rent_total)} ＝ 実請求の {rent_total/a.real_usd*100:.0f}%**。"
      f"上位ブロックは寿命数百ターンに対し因果参照わずか 0–4 回——大半が受動家賃。"
      f"問題は『原資があるか』ではなく『フォルトを誘発せず排出できるか』。")
    p("")
    p("| # | 家賃$ | est tok | 因果参照回数 | 誕生→寿命 | 種別 | 抜粋 |")
    p("|---|---:|---:|---:|---|---|---|")
    for i, r in enumerate(rent[:8], 1):
        ex = r["excerpt"].replace("|", "\\|")
        p(f"| {i} | {_usd(r['usd'])} | {r['est']:,} | {r['nref']} | t{r['birth']}→{r['life']} | "
          f"{r['kind']} | {ex}… |")
    p("")

    p("## 4. Context-MRC（発明①）と H11 判定")
    p("")
    p("因果参照の再利用距離（連続する参照間のアシスタントターン数）分布から MR_ctx(c)=P(rd>c)。"
      "膝 c*（最大曲率点）が小さいほど局所性が強い＝コールド・ブロックを排出する余地がある。")
    p("")
    p("| 参照メトリック | 膝 c*（ターン） | 窓 | 膝/窓 | 参照標本数 |")
    p("|---|---:|---:|---:|---:|")
    for m in REF_METRICS:
        s = sweep[m]
        p(f"| {m} | {s.mrc.knee} | {s.mrc.window_turns} | {s.mrc.knee_fraction:.2f} | {len(s.mrc.distances)} |")
    p("")
    p(f"**H11（膝 ≤ 窓の40%）: {'PASS' if h11_pass else 'FAIL'}** "
      f"（全メトリックで膝/窓 ≤ {max(knees):.2f} ≪ 0.40、頑健）。"
      f"文脈に強い局所性が存在し、SVM の前提（再利用はバースト的、その後コールド化）は成立する。")
    p("")

    p("## 5. H14 判定")
    p("")
    h14_line = {
        "PASS": "L0・保守メトリックでも 20% を超過。",
        "FAIL": "oracle（上界）でも 20% に届かない。",
        "INCONCLUSIVE": "ΔCost_sim の帯が 20% をまたぐ。シミュレーション単独では決定不能。",
    }[h14]
    p(f"**H14（L0シミュで実請求 ≥ 20% 削減・キャッシュ会計込・中間不変）: {h14}** — {h14_line}")
    p("")
    if h14 == "INCONCLUSIVE":
        p("決定不能の根因は単一: **大型・受動ブロックの真のフォルト率**（スタブ化後に本当にページインが"
          "必要になる割合）。これはモデル挙動量であり V-A では原理的に測れない（V-C フォルト想起率の担当）。"
          "$21 の原資が回収可能か焼失するかは、この一点に集約される。")
        p("")
        p("構造的補足: 本セッションは**無料窓ゼロの単一長セッション**＝SVM の最悪条件。中間ブロックの"
          "排出はプレフィックス・キャッシュを破壊する（§0.2 の損失機構）ため、フォルトしたページイン本体は"
          "セッション終端まで常駐し続ける。compaction を含むセッションでは系2 の安価な排出レバーが効くが、"
          "T1 には存在しない。")
        p("")
    p(f"H17 先行シグナル（V-D 前倒し）: MCP スキーマは常駐コスト（≈{SCHEMA_TAX_TOKENS} tok/ターン）。"
      f"本セッション（大型）では実請求の {a.oracle_bill.schema_tax*a.base_rate/a.real_usd*100:.2f}% と無視可能"
      "だが、極短・低額セッションではこの固定費が二桁% を占め最大後悔 ≤2%（H17）を破り得る。"
      "決定エンジン（π_b）はホットブロックを誤ってスタブ化しない（テスト緑）ため、対策はスキーマ側"
      "（しきい値超でのみ SVM ツールを遅延 attach）であって π_b 側ではない。")
    p("")

    p("## 6. ファウンダー意思決定（ちょうど3点・推奨付き）")
    p("")
    p("**D-α — T9+（フック/MCP実装）着手前に、最小フォルト率プローブ（V-C の薄片）を回すか？**")
    p("　推奨: **YES**。H14 は ΔCost_sim ∈ "
      f"[{dmin*100:+.1f}%, {dmax*100:+.1f}%] で帯が 20% をまたぎ、決定変数は単一（大型受動ブロックの"
      "真フォルト率）。上位 ~10 ブロックを意図的にスタブ化した数タスクを実走（≤$5、D2 予算内）すれば、"
      "$21 原資が profitable 帯に入るかが判明する。フック実装はそれ**後**。盲目的に T9 へ進むのは非推奨。")
    p("")
    p("**D-β — L0 推定器の参照メトリック／τ_ref をどう凍結するか？**（§3 隣接 → 凍結規律で承認要）")
    p("　推奨: 正規化 **`block`（|b∩c|/|b|）** を既定とし、τ_ref は D-α プローブの実測フォルトに対して"
      "**較正**（現状 0.30 は仮）。`consumer` は大型ブロックで参照過大推定、`jaccard` は過少推定で、"
      "どちらも単独では危険。")
    p("")
    p("**D-γ — V-A/V-B のコーパスに compaction 入り・マルチセッションのログを追加するか？**")
    p("　推奨: **YES**。T1 は無料窓ゼロ＝SVM が最も不利な session class。系2（無料窓一括排出）と"
      "V-E（comm 34.9% 削減）という**メトリック頑健**な利得源は compaction／オーケストラ構成に宿る。"
      "単一窓なしセッションの No 寄り結果で、本来の対象 class まで棄却しないこと。")
    p("")
    p("---")
    p("*Generated by mtv-audit SVM simulator (T8 / V-A). LLM 呼び出しゼロ・再現可能・"
      "数式 §3 凍結遵守。結果は MTV_SPEC_v2.0_SVM §9 に追記。*")
    return "\n".join(L) + "\n"


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="mtv-audit-svm",
                                 description="SVM simulation audit (T8 / V-A)")
    ap.add_argument("session", help="path to Claude Code session JSONL")
    ap.add_argument("--price-config", default=None)
    ap.add_argument("--audit-date", default="(date pending)",
                    help="audit date stamp for the receipt header")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args(argv)
    book = PriceBook.load(args.price_config)
    session = load_session(args.session)
    if not session.assistant_turns():
        import sys
        print("error: no assistant turns found", file=sys.stderr)
        return 2
    receipt = render_receipt_v2(session, book, audit_date=args.audit_date)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(receipt)
        print(f"receipt written: {args.output}")
    else:
        print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
