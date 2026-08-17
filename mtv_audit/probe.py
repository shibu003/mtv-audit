"""D-α probe harness (MTV_SPEC_v2.0_addendum_A §B) — the $0 parts.

The probe measures the one quantity T8/V-A honestly abstained on: the **true
fault rate** of large paged-out blocks during real task execution (how often a
stubbed block must really be `ctx_read` back). This module does everything that
does NOT require live spend, so the live arm-V/arm-S runs (B.3) can be fired
with a single decision once the founder green-lights the budget:

  B.1  select_probe_blocks  — stratified pick of high-rent blocks from a T8 audit
  B.2  make_stub            — deterministic 2-line / 0-line stubs (no LLM)
  B.4  scan_faults          — parse an arm-S transcript: ctx_read hits +
                              hallucination check (needed-but-not-read)
  B.6  recompute_profitable — re-run pi_b with the measured f̂ → profitable $ pool

The live runner (B.3) and ctx_read shim are intentionally not here: they spend
money / touch real repos and are gated on the founder's go (B.6 GREEN gate).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

from .model import Session
from .parser import load_session
from .pricing import PriceBook
from .svm import (
    P_READ, P_WRITE, RPRIME_DEFAULT, STUB_TOKENS, SvmAudit, decide_policy,
    fit_type_priors, passive_rent, pi_b, run_svm_audit,
)

# --------------------------------------------------------------------------
# B.1 — block-type stratification. The per-type fault rate is the probe's main
# product, so selection must cover the distinct types the spec calls out.
# --------------------------------------------------------------------------

REQUIRED_TYPES = ("file_read", "search", "summary", "skill_system", "other")
MIN_PER_TYPE = 2
TARGET_BLOCKS = 12          # 10–14 band, $5 budget
MAX_BLOCKS = 14


def _origin_tool(session: Session, block_id: str) -> str | None:
    """Tool that produced a tool_result block (block_id = '<tool_use_id>.result')."""
    if not block_id.endswith(".result"):
        return None
    origin = block_id[: -len(".result")]
    for t in session.turns:
        for b in t.blocks:
            if b.kind == "tool_use" and b.block_id == origin:
                return b.tool_name
    return None


_SYSTEM_MARKERS = (
    "Base directory for this skill", "<task-notification>", "<system-reminder>",
    "approved your plan", "Write isn't available", "Edit isn't available",
    "read-only plan mode", "You can now start coding", "<command-",
)
_SUMMARY_MARKERS = ("findings", "調査", "報告", "まとめ", "結論", "調査完了", "調査結果")


def classify_block(session: Session, block_id: str, kind: str, text: str) -> str:
    """Probe stratum for a block. Deterministic; origin-tool first, then content.

    Order matters: system/control text and subagent summaries are recognized
    before the file/search content-shape fallback, because they otherwise leak
    into 'search' (colon- or '==='-rich) and pollute the per-type fault rate.
    """
    head = text[:400]
    if any(m in head for m in _SYSTEM_MARKERS) or "skill" in (block_id or "").lower():
        return "skill_system"
    if any(m in head for m in _SUMMARY_MARKERS):
        return "summary"
    tool = (_origin_tool(session, block_id) or "").lower()
    if tool in ("read", "notebookread"):
        return "file_read"
    if tool in ("grep", "glob", "ls"):
        return "search"
    # content-shape fallback (origin tool unknown)
    stripped = text.lstrip()
    if stripped[:2] and stripped[0].isdigit() and ("\t" in stripped[:6] or stripped[1:3].strip().isdigit()):
        return "file_read"               # line-numbered file body ("1\t…")
    if "import " in head[:80] or "export " in head[:80] or "def " in head[:80]:
        return "file_read"
    if "===" in head[:160]:
        return "search"                  # grep-style section headers
    if kind == "text":
        return "other"
    return "search" if text.count(":") > 8 else "other"


@dataclass
class ProbeBlock:
    handle: str                 # short content handle for ctx_read (sha-ish slug)
    block_id: str
    type: str                   # probe stratum
    kind: str
    size_tok: int               # est tokens
    billed: float               # billed tokens (α · est)
    birth_turn: int
    lifespan_turns: int
    causal_refs: int            # measured 0-4 in T8
    passive_rent_usd: float
    stub_2line: str = ""
    stub_0line: str = ""
    full_text: str = ""         # ground truth for hallucination check (kept out of git)


def _handle(block_id: str) -> str:
    # deterministic short handle (no Math.random / hashing needs are fine here)
    base = block_id.replace(".result", "").replace("toolu_", "")
    return base[-8:] if len(base) >= 8 else base


def select_probe_blocks(audit: SvmAudit, target: int = TARGET_BLOCKS) -> list[ProbeBlock]:
    """Top-rent blocks, stratified by type with >= MIN_PER_TYPE per present type."""
    session = audit.session
    rent = passive_rent(audit)                 # already sorted by usd desc
    bytext = {bl.block_id: bl for bl in audit.lives}
    # annotate top-30 with type
    pool: list[ProbeBlock] = []
    for r in rent[:30]:
        bl = bytext[r["block_id"]]
        typ = classify_block(session, bl.block_id, bl.kind, bl.text)
        pool.append(ProbeBlock(
            handle=_handle(bl.block_id), block_id=bl.block_id, type=typ, kind=bl.kind,
            size_tok=bl.est_tokens, billed=bl.billed, birth_turn=bl.birth_turn,
            lifespan_turns=r["life"], causal_refs=r["nref"],
            passive_rent_usd=r["usd"], full_text=bl.text,
        ))
    # stratified fill: guarantee MIN_PER_TYPE per present type, then top up by rent
    chosen: list[ProbeBlock] = []
    by_type: dict[str, list[ProbeBlock]] = {}
    for pb in pool:
        by_type.setdefault(pb.type, []).append(pb)
    for typ, items in by_type.items():
        chosen.extend(items[:MIN_PER_TYPE])
    chosen_ids = {pb.block_id for pb in chosen}
    for pb in pool:                            # top up by rent until target
        if len(chosen) >= min(target, MAX_BLOCKS):
            break
        if pb.block_id not in chosen_ids:
            chosen.append(pb); chosen_ids.add(pb.block_id)
    chosen.sort(key=lambda pb: pb.passive_rent_usd, reverse=True)
    for pb in chosen:
        pb.stub_2line = make_stub(pb, lines=2)
        pb.stub_0line = make_stub(pb, lines=0)
    return chosen


# --------------------------------------------------------------------------
# B.2 — deterministic stub generation (§2 stub form). No LLM: the "summary" is
# a structural description, never a paraphrase of the body.
# --------------------------------------------------------------------------

def _first_meaningful_line(text: str) -> str:
    for ln in (text or "").splitlines():
        s = ln.strip()
        if len(s) >= 8:
            return s[:70]
    return (text or "").strip()[:70]


def make_stub(pb: ProbeBlock, lines: int = 2) -> str:
    head = (f"[SVM:paged h={pb.handle} type={pb.kind} size={pb.size_tok}tok "
            f"born=t{pb.birth_turn}]")
    if lines == 0:
        # address-only variant (design law 3): what + how to call, no summary
        return head + f"\n全文: ctx_read(\"{pb.handle}\")"
    summary = _first_meaningful_line(pb.full_text)
    return (head
            + f"\n要約: {pb.type} ブロック（先頭: {summary}…、≤2行・決定的生成）"
            + f"\n全文が必要なら ctx_read(\"{pb.handle}\")。部分: ctx_read(\"{pb.handle}\", lines=\"1-40\")")


# --------------------------------------------------------------------------
# B.4 — fault measurement (frozen definition). A "fault" = arm-S issued
# ctx_read(handle) for a probe block (the model needed the full body). The
# worst event = needed-but-not-read + hallucinated body.
# --------------------------------------------------------------------------

@dataclass
class FaultScan:
    handle: str
    ctx_read_count: int = 0          # faults served by paging it back (cheap, controlled)
    disk_reacquired: int = 0         # faults served by re-Read/Grep from disk (no ctx_read)
    distinctive_echoes: int = 0      # turns echoing block-distinctive tokens w/o ctx_read
    hallucinated: bool = False       # echoed body content but never paged it in

    def needed(self) -> bool:
        return self.ctx_read_count > 0 or self.disk_reacquired > 0 or self.distinctive_echoes > 0

    def fault(self) -> bool:
        # B.4 (live-repo correction): the block was needed back, served by EITHER
        # ctx_read OR a disk re-read. Both mean the passive rent was NOT saved.
        return self.ctx_read_count > 0 or self.disk_reacquired > 0


def scan_faults(transcript_path: str, blocks: list[ProbeBlock],
                distinctive_min: int = 6) -> dict[str, FaultScan]:
    """Scan one arm-S run transcript (Claude Code JSONL) for faults.

    A ctx_read tool_use whose input/handle names a probe block counts as a
    fault. Independently, if a later assistant turn reproduces >= distinctive_min
    of a block's *distinctive* tokens (present in the block, rare elsewhere)
    without any ctx_read for it, that is a hallucination candidate (worst event).
    """
    from .svm import _toks
    scans = {pb.handle: FaultScan(handle=pb.handle) for pb in blocks}
    # distinctive tokens = block tokens minus tokens shared across many blocks
    from collections import Counter
    all_tok = Counter()
    per_block = {}
    for pb in blocks:
        tk = _toks(pb.full_text)
        per_block[pb.handle] = tk
        all_tok.update(tk)
    distinctive = {h: {w for w in tk if all_tok[w] <= 2} for h, tk in per_block.items()}

    sess = load_session(transcript_path)
    # body re-entering context via a NON-ctx_read tool_result (Read/Grep/Bash cat)
    # is a disk re-acquisition. We attribute by elimination: if the body shows up
    # in a tool_result but ctx_read was never issued for it, it came from disk.
    tool_result_hits = {pb.handle: 0 for pb in blocks}
    for turn in sess.turns:
        for b in turn.blocks:
            if b.kind == "tool_use" and (b.tool_name or "").lower().startswith("ctx_read"):
                payload = json.dumps(b.tool_input or {})
                for pb in blocks:
                    if pb.handle in payload:
                        scans[pb.handle].ctx_read_count += 1
            elif b.kind == "tool_result":
                rwords = _toks(b.text)
                for pb in blocks:
                    if len(distinctive[pb.handle] & rwords) >= distinctive_min:
                        tool_result_hits[pb.handle] += 1
            elif b.kind == "text":
                owords = _toks(b.text)
                for pb in blocks:
                    if len(distinctive[pb.handle] & owords) >= distinctive_min:
                        scans[pb.handle].distinctive_echoes += 1
    for pb in blocks:
        s = scans[pb.handle]
        if s.ctx_read_count == 0 and tool_result_hits[pb.handle] > 0:
            s.disk_reacquired = tool_result_hits[pb.handle]
        # hallucination = body echoed in assistant text with NO source at all
        s.hallucinated = (s.ctx_read_count == 0 and s.disk_reacquired == 0
                          and s.distinctive_echoes > 0)
    return scans


# --------------------------------------------------------------------------
# B.6 — recompute the profitable pool with the measured per-type fault rate.
# --------------------------------------------------------------------------

@dataclass
class ProfitVerdict:
    profitable_usd: float
    pool_usd: float
    profitable_pct: float
    worst_events: int                # hallucinations across the probe
    healthy_recall: float | None     # ctx_read-when-needed rate, if measured
    gate: str                        # GREEN | AMBER | RED
    per_type_fhat: dict[str, float] = field(default_factory=dict)


def recompute_profitable(audit: SvmAudit, blocks: list[ProbeBlock],
                         fhat_by_type: dict[str, float],
                         rprime: float = float(RPRIME_DEFAULT),
                         worst_events: int = 0,
                         healthy_recall: float | None = None) -> ProfitVerdict:
    """Re-run pi_b with n̂_b := f̂(type) for the full passive-rent pool, and
    apply the B.6 GREEN/AMBER/RED gate."""
    rent = passive_rent(audit)
    bytext = {bl.block_id: bl for bl in audit.lives}
    assist = [t.index for t in audit.session.assistant_turns()]

    def residency(birth: int) -> int:
        return sum(1 for i in assist if i > birth)

    pool_usd = sum(r["usd"] for r in rent)
    profitable = 0.0
    for r in rent:
        bl = bytext[r["block_id"]]
        typ = classify_block(audit.session, bl.block_id, bl.kind, bl.text)
        n_hat = fhat_by_type.get(typ, fhat_by_type.get("other", 1.0))
        if pi_b(bl.billed, residency(bl.birth_turn), n_hat, rprime) > 0:
            profitable += r["usd"]
    pct = profitable / pool_usd if pool_usd else 0.0
    # B.6 gate
    if worst_events >= 1 or pct < 0.30:
        gate = "RED"
    elif pct >= 0.60 and worst_events == 0 and (healthy_recall is None or healthy_recall >= 0.90):
        gate = "GREEN"
    else:
        gate = "AMBER"
    return ProfitVerdict(profitable_usd=profitable, pool_usd=pool_usd,
                         profitable_pct=pct, worst_events=worst_events,
                         healthy_recall=healthy_recall, gate=gate,
                         per_type_fhat=dict(fhat_by_type))


# --------------------------------------------------------------------------
# Manifest writer (real provenance -> *_real* -> git-ignored).
# --------------------------------------------------------------------------

def write_manifest(blocks: list[ProbeBlock], path: str,
                   include_full_text: bool = True) -> dict:
    payload = {
        "spec": "MTV_SPEC_v2.0_addendum_A §B",
        "n_blocks": len(blocks),
        "budget_usd_cap": 5.0,
        "blocks": [],
    }
    for pb in blocks:
        d = asdict(pb)
        if not include_full_text:
            d.pop("full_text", None)
        payload["blocks"].append(d)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def build_probe_manifest(session_path: str, book: PriceBook | None = None,
                         target: int = TARGET_BLOCKS) -> list[ProbeBlock]:
    audit = run_svm_audit(load_session(session_path), book or PriceBook.load())
    return select_probe_blocks(audit, target=target)


def load_manifest(path: str) -> list[ProbeBlock]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [ProbeBlock(**b) for b in data.get("blocks", [])]


# --------------------------------------------------------------------------
# Scoring driver (B.7). Consumes the founder's filled results.json after the
# arm-V/arm-S runs and emits every B.7 deliverable + a ready §9 entry.
#
# results.json shape:
#   {
#     "tasks": {"T1": ["9UCuRTj9"], "T2": ["acy27aU5"], ...},   # task -> handles
#     "runs": [
#       {"task":"T1","arm":"S","variant":"2line","rep":1,
#        "transcript":"runs/T1_S_2line_1.jsonl","success":true},
#       {"task":"T1","arm":"V","variant":"-","rep":1,
#        "transcript":"runs/T1_V_1.jsonl","success":true}, ...
#     ]
#   }
# --------------------------------------------------------------------------

def tally_cost_usd(transcript: str, book: PriceBook | None = None) -> float:
    book = book or PriceBook.load()
    s = load_session(transcript)
    return sum(book.turn_cost_usd(t.model, t.usage) for t in s.assistant_turns())


def score_probe(results: dict, blocks: list[ProbeBlock], audit: SvmAudit,
                book: PriceBook | None = None) -> dict:
    book = book or PriceBook.load()
    by_handle = {b.handle: b for b in blocks}
    type_of = {b.handle: b.type for b in blocks}
    task_handles = results.get("tasks", {})

    # per (type) tallies and per (variant) tallies for B.5
    from collections import defaultdict
    trials = defaultdict(int); faults = defaultdict(int)
    hallu_by_type = defaultdict(int)
    served_ctx = defaultdict(int); served_disk = defaultdict(int)
    variant_trials = defaultdict(int); variant_faults = defaultdict(int)
    variant_hallu = defaultdict(int)
    succ = {"V": [], "S": []}
    total_cost = 0.0
    cost_by_arm = defaultdict(float)

    for run in results.get("runs", []):
        arm = run["arm"]; variant = run.get("variant", "-")
        handles = task_handles.get(run["task"], [])
        relevant = [by_handle[h] for h in handles if h in by_handle]
        cost = tally_cost_usd(run["transcript"], book)
        total_cost += cost; cost_by_arm[arm] += cost
        succ[arm].append(1 if run.get("success") else 0)
        if arm != "S":
            continue
        scans = scan_faults(run["transcript"], relevant)
        for h in handles:
            sc = scans.get(h)
            if sc is None:
                continue
            t = type_of.get(h, "other")
            trials[t] += 1
            variant_trials[(t, variant)] += 1
            if sc.fault():
                faults[t] += 1; variant_faults[(t, variant)] += 1
                if sc.ctx_read_count > 0:
                    served_ctx[t] += 1
                else:
                    served_disk[t] += 1
            if sc.hallucinated:
                hallu_by_type[t] += 1; variant_hallu[(t, variant)] += 1

    fhat = {t: (faults[t] / trials[t] if trials[t] else 0.0) for t in trials}
    worst_events = sum(hallu_by_type.values())
    needed = sum(faults.values()) + worst_events
    healthy_recall = (sum(faults.values()) / needed) if needed else None
    delta_success = ((sum(succ["S"]) / len(succ["S"]) if succ["S"] else 0.0)
                     - (sum(succ["V"]) / len(succ["V"]) if succ["V"] else 0.0))

    verdict = recompute_profitable(audit, blocks, fhat, worst_events=worst_events,
                                   healthy_recall=healthy_recall)

    # B.5 stub-form sub-experiment: fault rate 2line vs 0line
    b5 = {}
    for variant in ("2line", "0line"):
        tt = sum(v for (t, vv), v in variant_trials.items() if vv == variant)
        ff = sum(v for (t, vv), v in variant_faults.items() if vv == variant)
        hh = sum(v for (t, vv), v in variant_hallu.items() if vv == variant)
        b5[variant] = {"trials": tt, "fault_rate": (ff / tt if tt else None),
                       "hallucinations": hh}

    return {
        "fhat_by_type": fhat, "trials_by_type": dict(trials),
        "healthy_recall": healthy_recall, "worst_events": worst_events,
        "delta_success_pt": delta_success * 100.0,
        "total_cost_usd": total_cost, "cost_by_arm": dict(cost_by_arm),
        "within_budget": total_cost <= 5.0,
        "fault_served": {"ctx_read": sum(served_ctx.values()),
                         "disk_reread": sum(served_disk.values())},
        "profitable_pct": verdict.profitable_pct, "profitable_usd": verdict.profitable_usd,
        "pool_usd": verdict.pool_usd, "gate": verdict.gate, "b5_stub_form": b5,
    }


def render_section9_entry(score: dict, audit: SvmAudit) -> str:
    f = score["fhat_by_type"]
    def g(t): return f"{f[t]*100:.0f}%" if t in f else "n/a"
    hr = score["healthy_recall"]
    L = [
        "[2026-06-__] D-α プローブ結果",
        f"- f̂(type): file_read={g('file_read')} search={g('search')} "
        f"summary={g('summary')} skill_system={g('skill_system')}",
        f"- 健全フォルト率={'%.0f%%'%(hr*100) if hr is not None else 'n/a'}  "
        f"最悪事象(幻覚)={score['worst_events']}件",
        f"- フォルト供給源: ctx_read={score['fault_served']['ctx_read']}件 / "
        f"disk再読={score['fault_served']['disk_reread']}件"
        f"（disk再読は家賃を節約せず arm V 相当）",
        f"- π_b再計算: profitable原資={score['profitable_pct']*100:.0f}% "
        f"(${score['profitable_usd']:.2f}/${score['pool_usd']:.2f})",
        f"- Δsuccess={score['delta_success_pt']:+.0f}pt  "
        f"総支出=${score['total_cost_usd']:.2f}（予算内={score['within_budget']}）",
        f"- B.5 スタブ形式: 2line fault={_pct(score['b5_stub_form']['2line']['fault_rate'])} "
        f"/ 0line fault={_pct(score['b5_stub_form']['0line']['fault_rate'])}",
        f"- 判定: {score['gate']}",
    ]
    return "\n".join(L)


def _pct(x):
    return f"{x*100:.0f}%" if isinstance(x, (int, float)) else "n/a"


def _score_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mtv-audit-probe-score",
                                 description="Score a D-α probe run (B.7)")
    ap.add_argument("results", help="founder-filled results.json")
    ap.add_argument("--manifest", required=True, help="dalpha_manifest_real.json")
    ap.add_argument("--session", required=True, help="T1 JSONL (for pi_b recompute base)")
    args = ap.parse_args(argv)
    with open(args.results, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    blocks = load_manifest(args.manifest)
    audit = run_svm_audit(load_session(args.session))
    score = score_probe(results, blocks, audit)
    print(json.dumps(score, ensure_ascii=False, indent=2))
    print("\n--- §9 entry (paste into MTV_SPEC_v2.0_SVM.md) ---")
    print(render_section9_entry(score, audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(_score_main())
