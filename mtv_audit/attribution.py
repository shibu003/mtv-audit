"""MTV waste attribution rules (Stage 1 core).

Implements the operational definitions of §6 of MTV_META_PROMPT.md:

  retry  contaminated retries: failure traces re-sent after a failure marker
  clean  stale context re-reads: re-sent blocks irrelevant to the current goal
  comm   full-state rebroadcast: subagent payloads duplicating known context
  deep   overthinking: thinking tokens on ex-post trivial steps
  model  over-tier: top-tier model on trivial steps (FLAG ONLY, needs replay)
  stop   no-circuit-breaker cost: tokens after the job was already done

Accounting rules:
  * No double counting. Each (turn, block) prompt-side token is claimed by at
    most one channel. Precedence: retry > clean > comm > deep > stop.
    `model` is flag-only and never enters the recoverable sum.
  * Block tokens are estimates (chars/4) scaled per turn against reported
    API usage (turn_scale), so ledger numbers live in billed-token units.
  * Context-side waste is valued at the turn's *blended* prompt rate
    (cache-aware); output-side waste at the model's output rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .model import Block, Session, Turn
from .pricing import PriceBook, is_top_tier

# --------------------------------------------------------------------------
# Dials (λ(U)): saver / balanced / optimizer — unified vocabulary everywhere.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DialParams:
    name: str
    clean_relevance_threshold: float   # relevance below this => stale
    clean_min_block_tokens: int        # ignore tiny blocks
    retry_grace_turns: int             # assistant turns after failure that are tolerated
    comm_overlap_threshold: float      # payload/context overlap above this => rebroadcast
    deep_thinking_allowance: int       # free thinking tokens per trivial step
    trivial_text_max_tokens: int       # max assistant prose for a step to count as trivial
    trivial_edit_max_chars: int        # max edit size still considered mechanical


DIALS: dict[str, DialParams] = {
    "saver": DialParams("saver", 0.30, 80, 0, 0.35, 50, 80, 400),
    "balanced": DialParams("balanced", 0.15, 120, 0, 0.50, 100, 60, 300),
    "optimizer": DialParams("optimizer", 0.05, 200, 1, 0.70, 300, 40, 200),
}

RECOVERABLE_CHANNELS = ("retry", "clean", "comm", "deep", "stop")
ALL_CHANNELS = RECOVERABLE_CHANNELS + ("model",)

TRIVIAL_TOOLS = {"Read", "LS", "Glob"}
STATE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
SUBAGENT_TOOLS = {"Task", "dispatch_agent", "Agent"}
READONLY_BASH_PREFIXES = (
    "ls", "cat", "grep", "rg ", "find ", "head", "tail", "echo", "pwd",
    "git status", "git diff", "git log", "pytest", "python -m pytest", "wc ",
)

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_\.]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "has",
    "are", "was", "were", "you", "your", "please", "run", "use", "can",
    "will", "should", "would", "into", "then", "when", "what", "how",
    "all", "not", "but", "out", "they", "their", "there", "here",
}

# --------------------------------------------------------------------------
# Ledger structures
# --------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    channel: str
    turn_index: int
    block_id: str
    tokens_est: int        # estimator-space tokens
    tokens_scaled: float   # billed-token-equivalent after turn scaling
    usd: float
    note: str = ""
    excerpt: str = ""
    flagged: bool = False  # model channel: flagged, pending replay confirmation


@dataclass
class Ledger:
    dial: str
    entries: list[LedgerEntry] = field(default_factory=list)

    def channel_totals(self) -> dict[str, dict[str, float]]:
        out = {c: {"tokens": 0.0, "usd": 0.0, "count": 0} for c in ALL_CHANNELS}
        for e in self.entries:
            out[e.channel]["tokens"] += e.tokens_scaled
            out[e.channel]["usd"] += e.usd
            out[e.channel]["count"] += 1
        return out

    def recoverable_usd(self) -> float:
        return sum(e.usd for e in self.entries if e.channel in RECOVERABLE_CHANNELS)

    def recoverable_tokens(self) -> float:
        return sum(e.tokens_scaled for e in self.entries if e.channel in RECOVERABLE_CHANNELS)

    def top_items(self, n: int = 10) -> list[dict]:
        grouped: dict[tuple[str, str], dict] = {}
        for e in self.entries:
            if e.channel not in RECOVERABLE_CHANNELS:
                continue
            key = (e.channel, e.block_id)
            g = grouped.setdefault(key, {
                "channel": e.channel, "block_id": e.block_id, "usd": 0.0,
                "tokens": 0.0, "turns": [], "excerpt": e.excerpt, "note": e.note,
            })
            g["usd"] += e.usd
            g["tokens"] += e.tokens_scaled
            g["turns"].append(e.turn_index)
        items = sorted(grouped.values(), key=lambda g: g["usd"], reverse=True)[:n]
        for g in items:
            ts = sorted(set(g["turns"]))
            g["turn_span"] = f"{ts[0]}–{ts[-1]}" if len(ts) > 1 else str(ts[0])
            g["repeat"] = len(g["turns"])
        return items


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")} - _STOPWORDS


def goal_keywords(session: Session, t_idx: int, lookback: int = 4) -> set[str]:
    """Goal of the current step: latest user instruction + recent tool targets."""
    kws: set[str] = set()
    for turn in reversed(session.turns[:t_idx]):
        if turn.role == "user" and any(b.kind == "text" for b in turn.blocks):
            for b in turn.blocks:
                if b.kind == "text":
                    kws |= tokenize(b.text)
            break
    lo = max(0, t_idx - lookback)
    for turn in session.turns[lo:t_idx]:
        for b in turn.tool_uses():
            ti = b.tool_input or {}
            for key in ("file_path", "path", "pattern", "command", "notebook_path"):
                if key in ti:
                    kws |= tokenize(str(ti[key]))
    return kws


def relevance(block: Block, keywords: set[str], extra_words: set[str] | None = None) -> float:
    """Fraction of goal keywords present in the block (lexical proxy, v0)."""
    if not keywords:
        return 1.0  # no goal signal -> do not accuse
    words = tokenize(block.text) | (extra_words or set())
    return len(keywords & words) / len(keywords)


def _excerpt(text: str, n: int = 110) -> str:
    return " ".join((text or "").split())[:n]


def est_context_tokens(session: Session, t_idx: int) -> int:
    return sum(b.est_tokens for turn in session.turns[:t_idx] for b in turn.blocks)


def turn_scale(session: Session, turn: Turn) -> float:
    """Map estimator-space tokens to billed-token units for this turn."""
    est = est_context_tokens(session, turn.index)
    reported = turn.reported_prompt_tokens()
    if est <= 0 or reported <= 0:
        return 1.0
    return max(0.25, min(4.0, reported / est))


def _is_readonly_bash(command: str) -> bool:
    c = (command or "").strip().lower()
    return any(c.startswith(p) for p in READONLY_BASH_PREFIXES)


def _tool_result_for(session: Session, turn: Turn, tool_use_id: str) -> Block | None:
    for later in session.turns[turn.index + 1: turn.index + 3]:
        for b in later.blocks:
            if b.kind == "tool_result" and b.block_id.startswith(tool_use_id):
                return b
    return None


def is_trivial_turn(session: Session, turn: Turn, p: DialParams) -> bool:
    """Ex-post trivial: one mechanical tool call, little prose, succeeded."""
    uses = turn.tool_uses()
    if len(uses) != 1 or turn.text_tokens() > p.trivial_text_max_tokens:
        return False
    u = uses[0]
    ti = u.tool_input or {}
    mechanical = (
        u.tool_name in TRIVIAL_TOOLS
        or (u.tool_name == "Bash" and _is_readonly_bash(str(ti.get("command", ""))))
        or (u.tool_name in {"Edit", "MultiEdit"}
            and len(str(ti.get("old_string", "")) + str(ti.get("new_string", ""))) <= p.trivial_edit_max_chars)
    )
    if not mechanical:
        return False
    result = _tool_result_for(session, turn, u.block_id)
    return not (result and result.is_failure)


def is_state_changing_turn(turn: Turn) -> bool:
    for u in turn.tool_uses():
        if u.tool_name in STATE_TOOLS:
            return True
        if u.tool_name == "Bash" and not _is_readonly_bash(str((u.tool_input or {}).get("command", ""))):
            return True
    return False


# --------------------------------------------------------------------------
# Rule 1 — retry: contaminated retries
# --------------------------------------------------------------------------

def attribute_retry(session: Session, p: DialParams, book: PriceBook,
                    claimed: set[tuple[int, str]]) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    failures: list[tuple[int, Block]] = [
        (turn.index, b) for turn in session.turns for b in turn.blocks
        if b.kind == "tool_result" and b.is_failure
    ]
    if not failures:
        return entries
    for turn in session.assistant_turns():
        assistants_between = 0
        scale = None
        for f_idx, fblock in failures:
            if f_idx >= turn.index:
                continue
            assistants_between = sum(
                1 for t in session.turns[f_idx + 1: turn.index] if t.role == "assistant"
            )
            if assistants_between < p.retry_grace_turns:
                continue
            key = (turn.index, fblock.block_id)
            if key in claimed:
                continue
            claimed.add(key)
            scale = scale if scale is not None else turn_scale(session, turn)
            tokens_scaled = fblock.est_tokens * scale
            rate = book.blended_prompt_rate(turn.model, turn.usage)
            entries.append(LedgerEntry(
                channel="retry", turn_index=turn.index, block_id=fblock.block_id,
                tokens_est=fblock.est_tokens, tokens_scaled=tokens_scaled,
                usd=tokens_scaled * rate,
                note=f"failure trace from turn {f_idx} re-sent",
                excerpt=_excerpt(fblock.text),
            ))
    return entries


# --------------------------------------------------------------------------
# Rule 2 — clean: stale context re-reads
# --------------------------------------------------------------------------

def _origin_words(session: Session, result_block: Block) -> set[str]:
    """Words of the tool_use input that produced a tool_result block.

    Lets file paths / commands count toward relevance, so a read of
    src/payments.py is not misclassified as stale during payments work.
    """
    if not result_block.block_id.endswith(".result"):
        return set()
    origin_id = result_block.block_id[: -len(".result")]
    for turn in session.turns:
        for b in turn.blocks:
            if b.kind == "tool_use" and b.block_id == origin_id:
                return tokenize(b.text)
    return set()


def attribute_clean(session: Session, p: DialParams, book: PriceBook,
                    claimed: set[tuple[int, str]]) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for turn in session.assistant_turns():
        kws = goal_keywords(session, turn.index)
        scale = None
        for prior in session.turns[:turn.index]:
            for b in prior.blocks:
                if b.kind not in ("tool_result", "text"):
                    continue
                if b.est_tokens < p.clean_min_block_tokens:
                    continue
                key = (turn.index, b.block_id)
                if key in claimed:
                    continue
                rel = relevance(b, kws, extra_words=_origin_words(session, b))
                if rel >= p.clean_relevance_threshold:
                    continue
                claimed.add(key)
                scale = scale if scale is not None else turn_scale(session, turn)
                tokens_scaled = b.est_tokens * scale
                rate = book.blended_prompt_rate(turn.model, turn.usage)
                entries.append(LedgerEntry(
                    channel="clean", turn_index=turn.index, block_id=b.block_id,
                    tokens_est=b.est_tokens, tokens_scaled=tokens_scaled,
                    usd=tokens_scaled * rate,
                    note=f"relevance {rel:.2f} < {p.clean_relevance_threshold} (from turn {prior.index})",
                    excerpt=_excerpt(b.text),
                ))
    return entries


# --------------------------------------------------------------------------
# Rule 3 — comm: full-state rebroadcast to subagents
# --------------------------------------------------------------------------

def attribute_comm(session: Session, p: DialParams, book: PriceBook,
                   claimed: set[tuple[int, str]]) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for turn in session.assistant_turns():
        for u in turn.tool_uses():
            if u.tool_name not in SUBAGENT_TOOLS:
                continue
            ti = u.tool_input or {}
            payload = f"{ti.get('description', '')}\n{ti.get('prompt', '')}"
            payload_words = tokenize(payload)
            if not payload_words:
                continue
            context_words: set[str] = set()
            for prior in session.turns[:turn.index]:
                for b in prior.blocks:
                    context_words |= tokenize(b.text)
            overlap = len(payload_words & context_words) / len(payload_words)
            if overlap <= p.comm_overlap_threshold:
                continue
            key = (turn.index, u.block_id)
            if key in claimed:
                continue
            claimed.add(key)
            payload_tokens = max(1, len(payload) // 4)
            waste_tokens = payload_tokens * overlap
            rate = book.output_rate(turn.model)
            entries.append(LedgerEntry(
                channel="comm", turn_index=turn.index, block_id=u.block_id,
                tokens_est=int(waste_tokens), tokens_scaled=waste_tokens,
                usd=waste_tokens * rate,
                note=f"subagent payload overlap {overlap:.2f} > {p.comm_overlap_threshold} (diff-sync possible)",
                excerpt=_excerpt(payload),
            ))
    return entries


# --------------------------------------------------------------------------
# Rule 4 — deep: overthinking on ex-post trivial steps
# --------------------------------------------------------------------------

def attribute_deep(session: Session, p: DialParams, book: PriceBook,
                   claimed_output: dict[int, float]) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for turn in session.assistant_turns():
        think = turn.thinking_tokens()
        if think <= p.deep_thinking_allowance:
            continue
        if not is_trivial_turn(session, turn, p):
            continue
        waste = think - p.deep_thinking_allowance
        claimed_output[turn.index] = claimed_output.get(turn.index, 0.0) + waste
        rate = book.output_rate(turn.model)
        think_block = next((b for b in turn.blocks if b.kind == "thinking"), None)
        entries.append(LedgerEntry(
            channel="deep", turn_index=turn.index,
            block_id=think_block.block_id if think_block else f"t{turn.index}.thinking",
            tokens_est=waste, tokens_scaled=float(waste), usd=waste * rate,
            note=f"{think} thinking tokens on trivial step (allowance {p.deep_thinking_allowance})",
            excerpt=_excerpt(think_block.text if think_block else ""),
        ))
    return entries


# --------------------------------------------------------------------------
# Rule 5 — model: over-tier (FLAG ONLY; counted after replay confirms)
# --------------------------------------------------------------------------

def attribute_model_flags(session: Session, p: DialParams, book: PriceBook) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for turn in session.assistant_turns():
        if not is_top_tier(turn.model):
            continue
        if not is_trivial_turn(session, turn, p):
            continue
        out_tokens = turn.reported_output_tokens() or turn.est_output_tokens()
        entries.append(LedgerEntry(
            channel="model", turn_index=turn.index, block_id=f"t{turn.index}.model",
            tokens_est=out_tokens, tokens_scaled=float(out_tokens),
            usd=out_tokens * book.output_rate(turn.model),
            note=f"top-tier model ({turn.model}) on trivial step — pending lower-tier replay",
            excerpt=_excerpt(" / ".join(u.tool_name or "" for u in turn.tool_uses())),
            flagged=True,
        ))
    return entries


# --------------------------------------------------------------------------
# Rule 6 — stop: tokens after the job was done (no circuit breaker)
# --------------------------------------------------------------------------

def find_stop_boundary(session: Session) -> int | None:
    """Turn index after which spend counts as stop waste, or None."""
    last_state = None
    for turn in session.turns:
        if turn.role == "assistant" and is_state_changing_turn(turn):
            last_state = turn.index
    if last_state is None:
        return None
    for turn in session.turns[last_state + 1:]:
        if any(b.is_success for b in turn.blocks):
            return turn.index  # post-completion burn starts after this
    # abandonment branch: never reached success after last state change
    trailing = [t for t in session.turns[last_state + 1:] if t.role == "assistant"]
    return last_state if trailing else None


def attribute_stop(session: Session, p: DialParams, book: PriceBook,
                   claimed: set[tuple[int, str]],
                   claimed_output: dict[int, float]) -> list[LedgerEntry]:
    boundary = find_stop_boundary(session)
    if boundary is None:
        return []
    entries: list[LedgerEntry] = []
    for turn in session.assistant_turns():
        if turn.index <= boundary:
            continue
        scale = turn_scale(session, turn)
        claimed_prompt = sum(
            b.est_tokens * scale
            for prior in session.turns[:turn.index] for b in prior.blocks
            if (turn.index, b.block_id) in claimed
        )
        prompt_total = (turn.reported_prompt_tokens()
                        or est_context_tokens(session, turn.index) * scale)
        prompt_waste = max(0.0, prompt_total - claimed_prompt)
        out_total = float(turn.reported_output_tokens() or turn.est_output_tokens())
        out_waste = max(0.0, out_total - claimed_output.get(turn.index, 0.0))
        usd = (prompt_waste * book.blended_prompt_rate(turn.model, turn.usage)
               + out_waste * book.output_rate(turn.model))
        first_text = next((b.text for b in turn.blocks if b.kind in ("text", "tool_use")), "")
        entries.append(LedgerEntry(
            channel="stop", turn_index=turn.index, block_id=f"t{turn.index}.stop",
            tokens_est=int(prompt_waste + out_waste),
            tokens_scaled=prompt_waste + out_waste, usd=usd,
            note=f"spend after completion boundary (turn {boundary}); breaker absent",
            excerpt=_excerpt(first_text),
        ))
    return entries


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_audit(session: Session, dial: str = "balanced",
              book: PriceBook | None = None) -> Ledger:
    p = DIALS[dial]
    book = book or PriceBook.load()
    claimed: set[tuple[int, str]] = set()
    claimed_output: dict[int, float] = {}
    ledger = Ledger(dial=dial)
    ledger.entries += attribute_retry(session, p, book, claimed)
    ledger.entries += attribute_clean(session, p, book, claimed)
    ledger.entries += attribute_comm(session, p, book, claimed)
    ledger.entries += attribute_deep(session, p, book, claimed_output)
    ledger.entries += attribute_model_flags(session, p, book)
    ledger.entries += attribute_stop(session, p, book, claimed, claimed_output)
    return ledger


def audit_all_dials(session: Session, book: PriceBook | None = None) -> dict[str, Ledger]:
    book = book or PriceBook.load()
    return {name: run_audit(session, name, book) for name in DIALS}
