# mtv-audit

**MTV (Marginal Token Value) waste audit for agent session logs.**
Stage 1 of the MTV business (see `MTV_META_PROMPT.md`): detect token waste in
Claude Code / agent sessions, attribute it to spend channels, and emit an
auditable **receipt** stating how much of the bill is recoverable.

Zero runtime dependencies (Python ≥ 3.10 stdlib only). `pytest` for tests.

## Quickstart

```bash
# from the repo root
python fixtures/generate_fixture.py                    # build the bundled fixture
python -m mtv_audit.cli fixtures/session_fixture.jsonl \
    --dial balanced --sessions-per-month 100 \
    -o reports/sample_receipt.md

python -m pytest tests/ -q                             # 67 tests, all green
```

Input: a Claude Code session JSONL (`~/.claude/projects/.../*.jsonl`).
The parser is tolerant: unknown line types are skipped and counted.

## CLI

```
python -m mtv_audit.cli SESSION.jsonl
    --dial {saver,balanced,optimizer}   λ(U) dial for the detailed ledger (default: balanced)
    --sessions-per-month N              monthly projection assumption (default: 100)
    --price-config prices.json          override the built-in price table
    -o receipt.md                       output path (default: stdout)
```

**Pricing:** the built-in table was checked against the published price list on
2026-08-17 (platform.claude.com → pricing). Prices still change, so re-check it,
or pass `--price-config`, before any customer-facing report — `tests/test_pricing.py`
pins the current figures so a stale table fails the suite rather than quietly
producing a plausible wrong number.

Two things the table cannot infer from a transcript, and does not pretend to:
a 1-hour-TTL cache write costs 2x base input rather than the 1.25x used here
(the log does not record which TTL was used), and matching is by substring, so
a new model whose id extends an existing key needs its own entry *above* that
key — see the ordering note in `mtv_audit/pricing.py`.

## Attribution rules (→ §6 of MTV_META_PROMPT.md)

| Channel | §6 definition | Implementation (all unit-tested) |
|---|---|---|
| `retry` | contaminated retries | failure-marker blocks (traceback / FAILED / AssertionError / non-zero exit) re-sent in later prompts |
| `clean` | stale context re-reads | re-sent blocks whose lexical relevance to the current goal (latest user instruction + recent tool targets) is below the dial threshold |
| `comm` | full-state rebroadcast | subagent (`Task`) payload overlap with accumulated context above the dial threshold — diff-sync was possible |
| `deep` | overthinking | thinking tokens beyond the dial allowance on ex-post trivial steps (single mechanical tool call that succeeded) |
| `model` | over-tier | top-tier model on a trivial step — **flag only**, excluded from the recoverable sum until a lower-tier replay confirms it |
| `stop` | no circuit breaker | spend after the first success marker following the last state-changing action (or after the last state change, on abandonment) |

**Accounting guarantees** (enforced by tests):
no double counting — each (turn, block) is claimed by at most one channel with
precedence `retry > clean > comm > deep > stop`; block-level token estimates
(chars/4) are scaled per turn against reported API usage; context-side waste is
valued at the turn's cache-aware *blended* prompt rate; recovered tokens never
exceed the reported session total; output is deterministic; dials are
monotonic (`saver ≥ balanced ≥ optimizer`).

## Replay verification

`mtv_audit/replay.py` defines the counterfactual-replay interface
(`ReplayHarness.plan/run`). Stage 1 v0 ships a stub runner that builds a real
removal plan (retry + clean blocks) but reports `NOT_RUN`; wiring the live API
runner is gated on founder decision **D2** (budget & execution environment).
Once a replay passes, the receipt's "upper-bound estimate" becomes a
"verified saving".

## Repository layout

```
mtv_audit/        core package (model, parser, pricing, attribution, report, replay, synth, cli)
fixtures/         deterministic fixture generator + generated JSONL
tests/            per-rule unit tests + end-to-end (67 tests)
reports/          generated receipts
```

## License

**Apache License 2.0** — see [LICENSE](LICENSE).

Use it, modify it, ship it commercially, host it. The patent grant matters
here: the attribution channels are meant to be implemented by other tools.

Contributions need a sign-off: see [CLA.md](CLA.md).
