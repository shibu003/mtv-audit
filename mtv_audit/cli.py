"""mtv-audit CLI.

Usage:
    python -m mtv_audit.cli SESSION.jsonl [--dial balanced]
        [--sessions-per-month 100] [--price-config prices.json]
        [-o reports/receipt.md]
"""
from __future__ import annotations

import argparse
import sys

from .attribution import DIALS, audit_all_dials
from .parser import load_session
from .pricing import PriceBook
from .replay import StubReplayRunner
from .report import render_receipt


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mtv-audit",
                                 description="MTV waste audit for agent session logs")
    ap.add_argument("session", help="path to Claude Code session JSONL")
    ap.add_argument("--dial", choices=sorted(DIALS), default="balanced",
                    help="λ dial for the detailed ledger (default: balanced)")
    ap.add_argument("--sessions-per-month", type=int, default=100,
                    help="projection assumption for monthly recoverable (default: 100)")
    ap.add_argument("--price-config", default=None,
                    help="JSON price table overriding built-in placeholder defaults")
    ap.add_argument("-o", "--output", default=None,
                    help="write receipt markdown here (default: stdout)")
    ap.add_argument("--data-provenance",
                    choices=["fixture", "real", "redacted-real"],
                    default=None,
                    help="data provenance tag for the receipt (auto-detected if omitted)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    book = PriceBook.load(args.price_config)
    session = load_session(args.session)
    if not session.assistant_turns():
        print("error: no assistant turns found — is this a Claude Code session JSONL?",
              file=sys.stderr)
        return 2
    provenance = args.data_provenance or (
        "fixture" if "fixture" in args.session else "real"
    )
    ledgers = audit_all_dials(session, book)
    runner = StubReplayRunner()
    plan = runner.plan(session, ledgers[args.dial])
    replay_result = runner.run(session, plan)
    receipt = render_receipt(session, ledgers, book, detail_dial=args.dial,
                             sessions_per_month=args.sessions_per_month,
                             replay=replay_result,
                             data_provenance=provenance)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(receipt)
        print(f"receipt written: {args.output}")
    else:
        print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
