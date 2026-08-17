"""Machine-readable receipts.

The markdown receipt is for a person. This is the same audit as an object
another tool can consume — a CI gate, a dashboard, a different agent runtime
emitting MTV receipts of its own.

Two rules hold this shape honest:

  * **No free text.** Every field is a number, an identifier, or a bounded
    enum. That is what lets a receipt be shared without a redaction pass —
    there is nothing here to redact.
  * **Every channel key is always present.** A channel with no findings emits
    zeros. An absent key would be indistinguishable from a rule that never ran.

The contract is `schema/receipt-v1.schema.json`; `tests/test_receipt_json.py`
checks emitted receipts against it.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any

from .attribution import ALL_CHANNELS, Ledger
from .model import Session
from .pricing import PRICE_TABLE_VERSION, PriceBook
from .redact import scrub_path
from .replay import ReplayResult

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "build_receipt", "dump_receipt"]


def build_receipt(
    session: Session,
    ledgers: dict[str, Ledger],
    book: PriceBook,
    *,
    detail_dial: str = "balanced",
    sessions_per_month: int | None = None,
    replay: ReplayResult | None = None,
    data_provenance: str = "fixture",
    top_n: int = 10,
    emitter: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Build the receipt object. Mirrors `report.render_receipt`'s inputs."""
    totals = session.total_reported()
    total_usd = sum(book.turn_cost_usd(t.model, t.usage) for t in session.assistant_turns())
    detail = ledgers[detail_dial]
    channel_totals = detail.channel_totals()

    def pct(usd: float) -> float:
        return round(usd / total_usd * 100, 4) if total_usd else 0.0

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session": {
            "id": str(session.meta.get("session_id", "unknown")),
            "source_filename": scrub_path(str(session.source_path)),
            "turns": len(session.turns),
            "assistant_turns": len(session.assistant_turns()),
            "audited_on": _dt.date.today().isoformat(),
            # Redaction is unconditional, so anything not a fixture is
            # redacted-real by construction rather than by assertion.
            "data_provenance": "fixture" if data_provenance == "fixture" else "redacted-real",
        },
        "totals": {
            "input_tokens": int(totals["input_tokens"]),
            "output_tokens": int(totals["output_tokens"]),
            "cache_read_input_tokens": int(totals["cache_read_input_tokens"]),
            "cache_creation_input_tokens": int(totals["cache_creation_input_tokens"]),
            "prompt_total_tokens": int(totals["prompt_total"]),
            "grand_total_tokens": int(totals["grand_total"]),
            "usd": round(total_usd, 6),
            "price_table_version": str(PRICE_TABLE_VERSION),
        },
        "channels": {
            c: {
                "tokens": round(float(channel_totals[c]["tokens"]), 3),
                "usd": round(float(channel_totals[c]["usd"]), 6),
                "count": int(channel_totals[c]["count"]),
                "percent_of_total": pct(float(channel_totals[c]["usd"])),
            }
            for c in ALL_CHANNELS
        },
        "recoverable": {
            "dial": detail_dial,
            "usd": round(detail.recoverable_usd(), 6),
            "percent_of_total": pct(detail.recoverable_usd()),
            # An estimate until a counterfactual replay says otherwise. Saying
            # "verified" without one would be the single most misleading thing
            # this tool could claim.
            "verified": bool(replay is not None and getattr(replay, "status", "") == "PASSED"),
            "replay_status": str(getattr(replay, "status", "NOT_RUN")) if replay else "NOT_RUN",
        },
        "top_items": [
            {
                "channel": item["channel"],
                "block_id": str(item["block_id"]),
                "repeat": int(item["repeat"]),
                "turn_span": str(item["turn_span"]),
                "tokens": round(float(item["tokens"]), 3),
                "usd": round(float(item["usd"]), 6),
            }
            for item in detail.top_items(top_n)
        ],
    }

    if sessions_per_month:
        receipt["projection"] = {
            "sessions_per_month": int(sessions_per_month),
            "by_dial": {
                name: round(ledger.recoverable_usd() * sessions_per_month, 6)
                for name, ledger in ledgers.items()
            },
        }

    if emitter:
        receipt["emitter"] = {"name": emitter[0], "version": emitter[1]}

    return receipt


def dump_receipt(*args: Any, indent: int = 2, **kwargs: Any) -> str:
    """`build_receipt` straight to JSON."""
    return json.dumps(build_receipt(*args, **kwargs), indent=indent, ensure_ascii=False)
