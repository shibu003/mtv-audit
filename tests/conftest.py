"""Shared test helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtv_audit.model import Session            # noqa: E402
from mtv_audit.parser import load_session      # noqa: E402
from mtv_audit.pricing import PriceBook        # noqa: E402
from mtv_audit.synth import SessionBuilder     # noqa: E402


@pytest.fixture
def book() -> PriceBook:
    return PriceBook.load()


def parse_built(builder: SessionBuilder, tmp_path) -> Session:
    """Round-trip a built session through JSONL + the real parser."""
    p = tmp_path / "session.jsonl"
    builder.write_jsonl(str(p))
    return load_session(str(p))


IRRELEVANT_BIG = ("Quarterly newsletter about the office ficus and parking garage "
                  "renovation schedules with absolutely unrelated trivia. " * 20)

RELEVANT_BIG = ("def refund(order, amount): adjust src/payments.py refund ledger "
                "for tests/test_payments.py::test_refund verification. " * 20)

TRACE = ("Traceback (most recent call last):\n"
         "  File 'tests/test_payments.py', line 42, in test_refund\n"
         "AssertionError: assert 60 == 70\n"
         + ("E       refund ledger diff line for tests/test_payments.py debugging\n" * 20)
         + "1 failed in 0.2s\n")

GOAL = ("Fix the failing test in tests/test_payments.py::test_refund. "
        "Run pytest and repair src/payments.py refund logic.")
