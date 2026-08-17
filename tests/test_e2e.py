"""End-to-end: CLI on the bundled fixture + global accounting invariants."""
import importlib.util
import sys
from pathlib import Path

import pytest
from conftest import ROOT

from mtv_audit.attribution import RECOVERABLE_CHANNELS, audit_all_dials, run_audit
from mtv_audit.cli import main as cli_main
from mtv_audit.parser import load_session
from mtv_audit.pricing import PriceBook
from mtv_audit.report import render_receipt


def _load_fixture_builder():
    spec = importlib.util.spec_from_file_location(
        "generate_fixture", ROOT / "fixtures" / "generate_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_fixture"] = mod
    spec.loader.exec_module(mod)
    return mod.build()


@pytest.fixture(scope="module")
def fixture_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("fixture") / "session_fixture.jsonl"
    _load_fixture_builder().write_jsonl(str(p))
    return p


@pytest.fixture(scope="module")
def fixture_session(fixture_path):
    return load_session(str(fixture_path))


REQUIRED_HEADINGS = (
    "MTV無駄監査 領収書",
    "## 1. セッション総計",
    "## 2. チャネル別無駄台帳",
    "## 3. 最も無駄なループ Top 10",
    "## 4. 月次回収見込み",
    "## 5. リプレイ検証ステータス",
    "## 6. 手法注記",
)


def test_cli_end_to_end_produces_receipt(fixture_path, tmp_path):
    out = tmp_path / "receipt.md"
    rc = cli_main([str(fixture_path), "--dial", "balanced",
                   "--sessions-per-month", "100", "-o", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    for heading in REQUIRED_HEADINGS:
        assert heading in text
    assert "NOT_RUN" in text            # replay harness is a stub in Stage 1 v0
    assert "saver" in text and "balanced" in text and "optimizer" in text


def test_all_recoverable_channels_fire_on_fixture(fixture_session, book):
    ledger = run_audit(fixture_session, "balanced", book)
    totals = ledger.channel_totals()
    for ch in RECOVERABLE_CHANNELS:
        assert totals[ch]["usd"] > 0, f"channel {ch} should fire on the fixture"
    assert totals["model"]["count"] >= 1
    assert all(e.flagged for e in ledger.entries if e.channel == "model")


def test_recovered_tokens_never_exceed_reported_total(fixture_session, book):
    tot = fixture_session.total_reported()["grand_total"]
    for name, ledger in audit_all_dials(fixture_session, book).items():
        assert ledger.recoverable_tokens() <= tot, name


def test_deterministic_output(fixture_session, book):
    l1 = run_audit(fixture_session, "balanced", book)
    l2 = run_audit(fixture_session, "balanced", book)
    assert l1.entries == l2.entries
    ledgers = audit_all_dials(fixture_session, book)
    r1 = render_receipt(fixture_session, ledgers, book)
    r2 = render_receipt(fixture_session, ledgers, book)
    assert r1 == r2


def test_dial_monotonicity(fixture_session, book):
    """Stricter dials must never claim less: saver >= balanced >= optimizer."""
    ledgers = audit_all_dials(fixture_session, book)
    saver = ledgers["saver"].recoverable_usd()
    balanced = ledgers["balanced"].recoverable_usd()
    optimizer = ledgers["optimizer"].recoverable_usd()
    eps = 1e-9
    assert saver + eps >= balanced >= optimizer - eps
