"""Counterfactual replay harness (interface + stub).

Purpose: turn the ledger from a *claim* into a *verified receipt*.
A replay removes low-MTV tokens (per the ledger) from the session,
re-runs it against the live API, and checks that task success holds.

Stage 1 v0 ships the interface and a stub runner. Wiring the live API
runner is gated on founder decision D2 (replay budget & environment).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .attribution import Ledger
from .model import Session


@dataclass
class ReplayPlan:
    """What to remove and what defines success for the re-run."""
    session_id: str
    dial: str
    remove: list[tuple[int, str]] = field(default_factory=list)  # (turn, block_id)
    success_criteria: str = "all originally-passing checks still pass"
    estimated_tokens_removed: float = 0.0


@dataclass
class ReplayResult:
    status: str            # VERIFIED | FAILED | NOT_RUN
    success_preserved: bool | None = None
    tokens_saved: float | None = None
    note: str = ""


class ReplayHarness(ABC):
    @abstractmethod
    def plan(self, session: Session, ledger: Ledger) -> ReplayPlan: ...

    @abstractmethod
    def run(self, session: Session, plan: ReplayPlan) -> ReplayResult: ...


class StubReplayRunner(ReplayHarness):
    """Builds a real plan; execution is stubbed (no API calls)."""

    def plan(self, session: Session, ledger: Ledger) -> ReplayPlan:
        remove = [(e.turn_index, e.block_id) for e in ledger.entries
                  if e.channel in ("retry", "clean") and not e.flagged]
        return ReplayPlan(
            session_id=session.meta.get("session_id", "unknown"),
            dial=ledger.dial,
            remove=remove,
            estimated_tokens_removed=sum(
                e.tokens_scaled for e in ledger.entries
                if e.channel in ("retry", "clean")
            ),
        )

    def run(self, session: Session, plan: ReplayPlan) -> ReplayResult:
        return ReplayResult(
            status="NOT_RUN",
            note=("stub runner — live counterfactual replay pending founder "
                  "decision D2 (budget & execution environment)"),
        )
