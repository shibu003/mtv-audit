"""SVM simulator (T8 / V-A) — invariants, π_b economics, MRC, harmlessness.

These tests use only synthetic sessions (the real T1 JSONL is private and out
of tree). They lock the *structural* guarantees the receipt rests on, not the
particular T1 numbers.
"""
from conftest import GOAL, parse_built

from mtv_audit.svm import (
    P_READ, P_WRITE, STUB_TOKENS, build_mrc, build_block_lives, calibrate,
    decide_policy, fit_type_priors, pi_b, run_svm_audit, simulate, BlockLife,
)
from mtv_audit.synth import SessionBuilder

MODEL = "claude-opus-4-8"

# A big block whose vocabulary never recurs in later turns (pure passive rent).
NEWSLETTER = ("Quarterly newsletter ficus parking garage renovation potluck "
              "stapler thermostat elevator carpet beverage vending lobby. " * 60)
# A block whose tokens DO recur every turn (genuinely re-used, must be kept).
PAYMENTS = ("refund payments ledger order amount captured src/payments.py "
            "test_refund verification reconcile invoice charge. " * 18)


def _passive_session(tmp_path, n_idle: int = 16):
    """Read one huge irrelevant block, then many turns that never touch it."""
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Read", {"file_path": "NEWSLETTER.md"})
    b.assistant(MODEL, [t1])
    b.tool_result(t1["id"], NEWSLETTER)
    for i in range(n_idle):
        b.assistant(MODEL, [b.text(f"Step {i}: editing src/payments.py refund "
                                   "ledger to reconcile the captured amount.")])
    return parse_built(b, tmp_path)


# --------------------------------------------------------------------------
# Calibration + append-only invariant
# --------------------------------------------------------------------------

def test_calibration_fits_linear_usage(tmp_path):
    session = _passive_session(tmp_path)
    cal = calibrate(session)
    assert cal.r2 > 0.9           # synth usage is ~linear in cumulative content
    assert cal.scale > 0
    assert cal.overhead >= 0


def test_append_only_invariant_never_broken(tmp_path, book):
    """The intermediate-invariant assert (§4 V-A step 3): our own policy must
    never restructure the prefix. All three sims must report zero breaks."""
    a = run_svm_audit(_passive_session(tmp_path), book)
    assert a.base_bill.invariant_breaks == 0
    assert a.oracle_bill.invariant_breaks == 0
    assert a.l0_bill.invariant_breaks == 0


# --------------------------------------------------------------------------
# π_b economics (§3.2)
# --------------------------------------------------------------------------

def test_pi_b_pure_win_when_never_referenced():
    """n̂=0 → cost_stub has no fault term → stubbing a block bigger than σ is
    an unconditional win (π_b > 0). This is the source of all T1 savings."""
    s = 3440.0  # the §3.2 example block, billed tokens
    assert pi_b(s, life=291, n_hat=0, rprime=0) > 0


def test_pi_b_example_matches_spec_sign():
    """§3.2 worked example: s≈3440, life≈291, n̂=3, R̂'=20 → π≈+67.5k (stub)."""
    p = pi_b(3440.0, life=291, n_hat=3, rprime=20)
    assert p > 0
    assert abs(p - 67_500) < 5_000   # within rounding of the spec's figure


def test_pi_b_harmless_on_small_or_hot_block():
    """A block barely larger than the stub, faulted repeatedly, must NOT be
    stubbed (π_b ≤ 0). This is the no-harm guarantee (V-D / H17)."""
    s = STUB_TOKENS + 40.0          # only marginally worth paging
    assert pi_b(s, life=30, n_hat=5, rprime=25) <= 0


def test_big_passive_block_is_stubbed_and_saves(tmp_path, book):
    """End-to-end: a huge never-re-used block is stubbed by the oracle policy
    and the simulated bill drops below the baseline."""
    a = run_svm_audit(_passive_session(tmp_path), book)
    assert a.n_stub_oracle >= 1
    assert a.oracle_usd < a.base_usd       # real win on this favorable case
    big = max(a.lives, key=lambda bl: bl.est_tokens)
    assert big.est_tokens > 800            # the newsletter block is the big one


# --------------------------------------------------------------------------
# Context-MRC (§3.3)
# --------------------------------------------------------------------------

def test_mrc_curve_monotone_and_knee_in_window():
    lives = [
        BlockLife("b1", "tool_result", None, 1000, 3300.0, 0, "x",
                  ref_turns=[2, 3, 10, 40]),
        BlockLife("b2", "tool_result", None, 500, 1650.0, 1, "y",
                  ref_turns=[3, 5]),
    ]
    mrc = build_mrc(lives, window_turns=50)
    miss = [m for _, m in mrc.curve]
    assert all(miss[i] >= miss[i + 1] for i in range(len(miss) - 1))  # non-increasing
    assert 0 <= mrc.knee <= mrc.window_turns
    assert mrc.curve[0][1] <= 1.0 and mrc.curve[-1][1] >= 0.0


def test_mrc_empty_when_no_references():
    lives = [BlockLife("b", "text", None, 100, 330.0, 0, "z", ref_turns=[])]
    mrc = build_mrc(lives, window_turns=10)
    assert mrc.distances == []
    assert mrc.knee == 0


# --------------------------------------------------------------------------
# Schema tax (design law 1) and harmlessness bound (V-D / H17)
# --------------------------------------------------------------------------

def test_schema_tax_present_in_svm_absent_in_baseline(tmp_path, book):
    a = run_svm_audit(_passive_session(tmp_path), book)
    assert a.base_bill.schema_tax == 0.0
    assert a.oracle_bill.schema_tax > 0.0
    # design law 1: the MCP schema is a tiny, bounded standing cost
    assert a.oracle_bill.schema_tax * a.base_rate < 0.25 * a.real_usd


def test_adversarial_no_bad_stubs_loss_is_only_schema_tax(tmp_path, book):
    """V-D flavour. On a short session of hot blocks the *decision* engine is
    harmless: L0 stubs nothing (no fault-inducing mistakes). The only residual
    loss is the standing MCP schema tax — and it is fully accounted for.

    NB (H17 finding): on a tiny/cheap session that fixed schema cost can be a
    double-digit % of the bill, i.e. the schema tax — not bad stubbing — is the
    thing that can break the ≤2% max-regret guarantee. The mitigation lives in
    the schema (lazy-attach SVM tools above a size threshold), not in π_b."""
    b = SessionBuilder()
    b.user_text(GOAL)
    for i in range(6):
        t = b.tool_use("Read", {"file_path": "src/payments.py"})
        b.assistant(MODEL, [t])
        b.tool_result(t["id"], PAYMENTS)   # recurs every turn → kept
        b.assistant(MODEL, [b.text("reconcile refund payments ledger amount.")])
    session = parse_built(b, tmp_path)
    a = run_svm_audit(session, book)
    assert a.n_stub_l0 == 0                       # no wrong stubs on hot blocks
    assert a.l0_bill.faults == 0
    # the entire L0 vs baseline delta is the schema tax — nothing else
    schema_usd = a.l0_bill.schema_tax * a.base_rate
    assert abs((a.l0_usd - a.base_usd) - schema_usd) < 1e-6


def test_oracle_at_least_as_good_as_doing_nothing(tmp_path, book):
    """The oracle policy, which knows the true future, can never lose money on
    the favorable passive session relative to baseline."""
    a = run_svm_audit(_passive_session(tmp_path), book)
    assert a.oracle_usd <= a.base_usd + 1e-9
