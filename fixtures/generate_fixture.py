"""Generate fixtures/session_fixture.jsonl (deterministic).

Narrative: "fix the failing refund test" — crafted so every channel fires:
  clean  large irrelevant README read early, re-sent every turn after
  retry  two failing pytest tracebacks re-sent across subsequent turns
  comm   Task subagent payload that copies code + tracebacks verbatim
  deep   long thinking before a trivial Read of a tiny config file
  model  that trivial step runs on the top-tier (opus) model -> flag
  stop   after tests pass, two redundant verification turns keep burning
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtv_audit.synth import SessionBuilder  # noqa: E402

OPUS = "claude-opus-4-5-20251101"

README = (
    "# Acme Webshop Monorepo\n\nWelcome! This repository contains the storefront, "
    "the marketing site, the design system, and assorted infrastructure glue. "
    "Below you will find onboarding instructions, brand guidelines, a glossary of "
    "team acronyms, the holiday on-call rotation, and a long history section about "
    "how the company migrated from SVN to Git in 2014. "
) + ("This section is purely historical onboarding trivia about office plants. " * 40)

PAYMENTS_SRC = (
    "def refund(order, amount):\n"
    "    \"\"\"Refund `amount` from order balance; raises on over-refund.\"\"\"\n"
    "    if amount <= 0:\n"
    "        raise ValueError('amount must be positive')\n"
    "    if amount > order.captured:\n"
    "        raise ValueError('refund exceeds captured amount')\n"
    "    order.captured -= amount\n"
    "    order.refunds.append(amount)\n"
    "    return order.captured\n"
    + ("# payments helper utilities for refund ledger reconciliation\n" * 30)
)

TRACEBACK_1 = (
    "============================= test session starts ==============================\n"
    "tests/test_payments.py::test_refund FAILED\n"
    "=================================== FAILURES ===================================\n"
    "Traceback (most recent call last):\n"
    "  File 'tests/test_payments.py', line 42, in test_refund\n"
    "    assert refund(order, 30) == 70\n"
    "AssertionError: assert 60 == 70\n"
    + ("E       +  where 60 = refund(order, 30) in refund ledger state dump line\n" * 28)
    + "=========================== 1 failed in 0.21s ===========================\n"
)

TRACEBACK_2 = (
    "tests/test_payments.py::test_refund FAILED\n"
    "Traceback (most recent call last):\n"
    "  File 'src/payments.py', line 7, in refund\n"
    "    order.refunds.append(amount)\n"
    "AttributeError: 'Order' object has no attribute 'refunds'\n"
    + ("E       full Order repr with forty fields dumped for debugging purposes here\n" * 24)
    + "=========================== 1 failed in 0.19s ===========================\n"
)

SUBAGENT_PROMPT = (
    "Review the refund implementation below and the two failing traces, then "
    "suggest edge cases for tests/test_payments.py.\n\n--- src/payments.py ---\n"
    + PAYMENTS_SRC
    + "\n--- failure trace 1 ---\n" + TRACEBACK_1
    + "\n--- failure trace 2 ---\n" + TRACEBACK_2
)

LONG_THINKING = (
    "Let me reason very carefully about whether to open the editor config. "
    "There are many philosophical considerations about configuration files. "
) + ("Continuing to deliberate at length about a trivial file read. " * 24)


def build() -> SessionBuilder:
    b = SessionBuilder()

    # 0 user instruction
    b.user_text("Fix the failing test in tests/test_payments.py::test_refund. "
                "Run pytest, find the bug in src/payments.py refund logic, fix it.")

    # 1-2 irrelevant README read (clean source)
    t_readme = b.next_tool_id()
    b.assistant(OPUS, [b.text("Let me get oriented in the repo first."),
                       b.tool_use("Read", {"file_path": "README.md"}, t_readme)],
                cache_hit=False)
    b.tool_result(t_readme, README)

    # 3-4 relevant source read
    t_src = b.tool_use("Read", {"file_path": "src/payments.py"})
    b.assistant(OPUS, [b.text("Now the refund implementation."), t_src])
    b.tool_result(t_src["id"], PAYMENTS_SRC)

    # 5-6 first test run -> failure 1
    t_test1 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(OPUS, [t_test1])
    b.tool_result(t_test1["id"], TRACEBACK_1)

    # 7-8 first fix attempt (contaminated retry context from here on)
    t_edit1 = b.tool_use("Edit", {"file_path": "src/payments.py",
                                  "old_string": "order.captured -= amount",
                                  "new_string": "order.captured = order.captured - amount"})
    b.assistant(OPUS, [b.thinking("The assertion expects 70; ledger math is off by 10."),
                       t_edit1])
    b.tool_result(t_edit1["id"], "The file src/payments.py has been updated.")

    # 9-10 second test run -> failure 2
    t_test2 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(OPUS, [t_test2])
    b.tool_result(t_test2["id"], TRACEBACK_2)

    # 11-12 second fix
    t_edit2 = b.tool_use("Edit", {"file_path": "src/payments.py",
                                  "old_string": "order.refunds.append(amount)",
                                  "new_string": "getattr(order, 'refunds', []).append(amount)"})
    b.assistant(OPUS, [b.thinking("Order lacks a refunds attribute; guard it."), t_edit2])
    b.tool_result(t_edit2["id"], "The file src/payments.py has been updated.")

    # 13-14 subagent spawned with full-state rebroadcast (comm)
    t_task = b.tool_use("Task", {"description": "refund test edge cases",
                                 "prompt": SUBAGENT_PROMPT})
    b.assistant(OPUS, [b.text("Spawning a reviewer subagent for edge cases."), t_task])
    b.tool_result(t_task["id"], "Subagent: consider zero-amount, over-refund, and float drift cases.")

    # 15-16 third test run -> success (stop boundary)
    t_test3 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(OPUS, [t_test3])
    b.tool_result(t_test3["id"],
                  "============================= test session starts ==============================\n"
                  "tests/test_payments.py::test_refund PASSED\n"
                  "=========================== 2 passed in 0.18s ===========================\n")

    # 17-18 post-completion trivial step with overthinking (deep + model flag + stop)
    t_cfg = b.tool_use("Read", {"file_path": ".editorconfig"})
    b.assistant(OPUS, [b.thinking(LONG_THINKING), t_cfg])
    b.tool_result(t_cfg["id"], "root = true\n[*]\nindent_style = space\n")

    # 19-20 redundant re-verification (stop)
    t_test4 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(OPUS, [b.text("Re-running the suite once more just to be sure."), t_test4])
    b.tool_result(t_test4["id"], "=========================== 2 passed in 0.18s ===========================\n")

    # 21 final message (stop)
    b.assistant(OPUS, [b.text("All tests pass. The refund bug is fixed.")])
    return b


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "session_fixture.jsonl")
    build().write_jsonl(out)
    print(f"fixture written: {out}")


if __name__ == "__main__":
    main()
