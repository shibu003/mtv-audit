"""Redaction for anything that reaches a receipt.

An agent session log is one of the most sensitive files on a developer's disk:
it contains prompts, source code, tool output, filesystem paths, and whatever
secrets happened to be echoed along the way. The audit reads all of that.

The receipt must not carry it. A receipt is the artefact people paste into a
ticket, mail to a vendor, or attach to an invoice — it has to be safe to hand
over, and that safety has to come from the tool, not from the sender
remembering to check.

Design: *the audit reads everything locally, the receipt carries numbers.* The
only free text that ever reaches a receipt is the Top-10 excerpt, so that is
the surface this module guards, plus the source path in the header.

`scrub` is deliberately blunt. A false positive costs one unreadable excerpt;
a false negative leaks a credential.
"""
from __future__ import annotations

import re

__all__ = ["scrub", "scrub_path", "PLACEHOLDER"]

PLACEHOLDER = "[redacted]"

# Ordered: the most specific patterns first, so a token is not partly eaten by
# a broader rule before its own rule sees it.
_RULES: list[tuple[re.Pattern[str], str]] = [
    # --- provider-shaped credentials -------------------------------------
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"), "[redacted:anthropic-key]"),
    (re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{16,}"), "[redacted:openai-key]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "[redacted:api-key]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), "[redacted:google-key]"),
    (re.compile(r"\bgh[pousr]_[0-9A-Za-z]{20,}"), "[redacted:github-token]"),
    (re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}"), "[redacted:slack-token]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[redacted:aws-key-id]"),
    (re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
     "[redacted:jwt]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "[redacted:private-key]"),
    # --- credentials embedded in URLs ------------------------------------
    (re.compile(r"\b([a-z][a-z0-9+.\-]*)://[^\s/@:]+:[^\s/@]+@"), r"\1://[redacted]@"),
    # --- assignments: KEY=value / "token": "value" ------------------------
    (re.compile(
        r"(?i)\b([A-Za-z0-9_\-]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|"
        r"AUTH|COOKIE|SESSION)[A-Za-z0-9_\-]*)\s*[:=]\s*[\"']?[^\s\"',;)}]{4,}"),
     r"\1=[redacted]"),
    # --- direct identifiers ----------------------------------------------
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
     "[redacted:email]"),
    # --- home directories: keep the shape, drop the identity --------------
    (re.compile(r"/Users/[^/\s\"']+"), "/Users/[redacted]"),
    (re.compile(r"/home/[^/\s\"']+"), "/home/[redacted]"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\s\"']+"), r"C:\\Users\\[redacted]"),
]


def scrub(text: str) -> str:
    """Remove credential-shaped and identifying substrings from free text.

    Applied to every excerpt before it can reach a receipt. Order matters:
    see _RULES.
    """
    if not text:
        return text
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def scrub_path(path: str) -> str:
    """Reduce a session path to its filename.

    The directory tree above a session log routinely names the employer, the
    client, or the unreleased product. The filename is enough to tell two
    receipts apart, which is all the receipt needs it for.
    """
    if not path:
        return path
    tail = re.split(r"[/\\]", str(path))[-1]
    return scrub(tail)
