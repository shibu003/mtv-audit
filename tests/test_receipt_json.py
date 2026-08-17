"""The JSON receipt must match the schema it publishes.

A schema nobody validates against is documentation, not a contract. These
checks run every emitted receipt through `schema/receipt-v1.schema.json` using
a small validator written for the subset of JSON Schema the file actually uses
— the point is to have no dependency, so an emitter in any project can copy
the schema without pulling in a validator stack.
"""
from __future__ import annotations

import json
from pathlib import Path

from conftest import GOAL, TRACE, parse_built

from mtv_audit.attribution import ALL_CHANNELS, audit_all_dials
from mtv_audit.receipt import SCHEMA_VERSION, build_receipt, dump_receipt
from mtv_audit.synth import SessionBuilder

MODEL = "claude-sonnet-4-6"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "receipt-v1.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# a validator for the subset the schema uses


def validate(node, schema, path="$", defs=None):
    """Raise AssertionError on the first violation."""
    defs = defs if defs is not None else SCHEMA.get("$defs", {})
    if "$ref" in schema:
        ref = schema["$ref"].rsplit("/", 1)[-1]
        return validate(node, defs[ref], path, defs)
    if "const" in schema:
        assert node == schema["const"], f"{path}: expected {schema['const']!r}, got {node!r}"
    if "enum" in schema:
        assert node in schema["enum"], f"{path}: {node!r} not in {schema['enum']}"
    t = schema.get("type")
    if t == "object":
        assert isinstance(node, dict), f"{path}: expected object, got {type(node).__name__}"
        for key in schema.get("required", []):
            assert key in node, f"{path}: missing required key {key!r}"
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(node) - set(props)
            assert not extra, f"{path}: unexpected keys {sorted(extra)}"
        for key, value in node.items():
            if key in props:
                validate(value, props[key], f"{path}.{key}", defs)
    elif t == "array":
        assert isinstance(node, list), f"{path}: expected array"
        for i, item in enumerate(node):
            validate(item, schema["items"], f"{path}[{i}]", defs)
    elif t == "integer":
        assert isinstance(node, int) and not isinstance(node, bool), f"{path}: expected integer, got {node!r}"
    elif t == "number":
        assert isinstance(node, (int, float)) and not isinstance(node, bool), f"{path}: expected number"
    elif t == "string":
        assert isinstance(node, str), f"{path}: expected string, got {type(node).__name__}"
    elif t == "boolean":
        assert isinstance(node, bool), f"{path}: expected boolean"
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        if "minimum" in schema:
            assert node >= schema["minimum"], f"{path}: {node} < minimum {schema['minimum']}"
        if "maximum" in schema:
            assert node <= schema["maximum"], f"{path}: {node} > maximum {schema['maximum']}"


def _session(tmp_path):
    b = SessionBuilder()
    b.user_text(GOAL)
    t1 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(MODEL, [t1])
    b.tool_result(t1["id"], TRACE)
    t2 = b.tool_use("Edit", {"file_path": "src/payments.py", "old_string": "a", "new_string": "b"})
    b.assistant(MODEL, [t2])
    b.tool_result(t2["id"], "updated")
    t3 = b.tool_use("Bash", {"command": "pytest tests/test_payments.py -x"})
    b.assistant(MODEL, [t3])
    b.tool_result(t3["id"], "1 passed in 0.2s")
    return parse_built(b, tmp_path)


def _receipt(tmp_path, book, **kw):
    session = _session(tmp_path)
    return build_receipt(session, audit_all_dials(session, book), book, **kw)


# --------------------------------------------------------------------------
# the checks


def test_the_validator_actually_rejects_something():
    """Guard the guard: a validator that accepts everything proves nothing."""
    validate(SCHEMA_VERSION, SCHEMA["properties"]["schema_version"], "$.schema_version")
    try:
        validate(999, SCHEMA["properties"]["schema_version"], "$.schema_version")
    except AssertionError:
        return
    raise AssertionError("the validator accepted a wrong schema_version")


def test_receipt_matches_the_published_schema(tmp_path, book):
    validate(_receipt(tmp_path, book, sessions_per_month=100), SCHEMA)


def test_every_channel_key_is_present_even_at_zero(tmp_path, book):
    channels = _receipt(tmp_path, book)["channels"]
    assert set(channels) == set(ALL_CHANNELS), (
        "a missing channel is indistinguishable from a rule that never ran"
    )


def test_receipt_carries_no_free_text(tmp_path, book):
    """The reason this receipt needs no redaction pass: nothing in it is prose.

    Walk every string in the emitted object and assert it sits at one of the
    identifier-ish fields. If someone adds an excerpt later, this fails.
    """
    receipt = _receipt(tmp_path, book, sessions_per_month=10)
    allowed = {
        "session.id", "session.source_filename", "session.audited_on",
        "session.data_provenance", "totals.price_table_version",
        "recoverable.dial", "recoverable.replay_status",
        "emitter.name", "emitter.version",
    }

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for v in node:
                walk(v, f"{path}[]")
        elif isinstance(node, str):
            base = path.replace("[]", "")
            assert base in allowed or base.startswith("top_items"), (
                f"unexpected free text at {path}: {node!r}"
            )

    walk(receipt)


def test_top_items_identify_by_id_not_content(tmp_path, book):
    for item in _receipt(tmp_path, book)["top_items"]:
        assert "excerpt" not in item, "an excerpt would put session text in a shared artefact"
        assert item["block_id"], "an item must still be locatable in the local log"


def test_recoverable_is_not_claimed_verified_without_a_replay(tmp_path, book):
    r = _receipt(tmp_path, book)["recoverable"]
    assert r["verified"] is False
    assert r["replay_status"] == "NOT_RUN"


def test_dial_totals_are_monotonic(tmp_path, book):
    """A stricter dial must never claim less waste than a looser one.

    Direction per tests/test_e2e.py and the README: saver >= balanced >=
    optimizer. Same invariant, checked here on the JSON projection so an
    emitter cannot ship a receipt that violates it.
    """
    p = _receipt(tmp_path, book, sessions_per_month=1)["projection"]["by_dial"]
    eps = 1e-9
    assert p["saver"] + eps >= p["balanced"] >= p["optimizer"] - eps, p


def test_emitter_block_is_optional_and_round_trips(tmp_path, book):
    assert "emitter" not in _receipt(tmp_path, book)
    tagged = _receipt(tmp_path, book, emitter=("some-other-agent", "2.1.0"))
    assert tagged["emitter"] == {"name": "some-other-agent", "version": "2.1.0"}
    validate(tagged, SCHEMA)


def test_dump_receipt_emits_parseable_json(tmp_path, book):
    session = _session(tmp_path)
    text = dump_receipt(session, audit_all_dials(session, book), book)
    validate(json.loads(text), SCHEMA)
