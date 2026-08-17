"""pricing: every tier resolves, and the specific-before-general order holds.

These exist because the price table is matched by substring. That is fine
until a newer model's id contains an older key ("claude-sonnet-5" contains
"sonnet"), or a tier has no key at all and lands on "default" — both of which
mis-price silently, with a plausible-looking number and no error.
"""
import pytest

from mtv_audit.pricing import DEFAULT_PRICES, PriceBook, is_top_tier


@pytest.mark.parametrize(
    "model,inp,out",
    [
        ("claude-fable-5", 10.00, 50.00),
        ("claude-mythos-5", 10.00, 50.00),
        ("claude-opus-5", 5.00, 25.00),
        ("claude-opus-4-8", 5.00, 25.00),
        ("claude-opus-4-1-20250805", 15.00, 75.00),
        ("claude-sonnet-5", 2.00, 10.00),
        ("claude-sonnet-4-6", 3.00, 15.00),
        ("claude-haiku-4-5-20251001", 1.00, 5.00),
    ],
)
def test_each_model_prices_at_its_own_rate(model, inp, out):
    book = PriceBook.load()
    assert book.input_rate(model) == pytest.approx(inp / 1_000_000)
    assert book.output_rate(model) == pytest.approx(out / 1_000_000)


def test_no_current_tier_falls_through_to_default():
    """A tier landing on "default" is the silent-mis-pricing failure mode."""
    book = PriceBook.load()
    default = DEFAULT_PRICES["default"]
    for model in ("claude-fable-5", "claude-opus-5", "claude-sonnet-5",
                  "claude-haiku-4-5-20251001"):
        assert book._entry(model) is not default, f"{model} priced as default"


def test_specific_keys_precede_the_general_keys_they_extend():
    """The table is order-sensitive; assert the order rather than trusting it."""
    keys = list(DEFAULT_PRICES)
    for specific, general in (("sonnet-5", "sonnet"), ("opus-4-1", "opus")):
        assert keys.index(specific) < keys.index(general), (
            f'"{specific}" must precede "{general}" or the general key wins'
        )
    assert keys[-1] == "default", '"default" must be last'


def test_unknown_model_still_prices_rather_than_raising():
    book = PriceBook.load()
    assert book.output_rate("some-future-model") > 0
    assert book.output_rate(None) > 0


def test_top_tier_covers_every_model_priced_above_opus():
    assert is_top_tier("claude-opus-5")
    assert is_top_tier("claude-fable-5")
    assert is_top_tier("claude-mythos-5")
    assert not is_top_tier("claude-sonnet-5")
    assert not is_top_tier("claude-haiku-4-5-20251001")


def test_blended_prompt_rate_sits_between_cache_read_and_write():
    """Sanity-check the blend: a mixed turn costs more than an all-cache-read
    turn and less than an all-cache-write one, at the same model."""
    book = PriceBook.load()
    model = "claude-opus-5"
    base = book.input_rate(model)
    mixed = book.blended_prompt_rate(
        model, {"input_tokens": 100, "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 100})
    assert base * 0.10 < mixed < base * 1.25
