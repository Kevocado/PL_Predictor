"""The unfitted blend weight: what the three docstrings say, and what the code does.

The review flagged three docstrings in `player_goals.py` as disagreeing about
the direct-only fallback. They agree about the MECHANISM — `None` when nothing
was fitted, and `blend_contribution`'s `weight is None` branch serving the direct
model alone — and disagree about HISTORY: `_serving_blend_weight` says the
behaviour "is what `_fit_blend_weight`'s docstring has always promised", which
is a claim about a past that no test can check, and which is the sort of
sentence that stops being edited when the code changes under it.

So this file pins the behaviour the docs describe, which is the part that can be
checked, and leaves the historical claim where it is with a note about what it
can and cannot assert. A docstring that says "always" is unfalsifiable; a
docstring that says "here is the branch, and here is a test" is not.

The asymmetry is the load-bearing part, so it is asserted directly: an unfitted
weight is NOT the neutral 0.5. `NEUTRAL_BLEND_WEIGHT` blends 50/50 against an
*uncalibrated* union, so using it when no weight was fitted would be an
undisclosed behaviour change rather than a conservative default. Serving the
direct model alone is the choice that cannot be worse than inventing a weight.
"""
from pl_predictor.models.player_goals import (
    NEUTRAL_BLEND_WEIGHT,
    _serving_blend_weight,
    blend_contribution,
)


def test_an_unfitted_weight_is_none_and_not_the_neutral_share():
    """The distinction the two branches exist to make.

    Asserted as an inequality as well as an equality: a "harmless" refactor that
    made the unfitted case return the neutral share would pass a test that only
    checked `is None`, and would change what every unfitted fixture is served.
    """
    assert _serving_blend_weight(None) is None
    assert _serving_blend_weight(None) != NEUTRAL_BLEND_WEIGHT


def test_a_fitted_weight_still_maps_to_the_neutral_share():
    """The control, and the other half of the contract.

    The fitted value is real information but is not what serving applies, so it
    is recorded separately in the manifest. Without this the previous test would
    pass for a function that ignored its argument entirely.
    """
    assert _serving_blend_weight(0.31) == NEUTRAL_BLEND_WEIGHT
    assert _serving_blend_weight(0.87) == NEUTRAL_BLEND_WEIGHT
    assert NEUTRAL_BLEND_WEIGHT == 0.5


def test_the_unfitted_branch_serves_the_direct_model_alone():
    """What the three docstrings all claim, stated as behaviour.

    `weight is None` returns the direct probability UNCHANGED — not a blend, and
    not the union. A future edit that made this a 50/50 mix would be the exact
    undisclosed change the docstring warns against, and would be invisible from
    the manifest because the fitted weight is recorded separately.
    """
    direct, union = 0.62, 0.31
    assert blend_contribution(direct, union, None) == direct


def test_the_fitted_branch_does_blend():
    """The control for the one above: the branches are genuinely different."""
    direct, union = 0.62, 0.31
    blended = blend_contribution(direct, union, NEUTRAL_BLEND_WEIGHT)
    assert blended != direct
    assert blended == 0.5 * direct + 0.5 * union


def test_no_direct_probability_falls_back_to_the_union_either_way():
    """The third branch, and the one that is not about the weight at all."""
    assert blend_contribution(None, 0.31, None) == 0.31
    assert blend_contribution(None, 0.31, NEUTRAL_BLEND_WEIGHT) == 0.31
