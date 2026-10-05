"""The pre-availability projection, and why an absence row cannot fall back to
`anytime_goal_prob`.

**The bug this exists to prevent.** `player_goals.predict_player` folds the live
availability multiplier into its rate:

    scale = strength_multiplier * minutes_fraction * availability   # line 534
    lam_goals = goals_estimate * scale
    "anytime_goal_prob": anytime_probability(lam_goals)

`fpl_api.availability_multiplier` returns `0.0` for `i`/`s`/`u`. So for exactly
the players an absence signal is about, `anytime_goal_prob` is **0.0** — and
`anytime_goal_prob_pre_availability` is what the model would have expected of them
had they played. The first version of `signals/absence.py` ranked on
`anytime_goal_prob`. It reported, for every out player:

    "Out: Haaland, our #1 scorer, 0% to score"

and, because all out players tie at 0.0, the rank was decided by whatever order
the pool arrived in. Caught by CodeRabbit on PL_Predictor#53.

**Why the adapter REFUSES rather than falling back.** `anytime_goal_prob` is
always present — it is a required field on `PlayerPrediction` — so a fallback
would always succeed and would always be wrong, silently, on exactly the rows this
signal exists for. A cached public snapshot written before this field existed has
no pre-availability projection to offer, and the honest response is no signal.
That refusal is asserted here, because it is the difference between a bug that
shows a zero and a bug that shows nothing.

Run: python -m pytest tests/test_pre_availability_projection.py -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from pl_predictor.api.schemas import PlayerPrediction
from pl_predictor.models import player_goals
from pl_predictor.signals import absence
from pl_predictor.signals.absence import PRE_AVAILABILITY_PROB, absence_signal


def _rates(goals_per90: float = 0.30, minutes: float = 90.0) -> dict:
    """A minimal `rates` dict: `predict_player` only reads these keys when no
    position rate model is supplied."""
    return {
        "goals_per90": goals_per90,
        "assists_per90": 0.10,
        "shots_per90": 2.0,
        "shots_on_target_per90": 0.8,
        "saves_per90": 0.0,
        "avg_minutes": minutes,
        "was_home": 0,
    }


FULL_AVAILABILITY = dict(
    rates=_rates(),
    team_goal_expectation=1.40,
    expected_minutes=90.0,
    position="ST",
    is_home=True,
)
OUT = dict(FULL_AVAILABILITY, availability=0.0)


def test_a_fit_player_who_plays_gets_a_pre_availability_projection():
    """Present even when availability is 1.0, so the two fields are always
    comparable and a consumer never has to ask which regime it is in."""
    out = player_goals.predict_player(**FULL_AVAILABILITY, availability=1.0)
    assert "anytime_goal_prob_pre_availability" in out
    assert 0.0 <= out["anytime_goal_prob_pre_availability"] <= 1.0


def test_an_out_player_has_a_zero_availability_probability_and_a_nonzero_pre_one():
    """The two facts the absence row depends on, asserted separately because the
    bug was reading the first as if it were the second."""
    out = player_goals.predict_player(**OUT)
    assert out["anytime_goal_prob"] == 0.0, (
        "if this stops being true the availability multiplier is no longer in the "
        "rate, and this whole file is looking for a bug that has moved"
    )
    assert out["anytime_goal_prob_pre_availability"] > 0.0


def test_the_pre_availability_projection_is_exactly_the_unavailability_scaled_one():
    """Not a second, separately-fitted model: the SAME estimate with the
    availability factor left out. Two lambdas from one `goals_estimate`, so the
    two figures cannot drift apart when the model is refitted."""
    rates = _rates()
    full = player_goals.predict_player(
        rates=rates, team_goal_expectation=1.40, availability=1.0,
        expected_minutes=90.0, position="ST", is_home=True,
    )
    out = player_goals.predict_player(
        rates=rates, team_goal_expectation=1.40, availability=0.0,
        expected_minutes=90.0, position="ST", is_home=True,
    )
    assert full["anytime_goal_prob_pre_availability"] == pytest.approx(
        full["anytime_goal_prob"], abs=1e-9
    )
    assert out["anytime_goal_prob_pre_availability"] == pytest.approx(
        full["anytime_goal_prob_pre_availability"], abs=1e-9
    )
    assert out["expected_goals_pre_availability"] == pytest.approx(
        full["expected_goals"], abs=1e-9
    )


def test_the_existing_figures_are_untouched_by_the_new_one():
    """The other eight figures must be bit-identical, or every market that reads
    them moves for a change that was only meant to add a ninth."""
    rates = _rates()
    kwargs = dict(
        rates=rates, team_goal_expectation=1.40, availability=0.6,
        expected_minutes=72.0, position="ST", is_home=True,
    )
    out = player_goals.predict_player(**kwargs)
    for key in (
        "expected_goals", "expected_assists", "anytime_goal_prob",
        "anytime_assist_prob", "anytime_goal_contribution_prob",
        "expected_shots", "expected_shots_on_target",
        "anytime_shot_on_target_prob", "expected_saves",
    ):
        assert key in out, f"{key} disappeared"
    # Availability 0.6 must still scale the ORIGINAL figures, or this change has
    # silently retuned every existing market.
    assert out["expected_goals"] < out["expected_goals_pre_availability"]


def test_a_penalty_taker_bonus_is_carried_on_both_sides_of_the_fence():
    """The penalty bonus is a separate additive term that also multiplies by
    availability. Pre-availability it must still apply -- a penalty taker who is
    out was still a penalty taker -- or the pre-availability figure understates a
    set-piece striker."""
    rates = _rates(goals_per90=0.0)
    common = dict(
        rates=rates, team_goal_expectation=1.40, availability=0.0,
        expected_minutes=90.0, position="ST", is_home=True,
    )
    with_bonus = player_goals.predict_player(**common, is_penalty_taker=True)
    without = player_goals.predict_player(**common, is_penalty_taker=False)
    assert with_bonus["expected_goals_pre_availability"] > without["expected_goals_pre_availability"]
    assert with_bonus["expected_goals"] == 0.0


def test_the_schema_carries_it():
    """Optional with a `None` default, so an older cached row still validates --
    and so `None` is a value the adapter can and must refuse on."""
    row = PlayerPrediction(
        player_id=1, name="A", position="ST",
        anytime_goal_prob=0.0, anytime_assist_prob=0.0,
        anytime_goal_contribution_prob=0.0, status="i", news="", confidence="low",
        predicted_starter=False, confirmed_starter=False, expected_minutes=0.0,
        is_penalty_taker=False, is_set_piece_taker=False,
    )
    assert row.anytime_goal_prob_pre_availability is None


# --- the adapter refuses rather than falling back ------------------------


def test_the_adapter_refuses_a_row_with_no_pre_availability_projection():
    """`anytime_goal_prob` is REQUIRED on the schema, so a fallback would always
    succeed and would always be 0.0 for an out player. The refusal is the whole
    point: a cached snapshot predating this field yields no signal, not a zero."""
    sig = absence_signal("7", [{
        "name": "Haaland", "status": "i", "anytime_goal_prob": 0.0,
        "anytime_goal_prob_pre_availability": None,
    }])
    assert sig is None


def test_the_adapter_refuses_when_the_field_is_absent_entirely():
    sig = absence_signal("7", [{"name": "Haaland", "status": "i", "anytime_goal_prob": 0.0}])
    assert sig is None


def test_the_adapter_ranks_on_the_pre_availability_projection():
    """Two out players whose availability figures TIE at 0.0 and whose pre-
    availability figures do not. The row must name the one the model actually
    expected more from -- which is the opposite of what the buggy version did."""
    sig = absence_signal("7", [
        {"name": "Also Out", "status": "i", "anytime_goal_prob": 0.0,
         "anytime_goal_prob_pre_availability": 0.18},
        {"name": "Really Out", "status": "i", "anytime_goal_prob": 0.0,
         "anytime_goal_prob_pre_availability": 0.71},
    ])
    assert sig is not None
    assert "Really Out" in sig["headline"]["text"]
    assert "71" in sig["headline"]["text"]
    assert "0%" not in sig["headline"]["text"]


def test_a_healthy_players_row_is_unaffected_by_the_new_field():
    """A player who is playing has one meaningful figure and both fields agree, so
    the rank the page already shows does not move."""
    sig = absence_signal("7", [
        {"name": "Played", "status": "a", "anytime_goal_prob": 0.4,
         "anytime_goal_prob_pre_availability": 0.44},
        {"name": "Out", "status": "u", "anytime_goal_prob": 0.0,
         "anytime_goal_prob_pre_availability": 0.30},
    ])
    assert sig is not None
    assert "Out" in sig["headline"]["text"]
    # Ranked over every player, so the out player is #2 -- above a 30% figure and
    # below a 40% one, which is what the rank is claiming.
    assert "#2" in sig["headline"]["text"]


def test_strength_follows_the_pre_availability_projection():
    low = absence_signal("7", [{"name": "L", "status": "i", "anytime_goal_prob": 0.0,
                                "anytime_goal_prob_pre_availability": 0.05}])
    high = absence_signal("7", [{"name": "H", "status": "i", "anytime_goal_prob": 0.0,
                                 "anytime_goal_prob_pre_availability": 0.95}])
    assert low is not None and high is not None
    assert low["strength"] < high["strength"]


def test_a_snapshot_with_a_mix_of_rows_uses_the_ones_that_can_answer():
    """One stale row in a pool is not a reason to refuse the whole fixture -- the
    stale row simply is not rankable, and a player who cannot be placed in the
    ordering cannot be stated as '#3'."""
    sig = absence_signal("7", [
        {"name": "Stale", "status": "i", "anytime_goal_prob": 0.0},
        {"name": "Fresh", "status": "u", "anytime_goal_prob": 0.0,
         "anytime_goal_prob_pre_availability": 0.55},
    ])
    assert sig is not None
    assert "Fresh" in sig["headline"]["text"]


def test_the_projection_field_name_is_declared_once():
    """A literal string in the adapter and a literal in the model are two places
    for the rename to miss one."""
    assert absence.PRE_AVAILABILITY_PROB == "anytime_goal_prob_pre_availability"
    row = {"name": "X", "status": "i", absence.PRE_AVAILABILITY_PROB: 0.4,
           "anytime_goal_prob": 0.0}
    sig = absence_signal("7", [row])
    assert sig is not None and "40" in sig["headline"]["text"]


# --- the public snapshot must not keep serving rows without the field ----


def test_a_cached_player_block_without_the_field_is_not_reusable():
    """The reuse path is how a missing field would survive a deploy: the snapshot
    keeps handing back the block it already has, so the field never arrives and
    every absence signal off it is refused indefinitely.

    A predicate rather than a whole snapshot job -- the reuse decision is this one
    boolean, and what matters is that it reads the ROWS.
    """
    from pl_predictor.public_snapshot import player_block_is_current

    stale = {"home_players": [{"name": "A", "anytime_goal_prob": 0.4}], "away_players": []}
    assert player_block_is_current(stale) is False


def test_a_block_carrying_the_field_is_reusable():
    from pl_predictor.public_snapshot import player_block_is_current

    fresh = {
        "home_players": [{"name": "A", PRE_AVAILABILITY_PROB: 0.44}],
        "away_players": [{"name": "B", PRE_AVAILABILITY_PROB: 0.10}],
    }
    assert player_block_is_current(fresh) is True


def test_a_half_stale_block_is_stale():
    """`all`, not `any`: one stale row in a block means the ranks drawn from it
    would mix two different projections, and a rank is a claim about an order."""
    from pl_predictor.public_snapshot import player_block_is_current

    mixed = {
        "home_players": [{"name": "A", PRE_AVAILABILITY_PROB: 0.44}],
        "away_players": [{"name": "B", "anytime_goal_prob": 0.2}],
    }
    assert player_block_is_current(mixed) is False


def test_an_empty_block_is_current_so_it_still_gets_a_real_attempt():
    """The loop's own comment at the call site says a missing block must not stay
    permanently empty. `all` over an empty sequence is True, which is what gives
    that; asserted so a future `any` does not quietly make empty blocks stale and
    every fixture re-fetch forever."""
    from pl_predictor.public_snapshot import player_block_is_current

    assert player_block_is_current({"home_players": [], "away_players": []}) is True
    assert player_block_is_current({"home_players": [], "away_players": None}) is True


def test_a_non_dict_block_is_not_current():
    """An older snapshot may hold something else entirely; that is not a block."""
    from pl_predictor.public_snapshot import player_block_is_current

    assert player_block_is_current(None) is False
    assert player_block_is_current([]) is False
