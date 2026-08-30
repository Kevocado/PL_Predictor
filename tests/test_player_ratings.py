import pandas as pd
import pytest

from pl_predictor.features.player_form import build_historical_player_form
from pl_predictor.models import player_ratings


def test_role_component_ceilings_are_equalised_across_positions():
    """An equally maxed-out role must earn the same component total."""
    maxed_out = {
        "minutes": 900, "expected_goals": 100, "expected_assists": 100,
        "expected_goal_involvements": 100, "goals_scored": 100,
        "saves": 100, "clean_sheets": 100, "bps": 1000,
        "defensive_contribution": 1000, "threat": 2000, "creativity": 2000,
    }

    ceilings = {
        position: sum(player_ratings._role_components(maxed_out, position).values())
        for position in ("GK", "DEF", "MID", "FWD")
    }

    assert ceilings["DEF"] == 36.0
    for position in ("GK", "MID", "FWD"):
        assert ceilings[position] == pytest.approx(36.0, abs=0.01), position


def test_stale_prior_is_shrunk_toward_role_baseline_without_current_evidence():
    """Removing prior shrinkage would let an unused player's old score survive untouched."""
    quality, _ = player_ratings._quality_score({"minutes": 0, "starts": 0}, "GK", prior_quality=90.0)

    assert quality == pytest.approx(76.0, abs=0.1)
    assert quality < 90.0


def test_quality_without_a_historical_prior_keeps_the_role_baseline():
    """Only a real carried-over number is shrunk; absence of history is neutral."""
    quality, _ = player_ratings._quality_score({"minutes": 0, "starts": 0}, "MID", prior_quality=None)

    assert quality == 50.0


def _element(
    player_id,
    position=3,
    minutes=900,
    starts=10,
    goals=0,
    total_points=40,
    xg=0,
    xa=0,
    status="a",
    chance=None,
):
    return {
        "id": player_id,
        "element_type": position,
        "minutes": minutes,
        "starts": starts,
        "appearances": max(starts, 1),
        "goals_scored": goals,
        "total_points": total_points,
        "expected_goals": str(xg),
        "expected_assists": str(xa),
        "expected_goal_involvements": str(xg + xa),
        "clean_sheets": 0,
        "saves": 0,
        "defensive_contribution": 0,
        "bps": total_points * 3,
        "bonus": 0,
        "threat": xg * 100,
        "creativity": xa * 100,
        "status": status,
        "chance_of_playing_next_round": chance,
    }


def test_rating_scale_does_not_make_top_rank_elite():
    ratings = player_ratings.rate_bootstrap_elements(
        [_element(1, xg=2, xa=2), _element(2, xg=1, xa=1)], {3: "MID"}
    )

    assert max(row["overall_rating"] for row in ratings.values()) < 90


def test_form_needs_minutes_underlying_and_actual_evidence():
    breakout = _element(1, minutes=1_350, starts=15, goals=13, xg=11, xa=7, total_points=150)
    finishing_spike = _element(2, minutes=1_350, starts=15, goals=13, xg=1, xa=0, total_points=80)
    ratings = player_ratings.rate_bootstrap_elements([breakout, finishing_spike], {3: "MID"})

    assert ratings[1]["form_rating"] >= 10
    assert ratings[2]["form_rating"] < 10


def test_availability_changes_impact_only():
    available = _element(1, xg=5, xa=4, goals=5)
    doubtful = {**available, "id": 2, "status": "d", "chance_of_playing_next_round": 50}
    ratings = player_ratings.rate_bootstrap_elements([available, doubtful], {3: "MID"})

    assert ratings[1]["quality_rating"] == ratings[2]["quality_rating"]
    assert ratings[1]["form_rating"] == ratings[2]["form_rating"]
    assert ratings[2]["current_impact_rating"] < ratings[1]["current_impact_rating"]


def test_quality_uses_cached_prior_season_evidence_early_in_current_season():
    current = _element(1, minutes=180, starts=2, xg=1.4, xa=0.1, goals=2)
    current.update({"first_name": "Erling", "second_name": "Haaland"})

    ratings = player_ratings.rate_bootstrap_elements(
        [current], {3: "MID"}, historical_priors={"erling haaland": {"quality_rating": 90.0}}
    )

    # The carried-over signal still matters early, but is no longer treated
    # as a fully earned current-season 90 after only two starts.
    assert 70 < ratings[1]["quality_rating"] < 80


def test_quality_reserves_elite_scores_for_sustained_role_output():
    half_season_forward = _element(1, position=4, minutes=1_065, starts=10, goals=6, xg=7.25, xa=1.58)
    elite_forward = _element(2, position=4, minutes=2_953, starts=34, goals=27, xg=25.5, xa=2.67)

    half_season_quality, _ = player_ratings._quality_score(half_season_forward, "FWD")
    elite_quality, _ = player_ratings._quality_score(elite_forward, "FWD")

    assert half_season_quality < 75
    assert elite_quality >= 80


def test_live_fpl_form_is_separate_from_durable_quality_and_shrinks_tiny_samples():
    short_spell = _element(1, minutes=90, starts=1, goals=1, xg=0.2, xa=0.1)
    sustained_spell = _element(2, minutes=720, starts=8, goals=8, xg=4.0, xa=2.0)
    short_spell["form"] = "11.0"
    sustained_spell["form"] = "11.0"

    ratings = player_ratings.rate_bootstrap_elements([short_spell, sustained_spell], {3: "MID"})

    assert ratings[1]["live_form_rating"] < ratings[2]["live_form_rating"]


def test_live_fpl_form_changes_without_changing_durable_quality():
    cool = _element(1, minutes=900, starts=10, goals=4, xg=3.0, xa=1.0)
    hot = {**cool, "id": 2}
    cool["form"] = "2.0"
    hot["form"] = "10.0"

    ratings = player_ratings.rate_bootstrap_elements([cool, hot], {3: "MID"})

    assert ratings[1]["quality_rating"] == ratings[2]["quality_rating"]
    assert ratings[1]["live_form_rating"] < ratings[2]["live_form_rating"]


def test_exceptional_short_term_fpl_form_can_clear_quality_but_not_elite_ceiling():
    breakout = _element(1, minutes=108, starts=1, goals=2, xg=0.35, xa=1.05)
    breakout.update({"first_name": "Rayan", "second_name": "Cherki", "form": "11.0"})

    rating = player_ratings.rate_bootstrap_elements(
        [breakout], {3: "MID"}, historical_priors={"rayan cherki": {"quality_rating": 69.0}}
    )[1]

    assert rating["live_form_rating"] > rating["quality_rating"]
    assert rating["live_form_rating"] < 90


def test_two_game_form_does_not_cluster_at_an_80_rating():
    two_games = _element(1, minutes=180, starts=2, goals=3, xg=1.2, xa=0.4)
    two_games["form"] = "12.0"

    rating = player_ratings.rate_bootstrap_elements([two_games], {3: "MID"})[1]

    assert rating["live_form_rating"] < 80


def test_role_model_report_uses_role_specific_targets_and_strict_selection():
    rows = []
    for season_index, season in enumerate(("2022-23", "2023-24", "2024-25")):
        for position_index, position in enumerate(("GK", "DEF", "MID", "FWD")):
            for player_offset in range(2):
                element = season_index * 100 + position_index * 10 + player_offset
                for gameweek in range(1, 21):
                    rows.append(
                        {
                            "season": season,
                            "element": element,
                            "position": position,
                            "kickoff_time": pd.Timestamp("2022-08-01") + pd.Timedelta(days=season_index * 365 + gameweek * 7),
                            "minutes": 90,
                            "starts": 1,
                            "goals_scored": (gameweek + player_offset) % 3,
                            "assists": gameweek % 2,
                            "expected_goals": 0.2 + gameweek / 100,
                            "expected_assists": 0.1 + gameweek / 200,
                            "expected_goal_involvements": 0.3 + gameweek / 80,
                            "clean_sheets": int(gameweek % 3 == 0),
                            "saves": 3 + gameweek % 2,
                            "bps": 15 + gameweek,
                            "defensive_contribution": 8 + gameweek,
                        }
                    )

    report = player_ratings.evaluate_role_models(pd.DataFrame(rows))

    assert {"position", "target", "baseline_mae", "rich_mae", "baseline_rmse", "rich_rmse", "selected_model", "top_driver"} <= set(report.columns)
    assert set(report["target"]) == {"shot_prevention", "defence_and_attack", "creation_and_output", "finishing_and_output"}
    assert (report.loc[report["selected_model"] == "rich", "rich_mae"] < report.loc[report["selected_model"] == "rich", "baseline_mae"]).all()


def test_historical_form_does_not_carry_fpl_element_ids_between_seasons():
    history = pd.DataFrame(
        [
            {"season": "2023-24", "element": 9, "position": "MID", "kickoff_time": "2023-08-01", "minutes": 90, "goals_scored": 3, "assists": 1},
            {"season": "2024-25", "element": 9, "position": "MID", "kickoff_time": "2024-08-01", "minutes": 90, "goals_scored": 0, "assists": 0},
        ]
    )
    form, _ = build_historical_player_form(history)
    new_season_first_match = form[form["season"] == "2024-25"].iloc[0]

    assert pd.isna(new_season_first_match["goals_per90_last3"])
