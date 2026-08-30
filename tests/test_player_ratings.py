import pandas as pd
import pytest

from pl_predictor.features.player_form import build_historical_player_form
from pl_predictor.models import player_ratings


def _history_row(name, position, team, season, **overrides):
    row = {
        "name": name,
        "position": position,
        "team": team,
        "season": season,
        "minutes": 1_900,
        "starts": 22,
        "goals_scored": 8,
        "assists": 4,
        "expected_goals": 9.0,
        "expected_assists": 4.0,
        "expected_goal_involvements": 13.0,
        "clean_sheets": 5,
        "saves": 60,
        "goals_conceded": 30,
        "defensive_contribution": 80,
        "bps": 360,
        "total_points": 120,
    }
    row.update(overrides)
    return row


def test_multi_season_prior_rewards_sustained_elite_role_evidence_not_fpl_points():
    """Replacing observed role evidence with FPL points must not decide ability."""
    rows = []
    for season in ("2023-24", "2024-25", "2025-26"):
        rows.extend(
            [
                _history_row("Elite Forward", "FWD", "Arsenal", season, goals_scored=28, assists=8, expected_goals=25.0, expected_assists=6.0, expected_goal_involvements=31.0, bps=720, total_points=100),
                _history_row("Good Forward", "FWD", "Chelsea", season, goals_scored=12, assists=5, expected_goals=12.0, expected_assists=4.0, expected_goal_involvements=16.0, bps=360, total_points=260),
                _history_row("Average Forward", "FWD", "Everton", season, goals_scored=7, assists=2, expected_goals=7.0, expected_assists=2.0, expected_goal_involvements=9.0, bps=250, total_points=180),
                *[_history_row(f"Rotation {index}", "FWD", f"Club {index}", season, goals_scored=6, assists=2, expected_goals=6.0, expected_assists=2.0, expected_goal_involvements=8.0, bps=220, total_points=220) for index in range(4)],
            ]
        )

    priors = player_ratings.build_historical_priors(pd.DataFrame(rows))

    assert priors["elite forward"]["quality_rating"] >= 85
    assert priors["elite forward"]["quality_rating"] > priors["good forward"]["quality_rating"]
    assert priors["elite forward"]["rating_status"] == "established"


def test_goalkeeper_saves_without_prevention_advantage_cannot_become_elite():
    """Raw saves should not make a busy keeper appear world class."""
    rows = []
    for season in ("2023-24", "2024-25", "2025-26"):
        rows.extend(
            [
                _history_row("Busy Keeper", "GK", "Leeds", season, saves=500, clean_sheets=1, goals_conceded=62, bps=420),
                _history_row("Leeds Reserve", "GK", "Leeds", season, minutes=100, saves=20, clean_sheets=0, goals_conceded=3, bps=20),
                _history_row("Elite Keeper", "GK", "Arsenal", season, saves=80, clean_sheets=16, goals_conceded=20, bps=540),
                _history_row("Arsenal Reserve", "GK", "Arsenal", season, minutes=100, saves=4, clean_sheets=1, goals_conceded=1, bps=20),
                _history_row("Average Keeper", "GK", "Fulham", season, saves=95, clean_sheets=7, goals_conceded=33, bps=360),
            ]
        )

    priors = player_ratings.build_historical_priors(pd.DataFrame(rows))

    assert priors["busy keeper"]["quality_rating"] < 70
    assert priors["elite keeper"]["quality_rating"] > priors["busy keeper"]["quality_rating"]


def test_goalkeeper_quality_uses_xg_prevention_not_a_weaker_reserve_as_the_benchmark():
    """A weaker backup must not turn a short Darlow-like spell into the top keeper rating."""
    rows = []
    for season in ("2023-24", "2024-25", "2025-26"):
        rows.extend([
            _history_row("Elite Keeper", "GK", "Elite FC", season, clean_sheets=12, goals_conceded=25, expected_goals_conceded=40.0, bps=450),
            _history_row("League Average", "GK", "Average FC", season, clean_sheets=7, goals_conceded=33, expected_goals_conceded=33.0, bps=330),
            _history_row("Average Reserve", "GK", "Average FC", season, minutes=100, clean_sheets=0, goals_conceded=5, expected_goals_conceded=5.0, bps=10),
            _history_row("Weak Starter", "GK", "Weak FC", season, clean_sheets=4, goals_conceded=45, expected_goals_conceded=45.0, bps=290),
            _history_row("Weak Reserve", "GK", "Weak FC", season, minutes=100, clean_sheets=0, goals_conceded=10, expected_goals_conceded=10.0, bps=10),
        ])
    rows.extend([
        _history_row("Karl Darlow", "GK", "Leeds", "2025-26", clean_sheets=5, goals_conceded=27, expected_goals_conceded=27.5, bps=334),
        _history_row("Leeds Reserve", "GK", "Leeds", "2025-26", minutes=100, clean_sheets=0, goals_conceded=15, expected_goals_conceded=15.0, bps=10),
    ])

    priors = player_ratings.build_historical_priors(pd.DataFrame(rows))

    assert priors["elite keeper"]["quality_rating"] > priors["karl darlow"]["quality_rating"]


def test_insufficient_recent_history_is_provisional():
    """Removing the evidence threshold would silently rank a 300-minute player."""
    history = pd.DataFrame([
        _history_row("New Signing", "MID", "Chelsea", "2025-26", minutes=300, starts=3, goals_scored=4, expected_goals=4.0, expected_goal_involvements=5.0),
        _history_row("Established Mid", "MID", "Arsenal", "2025-26"),
        _history_row("Established Mid", "MID", "Arsenal", "2024-25"),
        _history_row("Established Mid", "MID", "Arsenal", "2023-24"),
    ])

    priors = player_ratings.build_historical_priors(history)

    assert priors["new signing"]["rating_status"] == "provisional"
    assert priors["new signing"]["evidence_minutes"] == 300


@pytest.mark.parametrize("position", ["DEF", "MID", "FWD"])
def test_one_strong_season_cannot_claim_a_sustained_elite_quality_rating(position):
    """Calling a one-season player established would recreate Darlow-like false rankings."""
    rows = []
    for season in ("2023-24", "2024-25", "2025-26"):
        rows.extend(
            [
                    _history_row("Good Player", position, "Chelsea", season, goals_scored=12, assists=5, expected_goals=12.0, expected_assists=4.0, expected_goal_involvements=16.0),
                    _history_row("Average Player", position, "Everton", season, goals_scored=7, assists=2, expected_goals=7.0, expected_assists=2.0, expected_goal_involvements=9.0),
            ]
        )
    rows.append(_history_row("Single Season Star", position, "Leeds", "2025-26", goals_scored=30, assists=10, expected_goals=28.0, expected_assists=8.0, expected_goal_involvements=36.0))

    prior = player_ratings.build_historical_priors(pd.DataFrame(rows))["single season star"]

    assert prior["rating_status"] == "limited"
    assert prior["quality_rating"] <= 76


def test_historical_prior_adds_unique_first_last_alias_for_current_fpl_name():
    """Dropping aliases would make Bruno Borges Fernandes vanish from Bruno Fernandes's prior."""
    history = pd.DataFrame([
        _history_row("Bruno Borges Fernandes", "MID", "Man United", "2023-24"),
        _history_row("Bruno Borges Fernandes", "MID", "Man United", "2024-25"),
        _history_row("Bruno Borges Fernandes", "MID", "Man United", "2025-26"),
        _history_row("Other Mid", "MID", "Arsenal", "2025-26"),
    ])

    priors = player_ratings.build_historical_priors(history)

    assert priors["bruno fernandes"]["rating_status"] == "established"
    assert priors["bruno fernandes"]["quality_rating"] == priors["bruno borges fernandes"]["quality_rating"]


def test_historical_name_variant_is_one_player_before_multi_season_evidence_is_scored():
    """Keeping name variants separate lets a stable-name backup outrank an established starter."""
    history = pd.DataFrame([
        _history_row("Alisson Ramses Becker", "GK", "Liverpool", "2023-24", minutes=1_900, clean_sheets=13),
        _history_row("Alisson Ramses Becker", "GK", "Liverpool", "2024-25", minutes=1_800, clean_sheets=12),
        _history_row("Alisson Becker", "GK", "Liverpool", "2025-26", minutes=1_850, clean_sheets=12),
        _history_row("Comparable Keeper", "GK", "Arsenal", "2023-24", minutes=1_900, clean_sheets=8),
        _history_row("Comparable Keeper", "GK", "Arsenal", "2024-25", minutes=1_900, clean_sheets=8),
        _history_row("Comparable Keeper", "GK", "Arsenal", "2025-26", minutes=1_900, clean_sheets=8),
    ])

    prior = player_ratings.build_historical_priors(history)["alisson becker"]

    assert prior["evidence_minutes"] == 5_550


def test_history_identity_variants_follow_the_shortest_safe_name_through_multiple_merges():
    """Leaving a transitive alias at an intermediate name silently re-splits evidence."""
    history = pd.DataFrame([
        _history_row("Alex John Paul Smith", "MID", "Arsenal", "2023-24", minutes=1_800),
        _history_row("Alex John Smith", "MID", "Arsenal", "2024-25", minutes=1_850),
        _history_row("Alex Smith", "MID", "Arsenal", "2025-26", minutes=1_900),
        _history_row("Comparable Mid", "MID", "Chelsea", "2023-24"),
        _history_row("Comparable Mid", "MID", "Chelsea", "2024-25"),
        _history_row("Comparable Mid", "MID", "Chelsea", "2025-26"),
    ])

    prior = player_ratings.build_historical_priors(history)["alex smith"]

    assert prior["evidence_minutes"] == 5_550


def test_provisional_player_has_no_rankable_overall_but_keeps_live_form_and_impact():
    """A missing prior must not silently turn a new signing into a 50-rated player."""
    newcomer = _element(1, minutes=90, starts=1, goals=1, xg=0.4, xa=0.1)
    newcomer["form"] = "7.0"

    rating = player_ratings.rate_bootstrap_elements([newcomer], {3: "MID"}, historical_priors={})[1]

    assert rating["rating_status"] == "provisional"
    assert rating["quality_rating"] is None
    assert rating["overall_rating"] is None
    assert rating["live_form_rating"] is not None
    assert rating["current_impact_rating"] > 0


def test_availability_only_changes_impact_for_an_established_player():
    """Changing injury status must not rewrite durable ability or Overall."""
    available = _element(1, minutes=900, starts=10, goals=5, xg=4.0, xa=2.0)
    doubtful = {**available, "id": 2, "status": "d", "chance_of_playing_next_round": 50}
    priors = {"": {"quality_rating": 80.0, "rating_status": "established", "evidence_minutes": 3_000}}

    ratings = player_ratings.rate_bootstrap_elements([available, doubtful], {3: "MID"}, historical_priors=priors)

    assert ratings[1]["rating_status"] == "established"
    assert ratings[1]["quality_rating"] == ratings[2]["quality_rating"]
    assert ratings[1]["overall_rating"] == ratings[2]["overall_rating"]
    assert ratings[2]["current_impact_rating"] < ratings[1]["current_impact_rating"]


def test_quality_ignores_fpl_points_and_projection_fields():
    """Adding a projection dependency would make the descriptive rating circular."""
    element = _element(1, minutes=900, starts=10, goals=5, xg=4.0, xa=2.0, total_points=20)
    priors = {"": {"quality_rating": 82.0, "rating_status": "established", "evidence_minutes": 4_000}}

    baseline = player_ratings.rate_bootstrap_elements([element], {3: "MID"}, historical_priors=priors)[1]
    changed = player_ratings.rate_bootstrap_elements(
        [{**element, "total_points": 999, "projected_points": 999}], {3: "MID"}, historical_priors=priors
    )[1]

    assert changed["quality_rating"] == baseline["quality_rating"]
    assert changed["overall_rating"] == baseline["overall_rating"]


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


def test_missing_history_does_not_make_the_current_top_rank_elite():
    ratings = player_ratings.rate_bootstrap_elements(
        [_element(1, xg=2, xa=2), _element(2, xg=1, xa=1)], {3: "MID"}
    )

    assert {row["rating_status"] for row in ratings.values()} == {"provisional"}
    assert {row["overall_rating"] for row in ratings.values()} == {None}


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


def test_established_quality_stays_with_the_validated_multi_season_prior_early_in_season():
    current = _element(1, minutes=180, starts=2, xg=1.4, xa=0.1, goals=2)
    current.update({"first_name": "Erling", "second_name": "Haaland"})

    ratings = player_ratings.rate_bootstrap_elements(
        [current], {3: "MID"}, historical_priors={"erling haaland": {"quality_rating": 90.0, "rating_status": "established", "evidence_minutes": 5_000}}
    )

    # Current matches earn the separate Form lift. They must not overwrite a
    # multi-season Ability assessment after two starts.
    assert ratings[1]["quality_rating"] == 90.0


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
