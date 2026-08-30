import pandas as pd

from pl_predictor.features.player_form import build_historical_player_form
from pl_predictor.models import player_ratings


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
