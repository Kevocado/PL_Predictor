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
