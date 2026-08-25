import pandas as pd

from pl_predictor.evaluate import in_season_learning


def test_control_returns_empty_when_target_season_is_unavailable(monkeypatch):
    monkeypatch.setattr(in_season_learning.football_data, "load_training_data", lambda: pd.DataFrame())
    monkeypatch.setattr(in_season_learning, "build_training_frame", lambda **_: (pd.DataFrame(columns=["season", "date"]), []))

    report = in_season_learning.run_in_season_learning_control(target_season="2099-2100")

    assert report == {"target_season": "2099-2100", "checkpoints": []}


def test_checkpoint_plan_never_trains_on_same_day_as_evaluation():
    target = pd.DataFrame(
        [
            {"date": pd.Timestamp("2025-08-01"), "team_home": "A", "team_away": "B"},
            {"date": pd.Timestamp("2025-08-01"), "team_home": "C", "team_away": "D"},
            {"date": pd.Timestamp("2025-08-02"), "team_home": "E", "team_away": "F"},
        ]
    )

    checkpoints = in_season_learning.checkpoint_plan(target, "every_10_matches", horizon_matches=2)

    assert checkpoints == [{"matches_seen": 0, "evaluation_start": 0, "evaluation_end": 2, "checkpoint": "preseason"}]


def test_per_club_checkpoint_waits_until_every_team_reaches_19_matches():
    rows = []
    for match_number in range(19):
        rows.extend(
            [
                {"date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=match_number), "team_home": "A", "team_away": "B"},
                {"date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=match_number), "team_home": "C", "team_away": "D"},
            ]
        )
    rows.append({"date": pd.Timestamp("2025-09-01"), "team_home": "A", "team_away": "C"})

    checkpoints = in_season_learning.checkpoint_plan(pd.DataFrame(rows), in_season_learning.PER_CLUB_CADENCE)

    assert checkpoints[0]["matches_seen"] == 0
    assert checkpoints[1]["matches_seen"] == 38
    assert checkpoints[1]["checkpoint"] == "all clubs reached 19 matches"
