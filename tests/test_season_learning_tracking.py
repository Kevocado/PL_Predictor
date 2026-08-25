import pandas as pd

from pl_predictor.tracking import season_learning, store


def test_study_snapshots_are_immutable_and_reconcile_from_confirmed_result(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "study.db")
    snapshot = pd.DataFrame(
        [
            {
                "event_id": "fixture-1", "team_home": "Arsenal", "team_away": "Chelsea",
                "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "cadence": "every_19_matches",
                "arm": "season_weighted", "current_season_matches_seen": 19,
                "home_win": 0.55, "draw": 0.25, "away_win": 0.20,
            }
        ]
    )

    assert season_learning.record_study_predictions(snapshot) == 1
    changed = snapshot.assign(home_win=0.99)
    assert season_learning.record_study_predictions(changed) == 0

    matches = pd.DataFrame(
        [{"team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"), "ftr": "H"}]
    )
    assert season_learning.reconcile_study_predictions(matches) == 1
    report = season_learning.live_study_summary()
    assert report[0]["n_fixtures"] == 1
    assert report[0]["rps"] >= 0
