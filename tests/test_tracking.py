"""Round-trip check for the live prediction track record store."""

import pandas as pd
import penaltyblog as pb
import pytest

from pl_predictor.tracking import store


@pytest.fixture
def clean_db(tmp_path, monkeypatch):
    # Must patch the name as imported into `store` (`from ..config import
    # TRACKING_DB_PATH`), not `config.TRACKING_DB_PATH` itself — that's a
    # separate binding and patching it wouldn't affect what `store._connect`
    # actually opens. Previously this fixture deleted and rebuilt the real
    # `config.TRACKING_DB_PATH` (data/tracking.db) directly — every pytest
    # run was silently wiping this install's actual live prediction history,
    # discovered when a real live-captured (non-backfilled) prediction lost
    # its "captured before kickoff" distinction after a test run rebuilt it
    # via the self-healing backfill path instead.
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "test_tracking.db")
    yield


def test_record_is_idempotent(clean_db):
    table = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"),
                "home_win_prob": 0.6,
                "draw_prob": 0.25,
                "away_win_prob": 0.15,
                "over_2_5_prob": 0.7,
                "under_2_5_prob": 0.3,
                "btts_yes_prob": 0.55,
            }
        ]
    )
    assert store.record_predictions(table) == 6
    assert store.record_predictions(table) == 0  # already logged, no-op


def test_reconcile_resolves_against_actual_result(clean_db):
    table = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"),
                "home_win_prob": 0.6,
                "draw_prob": 0.25,
                "away_win_prob": 0.15,
                "over_2_5_prob": 0.7,
                "under_2_5_prob": 0.3,
                "btts_yes_prob": 0.55,
                "top_scoreline": "2-1",
            }
        ]
    )
    store.record_predictions(table)

    matches_df = pd.DataFrame(
        [
            {
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "date": pd.Timestamp("2020-01-01"),
                "goals_home": 2,
                "goals_away": 1,
                "ftr": "H",
                "matchday": 21,
            }
        ]
    )
    assert store.reconcile_predictions(matches_df) == 6

    record = store.get_track_record()
    assert record["n_resolved_fixtures"] == 1
    assert record["pct_correct_overall"] == pytest.approx(1.0)  # home_win was the top pick and it happened
    assert record["current_gameweek"] == 21
    assert record["pct_correct_current_gameweek"] == pytest.approx(1.0)
    assert record["n_fixtures_current_gameweek"] == 1
    assert record["gameweek_trend"] == [{"gameweek": 21, "pct_correct": pytest.approx(1.0), "n_fixtures": 1}]

    upsets = store.get_biggest_upsets()
    assert len(upsets) == 1
    assert upsets[0]["team_home"] == "Arsenal"
    assert upsets[0]["actual_outcome"] == "home_win"
    assert upsets[0]["predicted_prob"] == pytest.approx(0.6)

    gameweeks = store.get_results_by_gameweek()
    assert len(gameweeks) == 1
    group = gameweeks[0]
    assert group["gameweek"] == 21
    assert group["pct_correct"] == pytest.approx(1.0)
    assert group["n_fixtures"] == 1
    row = group["fixtures"][0]
    assert row["team_home"] == "Arsenal"
    assert row["team_away"] == "Chelsea"
    assert row["actual_goals_home"] == 2
    assert row["actual_goals_away"] == 1
    assert row["actual_outcome"] == "home_win"
    assert row["predicted_scoreline"] == "2-1"
    assert row["predicted_home_win"] == pytest.approx(0.6)
    assert row["predicted_draw"] == pytest.approx(0.25)
    assert row["predicted_away_win"] == pytest.approx(0.15)
    assert row["hit"] is True
    assert row["backfilled"] is False


def test_post_match_review_keeps_market_and_player_snapshot_provenance(clean_db):
    table = pd.DataFrame(
        [{
            "event_id": "e1", "team_home": "Arsenal", "team_away": "Chelsea",
            "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.6,
            "draw_prob": 0.25, "away_win_prob": 0.15, "over_2_5_prob": 0.7,
            "under_2_5_prob": 0.3, "btts_yes_prob": 0.55, "top_scoreline": "2-1",
        }]
    )
    store.record_predictions(table)
    store.record_fixture_market_predictions("e1", "Arsenal", "Chelsea", table.iloc[0]["commence_time"], {
        "corners": {"lambda": 10.2, "line": 9.5, "over": 0.61, "under": 0.39},
        "cards": {"lambda": 3.5, "line": 3.5, "over": 0.45, "under": 0.55},
    })
    store.record_player_prediction_snapshots("e1", [{
        "player_id": 7, "name": "Saka", "team": "Arsenal", "confirmed_starter": True,
        "anytime_goal_prob": 0.32, "anytime_assist_prob": 0.24, "anytime_goal_contribution_prob": 0.48,
    }])
    matches = pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"),
        "goals_home": 2, "goals_away": 1, "ftr": "H", "hc": 7, "ac": 5, "hy": 1, "ay": 2, "hr": 0, "ar": 0,
    }])
    store.reconcile_predictions(matches)
    assert store.reconcile_fixture_market_predictions(matches) == 2
    assert store.reconcile_player_prediction_snapshots("e1", {7: {"goals": 1, "assists": 0}}) == 1
    review = store.get_fixture_post_match("e1")
    assert review is not None
    assert review["provenance"] == "snapshot"
    assert next(row for row in review["verdicts"] if row["label"] == "Corners O/U 9.5")["hit"] is True
    assert len(review["player_calls"]) == 1
    assert review["player_calls"][0]["contribution_hit"] is True
    assert review["player_calls"][0]["is_recommended"] is True
    assert store.has_player_prediction_snapshots("e1") is True
    player_review = store.get_fixture_player_review("e1")
    assert player_review is not None
    assert player_review["correct"][0]["name"] == "Saka"
    assert player_review["missed"] == []
    # An after-the-fact reconstruction cannot replace a stored live row.
    assert store.record_fixture_market_predictions("e1", "Arsenal", "Chelsea", table.iloc[0]["commence_time"], {"corners": {"lambda": 1, "line": 1.5, "over": 0.1, "under": 0.9}}, provenance="reconstructed") == 0
    accuracy = store.get_scorer_accuracy()
    assert accuracy["snapshot"]["calls"] == 1
    assert accuracy["snapshot"]["call_hit_rate"] == pytest.approx(1.0)


def test_has_player_prediction_snapshots_is_false_before_capture(clean_db):
    assert store.has_player_prediction_snapshots("missing") is False


def test_goal_probability_qualifies_a_player_call_even_when_ga_is_lower(clean_db):
    store.record_player_prediction_snapshots("e1", [{
        "player_id": 7, "name": "Saka", "team": "Arsenal", "confirmed_starter": True,
        "anytime_goal_prob": 0.46, "anytime_assist_prob": 0.08, "anytime_goal_contribution_prob": 0.19,
    }])
    with store._connect() as conn:
        qualifies_call, contribution_probability = conn.execute(
            "SELECT qualifies_call, contribution_probability FROM player_prediction_snapshots WHERE event_id = 'e1' AND player_id = 7"
        ).fetchone()
    assert qualifies_call == 1
    assert contribution_probability == pytest.approx(0.19)


def test_player_review_uses_relevant_signal_and_separates_long_shots(clean_db):
    store.record_player_prediction_snapshots("e1", [
        {
            "player_id": 7, "name": "Saka", "team": "Arsenal", "confirmed_starter": True,
            "anytime_goal_prob": 0.46, "anytime_assist_prob": 0.18, "anytime_goal_contribution_prob": 0.46,
        },
        {
            "player_id": 8, "name": "Odegaard", "team": "Arsenal", "confirmed_starter": True,
            "anytime_goal_prob": 0.12, "anytime_assist_prob": 0.24, "anytime_goal_contribution_prob": 0.24,
        },
        {
            "player_id": 9, "name": "Castagne", "team": "Fulham", "confirmed_starter": True,
            "anytime_goal_prob": 0.04, "anytime_assist_prob": 0.11, "anytime_goal_contribution_prob": 0.11,
        },
        {
            "player_id": 10, "name": "Toney", "team": "Brentford", "confirmed_starter": True,
            "anytime_goal_prob": 0.12, "anytime_assist_prob": 0.08, "anytime_goal_contribution_prob": 0.23,
        },
    ])
    store.reconcile_player_prediction_snapshots("e1", {
        7: {"goals": 1, "assists": 0},
        8: {"goals": 0, "assists": 0},
        9: {"goals": 0, "assists": 2},
        10: {"goals": 1, "assists": 0},
    })

    review = store.get_fixture_player_review("e1")

    assert review is not None
    assert [(player["name"], player["review_label"]) for player in review["correct"]] == [
        ("Saka", "Goal call"),
        ("Toney", "Recommended player"),
    ]
    assert [(player["name"], player["review_label"]) for player in review["missed"]] == [("Odegaard", "Recommended player")]
    assert [(player["name"], player["review_label"]) for player in review["overperformed"]] == [("Castagne", "Overperformer")]


def test_resolved_fixtures_missing_player_snapshots_filters_by_gameweek(clean_db):
    table = pd.DataFrame([{
        "event_id": "e1", "team_home": "Arsenal", "team_away": "Chelsea",
        "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.6,
        "draw_prob": 0.25, "away_win_prob": 0.15, "over_2_5_prob": 0.7,
        "under_2_5_prob": 0.3, "btts_yes_prob": 0.55, "top_scoreline": "2-1",
    }])
    store.record_predictions(table)
    matches = pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"),
        "goals_home": 2, "goals_away": 1, "ftr": "H", "matchday": 1,
    }])
    store.reconcile_predictions(matches)

    assert [fixture["event_id"] for fixture in store.resolved_fixtures_missing_player_snapshots(1)] == ["e1"]
    assert store.resolved_fixtures_missing_player_snapshots(2) == []


def test_reconcile_resolves_against_tz_aware_matches_df(clean_db):
    # football-data.org's `commence_time` (renamed to `date` before being
    # passed here — see api/routes.py::_run_tracking_bookkeeping) is
    # tz-aware, unlike this test suite's other synthetic naive timestamps.
    # A live-captured (non-backfilled) prediction whose match only shows up
    # in a tz-aware matches_df previously failed to reconcile at all —
    # comparing a naive commence_time against a tz-aware `date` column
    # raised inside reconcile_predictions, silently swallowed by its only
    # caller, so the fixture just vanished from the app instead of erroring
    # loudly. Regression test for that bug.
    table = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "team_home": "Newcastle",
                "team_away": "Liverpool",
                "commence_time": pd.Timestamp("2026-08-23T15:30:00Z"),
                "home_win_prob": 0.3,
                "draw_prob": 0.3,
                "away_win_prob": 0.4,
                "over_2_5_prob": 0.6,
                "under_2_5_prob": 0.4,
                "btts_yes_prob": 0.6,
                "top_scoreline": "2-2",
            }
        ]
    )
    store.record_predictions(table)

    matches_df = pd.DataFrame(
        [
            {
                "team_home": "Newcastle",
                "team_away": "Liverpool",
                "date": pd.Timestamp("2026-08-23T15:30:00", tz="UTC"),
                "goals_home": 2,
                "goals_away": 2,
                "ftr": "D",
                "matchday": 1,
            }
        ]
    )
    assert store.reconcile_predictions(matches_df) == 6

    record = store.get_track_record()
    assert record["n_resolved_fixtures"] == 1


def test_backfill_missing_predictions(clean_db):
    # A tiny synthetic league where "Strong" always beats "Weak" decisively
    # — same fixture-fitting pattern as test_projected_table.py.
    fit_rows = []
    for _ in range(10):
        fit_rows.append({"team_home": "Strong", "team_away": "Weak", "goals_home": 3, "goals_away": 0})
        fit_rows.append({"team_home": "Weak", "team_away": "Strong", "goals_home": 0, "goals_away": 3})
    fit_df = pd.DataFrame(fit_rows)
    model = pb.models.DixonColesGoalModel(
        fit_df["goals_home"].to_numpy().astype(float),
        fit_df["goals_away"].to_numpy().astype(float),
        fit_df["team_home"].to_numpy().astype(str),
        fit_df["team_away"].to_numpy().astype(str),
    )
    model.fit()

    # An already-finished match no prediction was ever snapshotted for —
    # exactly the situation backfill_missing_predictions exists to catch up.
    # No `matchday` column here (mirrors the football-data.co.uk fallback
    # path, which has no gameweek data) — gameweek should stay null.
    df = pd.DataFrame(
        [
            {
                "season": "2025-2026",
                "team_home": "Strong",
                "team_away": "Weak",
                "date": pd.Timestamp("2025-08-20"),
                "goals_home": 3,
                "goals_away": 0,
                "ftr": "H",
            }
        ]
    )

    assert store.has_unlogged_finished_matches(df) is True
    n = store.backfill_missing_predictions(df, model)
    assert n == 6  # 6 rows per fixture (MARKET_SPEC), same as record_predictions elsewhere
    assert store.has_unlogged_finished_matches(df) is False  # idempotent — nothing left to backfill

    gameweeks = store.get_results_by_gameweek()
    assert len(gameweeks) == 1
    group = gameweeks[0]
    assert group["gameweek"] is None  # no matchday column in the source data
    row = group["fixtures"][0]
    assert row["team_home"] == "Strong"
    assert row["actual_goals_home"] == 3
    assert row["actual_goals_away"] == 0
    assert row["actual_outcome"] == "home_win"
    assert row["backfilled"] is True
    assert row["hit"] is True
    # Strong should look like a heavy favourite given the lopsided fit data.
    assert row["predicted_home_win"] > 0.7
    # top_scorelines[0] should be a lopsided home win given the fit data.
    assert row["predicted_scoreline"] is not None
    home_goals, away_goals = row["predicted_scoreline"].split("-")
    assert int(home_goals) > int(away_goals)

    record = store.get_track_record()
    assert record["current_gameweek"] is None
    assert record["pct_correct_overall"] == pytest.approx(1.0)
