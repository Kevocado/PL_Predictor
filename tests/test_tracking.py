"""Round-trip check for the live prediction track record store."""

import sqlite3

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

    # A pick recorded after its own kickoff counts toward the track record like
    # any other recorded pick (2026-10-01 reversal). It was excluded from the
    # headline until then, which is why PL shipped a null headline over graded
    # picks — and why a record kept emptying out on every model change.
    record = store.get_track_record()
    assert record["current_gameweek"] is None
    assert record["pct_correct_overall"] == pytest.approx(1.0)
    assert record["n_resolved_fixtures"] == 1
    assert record["n_rebuilt_fixtures"] == 1, "retained: this one was made after its kickoff"
    assert record["all_picks"]["n_resolved"] == 1
    # The pre-kickoff subset is where the absence is reported. The backfill
    # mechanics this test exists for are untouched either way.
    assert record["pre_kickoff"]["n_resolved_fixtures"] == 0
    assert record["pre_kickoff"]["pct_correct_overall"] is None


def test_backfill_missing_gameweeks_repairs_null_gameweek_rows(clean_db):
    """Reproduces the exact race reconcile_predictions's docstring warns
    about: football-data.co.uk (no matchday column) resolves a fixture
    before football-data.org gets the chance to, permanently leaving
    gameweek NULL since reconcile_predictions only ever revisits unresolved
    rows. backfill_missing_gameweeks is the self-heal for that once the
    matchday-carrying source becomes available."""
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

    # football-data.co.uk-style source: no matchday column at all.
    co_uk_matches = pd.DataFrame(
        [
            {
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "date": pd.Timestamp("2020-01-01"),
                "goals_home": 2,
                "goals_away": 1,
                "ftr": "H",
            }
        ]
    )
    store.reconcile_predictions(co_uk_matches)
    assert store.get_results_by_gameweek()[0]["gameweek"] is None

    # football-data.org becomes available later (e.g. an API key added
    # after the fact) -- reconcile_predictions itself won't touch this row
    # again since it's already resolved, but the backfill should.
    fd_org_matches = pd.DataFrame(
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
    # 6 rows per fixture (one per market — same shape reconcile_predictions
    # itself resolves in one call, see test_reconcile_resolves_against_actual_result).
    assert store.backfill_missing_gameweeks(fd_org_matches) == 6
    assert store.get_results_by_gameweek()[0]["gameweek"] == 21

    # Idempotent: nothing left to backfill on a second call.
    assert store.backfill_missing_gameweeks(fd_org_matches) == 0


def test_backfill_missing_gameweeks_noop_without_matchday_column(clean_db):
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
    co_uk_matches = pd.DataFrame(
        [
            {
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "date": pd.Timestamp("2020-01-01"),
                "goals_home": 2,
                "goals_away": 1,
                "ftr": "H",
            }
        ]
    )
    store.reconcile_predictions(co_uk_matches)

    assert store.backfill_missing_gameweeks(co_uk_matches) == 0


def test_post_match_uses_plain_marginal_argmax_for_match_result(clean_db):
    """A prior version credited a "hit" whenever the top scoreline was a
    draw, even if marginal argmax favored a side (see outcomes.py's
    module docstring). A walk-forward backtest across 5 seasons (1,900
    fixtures) showed that rule was net-negative for real accuracy at every
    threshold with meaningful volume, so this now uses plain marginal
    argmax -- matching `_fixture_hit_table()` exactly, so a fixture's
    verdict here always agrees with the season-wide accuracy stats."""
    table = pd.DataFrame(
        [{
            "event_id": "e1", "team_home": "Arsenal", "team_away": "Chelsea",
            "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.40,
            "draw_prob": 0.30, "away_win_prob": 0.30, "over_2_5_prob": 0.4,
            "under_2_5_prob": 0.6, "btts_yes_prob": 0.5, "top_scoreline": "1-1",
        }]
    )
    store.record_predictions(table)
    matches = pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"),
        "goals_home": 2, "goals_away": 2, "ftr": "D",
    }])
    store.reconcile_predictions(matches)

    review = store.get_fixture_post_match("e1")
    match_result = next(row for row in review["verdicts"] if row["label"] == "Match result")
    # Marginal argmax picked home_win (highest prob at 0.40) even though the
    # top scoreline was "1-1" -- no draw credit, genuinely a miss.
    assert match_result["hit"] is False
    assert match_result["prediction"] == "home_win"


def test_post_match_does_not_credit_a_draw_call_when_actual_result_is_not_a_draw(clean_db):
    """The top-scoreline draw credit is specifically for "predicted a draw,
    got a draw" -- it must not turn a genuinely wrong match-result call
    into a hit just because the top scoreline happened to be 1-1."""
    table = pd.DataFrame(
        [{
            "event_id": "e1", "team_home": "Arsenal", "team_away": "Chelsea",
            "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.40,
            "draw_prob": 0.30, "away_win_prob": 0.30, "over_2_5_prob": 0.4,
            "under_2_5_prob": 0.6, "btts_yes_prob": 0.5, "top_scoreline": "1-1",
        }]
    )
    store.record_predictions(table)
    matches = pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"),
        "goals_home": 1, "goals_away": 3, "ftr": "A",
    }])
    store.reconcile_predictions(matches)

    review = store.get_fixture_post_match("e1")
    match_result = next(row for row in review["verdicts"] if row["label"] == "Match result")
    # Marginal argmax picked home_win (wrong -- actual was away_win), and
    # the top scoreline (1-1, implying draw) doesn't match the actual
    # result either -- genuinely a miss, not eligible for the draw credit.
    assert match_result["hit"] is False
    assert match_result["prediction"] == "home_win"


def test_single_fixture_review_and_season_wide_track_record_agree_on_the_same_fixture(clean_db):
    """`get_fixture_post_match` (single-fixture review) and
    `get_track_record`/`_fixture_hit_table` (season-wide) must never
    disagree about whether the same match was called correctly -- that
    mismatch (scoreline model vs percentage model verdicts diverging) was
    the original bug report this behavior was built to fix."""
    table = pd.DataFrame(
        [{
            "event_id": "e1", "team_home": "Arsenal", "team_away": "Chelsea",
            "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.40,
            "draw_prob": 0.30, "away_win_prob": 0.30, "over_2_5_prob": 0.4,
            "under_2_5_prob": 0.6, "btts_yes_prob": 0.5, "top_scoreline": "1-1", "gameweek": 1,
        }]
    )
    store.record_predictions(table)
    matches = pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"),
        "goals_home": 2, "goals_away": 2, "ftr": "D",
    }])
    store.reconcile_predictions(matches)

    single_fixture_hit = next(
        row for row in store.get_fixture_post_match("e1")["verdicts"] if row["label"] == "Match result"
    )["hit"]
    season_wide_hit = store.get_track_record()["pct_correct_overall"]

    assert single_fixture_hit is False
    assert season_wide_hit == 0.0


def test_by_market_track_record_agrees_with_single_fixture_review_on_every_market(clean_db):
    """`get_track_record()["by_market"]` must never disagree with
    `get_fixture_post_match`'s own per-market verdicts for the same
    fixture, for the same reason as the match-result-only test above --
    extended to all four markets the model calls (exact score, match
    result, Over/Under 2.5, BTTS), not just match result."""
    table = pd.DataFrame(
        [{
            "event_id": "e1", "team_home": "Arsenal", "team_away": "Chelsea",
            "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.40,
            "draw_prob": 0.30, "away_win_prob": 0.30, "over_2_5_prob": 0.4,
            "under_2_5_prob": 0.6, "btts_yes_prob": 0.5, "top_scoreline": "1-1", "gameweek": 1,
        }]
    )
    store.record_predictions(table)
    matches = pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"),
        # 2-2: a draw (misses the home_win call), 4 total goals (misses the
        # "under" call), both teams scored (hits the "yes" BTTS call), and
        # obviously isn't the predicted "1-1" scoreline.
        "goals_home": 2, "goals_away": 2, "ftr": "D",
    }])
    store.reconcile_predictions(matches)

    verdicts = {row["label"]: row["hit"] for row in store.get_fixture_post_match("e1")["verdicts"]}
    by_market = store.get_track_record()["by_market"]

    assert verdicts["Exact score"] is False and by_market["exact_score"]["pct_correct"] == 0.0
    assert verdicts["Match result"] is False and by_market["match_result"]["pct_correct"] == 0.0
    assert verdicts["Goals O/U 2.5"] is False and by_market["over_under_2_5"]["pct_correct"] == 0.0
    assert verdicts["BTTS"] is True and by_market["btts"]["pct_correct"] == 1.0


def _one_fixture(event_id, home, away, gameweek=1):
    """A fixture in 2020, so `record_predictions` stamps `snapshotted_at = now`
    — which is AFTER its kickoff. Under the 2026-10-01 rule that makes every
    pick written through this helper a pick made after kickoff, whatever the
    `backfilled=` argument says.

    Which is the point: `made_before_kickoff` is derived from the timestamps, not
    read from the flag. `_one_fixture_upcoming` is the same fixture with a
    kickoff in the future, for the tests that need a genuinely in-time pick.
    """
    return pd.DataFrame([{
        "event_id": event_id, "team_home": home, "team_away": away,
        "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.6,
        "draw_prob": 0.25, "away_win_prob": 0.15, "over_2_5_prob": 0.4,
        "under_2_5_prob": 0.6, "btts_yes_prob": 0.5, "top_scoreline": "1-0", "gameweek": gameweek,
    }])


def _recorded_in_time(event_id, home, away, gameweek=1):
    """A fixture recorded as a pick made BEFORE its own kickoff.

    `record_predictions` stamps `snapshotted_at = now`, and every fixture here
    kicked off in 2020 — so a pick written through it is always a pick made after
    kickoff, whatever the `backfilled` argument says. That is the derivation
    working, not a limitation of the test, and the flag is never read.

    An in-time pick is therefore staged the way one actually gets in: write the
    row, then move its `snapshotted_at` back before its own kickoff — which is
    what the scheduled CI snapshot job produces for a fixture it captured early.
    Done with SQL rather than a helper so it is obvious this is staging the
    stored timestamp, not asking the code under test to change it.
    """
    store.record_predictions(_one_fixture(event_id, home, away, gameweek=gameweek))
    with sqlite3.connect(str(store.TRACKING_DB_PATH)) as conn:
        conn.execute(
            "UPDATE predictions SET snapshotted_at = ? WHERE event_id = ?",
            ("2019-12-31T09:00:00+00:00", event_id),
        )
    return event_id


def test_track_record_counts_a_pick_made_after_kickoff_in_every_rate(clean_db):
    """The headline counts BOTH picks, and the pre-kickoff subset is beside it.

    Reversed on 2026-10-01. This asserted the opposite — the headline was the
    in-time pick's 0/1 alone and the later pick was excluded from it — which is
    the rule that emptied the record: the models are re-run constantly, so a
    re-run on an already-played game stopped counting, and the headline went null
    on every model change.

    Both picks are graded here (a miss and a hit), so the headline can only be
    the mixed 0.5. Which of the two was made in time is decided by TIMESTAMPS —
    one row's `snapshotted_at` is moved back before its kickoff — and the
    `backfilled` flag is left at its default on both, so nothing here can pass by
    reading it.
    """
    _recorded_in_time("in-time", "Arsenal", "Chelsea")
    store.record_predictions(_one_fixture("late", "Spurs", "Villa"))
    store.reconcile_predictions(pd.DataFrame([
        {"team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"), "goals_home": 0, "goals_away": 1, "ftr": "A"},
        {"team_home": "Spurs", "team_away": "Villa", "date": pd.Timestamp("2020-01-01"), "goals_home": 2, "goals_away": 0, "ftr": "H"},
    ]))

    record = store.get_track_record()

    assert record["n_resolved_fixtures"] == 2, "the headline counts every recorded pick"
    assert record["n_rebuilt_fixtures"] == 1, "retained: the one made at/after its own kickoff"
    assert record["pct_correct_overall"] == 0.5  # 1 hit of 2
    # And the honest live read is beside it, with its own n: the in-time pick's
    # 0/1. This is the pair the reversal produces, and the gap between them is
    # exactly the size of the late-pick contribution.
    assert record["pre_kickoff"]["n_resolved_fixtures"] == 1
    assert record["pre_kickoff"]["pct_correct_overall"] == 0.0
    assert record["all_picks"]["n_resolved"] == 2, "all_picks keeps its published meaning"
    assert record["all_picks"]["pct_correct"] == 0.5


def test_track_record_with_only_late_picks_reports_a_headline_rate(clean_db):
    """Was: "..._reports_no_headline_rate", asserting `None` over a rebuilt pick.

    That `None` is the shipped bug: a graded, correct pick contributing nothing
    to the record because it was captured after the fact. The headline is now
    1.0 over that pick, and `pre_kickoff` is where the absence is reported — as
    a null rate over zero picks, which is an absence rather than a score of
    nought.
    """
    store.record_predictions(_one_fixture("rebuilt", "Spurs", "Villa"), backfilled=True)
    store.reconcile_predictions(pd.DataFrame([
        {"team_home": "Spurs", "team_away": "Villa", "date": pd.Timestamp("2020-01-01"), "goals_home": 2, "goals_away": 0, "ftr": "H"},
    ]))

    record = store.get_track_record()
    groups = store.get_results_by_gameweek()

    assert record["n_resolved_fixtures"] == 1
    assert record["pct_correct_overall"] == pytest.approx(1.0)
    assert record["n_rebuilt_fixtures"] == 1
    assert record["pre_kickoff"]["n_resolved_fixtures"] == 0
    assert record["pre_kickoff"]["pct_correct_overall"] is None, (
        "a rate over zero picks made in time is an absence, not 0% and not the headline's 100%"
    )
    assert record["all_picks"]["n_resolved"] == 1
    assert record["all_picks"]["pct_correct"] == pytest.approx(1.0)
    assert groups[0]["pct_correct"] == pytest.approx(1.0)
    assert groups[0]["n_fixtures"] == 1
    assert groups[0]["n_rebuilt"] == 1
    assert len(groups[0]["fixtures"]) == 1


def test_current_gameweek_tracks_the_latest_gameweek_and_its_rate_follows_it(clean_db):
    """The anchor AND the number beside it, both over every counted pick.

    `current_gameweek` anchors navigation (routes._resolve_current_gameweek,
    public_snapshot's default view), so a pick made after kickoff must move it.
    Under B8 it did not move the rate with it — that split is what this reversal
    undoes: scoping the gameweek tile to the pre-kickoff subset while the
    headline counted everything would have shown two different populations under
    one page's heading.
    """
    _recorded_in_time("in-time", "Arsenal", "Chelsea", gameweek=4)
    store.record_predictions(_one_fixture("late", "Spurs", "Villa", gameweek=5))
    store.reconcile_predictions(pd.DataFrame([
        {"team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"), "goals_home": 1, "goals_away": 0, "ftr": "H", "matchday": 4},
        {"team_home": "Spurs", "team_away": "Villa", "date": pd.Timestamp("2020-01-01"), "goals_home": 2, "goals_away": 0, "ftr": "H", "matchday": 5},
    ]))

    record = store.get_track_record()

    assert record["current_gameweek"] == 5, "the anchor follows every counted fixture"
    # Gameweek 5 holds only a pick made after its kickoff, and it is a hit, so
    # the tile reports it like any other.
    assert record["pct_correct_current_gameweek"] == 1.0
    assert record["n_fixtures_current_gameweek"] == 1
    assert record["pct_correct_overall"] == 1.0, "both gameweeks' picks are hits"
    assert record["n_resolved_fixtures"] == 2
    assert record["pre_kickoff"]["n_resolved_fixtures"] == 1, "only gameweek 4's pick was in time"


def test_current_gameweek_is_set_even_when_every_pick_is_rebuilt(clean_db):
    store.record_predictions(_one_fixture("rebuilt", "Spurs", "Villa", gameweek=5), backfilled=True)
    store.reconcile_predictions(pd.DataFrame([
        {"team_home": "Spurs", "team_away": "Villa", "date": pd.Timestamp("2020-01-01"), "goals_home": 2, "goals_away": 0, "ftr": "H", "matchday": 5},
    ]))

    assert store.get_track_record()["current_gameweek"] == 5


def test_biggest_upsets_include_rebuilt_picks(clean_db):
    store.record_predictions(_one_fixture("rebuilt", "Spurs", "Villa"), backfilled=True)
    store.reconcile_predictions(pd.DataFrame([
        {"team_home": "Spurs", "team_away": "Villa", "date": pd.Timestamp("2020-01-01"), "goals_home": 0, "goals_away": 3, "ftr": "A"},
    ]))

    upsets = store.get_biggest_upsets()
    assert len(upsets) == 1
    assert upsets[0]["team_home"] == "Spurs"
    assert upsets[0]["actual_outcome"] == "away_win"


def test_overall_pct_can_never_disagree_with_the_match_result_market(clean_db, tmp_path):
    """`pct_correct_overall` and `by_market["match_result"]` are the same
    quantity computed twice: the first is `fixtures["hit"].mean()`, the second
    `_market_reliability(fixtures, "hit")`. They are rendered side by side as
    "Correct overall" and the "Match result" market card, so a reader compares
    them directly -- and a past report conflated the overall figure with the
    unrelated `confirmed_win_rate` and read the pair as a rounding bug.

    Nothing pinned them to each other, so any future edit that changes one
    aggregate's filter (e.g. a NaN guard added to only one of them) would ship
    two contradicting percentages with no test failing. This is the invariant
    that makes such an edit impossible to merge silently.

    Asserted over a mixed fixture set — two picks captured in time (one hit, one
    miss) and two made after kickoff — because the mixed set is where a filter
    disagreement shows up. Since the 2026-10-01 reversal the headline and the
    match-result card are both drawn from EVERY counted pick (4 fixtures, 2 hits
    = 0.5), and `pre_kickoff` carries its own pair over the in-time subset. The
    invariant has to hold within each of those blocks; what it no longer has to
    do is hold ACROSS them, because they are different populations on purpose and
    the page labels them differently.
    """
    _recorded_in_time("in-time-hit", "Arsenal", "Chelsea")
    _recorded_in_time("in-time-miss", "Spurs", "Villa")
    store.record_predictions(_one_fixture("late-hit", "Everton", "Ipswich"))
    store.record_predictions(_one_fixture("late-miss", "Brighton", "Arsenal"))
    store.reconcile_predictions(pd.DataFrame([
        {"team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"), "goals_home": 1, "goals_away": 0, "ftr": "H"},
        {"team_home": "Spurs", "team_away": "Villa", "date": pd.Timestamp("2020-01-01"), "goals_home": 0, "goals_away": 2, "ftr": "A"},
        {"team_home": "Everton", "team_away": "Ipswich", "date": pd.Timestamp("2020-01-01"), "goals_home": 2, "goals_away": 0, "ftr": "H"},
        {"team_home": "Brighton", "team_away": "Arsenal", "date": pd.Timestamp("2020-01-01"), "goals_home": 0, "goals_away": 1, "ftr": "A"},
    ]))

    record = store.get_track_record()
    match_result = record["by_market"]["match_result"]

    # The mixed set exercises the rule: picks made after kickoff are present and
    # resolved, so the headline and the card must agree over ALL FOUR. 2 hits of
    # 4 = 0.5. The in-time subset gets its own pair over its own 2.
    assert record["n_rebuilt_fixtures"] == 2
    assert record["n_resolved_fixtures"] == 4, "the headline counts every recorded pick"
    assert record["pct_correct_overall"] == pytest.approx(0.5)  # 2 hits of 4
    assert match_result["n_resolved"] == record["n_resolved_fixtures"]
    assert match_result["pct_correct"] == pytest.approx(record["pct_correct_overall"])
    # `all_picks` keeps its published meaning (every pick), so it is the same
    # population as the headline rather than a wider one.
    assert record["all_picks"]["n_resolved"] == 4
    assert record["all_picks"]["pct_correct"] == pytest.approx(0.5)
    # And the pre-kickoff block is internally consistent too: 1 hit of 2.
    pre = record["pre_kickoff"]
    assert pre["n_resolved_fixtures"] == 2
    assert pre["pct_correct_overall"] == pytest.approx(0.5)
    assert pre["by_market"]["match_result"]["n_resolved"] == 2
    assert pre["by_market"]["match_result"]["pct_correct"] == pytest.approx(0.5)

    # And when only late picks exist, the pair must still agree — both reporting
    # the real rate now, with `pre_kickoff` the pair that is empty. The failure
    # guarded against is unchanged ("one reports 0.0 and the other reports
    # None"), it just moved to the other block. A separate store, because the
    # four picks above are still in this one.
    rebuilt_only = tmp_path / "rebuilt_only.db"
    original_db = store.TRACKING_DB_PATH
    store.TRACKING_DB_PATH = rebuilt_only
    try:
        store.record_predictions(_one_fixture("rebuilt-only", "Leeds", "Palace"), backfilled=True)
        store.reconcile_predictions(pd.DataFrame([
            {"team_home": "Leeds", "team_away": "Palace", "date": pd.Timestamp("2020-01-01"), "goals_home": 1, "goals_away": 1, "ftr": "D"},
        ]))

        rebuilt_record = store.get_track_record()
    finally:
        store.TRACKING_DB_PATH = original_db

    # The Leeds draw was a pick made after its own kickoff, and a miss: 0.0 over
    # the one counted pick, reported by the headline and by every block that
    # summarises it. Nothing here is an absence any more.
    assert rebuilt_record["pct_correct_overall"] == pytest.approx(0.0)
    assert rebuilt_record["by_market"]["match_result"]["pct_correct"] == pytest.approx(0.0)
    assert rebuilt_record["by_market"]["match_result"]["n_resolved"] == 1
    assert rebuilt_record["n_resolved_fixtures"] == 1
    # `pre_kickoff` is where the absence is reported: a null rate over zero picks,
    # which is an absence rather than a score of nought.
    assert rebuilt_record["pre_kickoff"]["n_resolved_fixtures"] == 0
    assert rebuilt_record["pre_kickoff"]["pct_correct_overall"] is None
    assert rebuilt_record["all_picks"]["n_resolved"] == 1
    assert rebuilt_record["all_picks"]["pct_correct"] == pytest.approx(0.0)


# --- The two detail views' counting key -------------------------------------
#
# #44 aligned PL's counting keys and left `get_fixture_player_review` /
# `get_fixture_post_match` reading every player row without the dedupe,
# flagged as follow-up on the theory that a detail view rendering a player's
# scored picks is part of that player's record. The theory does not hold: a
# duplicate `(event_id, player_id)` row is unreachable, because the table is
# keyed on it and written only by `INSERT OR IGNORE`. So there is deliberately
# no dedupe in those readers (see their docstrings), and what is pinned here
# is the invariant that makes the dedupe unnecessary -- plus the proof that the
# detail view and the counting path agree per player today.


def test_detail_views_and_counting_path_share_one_key(clean_db):
    """The detail views' key IS the counting path's key, read from the schema.

    The detail views do not dedupe, so the key that bounds them is the table's
    own `PRIMARY KEY`. Read it out of the live schema rather than asserted
    about the two functions, and read the counting path's key out of what
    `_counted_player_picks` actually does rather than its source text, so that
    the two cannot drift apart unnoticed: a mismatch here is exactly the silent
    hole #44 fixed elsewhere.
    """
    with store._connect() as conn:
        (schema,) = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'player_prediction_snapshots'"
        ).fetchone()
    assert "PRIMARY KEY (event_id, player_id)" in " ".join(schema.split())

    # What the counting path keeps, shown by feeding it the duplicate it would
    # exist to resolve: one row per (event_id, player_id), first surviving.
    row = {
        "event_id": "e1", "player_id": 7, "provenance": "snapshot",
        "goal_probability": 0.5, "assist_probability": 0.2,
        "contribution_probability": 0.6, "qualifies_call": 1,
        "actual_goals": 1, "actual_assists": 0,
    }
    counted = store._counted_player_picks(
        pd.DataFrame([row, dict(row, goal_probability=0.99), row])
    )
    assert list(counted["player_id"]) == [7]
    # Same key, and it resolves to the same key the schema enforces.
    assert set(counted.columns) >= {"event_id", "player_id"}

    # And the schema refuses the duplicate outright -- which is what makes the
    # detail views' absent dedupe unreachable rather than merely untested.
    store.record_player_prediction_snapshots("e1", [{
        "player_id": 7, "name": "Saka", "team": "Arsenal", "confirmed_starter": True,
        "anytime_goal_prob": 0.5, "anytime_assist_prob": 0.2,
        "anytime_goal_contribution_prob": 0.6,
    }])
    with store._connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO player_prediction_snapshots (event_id, player_id, name,"
                " team, goal_probability, assist_probability, contribution_probability,"
                " confirmed_starter, qualifies_call, provenance)"
                " VALUES ('e1', 7, 'Saka', 'Arsenal', 0.5, 0.2, 0.6, 1, 1, 'snapshot')"
            )


def test_rerun_cannot_show_one_player_twice_in_either_detail_view(clean_db):
    """The reachable duplicate attempt: the model is re-run constantly.

    The refresh workflow re-runs every 20 minutes on matchdays against a
    restored `tracking.db`, so "the model recomputed and wrote again" is
    routine. Every write below is a real writer call, and none may leave two
    rows for one `(event_id, player_id)` -- the only way a detail view could
    render one player's run twice. The reruns carry CHANGED probabilities,
    because that is what a re-run of a changed model actually looks like.
    """
    table = pd.DataFrame([
        {"event_id": "e1", "team_home": "Arsenal", "team_away": "Chelsea",
         "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.6,
         "draw_prob": 0.25, "away_win_prob": 0.15, "over_2_5_prob": 0.7,
         "under_2_5_prob": 0.3, "btts_yes_prob": 0.55, "top_scoreline": "2-1"},
    ])
    store.record_predictions(table)
    starters = [
        {"player_id": 7, "name": "Saka", "team": "Arsenal", "confirmed_starter": True,
         "anytime_goal_prob": 0.32, "anytime_assist_prob": 0.24, "anytime_goal_contribution_prob": 0.48},
        {"player_id": 8, "name": "Odegaard", "team": "Arsenal", "confirmed_starter": True,
         "anytime_goal_prob": 0.24, "anytime_assist_prob": 0.20, "anytime_goal_contribution_prob": 0.36},
    ]

    store.record_player_prediction_snapshots("e1", starters)
    # A rerun that recomputed and disagreed with itself, twice over.
    store.record_player_prediction_snapshots("e1", [
        dict(player, anytime_goal_prob=0.05, anytime_goal_contribution_prob=0.06)
        for player in starters
    ])
    store.record_player_prediction_snapshots("e1", starters)
    # And the after-the-fact reconstruction, which routes through the same key.
    store.record_player_prediction_snapshots(
        "e1", starters, provenance="reconstructed"
    )
    store.reconcile_predictions(pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"),
        "goals_home": 2, "goals_away": 1, "ftr": "H",
    }]))
    store.reconcile_player_prediction_snapshots(
        "e1", {7: {"goals": 1, "assists": 0}, 8: {"goals": 0, "assists": 0}}
    )

    with store._connect() as conn:
        rows = conn.execute(
            "SELECT player_id, goal_probability, provenance "
            "FROM player_prediction_snapshots ORDER BY player_id"
        ).fetchall()
    # Two players, four writes each: still two rows.
    assert [row[0] for row in rows] == [7, 8]
    # The EARLIEST recorded pick survives, and stays immutable -- the same rule
    # the counting path keeps, enforced by INSERT OR IGNORE rather than by a
    # dedupe. The rerun's 0.05 must not have overwritten the original 0.32.
    assert [row[1] for row in rows] == pytest.approx([0.32, 0.24])
    assert {row[2] for row in rows} == {"snapshot"}

    # Each view renders each player exactly once -- never twice, and never zero.
    review = store.get_fixture_player_review("e1")
    shown = [p["name"] for p in review["correct"] + review["missed"] + review["overperformed"]]
    assert sorted(shown) == ["Odegaard", "Saka"]
    assert len(shown) == len(set(shown))

    post_match = store.get_fixture_post_match("e1")
    names = [call["name"] for call in post_match["player_calls"]]
    assert sorted(names) == ["Saka"]
    assert len(names) == len(set(names))


def test_detail_view_total_agrees_with_the_counting_path_for_the_same_fixture(clean_db):
    """The two paths must not report different totals for one fixture.

    They are independent readers, so nothing makes their totals agree but the
    key and the filter. `get_fixture_post_match` shows `qualifies_call AND
    (goals > 0 OR assists > 0)`, which is `_scorer_accuracy_group`'s
    `call_hits` population -- so the counted hits and the rendered calls are
    the same rows. If either filter moves, this is what notices.
    """
    table = pd.DataFrame([
        {"event_id": "e1", "team_home": "Arsenal", "team_away": "Chelsea",
         "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"), "home_win_prob": 0.6,
         "draw_prob": 0.25, "away_win_prob": 0.15, "over_2_5_prob": 0.7,
         "under_2_5_prob": 0.3, "btts_yes_prob": 0.55, "top_scoreline": "2-1"},
    ])
    store.record_predictions(table)
    store.record_player_prediction_snapshots("e1", [
        # qualifies, scored -> a counted hit
        {"player_id": 7, "name": "Saka", "team": "Arsenal", "confirmed_starter": True,
         "anytime_goal_prob": 0.55, "anytime_assist_prob": 0.30, "anytime_goal_contribution_prob": 0.70},
        # qualifies, blank -> counted, not a hit
        {"player_id": 8, "name": "Odegaard", "team": "Arsenal", "confirmed_starter": True,
         "anytime_goal_prob": 0.50, "anytime_assist_prob": 0.28, "anytime_goal_contribution_prob": 0.65},
        # does NOT qualify, scored -> shown by the review, never counted
        {"player_id": 9, "name": "Rice", "team": "Arsenal", "confirmed_starter": True,
         "anytime_goal_prob": 0.05, "anytime_assist_prob": 0.05, "anytime_goal_contribution_prob": 0.06},
    ])
    store.reconcile_predictions(pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2020-01-01"),
        "goals_home": 2, "goals_away": 1, "ftr": "H",
    }]))
    store.reconcile_player_prediction_snapshots("e1", {
        7: {"goals": 1, "assists": 0}, 8: {"goals": 0, "assists": 0}, 9: {"goals": 1, "assists": 0},
    })

    counted = store.get_scorer_accuracy()["snapshot"]
    rendered = store.get_fixture_post_match("e1")["player_calls"]

    assert counted["calls"] == 2          # Saka and Odegaard qualify
    assert counted["call_hits"] == 1      # only Saka scored
    assert len(rendered) == counted["call_hits"]
    assert [call["name"] for call in rendered] == ["Saka"]

    # The review view is a per-market review, not a count, and deliberately
    # keeps Rice -- but still once each, and never a player the store does not
    # hold exactly one row for.
    review = store.get_fixture_player_review("e1")
    shown = [p["name"] for p in review["correct"] + review["missed"] + review["overperformed"]]
    assert sorted(shown) == ["Odegaard", "Rice", "Saka"]
