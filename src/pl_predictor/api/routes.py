"""routes.py — thin JSON-serialization layer over the existing `pl_predictor`
package. No business logic lives here; every endpoint just calls the same
functions the notebooks/tests already use and reshapes the result for the
frontend.
"""

from __future__ import annotations

import time

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..data import fixtures as fixtures_mod
from ..data import fpl_api, fpl_history
from ..data import football_data
from ..data.odds_api import OddsAPIKeyMissing, fetch_epl_odds
from ..evaluate import backtest as backtest_lib
from ..evaluate import calibration as calibration_lib
from ..features import head_to_head, player_form, ratings as ratings_mod, rolling_form
from ..features.build import build_features_for_fixtures, build_training_frame
from ..models import manifest as manifest_lib
from ..models import player_goals, power_rankings as power_rankings_mod, projected_table, scoreline
from ..models.manifest import chronological_split
from ..odds import value_bets
from ..tracking import store as tracking_store
from .schemas import (
    FixtureDetail,
    FixturePlayers,
    FixtureSummary,
    H2HMeeting,
    MarketEdge,
    OverUnderPrediction,
    PlayerPrediction,
)

router = APIRouter(prefix="/api")

_CACHE_TTL_SECONDS = 1800
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, build_fn, ttl: float = _CACHE_TTL_SECONDS, force: bool = False):
    hit = _cache.get(key)
    if not force and hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    value = build_fn()
    _cache[key] = (time.time(), value)
    return value


def _clear_cache(*keys: str):
    for k in keys:
        _cache.pop(k, None)


def _get_matches_df() -> pd.DataFrame:
    def build():
        df = football_data.load_training_data()
        current = football_data.fetch_current_season_partial()
        if current is not None and not current.empty:
            df = pd.concat([df, current], ignore_index=True)
        return df.sort_values("date").reset_index(drop=True)

    return _cached("matches_df", build)


def _get_models() -> dict:
    if not manifest_lib.MANIFEST_PATH.exists():
        raise HTTPException(status_code=409, detail="No trained models yet — call POST /api/retrain first.")
    # matches_df is only actually used when the chosen scoreline model is
    # ml_scoreline (see manifest.load_models's docstring) — always passed
    # since it's cheap here (already cached separately) and keeps this
    # simple regardless of which model is currently chosen.
    return _cached("models", lambda: manifest_lib.load_models(matches_df=_get_matches_df()))


def _get_fixtures_df(force: bool = False) -> pd.DataFrame:
    return _cached("fixtures_df", fixtures_mod.get_upcoming_fixtures, force=force)


def _get_position_priors() -> dict:
    def build():
        df = fpl_history.load_player_gw_history()
        return player_form.position_rate_priors(df)

    return _cached("position_priors", build, ttl=24 * 3600)


def _get_bootstrap() -> dict:
    return _cached("bootstrap", fpl_api.fetch_bootstrap, ttl=3600)


def _get_odds_df(force: bool = False) -> pd.DataFrame:
    def build():
        try:
            return fetch_epl_odds(force_refresh=force)
        except OddsAPIKeyMissing:
            return pd.DataFrame()

    return _cached("odds_df", build, force=force)


def _none_if_nan(value):
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else value


def _row_to_summary(row: pd.Series) -> FixtureSummary:
    def edge(side: str) -> MarketEdge:
        return MarketEdge(
            prob=row[f"{side}_prob"],
            implied=_none_if_nan(row.get(f"{side}_implied")),
            edge=_none_if_nan(row.get(f"{side}_edge")),
        )

    return FixtureSummary(
        event_id=str(row["event_id"]),
        commence_time=row["commence_time"],
        team_home=row["team_home"],
        team_away=row["team_away"],
        home_win=edge("home_win"),
        draw=edge("draw"),
        away_win=edge("away_win"),
        over_2_5=edge("over_2_5"),
        under_2_5=edge("under_2_5"),
        btts_yes_prob=row["btts_yes_prob"],
        top_scoreline=row["top_scoreline"],
        is_fallback_prediction=bool(row["is_fallback_prediction"]),
        data_confidence=row.get("data_confidence"),
        value_bet_flags=row["value_bet_flags"],
        has_live_odds=row.get("home_win_implied") is not None and not pd.isna(row.get("home_win_implied")),
    )


def _value_bet_table() -> pd.DataFrame:
    fixtures_df = _get_fixtures_df()
    if fixtures_df.empty:
        return fixtures_df
    models = _get_models()
    odds_df = _get_odds_df()
    return value_bets.build_value_bet_table(fixtures_df, odds_df, models)


@router.get("/fixtures", response_model=list[FixtureSummary])
def list_fixtures():
    table = _value_bet_table()

    # Track record bookkeeping piggybacks on the fixtures list being fetched
    # regularly during normal use — never let a tracking hiccup break the
    # main response.
    try:
        tracking_store.reconcile_predictions(_get_matches_df())
        trained_at = manifest_lib.load_manifest().get("trained_at") if manifest_lib.MANIFEST_PATH.exists() else None
        tracking_store.record_predictions(table, model_trained_at=trained_at)
    except Exception as exc:  # noqa: BLE001
        print(f"[tracking] skipped: {exc}")

    return [_row_to_summary(row) for _, row in table.iterrows()]


@router.get("/fixtures/{event_id}", response_model=FixtureDetail)
def fixture_detail(event_id: str):
    table = _value_bet_table()
    matches = table[table["event_id"].astype(str) == event_id] if not table.empty else table
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No upcoming fixture with event_id={event_id}")
    row = matches.iloc[0]
    home, away = row["team_home"], row["team_away"]

    models = _get_models()
    matches_df = _get_matches_df()

    pred = scoreline.predict_fixture(models["scoreline"], home, away, max_goals=6)
    feature_row = build_features_for_fixtures(
        pd.DataFrame([{"team_home": home, "team_away": away}]), matches_df=matches_df
    ).iloc[0]
    market_preds = value_bets.predict_market_models_for_fixture(models, feature_row)

    summary = _row_to_summary(row)
    return FixtureDetail(
        **summary.model_dump(),
        score_grid=[list(r) for r in pred["grid"][:6, :6]],
        top_scorelines=pred["top_scorelines"],
        corners=OverUnderPrediction(
            lambda_=market_preds["corners"]["lambda"],
            line=market_preds["corners"]["line"],
            over=market_preds["corners"]["over"],
            under=market_preds["corners"]["under"],
        ),
        cards=OverUnderPrediction(
            lambda_=market_preds["cards"]["lambda"],
            line=market_preds["cards"]["line"],
            over=market_preds["cards"]["over"],
            under=market_preds["cards"]["under"],
        ),
        head_to_head=[H2HMeeting(**m) for m in head_to_head.recent_meetings(matches_df, home, away)],
        home_recent_form=rolling_form.recent_form(matches_df, home),
        away_recent_form=rolling_form.recent_form(matches_df, away),
    )


@router.get("/fixtures/{event_id}/players", response_model=FixturePlayers)
def fixture_players(event_id: str):
    table = _value_bet_table()
    matches = table[table["event_id"].astype(str) == event_id] if not table.empty else table
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No upcoming fixture with event_id={event_id}")
    row = matches.iloc[0]
    home, away = row["team_home"], row["team_away"]

    models = _get_models()
    pred = scoreline.predict_fixture(models["scoreline"], home, away)

    bootstrap = _get_bootstrap()
    current_event = fpl_api.get_current_event(bootstrap)
    position_priors = _get_position_priors()

    def rank(team: str, team_goal_expectation: float) -> list[PlayerPrediction]:
        ranked = player_goals.rank_team_players(team, team_goal_expectation, bootstrap, current_event, position_priors)
        return [PlayerPrediction(**{k: p[k] for k in PlayerPrediction.model_fields}) for p in ranked]

    return FixturePlayers(
        home_players=rank(home, pred["home_goal_expectation"]),
        away_players=rank(away, pred["away_goal_expectation"]),
    )


@router.get("/manifest")
def get_manifest():
    if not manifest_lib.MANIFEST_PATH.exists():
        raise HTTPException(status_code=409, detail="No trained models yet — call POST /api/retrain first.")
    return manifest_lib.load_manifest()


@router.get("/manifest/history")
def get_manifest_history():
    return {"history": manifest_lib.load_manifest_history()}


@router.get("/calibration")
def get_calibration():
    models = _get_models()
    df, _ = build_training_frame()
    train_df, val_df = chronological_split(df)
    return {
        "model": calibration_lib.model_calibration(models["scoreline"], val_df),
        "bookmaker": calibration_lib.bookmaker_calibration(val_df),
        "naive": calibration_lib.naive_favourite_baseline(val_df),
        "season": str(val_df["season"].iloc[0]) if not val_df.empty else None,
    }


@router.post("/backtest")
def run_backtest(edge_threshold: float = 0.05, staking: str = "kelly"):
    models = _get_models()
    df, _ = build_training_frame()
    _, val_df = chronological_split(df)
    start, end = str(val_df["date"].min().date()), str(val_df["date"].max().date())
    bt = backtest_lib.build_value_bet_backtest(
        val_df, models["scoreline"], start, end, edge_threshold=edge_threshold, staking=staking
    )
    return {"results": bt.results(), "bankroll_curve": bt.account.tracker, "staking": staking}


@router.post("/retrain")
def retrain():
    manifest = manifest_lib.train_all()
    _clear_cache("models")
    return manifest


@router.post("/refresh-odds")
def refresh_odds():
    _get_odds_df(force=True)
    _clear_cache("fixtures_df")
    return {"status": "ok"}


@router.post("/refresh-fixtures")
def refresh_fixtures():
    _clear_cache("fixtures_df")
    _get_fixtures_df(force=True)
    return {"status": "ok"}


def _current_season_label() -> str:
    """The real in-progress season (e.g. '2026-2027'), not just whichever
    season happens to be latest in the loaded training window — those are
    different once the new season has started but football-data.co.uk
    hasn't published any results for it yet (or, as now, before a single
    ball's been kicked): `matches_df["season"].max()` would silently pick
    the *previous*, fully-completed season instead."""
    return football_data.season_str(football_data.CURRENT_SEASON_START_YEAR)


@router.get("/hub/rankings")
def get_power_rankings():
    models = _get_models()
    matches_df = _get_matches_df()
    current_season = _current_season_label()

    rankings = power_rankings_mod.power_rankings(models["dixon_coles_for_rankings"])

    season_matches = matches_df[matches_df["season"] == current_season]
    history = ratings_mod.team_rating_timeseries(season_matches)
    if history.empty:
        ratings_history = {}
    else:
        history["date"] = history["date"].dt.date.astype(str)
        ratings_history = {
            team: group[["date", "elo", "pi"]].to_dict("records") for team, group in history.groupby("team")
        }

    return {"rankings": rankings, "ratings_history": ratings_history, "season": current_season}


@router.get("/hub/table")
def get_projected_table():
    models = _get_models()
    matches_df = _get_matches_df()
    current_season = _current_season_label()

    standings = projected_table.compute_standings(matches_df[matches_df["season"] == current_season])
    remaining_fixtures = fixtures_mod.get_all_remaining_fixtures()
    table = projected_table.project_table(models["scoreline"], remaining_fixtures, standings)
    return {"table": table, "season": current_season}


@router.get("/hub/track-record")
def get_hub_track_record():
    return {
        "summary": tracking_store.get_track_record(),
        "biggest_misses": tracking_store.get_biggest_misses(),
    }
