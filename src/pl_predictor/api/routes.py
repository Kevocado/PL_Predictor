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
from ..data import football_data_org
from ..data.football_data_org import FootballDataOrgKeyMissing
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
# Fixtures/results/live-serving state get a much shorter TTL than everything
# else: on a live matchday, results and "which fixtures are still upcoming"
# genuinely change within minutes, and a stale 30-minute cache reads as "the
# app doesn't know a match already happened." `load_models(matches_df=...)`
# (the expensive part of this refreshing) benchmarks at ~0.4s on top of the
# ~7.5s data reload — trivial to redo this often for a personal app.
_LIVE_CACHE_TTL_SECONDS = 300
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

    return _cached("matches_df", build, ttl=_LIVE_CACHE_TTL_SECONDS)


def _get_models() -> dict:
    if not manifest_lib.MANIFEST_PATH.exists():
        raise HTTPException(status_code=409, detail="No trained models yet — call POST /api/retrain first.")
    # matches_df is only actually used when the chosen scoreline model is
    # ml_scoreline (see manifest.load_models's docstring) — always passed
    # since it's cheap here (already cached separately) and keeps this
    # simple regardless of which model is currently chosen.
    return _cached("models", lambda: manifest_lib.load_models(matches_df=_get_matches_df()), ttl=_LIVE_CACHE_TTL_SECONDS)


def _get_fixtures_df(force: bool = False) -> pd.DataFrame:
    return _cached("fixtures_df", fixtures_mod.get_upcoming_fixtures, ttl=_LIVE_CACHE_TTL_SECONDS, force=force)


def _get_remaining_fixtures_df() -> pd.DataFrame:
    """The full current-season fixture list, not-yet-played only. Prefers
    football-data.org (one call, always current — see
    data/football_data_org.py's module docstring for why this replaced the
    FPL-API-based source, which needed two blocking calls and was never
    called cached before this) and falls back to the FPL API if
    FOOTBALL_DATA_KEY isn't configured, so the app still works before
    that's set up."""

    def build():
        try:
            matches = football_data_org.fetch_matches()
        except FootballDataOrgKeyMissing:
            return fixtures_mod.get_all_remaining_fixtures()
        if matches.empty:
            return matches
        remaining = matches[~matches["finished"]].drop(columns=["status", "finished", "goals_home", "goals_away", "ftr"])
        remaining["has_odds"] = False
        return remaining.reset_index(drop=True)

    return _cached("remaining_fixtures_df", build, ttl=_LIVE_CACHE_TTL_SECONDS)


def _get_fd_org_standings() -> pd.DataFrame:
    """Live current-season standings, already computed by football-data.org
    — see data/football_data_org.py::fetch_standings. Empty DataFrame (not
    an exception) if the key isn't configured, so callers can fall back to
    `models/projected_table.py::compute_standings` the same way they
    already handle "no data yet"."""

    def build():
        try:
            return football_data_org.fetch_standings()
        except FootballDataOrgKeyMissing:
            return pd.DataFrame()

    return _cached("fd_org_standings", build, ttl=_LIVE_CACHE_TTL_SECONDS)


def _get_fd_org_matches() -> pd.DataFrame:
    """Full current-season fixture list including already-played results —
    unlike `_get_remaining_fixtures_df`, this keeps finished matches too,
    for reconciliation and the auto-retrain freshness check. Empty
    DataFrame if the key isn't configured."""

    def build():
        try:
            return football_data_org.fetch_matches()
        except FootballDataOrgKeyMissing:
            return pd.DataFrame()

    return _cached("fd_org_matches", build, ttl=_LIVE_CACHE_TTL_SECONDS)


def _get_position_priors() -> dict:
    def build():
        df = fpl_history.load_player_gw_history()
        return player_form.position_rate_priors(df)

    return _cached("position_priors", build, ttl=24 * 3600)


def _get_player_reliability_coeffs() -> dict:
    """Small linear-regression coefficients (goals ~ [rate, threat],
    assists ~ [rate, creativity]) — see `player_goals.fit_reliability_
    coefficients`'s docstring for why these two specifically. Cheap to
    refit (sub-second), so this is refreshed on the same day-scale cadence
    as `_get_position_priors` (both read the same historical FPL archive)
    rather than persisted anywhere — this data source is decoupled from the
    match model's own (currently stalled-on-football-data.co.uk) retrain
    cycle entirely, by design."""
    return _cached("player_reliability_coeffs", player_goals.fit_reliability_coefficients, ttl=24 * 3600)


def _get_bootstrap() -> dict:
    return _cached("bootstrap", fpl_api.fetch_bootstrap, ttl=3600)


def _get_odds_df(force: bool = False) -> pd.DataFrame:
    def build():
        try:
            return fetch_epl_odds(force_refresh=force)
        except OddsAPIKeyMissing:
            return pd.DataFrame()

    return _cached("odds_df", build, force=force)


def warm_caches() -> None:
    """Pre-populates every cache that would otherwise be paid for by
    whichever real request happens to be first — called once at server
    startup (see api/main.py's lifespan handler) so opening the very first
    fixture in the UI doesn't eat a multi-second bootstrap cost that every
    later fixture skips. `_get_bootstrap()` (a live FPL API call) and
    `_get_position_priors()` (loads/aggregates historical FPL gameweek
    data) are the two long-TTL ones that make "first click slow, everything
    after fast" — `_get_models()`/`_get_matches_df()` have the shorter
    5-minute live TTL and would eventually need re-warming anyway, but
    warming them too means a server that's just started is fast immediately
    rather than on whatever request happens to land first."""
    for name, fn in [
        ("matches_df", _get_matches_df),
        ("models", _get_models),
        ("fixtures_df", _get_fixtures_df),
        ("remaining_fixtures_df", _get_remaining_fixtures_df),
        ("bootstrap", _get_bootstrap),
        ("position_priors", _get_position_priors),
        ("player_reliability_coeffs", _get_player_reliability_coeffs),
        ("odds_df", _get_odds_df),
    ]:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - warming is best-effort, never fatal
            print(f"[warm_caches] {name} skipped: {exc}")


def maybe_auto_retrain() -> None:
    """Retrains only if the current season has produced more played matches
    than the last training run saw — this is what makes "a team's result
    changes its power ranking" happen without a manual click. Compares
    against `n_current_season_matches` in the existing manifest.json rather
    than tracking state separately. Called on a repeating timer from api/
    main.py's lifespan handler (see `_AUTO_RETRAIN_INTERVAL_SECONDS`), so a
    new result shows up within one interval, not instantly — full retrains
    (fitting Dixon-Coles/Bivariate-Poisson/XGBoost from scratch) take real
    time (~10-30s), so this deliberately doesn't run on every request.

    `football_data.fetch_current_season_partial()` (called below) now falls
    back to `data/pulselive.py` when football-data.co.uk has no rows yet for
    the in-progress season — confirmed this can otherwise stall for a long
    time (zero rows even after a full gameweek). This function needed no
    changes for that: it only ever looks at the row *count*, agnostic to
    which source produced it."""
    if not manifest_lib.MANIFEST_PATH.exists():
        return
    try:
        current = football_data.fetch_current_season_partial()
        n_current = 0 if current is None or current.empty else len(current)
        manifest = manifest_lib.load_manifest()
        if n_current <= manifest.get("n_current_season_matches", 0):
            return  # nothing new since the last training run
        print(f"[auto_retrain] {n_current} current-season matches now available (was {manifest.get('n_current_season_matches', 0)}) — retraining...")
        manifest_lib.train_all()
        _clear_cache("models")
        print("[auto_retrain] done")
    except Exception as exc:  # noqa: BLE001 - never take the server down over this
        print(f"[auto_retrain] skipped: {exc}")


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


def _run_tracking_bookkeeping(table: pd.DataFrame) -> None:
    """Snapshot/reconcile/backfill the prediction track record. Originally
    piggybacked only on `/api/fixtures` being polled during normal use;
    also called from `/api/fixtures/gameweek` now that the Fixtures page's
    default view no longer fetches the flat `/api/fixtures` list at all —
    without a second call site, tracking would silently stop updating.
    Never let a tracking hiccup break the caller's main response."""
    try:
        # football-data.org first, deliberately: it's the only source with
        # a `matchday` column (see reconcile_predictions's docstring), and
        # reconcile_predictions only ever touches still-unresolved rows — if
        # football-data.co.uk's fallback below resolved a fixture first
        # (which it eventually can too, once its own scrape catches up a few
        # days later), this call would find nothing left to do for it and
        # that fixture's gameweek would be permanently NULL. Reconciling
        # football-data.org first avoids that race regardless of which
        # source is faster on any given day.
        # football-data.org's fetch is cached (_LIVE_CACHE_TTL_SECONDS), so
        # reusing it below for the backfill check costs nothing extra.
        fd_org_matches = _get_fd_org_matches()
        fd_org_finished = (
            fd_org_matches[fd_org_matches["finished"]].rename(columns={"commence_time": "date"})
            if not fd_org_matches.empty
            else fd_org_matches
        )
        if not fd_org_finished.empty:
            tracking_store.reconcile_predictions(fd_org_finished)
        # Fallback/mop-up: anything football-data.org didn't have yet (e.g.
        # its own scrape running behind) still gets resolved here, just
        # without a gameweek number.
        tracking_store.reconcile_predictions(_get_matches_df())
        trained_at = manifest_lib.load_manifest().get("trained_at") if manifest_lib.MANIFEST_PATH.exists() else None
        tracking_store.record_predictions(table, model_trained_at=trained_at)

        # Catch up on any match that finished before this app was ever
        # polling for tracking to snapshot it live (e.g. the handful of
        # matches already played when tracking first started this season,
        # or any result that lands while the app is down). Prefers
        # football-data.org — it tends to reflect a finished match faster
        # than football-data.co.uk's own scrape, which is exactly why
        # backfill_missing_predictions uses the model's live-serving
        # predict path rather than a historical training-frame row (see
        # its docstring). Falls back to football-data.co.uk's own
        # current-season results once that's all that's available.
        if not fd_org_finished.empty:
            finished_matches = fd_org_finished
        else:
            current_season = _current_season_label()
            matches_df = _get_matches_df()
            finished_matches = matches_df[(matches_df["season"] == current_season) & matches_df["goals_home"].notna()]
        if tracking_store.has_unlogged_finished_matches(finished_matches):
            tracking_store.backfill_missing_predictions(
                finished_matches, _get_models()["scoreline"], model_trained_at=trained_at
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[tracking] skipped: {exc}")


@router.get("/fixtures", response_model=list[FixtureSummary])
def list_fixtures():
    table = _value_bet_table()
    _run_tracking_bookkeeping(table)
    return [_row_to_summary(row) for _, row in table.iterrows()]


@router.get("/fixtures/gameweek")
def current_gameweek_fixtures(gameweek: int | None = None):
    """Every match in one gameweek, finished and upcoming together — the
    whole gameweek at a glance, not just "what's next." Defaults to the
    current gameweek when `gameweek` is omitted; pass an explicit number to
    browse another one (prev/next navigation in the frontend). "Current
    gameweek" uses the same definition `tracking_store.get_track_record()`
    already establishes for the Track Record tab (the highest gameweek
    number among resolved matches) for consistency between the two views;
    falls back to the lowest not-yet-finished matchday from football-data.org
    when nothing's resolved yet at all (e.g. brand new season, gameweek 1
    hasn't kicked off). Finished fixtures reuse `tracking_store`'s already-
    recorded pre-match predictions (honest — never recomputed live, which
    for an already-finished match could leak that match's own result into
    what's "predicted"); upcoming fixtures get a fresh live prediction the
    same way `/fixtures` does, joined against the odds-windowed value-bet
    table by team pair when a live market covers that fixture (so the same
    market-edge/value-bet-flag data `/fixtures` shows is available here
    too, not just a bare model prediction) — falls back to a model-only
    prediction for fixtures further out than the odds window. `event_id` is
    included on every fixture (the Odds API's id when a live-odds match was
    found, football-data.org's id otherwise for upcoming ones, and the id
    `fixture_detail` already knows how to resolve for finished ones) so
    each card can open the same fixture-detail view `/fixtures` does."""
    value_bet_table = _value_bet_table()
    _run_tracking_bookkeeping(value_bet_table)

    fd_org_matches = _get_fd_org_matches()

    track_summary = tracking_store.get_track_record()
    current_gameweek = track_summary["current_gameweek"]

    if current_gameweek is None and not fd_org_matches.empty:
        unfinished = fd_org_matches[~fd_org_matches["finished"]]
        current_gameweek = int(unfinished["matchday"].min()) if not unfinished.empty else None

    target_gameweek = gameweek if gameweek is not None else current_gameweek
    if target_gameweek is None:
        return {"gameweek": None, "fixtures": [], "is_current": True, "min_gameweek": None, "max_gameweek": None}

    has_matchday = not fd_org_matches.empty and fd_org_matches["matchday"].notna().any()
    min_gameweek = int(fd_org_matches["matchday"].min()) if has_matchday else 1
    max_gameweek = int(fd_org_matches["matchday"].max()) if has_matchday else 38

    fixtures = []

    groups = tracking_store.get_results_by_gameweek()
    completed_group = next((g for g in groups if g["gameweek"] == target_gameweek), None)
    if completed_group:
        for r in completed_group["fixtures"]:
            fixtures.append(
                {
                    "event_id": str(r["event_id"]),
                    "team_home": r["team_home"],
                    "team_away": r["team_away"],
                    "commence_time": r["commence_time"],
                    "finished": True,
                    "actual_goals_home": r["actual_goals_home"],
                    "actual_goals_away": r["actual_goals_away"],
                    "predicted_home_win": r["predicted_home_win"],
                    "predicted_draw": r["predicted_draw"],
                    "predicted_away_win": r["predicted_away_win"],
                    "predicted_scoreline": r["predicted_scoreline"],
                    "hit": r["hit"],
                    "backfilled": r["backfilled"],
                    "has_live_odds": False,
                    "value_bet_flags": [],
                }
            )

    if not fd_org_matches.empty:
        upcoming_rows = fd_org_matches[(fd_org_matches["matchday"] == target_gameweek) & (~fd_org_matches["finished"])]
        if not upcoming_rows.empty:
            models = _get_models()
            preds = scoreline.predict_fixtures_batch(models["scoreline"], upcoming_rows)
            # The Odds API only covers a rolling ~1-2 gameweek window, so
            # not every upcoming fixture here will have a live-odds match —
            # prefer it (real event_id, model prob, value-bet flags) when
            # one exists for this exact team pair, same model-only
            # prediction otherwise (has_live_odds=False, same as
            # `_team_fixture_to_summary`'s fallback for team-lookup
            # fixtures further out than the odds window).
            for (_, row), pred in zip(upcoming_rows.iterrows(), preds):
                vb_match = (
                    value_bet_table[
                        (value_bet_table["team_home"] == row["team_home"])
                        & (value_bet_table["team_away"] == row["team_away"])
                    ]
                    if not value_bet_table.empty
                    else value_bet_table
                )
                if not vb_match.empty:
                    vb_row = vb_match.iloc[0]
                    fixtures.append(
                        {
                            "event_id": str(vb_row["event_id"]),
                            "team_home": row["team_home"],
                            "team_away": row["team_away"],
                            "commence_time": row["commence_time"].isoformat(),
                            "finished": False,
                            "actual_goals_home": None,
                            "actual_goals_away": None,
                            "predicted_home_win": float(vb_row["home_win_prob"]),
                            "predicted_draw": float(vb_row["draw_prob"]),
                            "predicted_away_win": float(vb_row["away_win_prob"]),
                            "predicted_scoreline": vb_row["top_scoreline"],
                            "hit": None,
                            "backfilled": False,
                            "has_live_odds": bool(vb_row["home_win_implied"] is not None and not pd.isna(vb_row["home_win_implied"])),
                            "value_bet_flags": list(vb_row["value_bet_flags"]),
                        }
                    )
                    continue

                top = pred["top_scorelines"][0]
                fixtures.append(
                    {
                        "event_id": str(row["event_id"]),
                        "team_home": row["team_home"],
                        "team_away": row["team_away"],
                        "commence_time": row["commence_time"].isoformat(),
                        "finished": False,
                        "actual_goals_home": None,
                        "actual_goals_away": None,
                        "predicted_home_win": pred["home_win"],
                        "predicted_draw": pred["draw"],
                        "predicted_away_win": pred["away_win"],
                        "predicted_scoreline": f"{top['home']}-{top['away']}",
                        "hit": None,
                        "backfilled": False,
                        "has_live_odds": False,
                        "value_bet_flags": [],
                    }
                )

    fixtures.sort(key=lambda f: f["commence_time"])
    return {
        "gameweek": target_gameweek,
        "fixtures": fixtures,
        "is_current": target_gameweek == current_gameweek,
        "min_gameweek": min_gameweek,
        "max_gameweek": max_gameweek,
    }


def _build_fixture_detail(summary: FixtureSummary, home: str, away: str) -> FixtureDetail:
    models = _get_models()
    matches_df = _get_matches_df()

    pred = scoreline.predict_fixture(models["scoreline"], home, away, max_goals=6)
    feature_row = build_features_for_fixtures(
        pd.DataFrame([{"team_home": home, "team_away": away}]), matches_df=matches_df
    ).iloc[0]
    market_preds = value_bets.predict_market_models_for_fixture(models, feature_row)

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


@router.get("/fixtures/{event_id}", response_model=FixtureDetail)
def fixture_detail(event_id: str):
    table = _value_bet_table()
    matches = table[table["event_id"].astype(str) == event_id] if not table.empty else table
    if not matches.empty:
        row = matches.iloc[0]
        return _build_fixture_detail(_row_to_summary(row), row["team_home"], row["team_away"])

    # Not in the odds-windowed value-bet table — might be a further-out
    # fixture reached via team lookup (GET /api/teams/{team}/fixtures uses
    # FPL fixture ids, a different id space than the Odds API's event_id).
    all_remaining = _get_remaining_fixtures_df()
    remaining_match = (
        all_remaining[all_remaining["event_id"].astype(str) == event_id] if not all_remaining.empty else all_remaining
    )
    if not remaining_match.empty:
        fixture = remaining_match.iloc[0]
        home, away = fixture["team_home"], fixture["team_away"]
        models = _get_models()
        pred = scoreline.predict_fixture(models["scoreline"], home, away)
        return _build_fixture_detail(_team_fixture_to_summary(fixture, pred), home, away)

    # Neither odds-windowed nor still-upcoming — likely a finished fixture
    # reached via the gameweek view. Pull its honestly pre-match-recorded
    # prediction from tracking_store instead of recomputing live (which for
    # an already-played match risks that match's own result leaking into
    # "current model state"); the score grid/h2h/corners/cards/recent-form
    # sections below still come from a live recompute either way, same as
    # every other path through this function — those are descriptive
    # current-model context, not "the prediction" itself.
    recorded = tracking_store.get_fixture_prediction(event_id)
    if recorded is not None:
        return _build_fixture_detail(
            _recorded_fixture_to_summary(recorded), recorded["team_home"], recorded["team_away"]
        )

    raise HTTPException(status_code=404, detail=f"No fixture with event_id={event_id}")


def _resolve_fixture_teams(event_id: str) -> tuple[str, str] | None:
    """Same three-tier lookup as `fixture_detail` (odds-windowed table,
    then still-upcoming fixtures, then a finished fixture's recorded
    prediction), reduced to just the team names — shared so player
    predictions work for any fixture reachable via the gameweek view, not
    only the odds-windowed ones."""
    table = _value_bet_table()
    matches = table[table["event_id"].astype(str) == event_id] if not table.empty else table
    if not matches.empty:
        row = matches.iloc[0]
        return row["team_home"], row["team_away"]

    all_remaining = _get_remaining_fixtures_df()
    remaining_match = (
        all_remaining[all_remaining["event_id"].astype(str) == event_id] if not all_remaining.empty else all_remaining
    )
    if not remaining_match.empty:
        fixture = remaining_match.iloc[0]
        return fixture["team_home"], fixture["team_away"]

    recorded = tracking_store.get_fixture_prediction(event_id)
    if recorded is not None:
        return recorded["team_home"], recorded["team_away"]

    return None


@router.get("/fixtures/{event_id}/players", response_model=FixturePlayers)
def fixture_players(event_id: str):
    resolved = _resolve_fixture_teams(event_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"No fixture with event_id={event_id}")
    home, away = resolved

    models = _get_models()
    pred = scoreline.predict_fixture(models["scoreline"], home, away)

    bootstrap = _get_bootstrap()
    current_event = fpl_api.get_current_event(bootstrap)
    position_priors = _get_position_priors()
    reliability_coeffs = _get_player_reliability_coeffs()

    def rank(team: str, team_goal_expectation: float) -> list[PlayerPrediction]:
        ranked = player_goals.rank_team_players(
            team, team_goal_expectation, bootstrap, current_event, position_priors, reliability_coeffs=reliability_coeffs
        )
        return [PlayerPrediction(**{k: p[k] for k in PlayerPrediction.model_fields}) for p in ranked]

    return FixturePlayers(
        home_players=rank(home, pred["home_goal_expectation"]),
        away_players=rank(away, pred["away_goal_expectation"]),
    )


def _team_fixture_to_summary(fixture: pd.Series, pred: dict) -> FixtureSummary:
    """Same shape as `_row_to_summary`, for a fixture pulled from
    `get_all_remaining_fixtures()` rather than the odds-windowed value-bet
    table — those use the FPL API's own fixture ids, which don't line up
    with the Odds API's `event_id`s, so there's no reliable way to join a
    live market onto them here. Always model-only (`has_live_odds=False`),
    same graceful no-odds path already proven in `odds/value_bets.py`."""

    def edge(prob_key: str) -> MarketEdge:
        return MarketEdge(prob=pred[prob_key], implied=None, edge=None)

    return FixtureSummary(
        event_id=str(fixture["event_id"]),
        commence_time=fixture["commence_time"],
        team_home=fixture["team_home"],
        team_away=fixture["team_away"],
        home_win=edge("home_win"),
        draw=edge("draw"),
        away_win=edge("away_win"),
        over_2_5=edge("over_2_5"),
        under_2_5=edge("under_2_5"),
        btts_yes_prob=pred["btts_yes"],
        top_scoreline=f"{pred['top_scorelines'][0]['home']}-{pred['top_scorelines'][0]['away']}",
        is_fallback_prediction=pred["fallback"],
        data_confidence=pred["data_confidence"],
        value_bet_flags=[],
        has_live_odds=False,
    )


def _recorded_fixture_to_summary(rec: dict) -> FixtureSummary:
    """Same shape as `_row_to_summary`/`_team_fixture_to_summary`, for an
    already-finished fixture reached via the gameweek view — uses the
    honestly pre-match-recorded probabilities from `tracking_store` instead
    of recomputing live. No live odds/edge to show (this isn't the
    odds-windowed table), same graceful no-odds shape as
    `_team_fixture_to_summary`."""

    def edge(prob_key: str) -> MarketEdge:
        return MarketEdge(prob=rec[prob_key], implied=None, edge=None)

    return FixtureSummary(
        event_id=rec["event_id"],
        commence_time=rec["commence_time"],
        team_home=rec["team_home"],
        team_away=rec["team_away"],
        home_win=edge("home_win"),
        draw=edge("draw"),
        away_win=edge("away_win"),
        over_2_5=edge("over_2_5"),
        under_2_5=edge("under_2_5"),
        btts_yes_prob=rec["btts_yes"],
        top_scoreline=rec["predicted_scoreline"] or "?",
        is_fallback_prediction=False,
        data_confidence=None,
        value_bet_flags=[],
        has_live_odds=False,
    )


@router.get("/teams")
def list_teams():
    all_remaining = _get_remaining_fixtures_df()
    if all_remaining.empty:
        return {"teams": []}
    return {"teams": sorted(set(all_remaining["team_home"]) | set(all_remaining["team_away"]))}


@router.get("/teams/{team}/fixtures", response_model=list[FixtureSummary])
def team_fixtures(team: str, limit: int = 5):
    """A team's next `limit` remaining fixtures with predictions —
    always finds one regardless of how far out it is, unlike the main
    `/api/fixtures` list which is limited to whatever window The Odds API
    currently has pre-match lines for (typically 1-2 gameweeks)."""
    all_remaining = _get_remaining_fixtures_df()
    if all_remaining.empty:
        return []
    team_df = (
        all_remaining[(all_remaining["team_home"] == team) | (all_remaining["team_away"] == team)]
        .sort_values("commence_time")
        .head(limit)
    )
    if team_df.empty:
        return []

    models = _get_models()
    return [
        _team_fixture_to_summary(
            fixture, scoreline.predict_fixture(models["scoreline"], fixture["team_home"], fixture["team_away"])
        )
        for _, fixture in team_df.iterrows()
    ]


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

    remaining_df = _get_remaining_fixtures_df()
    current_teams = (set(remaining_df["team_home"]) | set(remaining_df["team_away"])) if not remaining_df.empty else None
    # All-time appearance counts (same series FixtureFeatureContext uses for
    # live prediction confidence) — this is why a newly-promoted team's
    # confidence tier can improve the moment they've played, well before
    # the next retrain gives them a real fitted rating. The current
    # season's slice of this count comes from football-data.org's
    # standings when available (fresher than waiting on
    # football-data.co.uk's own scrape) layered on top of the historical
    # (pre-this-season) count from matches_df, so a newly-promoted team's
    # confidence can move the moment their first match shows up there.
    historical_played = (
        matches_df[matches_df["season"] != current_season]
        .melt(value_vars=["team_home", "team_away"])["value"]
        .value_counts()
    )
    fd_org_standings = _get_fd_org_standings()
    if not fd_org_standings.empty:
        current_played = fd_org_standings.set_index("team")["played"]
    else:
        current_played = (
            matches_df[matches_df["season"] == current_season]
            .melt(value_vars=["team_home", "team_away"])["value"]
            .value_counts()
        )
    games_played = {
        team: int(historical_played.get(team, 0)) + int(current_played.get(team, 0))
        for team in set(historical_played.index) | set(current_played.index)
    }
    rankings = power_rankings_mod.power_rankings(
        models["dixon_coles_for_rankings"], current_teams=current_teams, games_played=games_played
    )

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

    # Prefer football-data.org's already-computed live standings (fresher,
    # one call, no reconstruction needed — see
    # data/football_data_org.py::fetch_standings) over rebuilding them from
    # football-data.co.uk's own current-season results, which is what
    # compute_standings does; same column shape either way so
    # project_table doesn't need to know which source it got.
    standings = _get_fd_org_standings()
    if standings.empty:
        standings = projected_table.compute_standings(matches_df[matches_df["season"] == current_season])
    remaining_fixtures = _get_remaining_fixtures_df()
    table = projected_table.project_table(models["scoreline"], remaining_fixtures, standings)
    return {"table": table, "season": current_season}


@router.get("/hub/track-record")
def get_hub_track_record():
    return {
        "summary": tracking_store.get_track_record(),
        "biggest_upsets": tracking_store.get_biggest_upsets(),
        "gameweeks": tracking_store.get_results_by_gameweek(),
    }
