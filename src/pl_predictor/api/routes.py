"""routes.py — thin JSON-serialization layer over the existing `pl_predictor`
package. No business logic lives here; every endpoint just calls the same
functions the notebooks/tests already use and reshapes the result for the
frontend.
"""

from __future__ import annotations

import json
import time
from threading import Lock, Thread

import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException

from ..config import PUBLIC_MODE, PUBLIC_SNAPSHOT_PATH
from ..data import fixtures as fixtures_mod
from ..data import espn, fpl_api, fpl_history
from ..data.team_names import to_canonical
from ..data import football_data
from ..data import football_data_org
from ..data.football_data_org import FootballDataOrgKeyMissing
from ..data.odds_api import OddsAPIKeyMissing, fetch_epl_odds
from ..evaluate import backtest as backtest_lib
from ..evaluate import betting_validation
from ..evaluate import calibration as calibration_lib
from ..evaluate import odds_benchmark
from ..features import head_to_head, player_form, ratings as ratings_mod, rolling_form, squad_change
from ..features.build import build_features_for_fixtures, build_training_frame
from ..models import manifest as manifest_lib
from ..models import fpl as fpl_model, player_goals, power_rankings as power_rankings_mod, projected_table, scoreline
from ..models.manifest import chronological_split
from ..odds import value_bets
from ..tracking import store as tracking_store
from ..tracking import value_bet_ledger
from .schemas import (
    FixtureDetail,
    FixtureActualStats,
    FixturePlayers,
    FixtureSummary,
    FixtureTeamContext,
    H2HMeeting,
    MarketEdge,
    OverUnderPrediction,
    PlayerPrediction,
    SingleBetRecommendation,
    FPLManualSquadRequest,
    FPLTransferRequest,
)
from . import hub_analytics

router = APIRouter(prefix="/api")


def _admin_only() -> None:
    """Dependency for the four state-changing endpoints (retrain/refresh/
    backtest) — 404s them unconditionally on the public deployment.
    Pretending the route doesn't exist, rather than 403, avoids
    advertising an admin surface to a public visitor at all."""
    if PUBLIC_MODE:
        raise HTTPException(status_code=404)


_public_snapshot_cache: dict | None = None


def _public_snapshot() -> dict:
    """The precomputed data public_snapshot.py generates — read once per
    process (it only changes on a redeploy, which starts a fresh process
    anyway) rather than re-reading the file on every request. Empty dict if
    none has been generated yet (a public deploy before its first snapshot
    exists), so every PUBLIC_MODE branch degrades to empty results instead
    of a 500."""
    global _public_snapshot_cache
    if _public_snapshot_cache is None:
        _public_snapshot_cache = (
            json.loads(PUBLIC_SNAPSHOT_PATH.read_text()) if PUBLIC_SNAPSHOT_PATH.exists() else {}
        )
    return _public_snapshot_cache


_CACHE_TTL_SECONDS = 1800
# Fixtures/results/live-serving state get a much shorter TTL than everything
# else: on a live matchday, results and "which fixtures are still upcoming"
# genuinely change within minutes, and a stale 30-minute cache reads as "the
# app doesn't know a match already happened." `load_models(matches_df=...)`
# (the expensive part of this refreshing) benchmarks at ~0.4s on top of the
# ~7.5s data reload — trivial to redo this often for a personal app.
_LIVE_CACHE_TTL_SECONDS = 300
_cache: dict[str, tuple[float, object]] = {}
_cache_locks: dict[str, Lock] = {}
_player_outcome_backfill_lock = Lock()


def _cached(key: str, build_fn, ttl: float = _CACHE_TTL_SECONDS, force: bool = False):
    hit = _cache.get(key)
    if not force and hit and (time.time() - hit[0]) < ttl:
        return hit[1]

    lock = _cache_locks.setdefault(key, Lock())
    with lock:
        hit = _cache.get(key)
        if not force and hit and (time.time() - hit[0]) < ttl:
            return hit[1]
        value = build_fn()
        _cache[key] = (time.time(), value)
        return value


def _clear_cache(*keys: str):
    for k in keys:
        _cache.pop(k, None)


def _refresh_player_outcomes_in_background(gameweek: int) -> None:
    """Refresh missing player events without making fixture cards wait."""
    if not _player_outcome_backfill_lock.acquire(blocking=False):
        return

    def run() -> None:
        try:
            backfill_completed_player_reviews(gameweek)
        finally:
            _player_outcome_backfill_lock.release()

    Thread(target=run, name="player-outcome-backfill", daemon=True).start()


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


def _get_lineup_model():
    return _cached("lineup_model", player_goals.fit_lineup_model, ttl=24 * 3600)


def _get_position_rate_models() -> dict:
    return _cached("position_rate_models", player_goals.fit_position_rate_models, ttl=24 * 3600)


def _get_goal_contribution_model() -> dict:
    return _cached("goal_contribution_model", player_goals.fit_goal_contribution_model, ttl=24 * 3600)


def _get_ready_goal_contribution_model() -> dict | None:
    """Return the direct G+A model only after startup warming has finished.

    Training it from historical player data takes around 20 seconds. Player
    requests use the already-valid Poisson-union baseline until this cached
    model is ready, rather than making the first fixture modal wait for it.
    """
    hit = _cache.get("goal_contribution_model")
    if hit and (time.time() - hit[0]) < 24 * 3600:
        return hit[1]
    return None


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
        _clear_cache("models", "value_bet_table")
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

    recommended_market = _none_if_nan(row.get("recommended_market"))
    recommendation = (
        SingleBetRecommendation(
            market=str(recommended_market),
            probability=float(row["recommended_prob"]),
            implied_probability=float(row["recommended_implied"]),
            edge=float(row["recommended_edge"]),
            price=float(row["recommended_price"]),
            bookmaker=str(row["recommended_bookmaker"]),
        )
        if recommended_market is not None
        else None
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
        predicted_total_goals=row["home_goal_expectation"] + row["away_goal_expectation"],
        predicted_margin=row["home_goal_expectation"] - row["away_goal_expectation"],
        home_2plus_prob=row["home_2plus_prob"],
        away_2plus_prob=row["away_2plus_prob"],
        value_bet_flags=row["value_bet_flags"],
        has_live_odds=row.get("home_win_implied") is not None and not pd.isna(row.get("home_win_implied")),
        odds_fetched_at=_none_if_nan(row.get("odds_fetched_at")),
        odds_is_stale=bool(row.get("odds_is_stale", False)),
        recommended_bet=recommendation,
    )


def _value_bet_table() -> pd.DataFrame:
    """Cached: this rebuilds every remaining fixture's scoreline prediction
    (a `predict_fixtures_batch` + full feature-row build per fixture), and
    is called from several call chains per single fixture lookup (e.g.
    `_resolve_fixture_kickoff`, hit twice per `/fixtures/{id}/players`
    request) — confirmed via profiling this was the dominant cost of
    `public_snapshot.py`'s per-fixture loop, recomputing the whole table
    from scratch on every call rather than reusing it across a run whose
    fixtures/odds/models don't change mid-flight."""

    def build() -> pd.DataFrame:
        fixtures_df = _get_fixtures_df()
        if fixtures_df.empty:
            return fixtures_df
        models = _get_models()
        odds_df = _get_odds_df()
        return value_bets.build_value_bet_table(fixtures_df, odds_df, models)

    return _cached("value_bet_table", build, ttl=_LIVE_CACHE_TTL_SECONDS)


def _capture_fixture_market_predictions(table: pd.DataFrame) -> None:
    """Save corners/cards predictions before kickoff, outside UI requests."""
    if table.empty:
        return
    models = _get_models()
    matches_df = _get_matches_df()
    feature_rows = build_features_for_fixtures(
        table[["team_home", "team_away", "commence_time"]].copy(), matches_df=matches_df, context=models.get("context")
    )
    for (_, fixture), (_, feature_row) in zip(table.iterrows(), feature_rows.iterrows()):
        predictions = value_bets.predict_market_models_for_fixture(models, feature_row)
        tracking_store.record_fixture_market_predictions(
            fixture["event_id"], fixture["team_home"], fixture["team_away"], fixture["commence_time"], predictions
        )


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
            value_bet_ledger.reconcile_value_bets(fd_org_finished, result_source="football-data.org")
        # Fallback/mop-up: anything football-data.org didn't have yet (e.g.
        # its own scrape running behind) still gets resolved here, just
        # without a gameweek number.
        tracking_store.reconcile_predictions(_get_matches_df())
        value_bet_ledger.reconcile_value_bets(_get_matches_df(), result_source="football-data.co.uk")
        trained_at = manifest_lib.load_manifest().get("trained_at") if manifest_lib.MANIFEST_PATH.exists() else None
        tracking_store.record_predictions(table, model_trained_at=trained_at)
        tracking_store.record_fixture_forecast_snapshots(table, "early")
        tracking_store.capture_due_odds_snapshots(table)
        _capture_fixture_market_predictions(table)
        tracking_store.reconcile_fixture_market_predictions(_get_matches_df())
        value_bet_ledger.record_value_bets(
            table,
            model_trained_at=trained_at,
            model_manifest_hash=manifest_lib.manifest_fingerprint(),
        )

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
            models = _get_models()
            tracking_store.backfill_missing_predictions(
                finished_matches,
                models["scoreline"],
                model_trained_at=trained_at,
                market_overrides=models.get("scoreline_market_overrides"),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[tracking] skipped: {exc}")


@router.get("/fixtures", response_model=list[FixtureSummary])
def list_fixtures():
    table = _value_bet_table()
    _run_tracking_bookkeeping(table)
    return [_row_to_summary(row) for _, row in table.iterrows()]


def _resolve_current_gameweek(current_gameweek: int | None, fd_org_matches: pd.DataFrame, now: pd.Timestamp | None = None) -> int | None:
    """`current_gameweek` (from `tracking_store.get_track_record()`, or the
    lowest-unfinished-matchday fallback) still points at a gameweek once
    every one of its matches is resolved, until the *next* gameweek produces
    its own first resolved match — which can be days later (e.g. a Sunday
    finish, Friday restart). Advances to the next gameweek a day early
    instead of waiting for kickoff, so the default view doesn't sit on a
    fully-finished gameweek for days. `now` is injectable for tests; real
    callers always use the default (actual current time)."""
    if current_gameweek is None or fd_org_matches.empty:
        return current_gameweek

    this_gw_matches = fd_org_matches[fd_org_matches["matchday"] == current_gameweek]
    this_gw_fully_finished = not this_gw_matches.empty and bool(this_gw_matches["finished"].all())
    if not this_gw_fully_finished:
        return current_gameweek

    next_gw_matches = fd_org_matches[fd_org_matches["matchday"] == current_gameweek + 1]
    if next_gw_matches.empty:
        return current_gameweek

    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    next_kickoff = pd.to_datetime(next_gw_matches["commence_time"], utc=True).min()
    if next_kickoff <= now + pd.Timedelta(days=1):
        return current_gameweek + 1
    return current_gameweek


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
    hasn't kicked off) — then `_resolve_current_gameweek` advances that a
    day early once it's fully finished, rather than waiting on the next
    gameweek's own first result. Finished fixtures reuse `tracking_store`'s
    already-recorded pre-match predictions (honest — never recomputed live,
    which for an already-finished match could leak that match's own result
    into what's "predicted"); upcoming fixtures get a fresh live prediction
    the same way `/fixtures` does, joined against the odds-windowed
    value-bet table by team pair when a live market covers that fixture (so
    the same market-edge/value-bet-flag data `/fixtures` shows is available
    here too, not just a bare model prediction) — falls back to a
    model-only prediction for fixtures further out than the odds window.
    `event_id` is included on every fixture (the Odds API's id when a
    live-odds match was found, football-data.org's id otherwise for
    upcoming ones, and the id `fixture_detail` already knows how to resolve
    for finished ones) so each card can open the same fixture-detail view
    `/fixtures` does."""
    if PUBLIC_MODE:
        snapshot = _public_snapshot()
        gw = gameweek if gameweek is not None else snapshot.get("current_gameweek")
        return snapshot.get("fixtures_by_gameweek", {}).get(
            str(gw), {"gameweek": gw, "fixtures": [], "is_current": False, "min_gameweek": None, "max_gameweek": None}
        )

    value_bet_table = _value_bet_table()
    _run_tracking_bookkeeping(value_bet_table)

    fd_org_matches = _get_fd_org_matches()

    track_summary = tracking_store.get_track_record()
    current_gameweek = track_summary["current_gameweek"]

    # football-data.org is optional. When its key/feed is unavailable, FPL's
    # cached full fixture list is enough to retain the future gameweek view.
    # Without this branch, the app could still compute a value-bet table yet
    # show an empty Fixtures tab after a completed round.
    fallback_remaining = pd.DataFrame()
    if fd_org_matches.empty:
        try:
            fallback_remaining = _get_remaining_fixtures_df()
        except Exception:  # noqa: BLE001 - retain completed tracking rows
            fallback_remaining = pd.DataFrame()

    if current_gameweek is None and not fd_org_matches.empty:
        unfinished = fd_org_matches[~fd_org_matches["finished"]]
        current_gameweek = int(unfinished["matchday"].min()) if not unfinished.empty else None

    current_gameweek = _resolve_current_gameweek(current_gameweek, fd_org_matches)
    if current_gameweek is None and not fallback_remaining.empty and fallback_remaining["gameweek"].notna().any():
        # With no tracked result yet (for example, before GW1), the next
        # scheduled round is necessarily the current view. Once tracking has
        # established a current gameweek, do not jump the user into a future
        # round merely because the fallback only contains unplayed fixtures.
        current_gameweek = int(pd.to_numeric(fallback_remaining["gameweek"], errors="coerce").dropna().min())

    target_gameweek = gameweek if gameweek is not None else current_gameweek
    if target_gameweek is None:
        return {"gameweek": None, "fixtures": [], "is_current": True, "min_gameweek": None, "max_gameweek": None}

    # Keep previously completed rounds reachable even when the live fixture
    # provider only has the remaining schedule (the FPL fallback starts at
    # the next unplayed gameweek).  Those rows have the original pre-match
    # snapshot and post-match detail in the tracking store.
    groups = tracking_store.get_results_by_gameweek()
    tracked_gameweeks = [
        int(group["gameweek"])
        for group in groups
        if group.get("gameweek") is not None
    ]

    has_matchday = not fd_org_matches.empty and fd_org_matches["matchday"].notna().any()
    if has_matchday:
        min_gameweek = int(fd_org_matches["matchday"].min())
        max_gameweek = int(fd_org_matches["matchday"].max())
    elif not fallback_remaining.empty and fallback_remaining["gameweek"].notna().any():
        remaining_gw = pd.to_numeric(fallback_remaining["gameweek"], errors="coerce").dropna()
        min_gameweek, max_gameweek = int(remaining_gw.min()), int(remaining_gw.max())
        if tracked_gameweeks:
            min_gameweek = min(min_gameweek, min(tracked_gameweeks))
            max_gameweek = max(max_gameweek, max(tracked_gameweeks))
    else:
        min_gameweek, max_gameweek = (
            (min(tracked_gameweeks), max(tracked_gameweeks))
            if tracked_gameweeks
            else (1, 38)
        )

    fixtures = []
    completed_team_pairs: set[tuple[str, str]] = set()

    completed_group = next((g for g in groups if g["gameweek"] == target_gameweek), None)
    if completed_group:
        completed_team_pairs = {(r["team_home"], r["team_away"]) for r in completed_group["fixtures"]}
        pending_event_ids = {
            str(row["event_id"])
            for row in completed_group["fixtures"]
            if not tracking_store.has_fixture_player_outcomes(str(row["event_id"]))
        }
        if pending_event_ids:
            _refresh_player_outcomes_in_background(target_gameweek)
        try:
            bootstrap = _get_bootstrap()
        except Exception:  # noqa: BLE001 - scores still render during an FPL outage
            bootstrap = {"elements": []}
        for r in completed_group["fixtures"]:
            player_events_pending = str(r["event_id"]) in pending_event_ids
            player_events = tracking_store.get_fixture_player_events(str(r["event_id"]), bootstrap)
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
                    "home_player_events": player_events["home"],
                    "away_player_events": player_events["away"],
                    "player_events_pending": player_events_pending,
                }
            )

    if not fd_org_matches.empty:
        upcoming_rows = fd_org_matches[(fd_org_matches["matchday"] == target_gameweek) & (~fd_org_matches["finished"])]
        if not upcoming_rows.empty and completed_team_pairs:
            # Our own tracking store can resolve a match as finished slightly
            # before football-data.org's own `finished` flag catches up (or
            # vice versa) — without this, that lag window shows both an
            # "upcoming prediction" card and a "finished result" card for the
            # same fixture. The finished result (from completed_group above)
            # always wins.
            upcoming_rows = upcoming_rows[
                ~upcoming_rows.apply(lambda row: (row["team_home"], row["team_away"]) in completed_team_pairs, axis=1)
            ]
    elif not fallback_remaining.empty:
        upcoming_rows = fallback_remaining[
            pd.to_numeric(fallback_remaining["gameweek"], errors="coerce") == target_gameweek
        ]
    else:
        upcoming_rows = pd.DataFrame()

    if not upcoming_rows.empty:
            models = _get_models()
            preds = scoreline.predict_fixtures_batch(
                models["scoreline"], upcoming_rows, market_overrides=models.get("scoreline_market_overrides")
            )
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


def _build_fixture_detail(summary: FixtureSummary, home: str, away: str, read_only: bool = False) -> FixtureDetail:
    """`read_only=True` (used by public_snapshot.py) skips the two tracking-
    store writes below — `reconcile_fixture_market_predictions` in
    particular re-scans the *entire* matches_df against every unresolved
    prediction on every call. Confirmed live: ~250ms per call, unconditionally,
    on every single fixture-detail request regardless of whether anything
    actually changed (measured with 476 unresolved rows, zero updates on two
    back-to-back calls) — this was the dominant cost of opening a fixture,
    repeated on every click rather than amortized. Throttled via the same
    `_cached` TTL wrapper every other expensive live lookup in this file
    already uses, so it actually reconciles at most once per TTL window
    instead of once per click. Neither write is needed for a static export:
    they exist to build up this project's own live prediction-tracking
    history, which the public deployment doesn't have (or need) a copy of."""
    models = _get_models()
    matches_df = _get_matches_df()

    pred = scoreline.predict_fixture(
        models["scoreline"], home, away, max_goals=6, market_overrides=models.get("scoreline_market_overrides")
    )
    feature_row = build_features_for_fixtures(
        pd.DataFrame([{"team_home": home, "team_away": away, "commence_time": summary.commence_time}]),
        matches_df=matches_df,
        context=models.get("context"),
    ).iloc[0]
    market_preds = value_bets.predict_market_models_for_fixture(models, feature_row)
    if not read_only:
        fixture_time = pd.Timestamp(summary.commence_time)
        if fixture_time.tzinfo is not None:
            fixture_time = fixture_time.tz_localize(None)
        provenance = "snapshot" if fixture_time > pd.Timestamp.now(tz="UTC").tz_localize(None) else "reconstructed"
        tracking_store.record_fixture_market_predictions(
            summary.event_id, home, away, summary.commence_time, market_preds, provenance=provenance
        )
        _cached(
            "reconcile_fixture_market_predictions",
            lambda: tracking_store.reconcile_fixture_market_predictions(matches_df),
            ttl=_LIVE_CACHE_TTL_SECONDS,
        )
    post_match = tracking_store.get_fixture_post_match(summary.event_id)
    actual_stats = _fixture_actual_stats(matches_df, home, away, summary.commence_time)
    # The live odds table stops carrying a fixture after kickoff.  The
    # ledger is therefore the only honest source for a completed fixture's
    # value-bet call: it preserves the first displayed price and edge rather
    # than silently recalculating (or forgetting) it afterwards.
    pre_match_value_bets = value_bet_ledger.get_fixture_value_bets(summary.event_id)

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
        home_recent_form=rolling_form.recent_form(matches_df[matches_df["season"] == _current_season_label()], home),
        away_recent_form=rolling_form.recent_form(matches_df[matches_df["season"] == _current_season_label()], away),
        home_context=FixtureTeamContext(
            rest_days=_none_if_nan(feature_row.get("rest_days_home")),
            xg_for_last_5=_none_if_nan(feature_row.get("home_xg_for_last_5")),
            xg_against_last_5=_none_if_nan(feature_row.get("home_xg_against_last_5")),
            corners_last_5=_none_if_nan(feature_row.get("home_last_5_corners_for")),
            cards_last_5=_none_if_nan(feature_row.get("home_last_5_cards_for")),
            set_piece_xg_share_last_5=_none_if_nan(feature_row.get("home_set_piece_xg_share_last_5")),
        ),
        away_context=FixtureTeamContext(
            rest_days=_none_if_nan(feature_row.get("rest_days_away")),
            xg_for_last_5=_none_if_nan(feature_row.get("away_xg_for_last_5")),
            xg_against_last_5=_none_if_nan(feature_row.get("away_xg_against_last_5")),
            corners_last_5=_none_if_nan(feature_row.get("away_last_5_corners_for")),
            cards_last_5=_none_if_nan(feature_row.get("away_last_5_cards_for")),
            set_piece_xg_share_last_5=_none_if_nan(feature_row.get("away_set_piece_xg_share_last_5")),
        ),
        post_match=post_match,
        actual_stats=actual_stats,
        pre_match_value_bets=pre_match_value_bets,
    )


def _fixture_actual_stats(matches_df: pd.DataFrame, home: str, away: str, kickoff) -> FixtureActualStats | None:
    """Confirmed team totals for a finished fixture; no live-context leakage."""
    if matches_df.empty:
        return None
    fixture_time = pd.Timestamp(kickoff)
    if fixture_time.tzinfo is not None:
        fixture_time = fixture_time.tz_localize(None)
    candidates = matches_df[(matches_df["team_home"] == home) & (matches_df["team_away"] == away)].copy()
    if candidates.empty:
        return None
    candidates["date"] = pd.to_datetime(candidates["date"]).map(lambda value: value.tz_localize(None) if value.tzinfo is not None else value)
    candidates = candidates[(candidates["date"] - fixture_time).abs() <= pd.Timedelta(days=3)]
    if candidates.empty:
        return None
    match = candidates.sort_values("date").iloc[-1]
    if pd.isna(match.get("goals_home")) or pd.isna(match.get("goals_away")):
        return None

    def number(value):
        return None if value is None or pd.isna(value) else float(value)

    pairs = [
        ("Goals", "goals_home", "goals_away"), ("Shots", "hs", "as"),
        ("On target", "hst", "ast"), ("Possession %", "hp", "ap"),
        ("Corners", "hc", "ac"), ("Fouls", "hf", "af"),
        ("Yellow cards", "hy", "ay"), ("Red cards", "hr", "ar"),
    ]
    home_stats, away_stats = {}, {}
    for label, home_column, away_column in pairs:
        home_value, away_value = number(match.get(home_column)), number(match.get(away_column))
        if home_value is not None or away_value is not None:
            home_stats[label], away_stats[label] = home_value, away_value
    return FixtureActualStats(home=home_stats, away=away_stats)


@router.get("/fixtures/{event_id}", response_model=FixtureDetail)
def fixture_detail(event_id: str, read_only: bool = False):
    """`read_only=True` is for public_snapshot.py's own bulk export use
    only (skips tracking-store writes — see `_build_fixture_detail`'s
    docstring); the frontend never sets it."""
    if PUBLIC_MODE:
        cached = _public_snapshot().get("fixture_detail_by_event_id", {}).get(event_id)
        if cached is None:
            raise HTTPException(status_code=404, detail=f"No fixture with event_id={event_id}")
        return cached

    table = _value_bet_table()
    matches = table[table["event_id"].astype(str) == event_id] if not table.empty else table
    if not matches.empty:
        row = matches.iloc[0]
        return _build_fixture_detail(_row_to_summary(row), row["team_home"], row["team_away"], read_only=read_only)

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
        pred = scoreline.predict_fixture(
            models["scoreline"], home, away, market_overrides=models.get("scoreline_market_overrides")
        )
        return _build_fixture_detail(_team_fixture_to_summary(fixture, pred), home, away, read_only=read_only)

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
            _recorded_fixture_to_summary(recorded), recorded["team_home"], recorded["team_away"], read_only=read_only
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


def _resolve_fixture_kickoff(event_id: str):
    table = _value_bet_table()
    matches = table[table["event_id"].astype(str) == event_id] if not table.empty else table
    if not matches.empty:
        return pd.to_datetime(matches.iloc[0]["commence_time"])
    all_remaining = _get_remaining_fixtures_df()
    matches = all_remaining[all_remaining["event_id"].astype(str) == event_id] if not all_remaining.empty else all_remaining
    if not matches.empty:
        return pd.to_datetime(matches.iloc[0]["commence_time"])
    recorded = tracking_store.get_fixture_prediction(event_id)
    return pd.to_datetime(recorded["commence_time"]) if recorded is not None else None


def _rank_fixture_players(
    event_id: str,
    home: str,
    away: str,
    confirmed_lineups: dict[str, list[str]] | None = None,
    confirmed_starter_ids: set[int] | None = None,
) -> FixturePlayers:
    models = _get_models()
    pred = scoreline.predict_fixture(
        models["scoreline"], home, away, market_overrides=models.get("scoreline_market_overrides")
    )

    bootstrap = _get_bootstrap()
    current_event = fpl_api.get_current_event(bootstrap)
    position_priors = _get_position_priors()
    reliability_coeffs = _get_player_reliability_coeffs()
    confirmed_lineups = confirmed_lineups if confirmed_lineups is not None else espn.fetch_confirmed_lineups(home, away, _resolve_fixture_kickoff(event_id))

    lineup_model = _get_lineup_model()
    position_rate_models = _get_position_rate_models()
    contribution_model = _get_ready_goal_contribution_model()

    def rank(team: str, team_goal_expectation: float, is_home: bool) -> list[PlayerPrediction]:
        ranked = player_goals.rank_team_players(
            team, team_goal_expectation, bootstrap, current_event, position_priors,
            reliability_coeffs=reliability_coeffs, lineup_model=lineup_model,
            position_rate_models=position_rate_models, contribution_model=contribution_model,
            is_home=is_home, confirmed_starters=confirmed_lineups.get(team),
            confirmed_starter_ids=confirmed_starter_ids,
        )
        return [PlayerPrediction(**{k: p[k] for k in PlayerPrediction.model_fields}) for p in ranked]

    return FixturePlayers(
        home_players=rank(home, pred["home_goal_expectation"], True),
        away_players=rank(away, pred["away_goal_expectation"], False),
    )


def _get_cached_fixture_players(event_id: str, home: str, away: str) -> FixturePlayers:
    """Reuse the background-built player payload when a fixture modal opens."""
    return _cached(
        f"fixture_players:{event_id}",
        lambda: _rank_fixture_players(event_id, home, away),
        ttl=_LIVE_CACHE_TTL_SECONDS,
    )


def prewarm_current_gameweek_player_details() -> None:
    """Build current completed-fixture player payloads outside modal requests."""
    current_gameweek = tracking_store.get_track_record().get("current_gameweek")
    if current_gameweek is None:
        return
    group = next(
        (item for item in tracking_store.get_results_by_gameweek() if item["gameweek"] == current_gameweek),
        None,
    )
    if group is None:
        return
    for fixture in group["fixtures"]:
        try:
            _get_cached_fixture_players(fixture["event_id"], fixture["team_home"], fixture["team_away"])
        except Exception as exc:  # noqa: BLE001 - one cache warm must not delay the rest
            print(f"[player_prewarm] {fixture['event_id']} skipped: {exc}")


def _fetch_and_record_fixture_player_outcomes(event_id: str, home: str, away: str, kickoff) -> dict[int, dict]:
    """Fetch official FPL outcomes before any expensive player-model work."""
    post_match = tracking_store.get_fixture_post_match(event_id)
    expected_score = tuple(map(int, post_match["final_score"].split("-"))) if post_match is not None else None
    outcomes: dict[int, dict] = {}
    try:
        outcomes = fpl_api.fixture_player_outcomes(home, away, kickoff, _get_bootstrap(), expected_score=expected_score)
    except Exception as exc:  # noqa: BLE001 - the cache fallback below may still resolve it
        print(f"[player_tracking] direct FPL outcomes skipped: {exc}")
    if not outcomes:
        try:
            outcomes = fpl_api.fixture_player_outcomes(
                home, away, kickoff, {"teams": [], "elements": []}, expected_score=expected_score
            )
        except Exception as exc:  # noqa: BLE001 - a missing outcome must not block scores/cards
            print(f"[player_tracking] cached FPL outcomes skipped: {exc}")
    if outcomes:
        tracking_store.record_fixture_player_outcomes(event_id, outcomes)
    return outcomes


def _snapshot_or_reconcile_player_predictions(
    event_id: str,
    home: str,
    away: str,
    players: FixturePlayers,
    outcomes: dict[int, dict] | None = None,
) -> None:
    kickoff = _resolve_fixture_kickoff(event_id)
    if kickoff is None:
        return
    fixture_time = pd.Timestamp(kickoff)
    if fixture_time.tzinfo is not None:
        fixture_time = fixture_time.tz_localize(None)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    combined = [dict(player.model_dump(), team=home) for player in players.home_players] + [dict(player.model_dump(), team=away) for player in players.away_players]
    post_match = tracking_store.get_fixture_post_match(event_id)
    if fixture_time > now and post_match is None:
        tracking_store.record_player_prediction_snapshots(event_id, combined)
        return
    outcomes = outcomes or _fetch_and_record_fixture_player_outcomes(event_id, home, away, kickoff)
    if not outcomes:
        return
    for player in combined:
        player["confirmed_starter"] = bool(outcomes.get(int(player["player_id"]), {}).get("started"))
    tracking_store.record_player_prediction_snapshots(event_id, combined, provenance="reconstructed")
    tracking_store.reconcile_player_prediction_snapshots(event_id, outcomes)


def _ensure_completed_player_review(
    event_id: str,
    home: str,
    away: str,
    outcomes: dict[int, dict] | None = None,
) -> None:
    if (
        tracking_store.has_player_prediction_snapshots(event_id)
        and tracking_store.has_fixture_player_outcomes(event_id)
    ):
        return
    players = _get_cached_fixture_players(event_id, home, away)
    _snapshot_or_reconcile_player_predictions(event_id, home, away, players, outcomes=outcomes)


def backfill_completed_player_reviews(gameweek: int | None = None) -> dict:
    fixtures = tracking_store.resolved_fixtures_missing_player_snapshots(gameweek)
    outcomes_by_event: dict[str, dict[int, dict]] = {}
    for fixture in fixtures:
        try:
            outcomes = _fetch_and_record_fixture_player_outcomes(
                fixture["event_id"], fixture["team_home"], fixture["team_away"], fixture["commence_time"]
            )
            if outcomes:
                outcomes_by_event[str(fixture["event_id"])] = outcomes
        except Exception as exc:  # noqa: BLE001 - one FPL fixture must not stop the rest
            print(f"[player_outcomes] {fixture['event_id']} skipped: {exc}")

    completed, unresolved = 0, 0
    for fixture in fixtures:
        try:
            outcomes = outcomes_by_event.get(str(fixture["event_id"]))
            if outcomes:
                _ensure_completed_player_review(
                    fixture["event_id"], fixture["team_home"], fixture["team_away"], outcomes=outcomes
                )
            if (
                tracking_store.has_player_prediction_snapshots(fixture["event_id"])
                and tracking_store.has_fixture_player_outcomes(fixture["event_id"])
            ):
                completed += 1
            else:
                unresolved += 1
        except Exception as exc:  # noqa: BLE001 - one fixture must not stop the remaining archive
            unresolved += 1
            print(f"[player_backfill] {fixture['event_id']} skipped: {exc}")
    return {
        "attempted": len(fixtures),
        "events_saved": len(outcomes_by_event),
        "completed": completed,
        "unresolved": unresolved,
    }


@router.get("/fixtures/{event_id}/players", response_model=FixturePlayers)
def fixture_players(event_id: str, read_only: bool = False):
    """`read_only=True` is for public_snapshot.py's own bulk export use
    only — skips the tracking-store snapshot/reconcile write below (for a
    finished match, that write can itself trigger a live outcomes fetch),
    same idea as `fixture_detail`'s own `read_only` param. The frontend
    never sets it."""
    if PUBLIC_MODE:
        cached = _public_snapshot().get("fixture_players_by_event_id", {}).get(event_id)
        if cached is None:
            raise HTTPException(status_code=404, detail=f"No fixture with event_id={event_id}")
        return cached

    resolved = _resolve_fixture_teams(event_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"No fixture with event_id={event_id}")
    home, away = resolved
    players = _get_cached_fixture_players(event_id, home, away)
    if not read_only:
        _snapshot_or_reconcile_player_predictions(event_id, home, away, players)
    return players


@router.get("/fixtures/{event_id}/player-review")
def fixture_player_review(event_id: str):
    if PUBLIC_MODE:
        # The live public deployment never accumulates tracking history
        # itself (background tracking is skipped entirely in PUBLIC_MODE —
        # see main.py's lifespan), but public_snapshot.py runs this same
        # route locally at build time, where real tracking history *does*
        # exist, and bakes the result in — same pattern as fixture_detail's
        # own post_match field, which is exactly what makes the frontend
        # call this route in the first place. None (not 404) when a
        # fixture's review genuinely isn't available yet, matching what
        # this returns locally in that case.
        return _public_snapshot().get("player_review_by_event_id", {}).get(event_id)

    resolved = _resolve_fixture_teams(event_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"No fixture with event_id={event_id}")
    home, away = resolved
    if not (
        tracking_store.has_player_prediction_snapshots(event_id)
        and tracking_store.has_fixture_player_outcomes(event_id)
    ):
        kickoff = _resolve_fixture_kickoff(event_id)
        if kickoff is not None:
            outcomes = _fetch_and_record_fixture_player_outcomes(event_id, home, away, kickoff)
            if outcomes:
                _ensure_completed_player_review(event_id, home, away, outcomes=outcomes)
    return tracking_store.get_fixture_player_review(event_id)


def background_tracking_tick() -> None:
    """Capture routine markets and confirmed-XI calls without a UI request."""
    try:
        table = _value_bet_table()
        _run_tracking_bookkeeping(table)
        backfill_completed_player_reviews()
        if table.empty:
            return
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        for _, fixture in table.iterrows():
            kickoff = pd.Timestamp(fixture["commence_time"])
            if kickoff.tzinfo is not None:
                kickoff = kickoff.tz_localize(None)
            if not (now <= kickoff <= now + pd.Timedelta(hours=4)):
                continue
            confirmed = espn.fetch_confirmed_lineups(fixture["team_home"], fixture["team_away"], fixture["commence_time"])
            if fixture["team_home"] not in confirmed or fixture["team_away"] not in confirmed:
                continue
            tracking_store.record_fixture_forecast_snapshots(
                table.loc[[fixture.name]],
                "confirmed_xi_baseline",
                {str(fixture["event_id"]): confirmed},
            )
            players = _rank_fixture_players(str(fixture["event_id"]), fixture["team_home"], fixture["team_away"], confirmed)
            _snapshot_or_reconcile_player_predictions(str(fixture["event_id"]), fixture["team_home"], fixture["team_away"], players)
    except Exception as exc:  # noqa: BLE001 - periodic tracking is best effort
        print(f"[background_tracking] skipped: {exc}")


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
        predicted_total_goals=pred["home_goal_expectation"] + pred["away_goal_expectation"],
        predicted_margin=pred["home_goal_expectation"] - pred["away_goal_expectation"],
        home_2plus_prob=pred["home_2plus_prob"],
        away_2plus_prob=pred["away_2plus_prob"],
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
            fixture,
            scoreline.predict_fixture(
                models["scoreline"],
                fixture["team_home"],
                fixture["team_away"],
                market_overrides=models.get("scoreline_market_overrides"),
            ),
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


@router.get("/squad-continuity")
def get_squad_continuity():
    """Per-team squad continuity for the current season (EXP-2026-18, see
    docs/AI_CONTINUITY.md and features/squad_change.py) — the fraction of
    last season's playing-time minutes retained by players still
    registered this season, lowest first (biggest off-season turnover).
    Feeds `ml_scoreline`'s feature set directly (see features/build.py);
    this endpoint just exposes the raw per-team numbers for display,
    independent of which scoreline model happens to be `chosen_model`."""
    season = football_data.season_str(football_data.CURRENT_SEASON_START_YEAR)

    def build():
        try:
            table = squad_change.team_season_continuity_table([season])
        except RuntimeError:
            return []
        return table.sort_values("squad_continuity")[["team", "squad_continuity"]].to_dict("records")

    return {"season": season, "teams": _cached("squad_continuity", build, ttl=24 * 3600)}


@router.post("/backtest", dependencies=[Depends(_admin_only)])
def run_backtest(edge_threshold: float = 0.05, staking: str = "kelly"):
    models = _get_models()
    df, _ = build_training_frame()
    _, val_df = chronological_split(df)
    start, end = str(val_df["date"].min().date()), str(val_df["date"].max().date())
    selections: list[dict] = []
    bt = backtest_lib.build_value_bet_backtest(
        val_df,
        models["scoreline"],
        start,
        end,
        edge_threshold=edge_threshold,
        staking=staking,
        selections=selections,
        market_overrides=models.get("scoreline_market_overrides"),
    )
    return {
        "results": bt.results(),
        "bankroll_curve": bt.account.tracker,
        "staking": staking,
        "season": str(val_df["season"].iloc[0]) if not val_df.empty else None,
        "selections": selections,
    }


@router.get("/value-bets/walk-forward")
def get_walk_forward_value_bet_validation():
    """Periodic multi-season validation for the live single-bet rule."""
    return _cached(
        "value_bet_walk_forward",
        betting_validation.run_walk_forward_value_bet_validation,
        ttl=24 * 3600,
    )


@router.get("/value-bets/track-record")
def get_value_bet_track_record(staking: str = "kelly"):
    """Live counterpart to /backtest: not a synthetic held-out-season
    replay, but the real fixtures the app actually flagged as "Value bet"
    (via /api/fixtures and /api/fixtures/gameweek's shared bookkeeping),
    tracked from the moment each was first flagged and reconciled once the
    match finishes. Answers "have the value-bet recommendations actually
    shown to the user been worth following," not "is this a good strategy
    in principle." """
    _run_tracking_bookkeeping(_value_bet_table())
    return value_bet_ledger.get_value_bet_track_record(staking=staking)


@router.post("/retrain", dependencies=[Depends(_admin_only)])
def retrain():
    manifest = manifest_lib.train_all()
    _clear_cache("models", "value_bet_table")
    return manifest


@router.post("/refresh-odds", dependencies=[Depends(_admin_only)])
def refresh_odds():
    _get_odds_df(force=True)
    _clear_cache("fixtures_df", "value_bet_table")
    return {"status": "ok"}


@router.post("/refresh-fixtures", dependencies=[Depends(_admin_only)])
def refresh_fixtures():
    _clear_cache("fixtures_df", "value_bet_table")
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


def _fpl_projections(gameweek: int | None = None) -> dict:
    """Build the FPL view from the official live pool and the independent
    scoreline model.  It is cached separately from fixtures because an FPL
    page should not trigger one element-summary request per player."""
    bootstrap = _get_bootstrap()
    event_id = gameweek or fpl_api.get_current_event(bootstrap)
    if event_id is None:
        raise HTTPException(status_code=409, detail="FPL has not published a current or next gameweek yet.")

    def build():
        fixtures = fpl_api.fetch_fixtures()
        models = _get_models()
        teams = {int(team["id"]): team["name"] for team in bootstrap.get("teams", [])}
        event_fixtures = [item for item in fixtures if item.get("event") == int(event_id) and not item.get("finished")]
        batch_input = pd.DataFrame([
            {
                "team_home": to_canonical(teams.get(int(item["team_h"]), str(item["team_h"])), source="fpl"),
                "team_away": to_canonical(teams.get(int(item["team_a"]), str(item["team_a"])), source="fpl"),
                "commence_time": item.get("kickoff_time"),
            }
            for item in event_fixtures
        ])
        batch_predictions = scoreline.predict_fixtures_batch(
            models["scoreline"], batch_input, market_overrides=models.get("scoreline_market_overrides")
        ) if not batch_input.empty else []
        prediction_by_pair = {
            (row.team_home, row.team_away): {**prediction, "model_source": "independent_scoreline"}
            for row, prediction in zip(batch_input.itertuples(index=False), batch_predictions)
        }

        def predict(home: str, away: str) -> dict:
            return prediction_by_pair.get(
                (to_canonical(home, source="fpl"), to_canonical(away, source="fpl")),
                {"home_goal_expectation": 1.35, "away_goal_expectation": 1.15, "model_source": "fpl_fallback"},
            )

        return fpl_model.build_projections(bootstrap, fixtures, int(event_id), predict)

    return _cached(f"fpl_projections_{event_id}", build, ttl=_LIVE_CACHE_TTL_SECONDS)


@router.get("/fpl/projections")
def fpl_projections(gameweek: int | None = None):
    if PUBLIC_MODE:
        return _public_snapshot().get("fpl", {}).get("projections", {"gameweek": gameweek, "players": []})
    return _fpl_projections(gameweek)


@router.get("/fpl/optimal-xi")
def fpl_optimal_xi(gameweek: int | None = None, formation: str | None = None):
    if PUBLIC_MODE:
        return _public_snapshot().get("fpl", {}).get("optimal_xi", {})
    data = _fpl_projections(gameweek)
    try:
        recommendation = fpl_model.optimal_xi(data["players"], formation=formation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"gameweek": data["gameweek"], "model_source": data["model_source"], **recommendation}


@router.get("/fpl/squad")
def fpl_squad(gameweek: int | None = None, budget: float = 100.0):
    if not 80.0 <= budget <= 120.0:
        raise HTTPException(status_code=422, detail="Budget must be between £80m and £120m.")
    if PUBLIC_MODE:
        return _public_snapshot().get("fpl", {}).get("squad", {})
    data = _fpl_projections(gameweek)
    try:
        recommendation = fpl_model.build_squad(data["players"], budget)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"gameweek": data["gameweek"], "model_source": data["model_source"], **recommendation}


@router.post("/fpl/transfers/manual")
def fpl_manual_transfers(request: FPLManualSquadRequest = Body(...), gameweek: int | None = None):
    if len(set(request.player_ids)) != 15:
        raise HTTPException(status_code=422, detail="A manual FPL squad must contain exactly 15 distinct player ids.")
    data = _fpl_projections(gameweek)
    try:
        result = fpl_model.transfer_recommendations(data["players"], request.player_ids, request.bank, request.free_transfers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"gameweek": data["gameweek"], "source": "manual", **result}


@router.get("/fpl/entry/{entry_id}/transfers")
def fpl_entry_transfers(entry_id: int, gameweek: int | None = None, free_transfers: int = 1):
    """Public-entry convenience endpoint.  The ID is only used for this
    request; no entry or squad is written to our tracking database.

    This is reachable in PUBLIC_MODE (a random visitor entering their own
    public entry ID is exactly the intended use) with no admin surface or
    write involved, so it isn't gated like `_admin_only` routes — but
    unlike every other FPL endpoint here, it makes a fresh outbound FPL
    request on every call with no cache at all, which an anonymous public
    deployment has no rate limit in front of. A given entry's picks for an
    already-reached gameweek never change, so caching by
    (entry_id, gameweek) removes the repeat-request cost/abuse surface
    without changing the result."""
    data = _fpl_projections(gameweek)
    try:
        picks, source_gameweek = _cached(
            f"fpl_entry_picks_{entry_id}_{data['gameweek']}",
            lambda: fpl_api.fetch_latest_entry_picks(entry_id, data["gameweek"], _get_bootstrap()),
            ttl=_LIVE_CACHE_TTL_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - FPL uses several failure shapes
        raise HTTPException(status_code=502, detail="FPL could not load that public entry for this gameweek.") from exc
    player_ids = [int(item["element"]) for item in picks.get("picks", [])]
    bank = float(picks.get("entry_history", {}).get("bank", 0) or 0) / 10.0
    try:
        result = fpl_model.transfer_recommendations(data["players"], player_ids, bank, free_transfers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"gameweek": data["gameweek"], "source_gameweek": source_gameweek, "source": "public_entry", "entry_id": entry_id, **result}


@router.get("/hub/rankings")
def get_power_rankings():
    if PUBLIC_MODE:
        return _public_snapshot().get("hub", {}).get("rankings", {"rankings": [], "ratings_history": {}, "season": None})
    matches_df = _get_matches_df()
    current_season = _current_season_label()
    season_matches = matches_df[matches_df["season"] == current_season]

    if not season_matches.empty:
        current_teams = set(season_matches["team_home"]) | set(season_matches["team_away"])
    else:
        remaining_df = _get_remaining_fixtures_df()
        current_teams = (set(remaining_df["team_home"]) | set(remaining_df["team_away"])) if not remaining_df.empty else None
    prior_matches = matches_df[matches_df["season"] != current_season]
    prior_model = _cached(
        f"rankings_preseason_prior_{current_season}", lambda: scoreline.fit_dixon_coles(prior_matches), ttl=24 * 3600
    )
    games_played = (
        pd.concat([season_matches["team_home"], season_matches["team_away"]]).value_counts().to_dict()
        if not season_matches.empty
        else {}
    )
    # A display-only ranking can update after each result without changing
    # scoreline probabilities or value-bet decisions. The fixed 75% online
    # Elo/Pi form blend won the offline ranking comparison; its remaining 25%
    # pre-season prior stops an isolated result from completely erasing the
    # history-seeded baseline.
    rankings = _cached(
        f"rankings_live_form_{current_season}_{len(season_matches)}",
        lambda: power_rankings_mod.blended_form_power_rankings(
            prior_model,
            ratings_mod.fit_elo(matches_df),
            ratings_mod.fit_pi_ratings(matches_df),
            current_teams=current_teams or set(),
            form_weight=0.75,
            games_played=games_played,
        ),
        ttl=_LIVE_CACHE_TTL_SECONDS,
    )

    history = ratings_mod.team_rating_timeseries(season_matches, fd_org_matches=_get_fd_org_matches())
    if history.empty:
        ratings_history = {}
    else:
        history["date"] = history["date"].dt.date.astype(str)
        history["gameweek"] = history["gameweek"].apply(lambda v: int(v) if pd.notna(v) else None)
        ratings_history = {
            team: group[["date", "elo", "pi", "gameweek"]].to_dict("records") for team, group in history.groupby("team")
        }

    return {"rankings": rankings, "ratings_history": ratings_history, "season": current_season}


@router.get("/hub/table")
def get_projected_table():
    if PUBLIC_MODE:
        return _public_snapshot().get("hub", {}).get("table", {"table": [], "season": None})
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
    if PUBLIC_MODE:
        return _public_snapshot().get("hub", {}).get(
            "track_record", {"summary": {}, "biggest_upsets": [], "gameweeks": []}
        )
    return {
        "summary": tracking_store.get_track_record(),
        "biggest_upsets": tracking_store.get_biggest_upsets(),
        "gameweeks": tracking_store.get_results_by_gameweek(),
    }


@router.get("/hub/teams")
def get_team_hub():
    if PUBLIC_MODE:
        return _public_snapshot().get("hub", {}).get("teams", {})
    matches_df = _get_matches_df()
    season = _current_season_label()
    current_match_count = len(matches_df[matches_df["season"] == season])
    return _cached(
        f"team_hub_{season}_{current_match_count}",
        lambda: hub_analytics.build_team_hub(matches_df, season, bootstrap=_get_bootstrap()),
        ttl=_LIVE_CACHE_TTL_SECONDS,
    )


@router.get("/hub/players")
def get_player_hub():
    if PUBLIC_MODE:
        return _public_snapshot().get("hub", {}).get("players", {})
    return _cached("player_hub", lambda: hub_analytics.build_player_hub(_get_bootstrap()), ttl=300)


@router.get("/scorer-track-record")
def get_scorer_track_record():
    """Keep player-model evaluation with the calibration surfaces, not discovery."""
    return tracking_store.get_scorer_accuracy()


@router.get("/research/confirmed-xi")
def get_confirmed_xi_experiment():
    return tracking_store.confirmed_xi_experiment_status()


@router.get("/research/odds-snapshots")
def get_odds_snapshot_experiment():
    return tracking_store.odds_snapshot_status()


@router.get("/research/historical-closing-odds")
def get_historical_closing_odds_benchmark():
    return _cached(
        "historical_closing_odds_benchmark",
        lambda: odds_benchmark.closing_odds_benchmark(_get_matches_df()),
        ttl=24 * 3600,
    )
