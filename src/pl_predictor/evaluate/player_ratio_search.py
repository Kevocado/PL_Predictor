"""Pre-registered search for ratio features in player markets.

Procedure, seasons and thresholds are fixed in
``docs/player_ratio_search_preregistration.md``. If this file and that
document disagree, the document wins.

Nothing here changes a production model. ``run_search`` writes the raw JSON
record and evaluates the holdout only for candidates that pass selection.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import UNDERSTAT_SHOTS_CACHE_DIR
from ..data import fpl_history
from ..data.team_names import to_canonical
from ..data.understat_shots import _normalise_crosswalk_name, build_understat_fpl_crosswalk
from ..features import player_form
from ..models.player_goals import RATE_FEATURES

# ---- pre-registered constants ----------------------------------------------
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SELECTION_FOLDS = ["2022-23", "2023-24", "2024-25"]
HOLDOUT_SEASON = "2025-26"
WINDOWS = (3, 5, 10)
PAIR_WINDOWS = (5, 10)
MAX_KEPT_PER_MARKET = 3
ALPHA = 0.05
N_BOOT = 2000
SEED = 20261030
POOL_CAP = 300
RATIO_EPS = 0.01
CLIP_Q = 0.995

MARKETS = {"scorer": "binary", "assist": "binary", "shots": "count", "sot": "count"}
_TARGET = {"scorer": "scorer_y", "assist": "assist_y", "shots": "shots_count", "sot": "sot_count"}

# Production position-model features (player_goals.RATE_FEATURES), the
# reliability extras production blends in, and the production lineup features.
PRODUCTION_BASE = list(RATE_FEATURES) + [
    "threat_last10", "creativity_last10",
    "starts_last3", "starts_last5", "starts_last10", "sub_rate_last3", "sub_rate_last5",
    "minutes_last3", "minutes_last5", "minutes_last10", "minutes_ema", "start_streak",
]
PRODUCTION_SHOT_RATES = [f"{stat}_per90_last{w}" for stat in ("shots", "shots_on_target") for w in WINDOWS]

_UNDERSTAT_ALIASES = {"Sheffield United": "Sheffield Utd"}
_PER90 = {  # short name -> strictly lagged rolling sum column
    "goals": "goals_scored",
    "assists": "assists",
    "xg": "expected_goals",
    "xa": "expected_assists",
    "shots": "shots",
    "sot": "shots_on_target",
    "threat": "threat",
    "creativity": "creativity",
    "xgi": "expected_goal_involvements",
}
_ROLL_COLS = list(dict.fromkeys(
    ["minutes", "team_for", "share_min", "sot_share_min"] + list(_PER90.values())
))


# ---- metrics ---------------------------------------------------------------

def binary_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.clip(mu, 1e-9, None)
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2 * (ylogy - (y - mu))


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in input order."""
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    bucket = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = bucket == b
        if mask.any():
            total += abs(y[mask].mean() - p[mask].mean()) * mask.sum()
    return float(total / len(y))


# ---- candidate definitions ---------------------------------------------------

def candidate_definitions() -> list[dict]:
    """Full pool, one entry per candidate, fixed order. Names carry a `cand_`
    prefix so they never collide with production columns of the same stem
    (e.g. production's own `shots_per90_last3`)."""
    defs = []
    for w in WINDOWS:
        singles = [
            ("shots_per90", "rate", "shots per 90 over prior appearances"),
            ("sot_per90", "rate", "shots on target per 90"),
            ("sot_over_shots", "ratio", "SoT / shots"),
            ("goals_over_sot", "ratio", "goals / SoT"),
            ("goals_over_shots", "ratio", "goals / shots"),
            ("xg_over_shot", "ratio", "xG / shot"),
            ("xg_per90", "rate", "xG per 90 (stand-in for npxG/90: FPL history has no penalty-xG)"),
            ("xa_per90", "rate", "xA per 90"),
            ("shots_share_team", "share", "player shots / team shots"),
            ("sot_share_team", "share", "player SoT / team shots"),
            ("mw_shot_share", "share", "minutes-weighted per-match shot share"),
            ("mw_sot_share", "share", "minutes-weighted per-match SoT share"),
            ("assists_over_xa", "ratio", "assists / xA"),
            ("goals_over_xg", "ratio", "goals / xG"),
            ("team_shots_for", "team", "own team's mean shots for, prior matches"),
            ("opp_shots_allowed", "team", "opponent's mean shots allowed, prior matches"),
            ("team_volume_over_opp_allowed", "ratio", "team shots for / opponent shots allowed"),
        ]
        defs += [{"name": f"cand_{n}_last{w}", "family": f, "description": d} for n, f, d in singles]
    for w in PAIR_WINDOWS:
        for a in _PER90:
            for b in _PER90:
                if a != b:
                    defs.append({"name": f"cand_pair_{a}_over_{b}_last{w}", "family": "pairwise",
                                 "description": f"per-90 {a} / per-90 {b}"})
    return defs


# ---- data --------------------------------------------------------------------

def _understat_fixtures() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = UNDERSTAT_SHOTS_CACHE_DIR / f"_fixtures_{int(season[:4])}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing cached Understat fixtures: {path}")
        fx = pd.read_csv(path)
        fx["season"] = season
        frames.append(fx)
    fx = pd.concat(frames, ignore_index=True)
    fx["home"] = fx["team_home"].map(lambda n: _UNDERSTAT_ALIASES.get(n, to_canonical(n, "understat")))
    fx["away"] = fx["team_away"].map(lambda n: _UNDERSTAT_ALIASES.get(n, to_canonical(n, "understat")))
    fx["u_date"] = pd.to_datetime(fx["date"].astype(str).str[:10])
    return fx[["season", "understat_id", "home", "away", "u_date"]]


def _fpl_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    side = raw.drop_duplicates(["season", "fixture", "tc"])[["season", "fixture", "date", "tc", "was_home"]]
    home = side[side["was_home"]].rename(columns={"tc": "home"}).drop(columns="was_home")
    away = side[~side["was_home"]][["season", "fixture", "tc"]].rename(columns={"tc": "away"})
    fx = home.merge(away, on=["season", "fixture"], how="inner")
    fx["date"] = pd.to_datetime(fx["date"])
    return fx


def _load_shot_rows(understat_ids: list[int]) -> tuple[pd.DataFrame, list[int]]:
    """Per-shot rows from the cached per-match Understat files. Returns the
    rows and the ids whose file is missing (reported, never invented)."""
    frames, missing = [], []
    for uid in understat_ids:
        path = UNDERSTAT_SHOTS_CACHE_DIR / f"{uid}.csv"
        if not path.exists():
            missing.append(int(uid))
            continue
        shots = pd.read_csv(path, usecols=["player", "player_id", "h_a", "result", "x_g"])
        shots["understat_id"] = int(uid)
        frames.append(shots)
    shot_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["player", "player_id", "h_a", "result", "x_g", "understat_id"])
    return shot_rows, missing


def _crosswalk_by_season(raw: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Understat player_id -> FPL element, per season, through the project's
    own crosswalk. FPL history has one `name` per element, so a bootstrap-
    shaped dict is built from it. Ambiguous or unmatched players are absent
    from the result, as the crosswalk already guarantees."""
    frames = []
    for season, season_raw in raw.groupby("season"):
        elements = season_raw.drop_duplicates("element")
        bootstrap = {"elements": []}
        for element, name in zip(elements["element"], elements["name"]):
            parts = str(name).split()
            bootstrap["elements"].append({
                "id": int(element),
                "first_name": parts[0],
                "second_name": " ".join(parts[1:]) or parts[0],
                "web_name": str(name),
            })
        season_shots = shots[shots["season"] == season]
        players = season_shots[["player", "player_id"]].drop_duplicates("player_id")
        crosswalk = build_understat_fpl_crosswalk(players, bootstrap)
        frames.append(pd.DataFrame({"season": season, "player_id": list(crosswalk.keys()),
                                    "element": list(crosswalk.values())}))
    return pd.concat(frames, ignore_index=True)


def _name_tokens(name: str) -> frozenset[str]:
    """Normalised word tokens, hyphens split. FPL history's `name` is often the
    long legal form ("Richarlison de Andrade"), so the project crosswalk's
    web_name stages cannot fire on it; token containment can."""
    parts = str(name).replace("-", " ").split()
    return frozenset(_normalise_crosswalk_name(p) for p in parts if _normalise_crosswalk_name(p))


def _team_constrained_fallback(raw: pd.DataFrame, shots: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """For shooters the project crosswalk leaves out: an Understat name whose
    tokens are all present in exactly one same-team FPL element of that
    season. Team comes from the shot's own side of the fixture, so a name
    shared across clubs cannot resolve to the wrong player. Ambiguity keeps
    the player unmapped, as the project crosswalk does."""
    mapped = set(zip(mapping["season"], mapping["player_id"]))
    pairs = shots[["season", "player_id", "player", "team"]].drop_duplicates(["season", "player_id", "team"])
    elements = raw.drop_duplicates(["season", "element"])[["season", "element", "name", "tc"]]
    pool: dict[tuple[str, str], list[tuple[int, frozenset[str]]]] = {}
    for season, team, element, name in zip(elements["season"], elements["tc"], elements["element"], elements["name"]):
        pool.setdefault((season, team), []).append((int(element), _name_tokens(name)))
    rows = []
    for season, player_id, player, team in pairs.itertuples(index=False):
        if (season, player_id) in mapped:
            continue
        tokens = _name_tokens(player)
        hits = [element for element, names in pool.get((season, team), []) if tokens and tokens <= names]
        if len(hits) == 1:
            rows.append({"season": season, "player_id": player_id, "element": hits[0]})
            mapped.add((season, player_id))
    return pd.DataFrame(rows, columns=["season", "player_id", "element"])


def match_fixtures(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """FPL fixture -> Understat fixture on canonical teams and date (+-1 day)."""
    fpl_fx = _fpl_fixtures(raw)
    us = _understat_fixtures()
    m = fpl_fx.merge(us, on=["season", "home", "away"], how="left")
    m["day_gap"] = (m["date"] - m["u_date"]).dt.days.abs()
    m = m[m["day_gap"].fillna(99) <= 1].sort_values(["season", "fixture", "day_gap"])
    m = m.drop_duplicates(["season", "fixture"])
    report = {"fpl_fixtures": int(len(fpl_fx)), "joined_fixtures": int(len(m))}
    return m[["season", "fixture", "understat_id", "home", "away"]].copy(), report


def build_frame(seasons: list[str] = SEASONS) -> tuple[pd.DataFrame, dict]:
    """One row per covered player appearance (minutes > 0). Every rolling
    feature is shifted, so a row sees only its prior appearances.
    Returns (frame, coverage_report)."""
    raw = fpl_history.load_player_gw_history(seasons=seasons).copy()
    raw["date"] = pd.to_datetime(raw["kickoff_time"], utc=True).dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    raw["tc"] = raw["team"].map(lambda n: to_canonical(str(n), "fpl"))
    raw = raw.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)

    matched, coverage = match_fixtures(raw)
    shot_rows, missing = _load_shot_rows(matched["understat_id"].astype(int).tolist())
    shot_rows = shot_rows.merge(matched[["understat_id", "season"]], on="understat_id", how="left")

    # Team shots per fixture side, all shooters included (unmapped ones too).
    side_counts = shot_rows.groupby(["understat_id", "h_a"]).size().unstack(fill_value=0)
    for col in ("h", "a"):
        if col not in side_counts:
            side_counts[col] = 0
    have_shots = matched["understat_id"].isin(side_counts.index)
    coverage["missing_match_files"] = int(len(missing))
    coverage["fixtures_without_shot_rows"] = int((~have_shots).sum())
    matched = matched[have_shots].reset_index(drop=True)

    shot_rows = shot_rows.merge(matched[["understat_id", "home", "away"]], on="understat_id", how="left")
    shot_rows["team"] = np.where(shot_rows["h_a"] == "h", shot_rows["home"], shot_rows["away"])
    mapping = _crosswalk_by_season(raw, shot_rows)
    coverage["unmapped_share_after_project_crosswalk"] = round(
        1 - shot_rows.merge(mapping, on=["season", "player_id"], how="inner").shape[0] / max(len(shot_rows), 1), 4)
    fallback = _team_constrained_fallback(raw, shot_rows, mapping)
    mapping = pd.concat([mapping, fallback], ignore_index=True)
    mapped = shot_rows.merge(mapping, on=["season", "player_id"], how="inner")
    coverage["shot_rows"] = int(len(shot_rows))
    coverage["team_fallback_players"] = int(len(fallback))
    coverage["unmapped_shot_row_share"] = round(1 - len(mapped) / max(len(shot_rows), 1), 4)

    player_agg = mapped.assign(on_target=mapped["result"].isin(["Goal", "SavedShot"]).astype(float)).groupby(
        ["season", "understat_id", "element"], as_index=False).agg(
        shots=("x_g", "size"), shots_on_target=("on_target", "sum"))

    covered = raw.merge(matched, on=["season", "fixture"], how="inner")
    covered = covered.merge(player_agg, on=["season", "understat_id", "element"], how="left")
    covered[["shots", "shots_on_target"]] = covered[["shots", "shots_on_target"]].fillna(0.0)
    covered = covered.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)
    is_home = covered["was_home"].astype(bool).to_numpy()
    covered["team"] = np.where(is_home, covered["home"], covered["away"])
    covered["opp"] = np.where(is_home, covered["away"], covered["home"])
    own_counts = side_counts.reindex(covered["understat_id"].astype(int).to_numpy())
    own_h = own_counts["h"].to_numpy(dtype=float)
    own_a = own_counts["a"].to_numpy(dtype=float)
    covered["team_for"] = np.where(is_home, own_h, own_a)
    covered["team_against"] = np.where(is_home, own_a, own_h)
    denom = covered["team_for"].where(covered["team_for"] > 0)
    covered["share_min"] = (covered["shots"] / denom).fillna(0.0) * covered["minutes"]
    covered["sot_share_min"] = (covered["shots_on_target"] / denom).fillna(0.0) * covered["minutes"]
    coverage["covered_appearance_rows"] = int(len(covered))
    coverage["covered_fixtures"] = int(len(matched))

    # Team-level prior-match means, one row per (fixture, team).
    team_rows = pd.concat([
        pd.DataFrame({"season": matched["season"], "fixture": matched["fixture"], "team": matched["home"],
                      "shots_for": own_counts_for(side_counts, matched, "h"),
                      "shots_against": own_counts_for(side_counts, matched, "a")}),
        pd.DataFrame({"season": matched["season"], "fixture": matched["fixture"], "team": matched["away"],
                      "shots_for": own_counts_for(side_counts, matched, "a"),
                      "shots_against": own_counts_for(side_counts, matched, "h")}),
    ], ignore_index=True)
    kickoff = raw.drop_duplicates(["season", "fixture"])[["season", "fixture", "kickoff_time"]]
    team_rows = team_rows.merge(kickoff, on=["season", "fixture"], how="left").sort_values(["season", "team", "kickoff_time"])
    tg = team_rows.groupby(["season", "team"])
    for w in WINDOWS:
        team_rows[f"team_sf_last{w}"] = tg["shots_for"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
        team_rows[f"opp_sa_last{w}"] = tg["shots_against"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    own_view = team_rows[["season", "fixture", "team"] + [f"team_sf_last{w}" for w in WINDOWS]]
    covered = covered.merge(own_view, on=["season", "fixture", "team"], how="left")
    opp_view = team_rows[["season", "fixture", "team"] + [f"opp_sa_last{w}" for w in WINDOWS]].rename(
        columns={"team": "opp", **{f"opp_sa_last{w}": f"opp_allowed_last{w}" for w in WINDOWS}})
    covered = covered.merge(opp_view, on=["season", "fixture", "opp"], how="left")
    covered = covered.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)

    # Production baseline features, built by the production builders.
    starts, start_cols = player_form.build_historical_start_features(covered)
    played, _form_cols = player_form.build_historical_player_form(covered)
    frame = played.join(starts[start_cols], how="left").copy()
    frame["was_home"] = frame["was_home"].astype(float)
    frame["scorer_y"] = (frame["goals_scored"] >= 1).astype(float)
    frame["assist_y"] = (frame["assists"] >= 1).astype(float)
    frame["shots_count"] = frame["shots"].astype(float)
    frame["sot_count"] = frame["shots_on_target"].astype(float)

    # Strictly lagged rolling sums over prior appearances, per season and player.
    frame = pd.concat([frame, lagged_rolling_sums(frame, ["season", "element"], _ROLL_COLS, WINDOWS, prefix="S")], axis=1)
    frame = _add_candidates(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame, coverage


def lagged_rolling_sums(frame: pd.DataFrame, keys: list[str], cols: list[str], windows: tuple[int, ...], prefix: str) -> pd.DataFrame:
    """Rolling sums over each group's PRIOR rows only: `shift(1)` before the
    window, so a row's own value never enters its own feature."""
    grouped = frame.groupby(keys, sort=False)
    rolled = {}
    for w in windows:
        for col in cols:
            rolled[f"{prefix}{w}_{col}"] = grouped[col].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
    return pd.DataFrame(rolled, index=frame.index)


def own_counts_for(side_counts: pd.DataFrame, matched: pd.DataFrame, side: str) -> np.ndarray:
    return side_counts.reindex(matched["understat_id"].astype(int).to_numpy())[side].to_numpy(dtype=float)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b > 0)


def _rate90(frame: pd.DataFrame, w: int, short: str) -> pd.Series:
    return _safe_div(frame[f"S{w}_{_PER90[short]}"] * 90, frame[f"S{w}_minutes"])


def _add_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for w in WINDOWS:
        def S(col, w=w):
            return frame[f"S{w}_{col}"]
        out[f"cand_shots_per90_last{w}"] = _rate90(frame, w, "shots")
        out[f"cand_sot_per90_last{w}"] = _rate90(frame, w, "sot")
        out[f"cand_sot_over_shots_last{w}"] = _safe_div(S("shots_on_target"), S("shots"))
        out[f"cand_goals_over_sot_last{w}"] = _safe_div(S("goals_scored"), S("shots_on_target"))
        out[f"cand_goals_over_shots_last{w}"] = _safe_div(S("goals_scored"), S("shots"))
        out[f"cand_xg_over_shot_last{w}"] = _safe_div(S("expected_goals"), S("shots"))
        out[f"cand_xg_per90_last{w}"] = _rate90(frame, w, "xg")
        out[f"cand_xa_per90_last{w}"] = _rate90(frame, w, "xa")
        out[f"cand_shots_share_team_last{w}"] = _safe_div(S("shots"), S("team_for"))
        out[f"cand_sot_share_team_last{w}"] = _safe_div(S("shots_on_target"), S("team_for"))
        out[f"cand_mw_shot_share_last{w}"] = _safe_div(S("share_min"), S("minutes"))
        out[f"cand_mw_sot_share_last{w}"] = _safe_div(S("sot_share_min"), S("minutes"))
        out[f"cand_assists_over_xa_last{w}"] = _safe_div(S("assists"), S("expected_assists"))
        out[f"cand_goals_over_xg_last{w}"] = _safe_div(S("goals_scored"), S("expected_goals"))
        out[f"cand_team_shots_for_last{w}"] = frame[f"team_sf_last{w}"]
        out[f"cand_opp_shots_allowed_last{w}"] = frame[f"opp_allowed_last{w}"]
        out[f"cand_team_volume_over_opp_allowed_last{w}"] = _safe_div(frame[f"team_sf_last{w}"], frame[f"opp_allowed_last{w}"])
    for w in PAIR_WINDOWS:
        rates = {short: _rate90(frame, w, short) for short in _PER90}
        for a in _PER90:
            for b in _PER90:
                if a != b:
                    out[f"cand_pair_{a}_over_{b}_last{w}"] = rates[a] / (rates[b] + RATIO_EPS)
    return pd.concat([frame, pd.DataFrame(out, index=frame.index)], axis=1)


# ---- models ------------------------------------------------------------------

def _design(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    base = frame[features].astype(float)
    positions = pd.get_dummies(frame["position"].fillna("UNK"), dtype=float, prefix="pos")
    return pd.concat([base, positions], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def _binary_predict(X_tr, y_tr, seasons_tr, X_te) -> np.ndarray:
    """Balanced logistic regression, Platt-calibrated on the last training
    season (the same convention as goal_contribution_research)."""
    cal = seasons_tr == seasons_tr.max()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X_tr[~cal], y_tr[~cal])
    raw_te = np.clip(model.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)
    if len(np.unique(y_tr[cal])) < 2:
        return raw_te
    raw_cal = np.clip(model.predict_proba(X_tr[cal])[:, 1], 1e-6, 1 - 1e-6)
    platt = LogisticRegression(max_iter=1000).fit(np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1), y_tr[cal])
    return platt.predict_proba(np.log(raw_te / (1 - raw_te)).reshape(-1, 1))[:, 1]


def _count_predict(X_tr, y_tr, X_te) -> np.ndarray:
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000))
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _per_row(kind: str, y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(primary loss, secondary loss): log loss and Brier for binary markets,
    Poisson deviance and absolute error for counts."""
    if kind == "binary":
        return binary_log_loss(y, pred), (pred - y) ** 2
    return poisson_deviance(y, pred), np.abs(y - pred)


def _winsorise(train: np.ndarray, test: np.ndarray, lower_q: float, upper_q: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip every column to its training-fold quantiles (NaN -> 0 after).
    Applied identically to the baseline and to every candidate arm, so the
    two arms differ only by the candidate column. (Run 1 clipped candidates
    only; that gave the candidate arm a robustness advantage unrelated to
    ratio information.)"""
    lo = np.nanquantile(train, lower_q, axis=0)
    hi = np.nanquantile(train, upper_q, axis=0)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi), hi, 0.0)
    train = np.nan_to_num(np.clip(train, lo, hi), nan=0.0)
    test = np.nan_to_num(np.clip(test, lo, hi), nan=0.0)
    return train, test


def _fold(frame: pd.DataFrame, market: str, kind: str, base_features: list[str],
          candidate_names: list[str], train_seasons: list[str], test_season: str) -> dict:
    """Baseline and every candidate, fit on train_seasons and scored on
    test_season. Every design column (baseline and candidate alike) is
    winsorised at the training-fold 0.5th and 99.5th percentiles."""
    target = _TARGET[market]
    tr = frame["season"].isin(train_seasons).to_numpy()
    te = (frame["season"] == test_season).to_numpy()
    seasons_tr = frame.loc[tr, "season"].to_numpy()
    y_tr = frame.loc[tr, target].to_numpy(dtype=float)
    y_te = frame.loc[te, target].to_numpy(dtype=float)

    X_base = _design(frame, base_features)
    X_base_tr, X_base_te = _winsorise(X_base[tr], X_base[te], 1 - CLIP_Q, CLIP_Q)

    def predict(X_tr, X_te):
        return _binary_predict(X_tr, y_tr, seasons_tr, X_te) if kind == "binary" else _count_predict(X_tr, y_tr, X_te)

    base_pred = predict(X_base_tr, X_base_te)
    base_primary, base_secondary = _per_row(kind, y_te, base_pred)

    cand_all = frame[candidate_names].to_numpy(dtype=float)
    cand_tr, cand_te = cand_all[tr], cand_all[te]

    def one(j):
        c_tr, c_te = _winsorise(cand_tr[:, [j]], cand_te[:, [j]], 1 - CLIP_Q, CLIP_Q)
        pred = predict(np.column_stack([X_base_tr, c_tr]), np.column_stack([X_base_te, c_te]))
        primary, secondary = _per_row(kind, y_te, pred)
        return primary, secondary, pred

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(len(candidate_names))))
    return {
        "y": y_te,
        "base_primary": base_primary,
        "base_secondary": base_secondary,
        "base_pred": base_pred,
        "cand_primary": np.column_stack([r[0] for r in results]),
        "cand_secondary": np.column_stack([r[1] for r in results]),
        "cand_pred": np.column_stack([r[2] for r in results]),
    }


def _base_features(market: str) -> list[str]:
    return PRODUCTION_BASE + PRODUCTION_SHOT_RATES if market in ("shots", "sot") else list(PRODUCTION_BASE)


def selection_pass(frame: pd.DataFrame, names: list[str]) -> dict:
    report = {}
    for market, kind in MARKETS.items():
        folds = [_fold(frame, market, kind, _base_features(market), names,
                       SEASONS[:SEASONS.index(fold)], fold) for fold in SELECTION_FOLDS]
        diffs = np.vstack([f["cand_primary"] - f["base_primary"][:, None] for f in folds])
        fold_means = np.vstack([(f["cand_primary"] - f["base_primary"][:, None]).mean(axis=0) for f in folds])
        n = diffs.shape[0]
        mean_diff = diffs.mean(axis=0)
        se = diffs.std(axis=0, ddof=1) / math.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = -mean_diff / se
        p_raw = np.where(np.isfinite(z), norm.sf(z), 1.0)
        p_holm = holm_adjust(p_raw)
        sign_stable = (fold_means < 0).all(axis=0)
        passes = (p_holm < ALPHA) & sign_stable

        rows = [{
            "name": name,
            "mean_diff": float(mean_diff[j]),
            "se": float(se[j]) if np.isfinite(se[j]) else None,
            "z": float(z[j]) if np.isfinite(z[j]) else None,
            "p_raw": float(p_raw[j]),
            "p_holm": float(p_holm[j]),
            "fold_mean_diffs": [float(v) for v in fold_means[:, j]],
            "sign_stable_all_folds": bool(sign_stable[j]),
            "passes_selection": bool(passes[j]),
        } for j, name in enumerate(names)]
        ranked = sorted((r for r in rows if r["passes_selection"]), key=lambda r: r["mean_diff"])
        report[market] = {
            "kind": kind,
            "n_selection_rows": int(n),
            "baseline_mean_loss": float(np.mean(np.concatenate([f["base_primary"] for f in folds]))),
            "n_candidates": len(names),
            "n_passing": int(passes.sum()),
            "kept": [r["name"] for r in ranked[:MAX_KEPT_PER_MARKET]],
            "candidates": rows,
        }
    return report


def holdout_pass(frame: pd.DataFrame, kept: dict[str, list[str]]) -> dict:
    """The single holdout evaluation of the frozen kept set."""
    total = sum(len(v) for v in kept.values())
    alpha_adj = ALPHA / total
    rng = np.random.default_rng(SEED)
    tests = []
    for market, names in kept.items():
        if not names:
            continue
        kind = MARKETS[market]
        fold = _fold(frame, market, kind, _base_features(market), names, SEASONS[:-1], HOLDOUT_SEASON)
        y = fold["y"]
        base_ece = expected_calibration_error(y, fold["base_pred"]) if kind == "binary" else None
        for j, name in enumerate(names):
            d = fold["cand_primary"][:, j] - fold["base_primary"]
            idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
            boot = d[idx].mean(axis=1)
            lo, hi = np.quantile(boot, [alpha_adj / 2, 1 - alpha_adj / 2])
            sec = float((fold["cand_secondary"][:, j] - fold["base_secondary"]).mean())
            entry = {
                "market": market,
                "name": name,
                "n_holdout_rows": int(len(d)),
                "n_events": int(y.sum()) if kind == "binary" else None,
                "mean_loss_diff": float(d.mean()),
                "ci_level": 1 - alpha_adj,
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "secondary_metric": "brier" if kind == "binary" else "mae",
                "secondary_diff": sec,
            }
            if kind == "binary":
                cand_ece = expected_calibration_error(y, fold["cand_pred"][:, j])
                entry["ece_baseline"] = base_ece
                entry["ece_candidate"] = cand_ece
                ece_ok = cand_ece <= base_ece
            else:
                ece_ok = True
            entry["passes"] = bool(hi < 0 and sec <= 0 and ece_ok)
            tests.append(entry)
    return {"alpha_per_test": alpha_adj, "n_kept_total": total, "tests": tests}


def run_search(output_path: Path, evaluate_holdout: bool = False) -> dict:
    """Selection always runs. The holdout runs only with `evaluate_holdout`,
    which must be passed exactly once per season: a second look at the same
    season is not a holdout any more (see the preregistration amendment)."""
    frame, coverage = build_frame()
    defs = candidate_definitions()
    names = [d["name"] for d in defs]
    if len(names) > POOL_CAP:
        raise ValueError(f"candidate pool {len(names)} exceeds cap {POOL_CAP}")
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise ValueError(f"candidates not built: {missing[:5]}")

    selection = selection_pass(frame, names)
    kept = {m: r["kept"] for m, r in selection.items()}
    if not evaluate_holdout:
        holdout = {"status": "not_evaluated", "reason": "selection-only run; holdout season already read by run 1 (see preregistration amendment)"}
        passing = []
        verdict = "selection_only"
    elif any(kept.values()):
        holdout = holdout_pass(frame, kept)
        passing = [t for t in holdout["tests"] if t["passes"]]
        verdict = "pass" if passing else "negative"
    else:
        holdout = {"status": "not_evaluated", "reason": "no candidate passed selection"}
        passing = []
        verdict = "negative"

    result = {
        "preregistration": "docs/player_ratio_search_preregistration.md",
        "seasons": SEASONS,
        "selection_folds": SELECTION_FOLDS,
        "holdout_season": HOLDOUT_SEASON,
        "coverage": coverage,
        "not_built": {
            "key_passes": "absent from FPL history for 2020-21 to 2025-26; key-pass candidates dropped",
            "npxG_per90": "FPL history has no penalty-xG column; xG per 90 used as stand-in",
            "minutes_weighted": "minutes-weighted shot and SoT shares are built; plain per-90 rates are already minutes-weighted",
        },
        "pool_size": len(names),
        "candidate_definitions": defs,
        "selection": selection,
        "holdout": holdout,
        "verdict": verdict,
        "passing_candidates": [f"{t['market']}:{t['name']}" for t in passing],
    }
    output_path.write_text(json.dumps(result, indent=2, default=float))
    return result


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "docs" / "player_ratio_search_results.json"
    res = run_search(target)
    print(json.dumps({"pool_size": res["pool_size"], "verdict": res["verdict"], "coverage": res["coverage"],
                      "kept": {m: r["kept"] for m, r in res["selection"].items()}}, indent=2, default=float))
