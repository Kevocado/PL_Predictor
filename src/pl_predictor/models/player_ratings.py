"""Role-aware, evidence-weighted Player Hub ratings.

The scores are descriptive, not forecasts.  They consume only the cached FPL
bootstrap response, so opening Player Hub never fans out into player-history
requests or changes a fixture prediction.  Quality is a fixed role-aware
scale, Form is a deliberately constrained short-term lift, and Impact is the
only score affected by availability for the next gameweek.
"""

from __future__ import annotations

from functools import lru_cache
import re
import unicodedata

import pandas as pd

from ..config import FPL_HISTORY_CACHE_DIR
from ..data import fpl_history


ROLE_PRIORS = {"GK": 50.0, "DEF": 50.0, "MID": 50.0, "FWD": 50.0}
# Every role's independently-tuned components must reach the same ceiling for
# equally elite play. DEF's former 36-point ceiling is the conservative floor:
# the other roles are only scaled down, never inflated above their old caps.
ROLE_CAP_SCALE = {"GK": 36.0 / 38.0, "DEF": 1.0, "MID": 36.0 / 39.0, "FWD": 36.0 / 43.0}
# A prior-season rating is useful, but a player may have changed club, role,
# manager, or game-time status. Before current-season evidence arrives it is
# therefore shrunk toward the neutral role baseline rather than trusted whole.
PRIOR_TRUST = 0.65
ROLE_LABELS = {
    "GK": {"saves": "Saves and shot prevention", "clean_sheets": "Clean sheets", "bps": "Bonus-point system"},
    "DEF": {"clean_sheets": "Clean sheets", "defensive_contribution": "Defensive contribution", "expected_goal_involvements": "Expected goal involvements"},
    "MID": {"expected_goal_involvements": "Expected goal involvements", "expected_assists": "Chance creation", "threat": "Attacking threat"},
    "FWD": {"expected_goals": "Expected goals", "expected_goal_involvements": "Expected goal involvements", "goals_scored": "Goals per 90"},
}


def _number(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _availability(element: dict) -> float:
    if element.get("status", "a") in {"i", "s", "u"}:
        return 0.0
    if element.get("status") == "d":
        return _number(element.get("chance_of_playing_next_round")) / 100 if element.get("chance_of_playing_next_round") is not None else 0.5
    return 1.0


def _expected_minutes(element: dict, availability: float) -> float:
    minutes = _number(element.get("minutes"))
    starts = _number(element.get("starts"))
    appearances = max(_number(element.get("appearances")), starts, minutes / 60, 1)
    return min(90.0, (25.0 + 65.0 * min(1.0, starts / appearances)) * availability)


def _per90(element: dict, field: str) -> float:
    return _number(element.get(field)) / max(_number(element.get("minutes")), 1.0) * 90.0


def _role_components(element: dict, position: str) -> dict[str, float]:
    """Fixed-scale components.  They are deliberately not percentiles."""
    xg = _per90(element, "expected_goals")
    xa = _per90(element, "expected_assists")
    xgi = _per90(element, "expected_goal_involvements")
    goals = _per90(element, "goals_scored")
    if position == "GK":
        raw = {
            "saves": min(14.0, _per90(element, "saves") * 3.0),
            "clean_sheets": min(14.0, _per90(element, "clean_sheets") * 14.0),
            "bps": min(10.0, _per90(element, "bps") * 1.1),
        }
    elif position == "DEF":
        raw = {
            "clean_sheets": min(12.0, _per90(element, "clean_sheets") * 12.0),
            "defensive_contribution": min(8.0, _per90(element, "defensive_contribution") * 0.08),
            "expected_goal_involvements": min(10.0, xgi * 16.0),
            "bps": min(6.0, _per90(element, "bps") * 0.6),
        }
    elif position == "FWD":
        raw = {
            "expected_goals": min(22.0, xg * 28.0),
            "expected_goal_involvements": min(14.0, xgi * 13.0),
            "goals_scored": min(7.0, goals * 7.0),
        }
    else:
        raw = {
            "expected_goal_involvements": min(18.0, xgi * 18.0),
            "expected_assists": min(8.0, xa * 8.0),
            "threat": min(4.0, _per90(element, "threat") * 0.03),
            "creativity": min(4.0, _per90(element, "creativity") * 0.03),
            "goals_scored": min(5.0, goals * 5.0),
        }
    return {name: value * ROLE_CAP_SCALE.get(position, 1.0) for name, value in raw.items()}


def _quality_score(element: dict, position: str, prior_quality: float | None = None) -> tuple[float, str]:
    components = _role_components(element, position)
    strongest = max(components, key=components.get, default="expected_goal_involvements")
    raw = min(92.0, ROLE_PRIORS.get(position, 50.0) + sum(components.values()))
    minutes, starts = _number(element.get("minutes")), _number(element.get("starts"))
    # A role score needs roughly a full season's opportunity to become a
    # durable Quality assessment.  This stops a good 10-match spell from
    # inheriting an elite score while still recognising a full elite season.
    confidence = min(1.0, (minutes / 1800.0 + starts / 20.0) / 2.0)
    role_baseline = ROLE_PRIORS.get(position, 50.0)
    prior = (
        prior_quality * PRIOR_TRUST + role_baseline * (1.0 - PRIOR_TRUST)
        if prior_quality is not None
        else role_baseline
    )
    quality = prior * (1.0 - confidence) + raw * confidence
    label = ROLE_LABELS.get(position, ROLE_LABELS["MID"]).get(strongest, strongest.replace("_", " ").title())
    return round(min(92.0, max(0.0, quality)), 1), label


def _form_score(element: dict, position: str) -> float:
    """Return an earned 0–15 lift, gated by real opportunity evidence."""
    minutes, starts = _number(element.get("minutes")), _number(element.get("starts"))
    if minutes < 360.0 or starts < 4.0:
        return 0.0
    xgi = _per90(element, "expected_goal_involvements")
    output = _per90(element, "goals_scored") + _per90(element, "assists")
    opportunity = min(1.0, (minutes / 900.0 + starts / 10.0) / 2.0)
    if position == "GK":
        underlying = min(1.0, (_per90(element, "saves") / 4.0 + _per90(element, "clean_sheets")) / 2.0)
        actual = min(1.0, (_per90(element, "clean_sheets") + _per90(element, "bps") / 12.0) / 2.0)
    elif position == "DEF":
        underlying = min(1.0, (xgi / 0.45 + _per90(element, "clean_sheets")) / 2.0)
        actual = min(1.0, (output / 0.45 + _per90(element, "clean_sheets")) / 2.0)
    elif position == "FWD":
        underlying = min(1.0, (_per90(element, "expected_goals") / 0.75 + xgi / 0.95) / 2.0)
        actual = min(1.0, output / 0.85)
    else:
        underlying = min(1.0, xgi / 0.75)
        actual = min(1.0, output / 0.75)
    return round(min(15.0, 6.0 * underlying + 5.0 * actual + 4.0 * opportunity), 1)


def _live_fpl_form_score(element: dict, position: str) -> float:
    """Recent official-FPL form on a common 0–100 scale.

    FPL's ``form`` is valuable immediate information, but a single standout
    appearance must not look elite.  Short samples can clear a player's
    durable Quality, but their live-score ceiling rises only with starts and
    minutes.
    """
    raw_form = _number(element.get("form"))
    minutes, starts = _number(element.get("minutes")), _number(element.get("starts"))
    role_baseline = {"GK": 3.5, "DEF": 3.5, "MID": 4.0, "FWD": 4.0}.get(position, 4.0)
    unshrunk = min(100.0, max(0.0, 50.0 + (raw_form - role_baseline) * 9.0))
    evidence = min(1.0, (minutes / 900.0 + starts / 10.0) / 2.0)
    ceiling = 70.0 + 30.0 * evidence
    return round(min(unshrunk, ceiling), 1)


def _normalise_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


def _element_name_key(element: dict) -> str:
    full_name = " ".join(str(element.get(part, "")).strip() for part in ("first_name", "second_name")).strip()
    return _normalise_name(full_name or str(element.get("web_name", "")))


@lru_cache(maxsize=1)
def cached_historical_priors() -> dict[str, dict[str, float]]:
    """Build one local-file prior index; never fetch player history on visit."""
    frames = []
    for season in fpl_history.default_completed_seasons(n=3):
        path = FPL_HISTORY_CACHE_DIR / f"{season}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["season"] = season
            frames.append(frame)
    if not frames:
        return {}
    return build_historical_priors(pd.concat(frames, ignore_index=True))


_HISTORICAL_PRIOR_COLUMNS = (
    "minutes", "starts", "goals_scored", "assists", "expected_goals",
    "expected_assists", "expected_goal_involvements", "clean_sheets",
    "saves", "goals_conceded", "defensive_contribution", "bps",
)
_HISTORICAL_RECENCY_WEIGHTS = (0.15, 0.30, 0.55)
_PROVISIONAL_MINUTES = 900.0
_FULL_EVIDENCE_MINUTES = 1800.0


def _latest_positioned_seasons(history: pd.DataFrame, limit: int = 3) -> list[str]:
    if "season" not in history or "position" not in history:
        return []
    positioned = history[history["position"].isin(ROLE_PRIORS)]
    return sorted(str(season) for season in positioned["season"].dropna().unique())[-limit:]


def _history_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _historical_name_aliases(name_key: str) -> set[str]:
    """Conservative aliases for FPL archive full names versus bootstrap names."""
    parts = name_key.split()
    aliases = {name_key}
    if len(parts) >= 2:
        aliases.add(f"{parts[0]} {parts[-1]}")
        aliases.add(" ".join(parts[:2]))
    return aliases


def _aggregate_player_seasons(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only observed pre-existing player-season data.

    Team defensive baselines are calculated from goalkeeper minutes before
    collapsing player rows. That prevents raw saves or a team clean-sheet
    record from being treated as an individual player's whole ability.
    """
    rows = history.copy()
    rows["name_key"] = rows["name"].astype(str).map(_normalise_name)
    rows = rows[(rows["name_key"] != "") & rows["position"].isin(ROLE_PRIORS)].copy()
    for column in _HISTORICAL_PRIOR_COLUMNS:
        rows[column] = _history_numeric(rows, column)
    rows = rows[rows["minutes"] > 0].copy()
    if rows.empty:
        return rows

    gk = rows[rows["position"] == "GK"]
    team_baseline = gk.groupby(["season", "team"], as_index=False).agg(
        team_gk_minutes=("minutes", "sum"), team_gk_goals_conceded=("goals_conceded", "sum"),
    )
    team_baseline["team_goals_conceded_per90"] = (
        team_baseline["team_gk_goals_conceded"] * 90.0 / team_baseline["team_gk_minutes"].clip(lower=1.0)
    )
    rows = rows.merge(
        team_baseline[["season", "team", "team_goals_conceded_per90"]],
        on=["season", "team"], how="left",
    )
    rows["team_goals_conceded_per90"] = rows["team_goals_conceded_per90"].fillna(
        rows["goals_conceded"] * 90.0 / rows["minutes"].clip(lower=1.0)
    )
    rows["prevention_above_team"] = (
        rows["team_goals_conceded_per90"] * rows["minutes"] / 90.0 - rows["goals_conceded"]
    )
    grouped = rows.groupby(["name_key", "position", "season"], as_index=False).agg(
        **{column: (column, "sum") for column in _HISTORICAL_PRIOR_COLUMNS},
        prevention_above_team=("prevention_above_team", "sum"),
    )
    grouped["prevention_per90"] = grouped["prevention_above_team"] * 90.0 / grouped["minutes"].clip(lower=1.0)
    return grouped


def _historical_role_evidence(row: pd.Series) -> tuple[float, str]:
    minutes = max(float(row["minutes"]), 1.0)

    def per90(column: str) -> float:
        return float(row.get(column, 0.0)) * 90.0 / minutes

    xg, xa, xgi = per90("expected_goals"), per90("expected_assists"), per90("expected_goal_involvements")
    output = per90("goals_scored") + per90("assists")
    bps = per90("bps")
    position = str(row["position"])
    if position == "GK":
        components = {
            "clean_sheets": per90("clean_sheets") * 1.8,
            "shot_prevention": max(-1.5, min(1.5, float(row["prevention_per90"]))) * 2.6,
            "bonus_point_system": bps * 0.04,
            "save_support": min(4.0, per90("saves")) * 0.05,
        }
    elif position == "DEF":
        components = {
            "team_adjusted_prevention": max(-1.5, min(1.5, float(row["prevention_per90"]))) * 1.8,
            "clean_sheets": per90("clean_sheets") * 1.4,
            "defensive_contribution": per90("defensive_contribution") * 0.025,
            "expected_goal_involvements": xgi * 1.5,
            "bonus_point_system": bps * 0.025,
        }
    elif position == "MID":
        components = {
            "expected_goal_involvements": xgi * 1.8,
            "chance_creation": xa * 1.1,
            "goal_contributions": output * 0.6,
            "bonus_point_system": bps * 0.02,
        }
    else:
        components = {
            "expected_goals": xg * 1.8,
            "expected_goal_involvements": xgi * 1.1,
            "goal_contributions": output * 0.6,
            "bonus_point_system": bps * 0.02,
        }
    driver = max(components, key=components.get)
    return sum(components.values()), driver


def _calibrate_role_priors(player_seasons: pd.DataFrame, seasons: list[str]) -> dict[str, dict[str, float | str]]:
    if player_seasons.empty:
        return {}
    evidence = player_seasons.copy()
    values = evidence.apply(_historical_role_evidence, axis=1)
    evidence["raw_role_evidence"] = [value for value, _ in values]
    evidence["rating_driver_key"] = [driver for _, driver in values]
    qualifying = evidence[evidence["minutes"] >= _PROVISIONAL_MINUTES]
    anchors = qualifying.groupby("position")["raw_role_evidence"].agg(
        role_median="median",
        role_q25=lambda series: series.quantile(0.25),
        role_q75=lambda series: series.quantile(0.75),
    )
    evidence = evidence.join(anchors, on="position")
    evidence["role_median"] = evidence["role_median"].fillna(evidence.groupby("position")["raw_role_evidence"].transform("median"))
    evidence["role_iqr"] = (evidence["role_q75"] - evidence["role_q25"]).fillna(0.0).clip(lower=0.20)
    evidence["season_quality"] = (
        # A one-IQR edge over a normal qualifying Premier League season is
        # strong (75), not automatically world class. The 25-point slope
        # leaves 85+ for repeated far-above-distribution evidence.
        50.0 + 25.0 * (evidence["raw_role_evidence"] - evidence["role_median"]) / evidence["role_iqr"]
    ).clip(lower=30.0, upper=95.0)

    season_weights = {season: weight for season, weight in zip(seasons, _HISTORICAL_RECENCY_WEIGHTS[-len(seasons):])}
    evidence["recency_weight"] = evidence["season"].map(season_weights).fillna(0.0)
    evidence["weighted_minutes"] = evidence["minutes"].clip(upper=_FULL_EVIDENCE_MINUTES) * evidence["recency_weight"]
    priors: dict[str, dict[str, float | str]] = {}
    for name_key, group in evidence.groupby("name_key"):
        evidence_minutes = float(group["minutes"].sum())
        weighted_minutes = float(group["weighted_minutes"].sum())
        if weighted_minutes:
            weighted_quality = float((group["season_quality"] * group["weighted_minutes"]).sum() / weighted_minutes)
        else:
            weighted_quality = 50.0
        confidence = min(1.0, evidence_minutes / _FULL_EVIDENCE_MINUTES)
        quality = 50.0 + (weighted_quality - 50.0) * confidence
        qualifying_seasons = int((group["minutes"] >= _PROVISIONAL_MINUTES).sum())
        status = "established" if evidence_minutes >= _PROVISIONAL_MINUTES else "provisional"
        # A single full season is enough to be useful, but not enough to
        # represent a sustained world-class level. Current Form can still
        # add a separate lift once the player has actually featured.
        if qualifying_seasons < 2:
            quality = min(76.0, quality)
        latest = group.sort_values("season").iloc[-1]
        priors[name_key] = {
            "quality_rating": round(float(min(95.0, max(30.0, quality))), 1),
            "evidence_minutes": round(evidence_minutes, 1),
            "rating_status": status,
            "rating_driver": ROLE_LABELS.get(str(latest["position"]), {}).get(
                str(latest["rating_driver_key"]), str(latest["rating_driver_key"]).replace("_", " ").title()
            ),
        }
    aliases: dict[str, set[str]] = {}
    for name_key in priors:
        for alias in _historical_name_aliases(name_key):
            aliases.setdefault(alias, set()).add(name_key)
    for alias, candidates in aliases.items():
        if len(candidates) == 1:
            priors.setdefault(alias, dict(priors[next(iter(candidates))]))
    return priors


def build_historical_priors(history: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    """Return a stable, local-file-only historical ability index."""
    seasons = _latest_positioned_seasons(history)
    if not seasons:
        return {}
    return _calibrate_role_priors(_aggregate_player_seasons(history[history["season"].astype(str).isin(seasons)]), seasons)


def rate_bootstrap_elements(
    elements: list[dict], positions: dict[int, str], historical_priors: dict[str, dict[str, float]] | None = None
) -> dict[int, dict]:
    """Return data-led Quality, Form, Overall, and Impact per element."""
    result: dict[int, dict] = {}
    for element in elements:
        position = positions.get(int(element.get("element_type", 0)), "MID")
        prior = (historical_priors or {}).get(_element_name_key(element), {})
        # A legacy caller may pass only a numeric prior, which remains an
        # established prior for backwards-compatible notebook use. Real
        # cached records always include an explicit evidence status.
        status = prior.get("rating_status", "established" if prior.get("quality_rating") is not None else "provisional")
        _, current_driver = _quality_score(element, position, prior_quality=prior.get("quality_rating"))
        # Quality is deliberately durable multi-season ability. Current
        # season output earns the separate Form lift, rather than replacing
        # a validated prior after one or two matches.
        quality = float(prior["quality_rating"]) if status == "established" else None
        driver = prior.get("rating_driver") or current_driver
        form = _form_score(element, position)
        live_form = _live_fpl_form_score(element, position)
        overall = round(min(95.0, quality + form), 1) if quality is not None else None
        availability = _availability(element)
        expected_minutes = _expected_minutes(element, availability)
        impact_source = overall if overall is not None else live_form
        impact = round(impact_source * availability * expected_minutes / 90.0, 1)
        result[int(element["id"])] = {
            "quality_rating": quality,
            "form_rating": form,
            "live_form_rating": live_form,
            "live_form_vs_quality": round(live_form - quality, 1) if quality is not None else None,
            "overall_rating": overall,
            "current_impact_rating": impact,
            "rating_driver": driver,
            "rating_expected_minutes": round(expected_minutes, 1),
            "rating_status": status,
            "rating_evidence_minutes": round(float(prior.get("evidence_minutes", 0.0)), 1),
            "rating_model_source": "data_led_multiseason_role_evidence",
        }
    return result


ROLE_TARGETS = {
    "GK": "shot_prevention",
    "DEF": "defence_and_attack",
    "MID": "creation_and_output",
    "FWD": "finishing_and_output",
}


def _historical_role_target(rows: pd.DataFrame) -> pd.Series:
    """Role-specific descriptive targets, never live-serving outputs."""
    minutes = rows["minutes"].clip(lower=1)
    def rate(column: str) -> pd.Series:
        values = rows[column] if column in rows else pd.Series(0.0, index=rows.index)
        return pd.to_numeric(values, errors="coerce").fillna(0.0) / minutes * 90.0
    position = rows["position"].fillna("MID")
    target = pd.Series(0.0, index=rows.index)
    target.loc[position == "GK"] = (rate("saves") * .40 + rate("clean_sheets") * 3.0 + rate("bps") * .08)[position == "GK"]
    target.loc[position == "DEF"] = (rate("clean_sheets") * 3.0 + rate("defensive_contribution") * .02 + rate("expected_goal_involvements") * 1.5)[position == "DEF"]
    target.loc[position == "MID"] = (rate("expected_goal_involvements") * 1.4 + (rate("goals_scored") + rate("assists")) * .8)[position == "MID"]
    target.loc[position == "FWD"] = (rate("expected_goals") * 1.4 + rate("expected_goal_involvements") + rate("goals_scored") * .8)[position == "FWD"]
    return target


def evaluate_role_models(history: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward role-model comparison using only role-specific targets.

    This offline report intentionally cannot influence serving.  Rich models
    must improve MAE and not regress RMSE in every chronological fold before
    their report row is labelled ``rich``.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from ..features.player_form import build_historical_player_form

    rows, feature_cols = build_historical_player_form(history)
    rows = rows.copy()
    rows["role_target"] = _historical_role_target(rows)
    seasons = sorted(rows["season"].dropna().unique())
    report = []
    for position in ("GK", "DEF", "MID", "FWD"):
        role = rows[rows["position"] == position].copy()
        folds, drivers = [], []
        for index in range(2, len(seasons)):
            train = role[role["season"].isin(seasons[:index])]
            validation = role[role["season"] == seasons[index]]
            if len(train) < 30 or validation.empty:
                continue
            X_train = train[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0)
            X_validation = validation[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0)
            baseline = float(train["role_target"].mean())
            baseline_prediction = [baseline] * len(validation)
            model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
            model.fit(X_train, train["role_target"])
            rich_prediction = model.predict(X_validation)
            folds.append(
                {
                    "season": seasons[index],
                    "baseline_mae": mean_absolute_error(validation["role_target"], baseline_prediction),
                    "rich_mae": mean_absolute_error(validation["role_target"], rich_prediction),
                    "baseline_rmse": mean_squared_error(validation["role_target"], baseline_prediction) ** .5,
                    "rich_rmse": mean_squared_error(validation["role_target"], rich_prediction) ** .5,
                    "n_train": len(train),
                    "n_validation": len(validation),
                }
            )
            coefficients = model.named_steps["ridge"].coef_
            drivers.append(feature_cols[int(abs(coefficients).argmax())])
        if not folds:
            continue
        fold_report = pd.DataFrame(folds)
        selected = bool(((fold_report["rich_mae"] < fold_report["baseline_mae"]) & (fold_report["rich_rmse"] <= fold_report["baseline_rmse"])).all())
        report.append(
            {
                "position": position,
                "target": ROLE_TARGETS[position],
                "validation_seasons": ",".join(fold_report["season"]),
                "baseline_mae": float(fold_report["baseline_mae"].mean()),
                "rich_mae": float(fold_report["rich_mae"].mean()),
                "baseline_rmse": float(fold_report["baseline_rmse"].mean()),
                "rich_rmse": float(fold_report["rich_rmse"].mean()),
                "selected_model": "rich" if selected else "baseline",
                "top_driver": pd.Series(drivers).mode().iat[0],
                "n_train": int(fold_report["n_train"].mean()),
                "n_validation": int(fold_report["n_validation"].mean()),
            }
        )
    return pd.DataFrame(report)
