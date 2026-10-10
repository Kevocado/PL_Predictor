"""Walk-forward audit of `player_goals.predict_player`'s shots / shots-on-target
fields (design spec 2026-09-04-player-shots-market-design.md section 7).

Run:  python -m pl_predictor.evaluate.player_shots_backtest --cache-dir data/cache

For every player-appearance (minutes > 0) in the held-out seasons, the
prediction uses only rows strictly before that match, through the real code
path: `player_form.blended_current_form` -> `player_goals.predict_player`.

Substitutions vs. the live app (documented, not hidden):
  * team_goal_expectation: live uses the scoreline model; here a trailing
    20-match attack x opposition-defence estimate from football-data results.
  * availability: FPL status is not archived historically -> 1.0.
  * expected_minutes: the real `fit_lineup_model` / `predict_lineup`, fit on the
    4 seasons before each test season (not refit within the season).
  * shots_scale: real `_shots_scale_from_context` fed the team's venue-specific
    last-10 shots-for (football-data hs/as) as of the match (shift(1)).
  * prior_season: last season's FPL totals for the same-named player (as
    `history_past` supplies live; it has no shots, so shots rates are
    current-season only, exactly as live).
  * Shots/SoT come from the Understat per-shot files, aggregated with the
    repo's own `_aggregate_player_shots`; matched to FPL rows by
    (team, date, normalised name). Unmatched shooters read as 0 shots.
"""
from __future__ import annotations

import argparse
import glob
import html
import math
import re
import types
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import fpl_history, understat_shots
from ..data.team_names import to_canonical
from ..features import player_form, rolling_form
from ..models import player_goals as pg

SEASON_FILES = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def _norm(s: str) -> str:
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKD", str(s)).casefold()
    return "".join(c for c in s if c.isalnum())


def _tokens(s: str) -> list[str]:
    s = html.unescape(str(s))
    return [t for t in re.split(r"\s+", unicodedata.normalize("NFKD", str(s)).casefold().strip()) if t]


# ---------------------------------------------------------------- data
def load_shots(cache: Path) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(cache / "understat_shots" / "[0-9]*.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        for h_a, grp in df.groupby("h_a"):
            team = grp["team_home"].iloc[0] if h_a == "h" else grp["team_away"].iloc[0]
            team = to_canonical(team, source="odds_api")
            date = pd.to_datetime(grp["date"].iloc[0]).date()
            for r in understat_shots._aggregate_player_shots(grp, date):
                r["team"] = team
                rows.append(r)
    return pd.DataFrame(rows)


def attach_shots(fpl: pd.DataFrame, us: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add shots / shots_on_target to FPL rows. Returns (frame, share of
    Understat shot-rows that matched an FPL row)."""
    fpl = fpl.copy()
    fpl["_date"] = pd.to_datetime(fpl["kickoff_time"]).dt.date
    fpl["_full"] = fpl["name"].map(_norm)
    fpl["_sur"] = fpl["name"].map(lambda n: _norm(_tokens(n)[-1]) if _tokens(n) else "")
    fpl["_fl"] = fpl["name"].map(lambda n: _norm(_tokens(n)[0] + _tokens(n)[-1]) if _tokens(n) else "")
    fpl["shots"] = 0.0
    fpl["shots_on_target"] = 0.0
    idx = {k: g.index for k, g in fpl[fpl.minutes > 0].groupby(["team", "_date"])}
    matched = total = 0
    for (team, date), g in us.groupby(["team", "date"]):
        cand = idx.get((team, date))
        if cand is None:
            continue
        total += len(g)
        c = fpl.loc[cand]
        for _, r in g.iterrows():
            nm = _norm(r["player"])
            toks = _tokens(r["player"])
            hit = c[c["_full"] == nm]
            if len(hit) != 1 and toks:
                hit = c[c["_fl"] == _norm(toks[0] + toks[-1])]
            if len(hit) != 1 and toks:
                hit = c[c["_sur"] == _norm(toks[-1])]
            if len(hit) != 1 and toks:  # every Understat token appears among the FPL name's tokens
                want = {_norm(t) for t in toks}
                hit = c[c["name"].map(lambda n: want <= {_norm(t) for t in _tokens(n)})]
            if len(hit) == 1:
                fpl.loc[hit.index[0], ["shots", "shots_on_target"]] = [r["shots"], r["shots_on_target"]]
                matched += 1
    return fpl.drop(columns=["_full", "_sur", "_fl"]), matched / max(total, 1)


def team_context(cache: Path) -> tuple[pd.DataFrame, dict]:
    """Per team-match: venue-specific last-10 shots-for (shift(1)) and a
    trailing-20 attack x opposition-defence goal expectation."""
    fd = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(cache / "football_data" / "20*.csv")))])
    fd["date"] = pd.to_datetime(fd["date"])
    fd = fd.sort_values("date").reset_index(drop=True)
    fd["goals_home"] = fd["goals_home"].fillna(fd["fthg"])
    fd["goals_away"] = fd["goals_away"].fillna(fd["ftag"])
    long, _ = rolling_form.build_rolling_form(fd)
    shots = {(r.team, r.date.date()): (r.home_last_10_shots_for, r.away_last_10_shots_for) for r in long.itertuples()}
    # goal expectation
    rows = []
    for r in fd.itertuples():
        rows.append((r.date, r.team_home, r.goals_home, r.goals_away))
        rows.append((r.date, r.team_away, r.goals_away, r.goals_home))
    g = pd.DataFrame(rows, columns=["date", "team", "gf", "ga"]).sort_values("date")
    g["gf20"] = g.groupby("team")["gf"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    g["ga20"] = g.groupby("team")["ga"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    lg = fd["goals_home"].shift(1).rolling(380, min_periods=50).mean()
    la = fd["goals_away"].shift(1).rolling(380, min_periods=50).mean()
    avg = ((lg + la) / 2).fillna(1.35)
    avg.index = fd.index
    fd["avg"] = avg
    gi = g.set_index(["team", "date"])
    gexp = {}
    for r in fd.itertuples():
        try:
            h, a = gi.loc[(r.team_home, r.date)], gi.loc[(r.team_away, r.date)]
            h, a = h.iloc[0] if isinstance(h, pd.DataFrame) else h, a.iloc[0] if isinstance(a, pd.DataFrame) else a
        except KeyError:
            continue
        av = r.avg
        def f(x, d):
            return d if pd.isna(x) else x
        gexp[(r.team_home, r.date.date())] = (av * (f(h.gf20, av) / av) * (f(a.ga20, av) / av))
        gexp[(r.team_away, r.date.date())] = (av * (f(a.gf20, av) / av) * (f(h.ga20, av) / av))
    return shots, gexp


# ---------------------------------------------------------------- predictions
def prior_season_dicts(prev: pd.DataFrame | None) -> dict:
    if prev is None:
        return {}
    out = {}
    for name, g in prev.groupby("name"):
        out[name] = {
            "minutes": g.minutes.sum(), "starts": g["starts"].sum() if "starts" in g else 0,
            "goals_scored": g.goals_scored.sum(), "assists": g.assists.sum(),
        }
    return out


def run_season(season: str, frames: dict, us_shots_fpl: dict, shots_ctx, gexp, lineup_model, priors) -> pd.DataFrame:
    df = us_shots_fpl[season].sort_values("kickoff_time").reset_index(drop=True)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    prior = prior_season_dicts(us_shots_fpl.get(frames[season]))
    out = []
    cols_needed = ["GW", "minutes", "goals_scored", "assists", "shots", "shots_on_target", "saves", "starts", "kickoff_time"]
    for el, g in df.groupby("element"):
        g = g.sort_values("kickoff_time").reset_index(drop=True)
        name, position = g["name"].iloc[0], g["position"].iloc[0]
        position = {"AM": "MID"}.get(position, position)
        for i in range(len(g)):
            row = g.iloc[i]
            if row.minutes <= 0:
                continue
            hist = g.iloc[:i]
            hist = hist[hist.kickoff_time < row.kickoff_time][[c for c in cols_needed if c in g.columns]]
            rates, _ = player_form.blended_current_form(hist, prior.get(name), position, priors)
            sf = player_form.current_start_features(hist, fallback_minutes=rates["avg_minutes"])
            exp_min = pg.predict_lineup(sf, lineup_model)["expected_minutes"]
            d = row.kickoff_time.date()
            is_home = bool(row.was_home)
            gx = gexp.get((row.team, d), pg.LEAGUE_AVERAGE_TEAM_GOALS)
            sh = shots_ctx.get((row.team, d), (np.nan, np.nan))[0 if is_home else 1]
            ctx = types.SimpleNamespace(form=pd.DataFrame({("home_last_10_shots_for" if is_home else "away_last_10_shots_for"): [sh]}, index=["home" if is_home else "away"]))
            kw = dict(expected_minutes=exp_min, position=position, is_home=is_home, team=row.team)
            full = pg.predict_player(rates, gx, 1.0, context=ctx, **kw)
            no_scale = pg.predict_player(rates, gx, 1.0, context=None, **kw)
            no_str = pg.predict_player(rates, pg.LEAGUE_AVERAGE_TEAM_GOALS, 1.0, context=ctx, **kw)
            neither = pg.predict_player(rates, pg.LEAGUE_AVERAGE_TEAM_GOALS, 1.0, context=None, **kw)
            mf = min(row.minutes / 90, 1.0)
            out.append(dict(
                season=season, element=el, position=position, team=row.team, date=d, minutes=row.minutes, exp_minutes=exp_min,
                shots=row.shots, sot=row.shots_on_target, games_before=int((hist.minutes > 0).sum()),
                rate_shots90=rates.get("shots_per90", 0.0), rate_sot90=rates.get("shots_on_target_per90", 0.0),
                pm_shots=full["expected_shots"], pm_sot=full["expected_shots_on_target"],
                noscale_shots=no_scale["expected_shots"], noscale_sot=no_scale["expected_shots_on_target"],
                nostr_shots=no_str["expected_shots"], nostr_sot=no_str["expected_shots_on_target"],
                own_shots=neither["expected_shots"], own_sot=neither["expected_shots_on_target"],
                # oracle-minutes variants: isolate the shot-RATE skill from minutes prediction
                pm_shots_om=full["expected_shots"] / max(min(exp_min / 90, 1.0), 1e-9) * mf if exp_min > 0 else 0.0,
                pm_sot_om=full["expected_shots_on_target"] / max(min(exp_min / 90, 1.0), 1e-9) * mf if exp_min > 0 else 0.0,
                own_shots_om=neither["expected_shots"] / max(min(exp_min / 90, 1.0), 1e-9) * mf if exp_min > 0 else 0.0,
                own_sot_om=neither["expected_shots_on_target"] / max(min(exp_min / 90, 1.0), 1e-9) * mf if exp_min > 0 else 0.0,
            ))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- metrics
def boot_diff(loss_a, loss_b, clusters, n=2000, seed=0):
    """Paired cluster bootstrap of mean(loss_a - loss_b); clusters = match ids."""
    d = pd.Series(np.asarray(loss_a) - np.asarray(loss_b)).groupby(np.asarray(clusters)).agg(["sum", "count"])
    s, c = d["sum"].values, d["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pois_ll(y, lam):
    lam = np.clip(lam, 1e-6, None)
    return lam - y * np.log(lam)  # Poisson NLL up to a constant


def prob_losses(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--test-seasons", default="2023-24,2024-25,2025-26")
    ap.add_argument("--out", default=None, help="optional CSV of per-row predictions")
    a = ap.parse_args()
    cache = Path(a.cache_dir)
    tests = a.test_seasons.split(",")

    us = load_shots(cache)
    print(f"understat shooter-rows: {len(us)}")
    fpl, frames = {}, {}
    cov = {}
    for s in SEASON_FILES:
        p = cache / "fpl_history" / f"{s}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["season"] = s
        d["team"] = d["team"].map(lambda t: to_canonical(t, source="fpl"))
        fpl[s], cov[s] = attach_shots(d, us)
    order = [s for s in SEASON_FILES if s in fpl]
    frames = {s: (order[order.index(s) - 1] if order.index(s) > 0 else None) for s in order}
    print("share of Understat shooter-rows matched to an FPL appearance:", {k: round(v, 3) for k, v in cov.items()})
    shots_ctx, gexp = team_context(cache)

    res = []
    for s in tests:
        if s not in fpl:
            print(f"skip {s}: no FPL file")
            continue
        train_seasons = order[max(0, order.index(s) - 4): order.index(s)]
        train = pd.concat([fpl[t] for t in train_seasons])
        train["kickoff_time"] = pd.to_datetime(train["kickoff_time"])
        orig = fpl_history.load_player_gw_history
        fpl_history.load_player_gw_history = lambda seasons=None, force_refresh=False, _t=train: _t.sort_values("kickoff_time")
        lm = pg.fit_lineup_model(seasons=train_seasons)
        fpl_history.load_player_gw_history = orig
        # position priors: live archive has no shots column -> none here either
        priors = player_form.position_rate_priors(train.drop(columns=["shots", "shots_on_target"]))
        r = run_season(s, frames, fpl, shots_ctx, gexp, lm, priors)
        # position-average shots per 90 from training seasons (naive baseline 1)
        t = train[train.minutes > 0]
        pa = t.groupby("position").apply(lambda g: pd.Series({
            "s90": g.shots.sum() / g.minutes.sum() * 90, "t90": g.shots_on_target.sum() / g.minutes.sum() * 90}), include_groups=False)
        r["posavg_shots"] = r.position.map(pa.s90) * (r.exp_minutes.clip(upper=90) / 90)
        r["posavg_sot"] = r.position.map(pa.t90) * (r.exp_minutes.clip(upper=90) / 90)
        r["posavg_shots_om"] = r.position.map(pa.s90) * (r.minutes.clip(upper=90) / 90)
        r["posavg_sot_om"] = r.position.map(pa.t90) * (r.minutes.clip(upper=90) / 90)
        r["const_shots"] = t.shots.sum() / t.minutes.sum() * 90 * (r.exp_minutes.clip(upper=90) / 90)
        r["const_sot"] = t.shots_on_target.sum() / t.minutes.sum() * 90 * (r.exp_minutes.clip(upper=90) / 90)
        res.append(r)
        print(f"{s}: {len(r)} appearances")
    R = pd.concat(res, ignore_index=True)
    R["match"] = R.team + "|" + R.date.astype(str)
    if a.out:
        R.to_csv(a.out, index=False)
    report(R)


def report(R):
    print(f"\nN appearances = {len(R)}  mean shots = {R.shots.mean():.3f}  mean SoT = {R.sot.mean():.3f}  zero-shot share = {(R.shots == 0).mean():.3f}")
    for tgt, y, preds in [
        ("SHOTS", R.shots, dict(model="pm_shots", no_shots_scale="noscale_shots", no_strength="nostr_shots",
                                own_trailing_per90="own_shots", position_avg="posavg_shots", constant="const_shots")),
        ("SHOTS ON TARGET", R.sot, dict(model="pm_sot", no_shots_scale="noscale_sot", no_strength="nostr_sot",
                                        own_trailing_per90="own_sot", position_avg="posavg_sot", constant="const_sot")),
    ]:
        print(f"\n=== {tgt}: MAE / RMSE / Poisson-NLL / bias(mean pred - mean actual) ===")
        for k, c in preds.items():
            e = R[c] - y
            print(f"{k:20s} MAE {e.abs().mean():.4f}  RMSE {math.sqrt((e**2).mean()):.4f}  NLL {pois_ll(y.values, R[c].values).mean():.4f}  bias {e.mean():+.4f}")
        print("-- paired cluster bootstrap, model minus X (negative = model better), 95% CI")
        for k, c in preds.items():
            if k == "model":
                continue
            for lname, fn in [("abs err", lambda p: (p - y).abs()), ("sq err", lambda p: (p - y) ** 2), ("NLL", lambda p: pd.Series(pois_ll(y.values, p.values)))]:
                m, lo, hi = boot_diff(fn(R["pm_shots" if tgt == "SHOTS" else "pm_sot"]), fn(R[c]), R.match)
                print(f"  vs {k:18s} d{lname:8s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        # oracle minutes
        om = "shots" if tgt == "SHOTS" else "sot"
        print("-- ORACLE minutes (actual minutes plugged in): shot-rate skill only")
        for k, c in [("model", f"pm_{om}_om"), ("own_trailing_per90", f"own_{om}_om"), ("position_avg", f"posavg_{om}_om")]:
            e = R[c] - y
            print(f"{k:20s} MAE {e.abs().mean():.4f}  RMSE {math.sqrt((e**2).mean()):.4f}  NLL {pois_ll(y.values, R[c].values).mean():.4f}")
        for k, c in [("own_trailing_per90", f"own_{om}_om"), ("position_avg", f"posavg_{om}_om")]:
            m, lo, hi = boot_diff((R[f"pm_{om}_om"] - y) ** 2, (R[c] - y) ** 2, R.match)
            print(f"  vs {k:18s} dsq err   {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        # calibration by decile
        pc = preds["model"]
        R["_dec"] = pd.qcut(R[pc].rank(method="first"), 10, labels=False)
        cal = R.groupby("_dec").agg(pred=(pc, "mean"), actual=(tgt.split()[0].lower() if False else ("shots" if tgt == "SHOTS" else "sot"), "mean"), n=(pc, "size"))
        print("-- calibration by prediction decile (mean pred vs mean actual)")
        print(cal.round(3).to_string())
        print(f"   corr(pred, actual) model {np.corrcoef(R[pc], y)[0,1]:.3f}  own_trailing {np.corrcoef(R[preds['own_trailing_per90']], y)[0,1]:.3f}  position_avg {np.corrcoef(R[preds['position_avg']], y)[0,1]:.3f}")

    print("\n=== P(SoT >= 1): Brier / log loss ===")
    yb = (R.sot >= 1).astype(float).values
    print(f"base rate actual {yb.mean():.3f}")
    ps = {k: 1 - np.exp(-R[c].values) for k, c in dict(model="pm_sot", no_shots_scale="noscale_sot", no_strength="nostr_sot",
                                                         own_trailing_per90="own_sot", position_avg="posavg_sot", constant="const_sot").items()}
    L = {}
    for k, p in ps.items():
        b, ll = prob_losses(yb, p)
        L[k] = (b, ll)
        print(f"{k:20s} Brier {b.mean():.4f}  logloss {ll.mean():.4f}  mean p {p.mean():.3f}")
    for k in ps:
        if k == "model":
            continue
        for i, n in enumerate(["Brier", "logloss"]):
            m, lo, hi = boot_diff(L["model"][i], L[k][i], R.match)
            print(f"  model - {k:18s} d{n:8s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    bins = [0, .05, .1, .2, .3, .4, .5, .6, 1.01]
    bk = pd.cut(ps["model"], bins, right=False)
    print("-- reliability (model):")
    print(pd.DataFrame({"p": ps["model"], "y": yb, "b": bk}).groupby("b", observed=True).agg(mean_p=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).to_string())

    # by experience
    R["exp"] = pd.cut(R.games_before, [-1, 0, 4, 9, 99], labels=["0 prior games", "1-4", "5-9", "10+"])
    print("\n=== by prior current-season appearances (shots): model MAE vs position_avg MAE vs own MAE, mean pred, mean actual ===")
    print(R.groupby("exp", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "model_mae": (g.pm_shots - g.shots).abs().mean(), "posavg_mae": (g.posavg_shots - g.shots).abs().mean(),
        "mean_pred": g.pm_shots.mean(), "mean_actual": g.shots.mean()}), include_groups=False).round(3).to_string())


if __name__ == "__main__":
    main()
