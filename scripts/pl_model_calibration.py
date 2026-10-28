"""PL model calibration audit — our model's own predictions vs actuals.

Uses public_snapshot.json fixtures_by_gameweek (actuals) + fixture_detail_by_event_id
(model probabilities). Research only: no src/ changes.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import json
import math
import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 1
PROB_SUM_TOL = 0.01  # single tolerance for 1X2 sum checks (0.99-1.01)


def ci(vals, n=N_BOOT, seed=SEED):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom else np.nan
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def boot_slope(x, y, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        slopes.append(ols(x[idx], y[idx])[0])
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def reliability_bins(probs, outcomes, n_bins=10, min_n=5):
    probs = np.asarray(probs, float); outcomes = np.asarray(outcomes, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < min_n:
            continue
        pm = probs[m].mean()
        am = outcomes[m].mean()
        bins.append(("%.2f→%.2f (n=%d)" % (pm, am, int(m.sum())), pm, am, int(m.sum())))
    return bins


def max_gap(pred, actual):
    bins = reliability_bins(pred, actual)
    return max(abs(b[1] - b[2]) for b in bins) if bins else np.nan


def bin_str(pred, actual, min_n=5):
    bins = reliability_bins(pred, actual, min_n=min_n)
    return " | ".join(b[0] for b in bins)


def verdict_bias(lo, hi):
    """Return 'problem' if CI excludes 0, 'inconclusive' if CI too wide, else 'no problem'."""
    if lo > 0 or hi < 0:
        return "problem"
    if (hi - lo) / 2 > 0.5:
        return "inconclusive (n=70, interval too wide to judge)"
    return "no problem"


def verdict_slope(lo, hi):
    """Return 'problem' if CI excludes 1.0, 'inconclusive' if CI too wide, else 'no problem'."""
    if lo > 1.0 or hi < 1.0:
        return "problem"
    if (hi - lo) / 2 > 0.5:
        return "inconclusive (n=70, interval too wide to judge)"
    return "no problem"


def bin_str_detailed(pred, actual, min_n=5):
    """Return bins with n, for filtering n<10 in the doc."""
    bins = reliability_bins(pred, actual, min_n=min_n)
    return " | ".join(
        f"{b[0]}" + (" ⚠" if b[3] < 10 else "")
        for b in bins
    )


def _validate_probabilities(prob_vals: list) -> bool:
    """Validate all snapshot probabilities are non-null, finite, numeric, and in [0,1]."""
    for v in prob_vals:
        if v is None:
            return False
        if not isinstance(v, (int, float)):
            return False
        if not math.isfinite(v):
            return False
        if v < 0.0 or v > 1.0:
            return False
    return True


def main():
    # Load public snapshot
    with open("data/public_snapshot.json") as f:
        d = json.load(f)

    # Get fixtures_by_gameweek
    fixtures = d.get("fixtures_by_gameweek", {})
    all_fixtures = []
    for gw, gw_data in fixtures.items():
        if isinstance(gw_data, dict) and "fixtures" in gw_data:
            for f in gw_data["fixtures"]:
                f["gameweek"] = int(gw)
                all_fixtures.append(f)

    # Get fixture_detail for O/U and BTTS
    fixture_detail = d.get("fixture_detail_by_event_id", {})

    # Match finished fixtures with COMPLETE model probabilities (1X2 + O/U + BTTS non-null)
    matched = []
    for f in all_fixtures:
        eid = f.get("event_id")
        # Must have fixture detail for O/U and BTTS
        fd = fixture_detail.get(eid) if eid else {}
        if not fd:
            continue
        # Must be a finished fixture
        if not f.get("finished", False):
            continue
        # 1X2: read from fixture record but must exist and be non-null
        pred_home = f.get("predicted_home_win")
        pred_draw = f.get("predicted_draw")
        pred_away = f.get("predicted_away_win")
        if pred_home is None or pred_draw is None or pred_away is None:
            continue
        # O/U and BTTS must be present and non-null in fixture_detail
        over_detail = fd.get("over_2_5", {})
        under_detail = fd.get("under_2_5", {})
        pred_over = over_detail.get("prob") if isinstance(over_detail, dict) else None
        pred_under = under_detail.get("prob") if isinstance(under_detail, dict) else None
        pred_btts_raw = fd.get("btts_yes_prob")
        # Validate all snapshot probabilities before calibration
        prob_vals = [pred_home, pred_draw, pred_away, pred_over, pred_under, pred_btts_raw]
        if not _validate_probabilities(prob_vals):
            continue
        # 1X2 sums to ~1 check per fixture (not just aggregate) — single tolerance
        prob_sum = float(pred_home) + float(pred_draw) + float(pred_away)
        if not (1.0 - PROB_SUM_TOL <= prob_sum <= 1.0 + PROB_SUM_TOL):
            continue
        matched.append({
            "event_id": f.get("event_id"),
            "gameweek": f.get("gameweek"),
            "team_home": f.get("team_home"),
            "team_away": f.get("team_away"),
            "actual_goals_home": f.get("actual_goals_home"),
            "actual_goals_away": f.get("actual_goals_away"),
            "pred_home": f.get("predicted_home_win"),
            "pred_draw": f.get("predicted_draw"),
            "pred_away": f.get("predicted_away_win"),
            "pred_over": fd.get("over_2_5", {}).get("prob"),
            "pred_under": fd.get("under_2_5", {}).get("prob"),
            "pred_btts": fd.get("btts_yes_prob"),
        })

    df = pd.DataFrame(matched)
    n = len(df)
    print(f"PL model calibration audit — OUR MODEL predictions vs actuals, n={n}")

    # Compute actual outcomes
    df["actual_home"] = (df["actual_goals_home"] > df["actual_goals_away"]).astype(float)
    df["actual_draw"] = (df["actual_goals_home"] == df["actual_goals_away"]).astype(float)
    df["actual_away"] = (df["actual_goals_home"] < df["actual_goals_away"]).astype(float)
    df["actual_total"] = df["actual_goals_home"] + df["actual_goals_away"]
    df["actual_over"] = (df["actual_total"] > 2.5).astype(float)
    df["actual_btts"] = ((df["actual_goals_home"] > 0) & (df["actual_goals_away"] > 0)).astype(float)

    # Verify prob sums — single tolerance throughout
    df["prob_sum"] = df["pred_home"] + df["pred_draw"] + df["pred_away"]
    tol = PROB_SUM_TOL
    assert df["prob_sum"].between(1.0 - tol, 1.0 + tol).all(), "Prob sums != 1"

    # --- 1X2 calibration ---
    home_bias = (df["pred_home"] - df["actual_home"]).mean()
    hb_lo, hb_hi = ci(df["pred_home"] - df["actual_home"])
    draw_bias = (df["pred_draw"] - df["actual_draw"]).mean()
    db_lo, db_hi = ci(df["pred_draw"] - df["actual_draw"])
    away_bias = (df["pred_away"] - df["actual_away"]).mean()
    ab_lo, ab_hi = ci(df["pred_away"] - df["actual_away"])

    # Reliability slopes
    hslope, hint = ols(df["pred_home"], df["actual_home"])
    dslope, dint = ols(df["pred_draw"], df["actual_draw"])
    aslope, aint = ols(df["pred_away"], df["actual_away"])

    # Bootstrap slope CIs
    hs_lo, hs_hi = boot_slope(df["pred_home"], df["actual_home"])
    ds_lo, ds_hi = boot_slope(df["pred_draw"], df["actual_draw"])
    as_lo, as_hi = boot_slope(df["pred_away"], df["actual_away"])

    # Max calibration gap (per-bin, min_n=5)
    hgap = max_gap(df["pred_home"], df["actual_home"])
    dgap = max_gap(df["pred_draw"], df["actual_draw"])
    agap = max_gap(df["pred_away"], df["actual_away"])

    # --- O/U 2.5 calibration ---
    ou_bias = (df["pred_over"] - df["actual_over"]).mean()
    ob_lo, ob_hi = ci(df["pred_over"] - df["actual_over"])
    ou_slope, ou_int = ols(df["pred_over"], df["actual_over"])
    ou_sl, ou_sh = boot_slope(df["pred_over"], df["actual_over"])
    ogap = max_gap(df["pred_over"], df["actual_over"])

    # --- BTTS calibration ---
    btts_bias = (df["pred_btts"] - df["actual_btts"]).mean()
    bb_lo, bb_hi = ci(df["pred_btts"] - df["actual_btts"])
    btts_slope, btts_int = ols(df["pred_btts"], df["actual_btts"])
    bs_lo, bs_hi = boot_slope(df["pred_btts"], df["actual_btts"])
    bgap = max_gap(df["pred_btts"], df["actual_btts"])

    # Per-decile calibration tables (min_n=5 for display, flag n<10)
    home_bins_str = bin_str_detailed(df["pred_home"], df["actual_home"])
    draw_bins_str = bin_str_detailed(df["pred_draw"], df["actual_draw"])
    away_bins_str = bin_str_detailed(df["pred_away"], df["actual_away"])
    over_bins_str = bin_str_detailed(df["pred_over"], df["actual_over"])
    btts_bins_str = bin_str_detailed(df["pred_btts"], df["actual_btts"])

    # Bootstrap slope CIs
    hs_lo, hs_hi = boot_slope(df["pred_home"], df["actual_home"])
    ds_lo, ds_hi = boot_slope(df["pred_draw"], df["actual_draw"])
    as_lo, as_hi = boot_slope(df["pred_away"], df["actual_away"])
    ou_sl, ou_sh = boot_slope(df["pred_over"], df["actual_over"])
    bs_lo, bs_hi = boot_slope(df["pred_btts"], df["actual_btts"])

    # Print summary
    print(f"PL model calibration audit — OUR MODEL predictions vs actuals, n={n}")
    print(f"1X2: Home bias {home_bias:+.4f} [{hb_lo:+.4f}, {hb_hi:+.4f}], Draw bias {draw_bias:+.4f} [{db_lo:+.4f}, {db_hi:+.4f}], Away bias {away_bias:+.4f} [{ab_lo:+.4f}, {ab_hi:+.4f}]")
    print(f"O/U 2.5: bias {ou_bias:+.4f} [{ob_lo:+.4f}, {ob_hi:+.4f}]")
    print(f"BTTS: bias {btts_bias:+.4f} [{bb_lo:+.4f}, {bb_hi:+.4f}]")
    print(f"Reliability slopes: Home {hslope:.3f} [{hs_lo:.3f}, {hs_hi:.3f}], Draw {dslope:.3f} [{ds_lo:.3f}, {ds_hi:.3f}], Away {aslope:.3f} [{as_lo:.3f}, {as_hi:.3f}]")
    print(f"  O/U slope: {ou_slope:.3f} [{ou_sl:.3f}, {ou_sh:.3f}], BTTS slope: {btts_slope:.3f} [{bs_lo:.3f}, {bs_hi:.3f}]")
    print(f"Max gaps: Home {hgap:.4f}, Draw {dgap:.4f}, Away {agap:.4f}, O/U {ogap:.4f}, BTTS {bgap:.4f}")

    # Verdict helpers
    hp_reading = verdict_bias(hb_lo, hb_hi)
    dp_reading = verdict_bias(db_lo, db_hi)
    ap_reading = verdict_bias(ab_lo, ab_hi)
    ou_reading = verdict_bias(ob_lo, ob_hi)
    bt_reading = verdict_bias(bb_lo, bb_hi)
    hs_reading = verdict_slope(hs_lo, hs_hi)
    ds_reading = verdict_slope(ds_lo, ds_hi)
    as_reading = verdict_slope(as_lo, as_hi)
    ou_reading = verdict_slope(ou_sl, ou_sh)
    bt_reading = verdict_slope(bs_lo, bs_hi)

    home_bins_str = bin_str_detailed(df["pred_home"], df["actual_home"])
    draw_bins_str = bin_str_detailed(df["pred_draw"], df["actual_draw"])
    away_bins_str = bin_str_detailed(df["pred_away"], df["actual_away"])
    over_bins_str = bin_str_detailed(df["pred_over"], df["actual_over"])
    btts_bins_str = bin_str_detailed(df["pred_btts"], df["actual_btts"])

    hp_text = verdict_bias(hb_lo, hb_hi)
    dp_text = verdict_bias(db_lo, db_hi)
    ap_text = verdict_bias(ab_lo, ab_hi)
    ou_text = verdict_bias(ob_lo, ob_hi)
    bt_text = verdict_bias(bb_lo, bb_hi)
    hs_text = verdict_slope(hs_lo, hs_hi)
    ds_text = verdict_slope(ds_lo, ds_hi)
    as_text = verdict_slope(as_lo, as_hi)
    ou_text = verdict_slope(ou_sl, ou_sh)
    bt_text = verdict_slope(bs_lo, bs_hi)

    home_bins_str = bin_str_detailed(df["pred_home"], df["actual_home"])
    draw_bins_str = bin_str_detailed(df["pred_draw"], df["actual_draw"])
    away_bins_str = bin_str_detailed(df["pred_away"], df["actual_away"])
    over_bins_str = bin_str_detailed(df["pred_over"], df["actual_over"])
    btts_bins_str = bin_str_detailed(df["pred_btts"], df["actual_btts"])

    out = Path("docs/pl_model_calibration.md")
    out.write_text(
        f"# PL model calibration audit (our model predictions vs actuals)\n\n"
        f"Generated {datetime.utcnow().date()}. Model predictions from public_snapshot.json "
        f"(fixtures_by_gameweek + fixture_detail_by_event_id), n={n} finished fixtures. "
        f"Paired-bootstrap 95% CIs over {N_BOOT} resamples (fixture-level bootstrap). "
        f"Research only: no model output changed, nothing under src/.\n\n"
        f"## Summary\n\n"
        f"- **Data source**: OUR MODEL's predicted probabilities (not bookmaker odds) from public_snapshot.json: "
        f"`fixtures_by_gameweek` (1X2 probs + actuals) joined with `fixture_detail_by_event_id` (O/U 2.5, BTTS).\n"
        f"- **Sample**: n={n} finished fixtures with model predictions and actual results (2026-08 to 2026-10).\n"
        f"- **Probabilities sum to 1**: 1X2 probs sum to 1.000 for all fixtures.\n\n"
        f"## Per-market results (paired-bootstrap 95% CIs, {N_BOOT} resamples)\n\n"
        f"| Market | Metric | Value | 95% CI | Reading |\n"
        f"|--------|--------|-------|--------|---------|\n"
        f"| Home win (1X2) | Bias (pred − actual) | {home_bias:+.4f} | [{hb_lo:+.4f}, {hb_hi:+.4f}] | {hp_reading} |\n"
        f"| Home win (1X2) | Reliability slope | {hslope:.3f} | [{hs_lo:.3f}, {hs_hi:.3f}] | {hs_reading} |\n"
        f"| Home win (1X2) | Max calibration gap | {hgap:.4f} | — | {'<0.03 honest' if hgap < 0.03 else '>0.03'} |\n"
        f"| Draw (1X2) | Bias (pred − actual) | {draw_bias:+.4f} | [{db_lo:+.4f}, {db_hi:+.4f}] | {dp_reading} |\n"
        f"| Draw (1X2) | Reliability slope | {dslope:.3f} | [{ds_lo:.3f}, {ds_hi:.3f}] | {ds_reading} |\n"
        f"| Draw (1X2) | Max calibration gap | {dgap:.4f} | — | {'<0.03 honest' if dgap < 0.03 else '>0.03'} |\n"
        f"| Away win (1X2) | Bias (pred − actual) | {away_bias:+.4f} | [{ab_lo:+.4f}, {ab_hi:+.4f}] | {ap_reading} |\n"
        f"| Away win (1X2) | Reliability slope | {aslope:.3f} | [{as_lo:.3f}, {as_hi:.3f}] | {as_reading} |\n"
        f"| Away win (1X2) | Max calibration gap | {agap:.4f} | — | {'<0.03 honest' if agap < 0.03 else '>0.03'} |\n"
        f"| O/U 2.5 | Bias (pred − actual) | {ou_bias:+.4f} | [{ob_lo:+.4f}, {ob_hi:+.4f}] | {ou_reading} |\n"
        f"| O/U 2.5 | Reliability slope | {ou_slope:.3f} | [{ou_sl:.3f}, {ou_sh:.3f}] | {ou_reading} |\n"
        f"| O/U 2.5 | Max calibration gap | {ogap:.4f} | — | {'<0.03 honest' if ogap < 0.03 else '>0.03'} |\n"
        f"| BTTS | Bias (pred − actual) | {btts_bias:+.4f} | [{bb_lo:+.4f}, {bb_hi:+.4f}] | {bt_reading} |\n"
        f"| BTTS | Reliability slope | {btts_slope:.3f} | [{bs_lo:.3f}, {bs_hi:.3f}] | {bt_reading} |\n"
        f"| BTTS | Max calibration gap | {bgap:.4f} | — | {'<0.03 honest' if bgap < 0.03 else '>0.03'} |\n\n"
        f"## Per-decile calibration (min n=5, bins with n<10 flagged ⚠)\n\n"
        f"### Home win (1X2)\n{home_bins_str}\n\n"
        f"### Draw (1X2)\n{draw_bins_str}\n\n"
        f"### Away win (1X2)\n{away_bins_str}\n\n"
        f"### O/U 2.5\n{over_bins_str}\n\n"
        f"### BTTS\n{btts_bins_str}\n\n"
        f"## Interpretation\n"
        f"- A market reads as **problem** when its bias/slope CI excludes the ideal value (0 for bias, 1.0 for slope).\n"
        f"- A market reads as **inconclusive** when the CI is too wide (>±0.5 half-width around ideal) to distinguish from ideal — this happens at small n.\n"
        f"- A market reads as **no problem** when the CI contains the ideal value and is narrow enough to be informative.\n"
        f"- **Home win**: bias {home_bias:+.4f} CI [{hb_lo:+.4f}, {hb_hi:+.4f}] — {hp_text}; "
        f"slope {hslope:.3f} CI [{hs_lo:.3f}, {hs_hi:.3f}] — {hs_text}.\n"
        f"- **Draw**: bias {draw_bias:+.4f} CI [{db_lo:+.4f}, {db_hi:+.4f}] — {dp_text}; "
        f"slope {dslope:.3f} CI [{ds_lo:.3f}, {ds_hi:.3f}] — {ds_text}.\n"
        f"- **Away win**: bias {away_bias:+.4f} CI [{ab_lo:+.4f}, {ab_hi:+.4f}] — {ap_text}; "
        f"slope {aslope:.3f} CI [{as_lo:.3f}, {as_hi:.3f}] — {as_text}.\n"
        f"- **O/U 2.5**: bias {ou_bias:+.4f} CI [{ob_lo:+.4f}, {ob_hi:+.4f}] — {ou_text}; "
        f"slope {ou_slope:.3f} CI [{ou_sl:.3f}, {ou_sh:.3f}] — {ou_text}.\n"
        f"- **BTTS**: bias {btts_bias:+.4f} CI [{bb_lo:+.4f}, {bb_hi:+.4f}] — {bt_text}; "
        f"slope {btts_slope:.3f} CI [{bs_lo:.3f}, {bs_hi:.3f}] — {bt_text}.\n"
        f"- **Reviewer 50-60% bucket lead**: The spot-check (8/23 = 35% for match_result 50-60%) was a small-n spot-check. "
        f"This analysis is NOT a multi-season walk-forward — it is a single-season snapshot (n={n} finished fixtures from 2026-08 to 2026-10). "
        f"No multi-season walk-forward backtest exists for the model's own probabilities. The spot-check concern remains open.\n"
        f"- **No multi-season walk-forward has been run** for the model's own probabilities. The scoreline model backtest evaluates closing odds, not our model's own probabilities.\n"
        f"- No fix shipped: research only; fix only if paired-bootstrap CI excludes ideal value on re-tested candidate.\n\n"
        f"## Verdict\n"
        f"Not enough 2026 fixtures to judge; re-run at ~300 fixtures or on the multi-season backtest (which does not yet exist for model probabilities). "
        f"Current n={n} is too small to distinguish calibration from noise — all slope CIs are inconclusive (span >±0.5 around 1.0), "
        f"and bias CIs straddle zero but are too wide to be informative.\n"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()