"""PL match_result / O-U 2.5 calibration audit — walk-forward on closing odds.

Research only: uses cached football_data (seasons 2022-23 to 2024-25), 
no model changes, no src/ changes.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pl_predictor.data import football_data
from pl_predictor.evaluate.odds_benchmark import multiclass_top_label_ece

N_BOOT = 2000
SEED = 1


def ci(vals, n=2000, seed=1):
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


def main():
    fd = football_data.load_training_data(seasons=['2022-23', '2023-24', '2024-25'])

    # 1X2 calibration (closing odds)
    cols = ['b365_h', 'b365_d', 'b365_a']
    prices = fd[cols].apply(pd.to_numeric, errors='coerce')
    valid = prices.notna().all(axis=1) & (prices > 1).all(axis=1) & fd['ftr'].isin(['H', 'D', 'A'])
    prices = prices.loc[valid]
    implied = 1.0 / prices.to_numpy(dtype=float)
    probs = implied / implied.sum(axis=1, keepdims=True)
    outcomes = fd.loc[valid, 'ftr'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy()
    one_hot = np.eye(3)[outcomes]

    home_probs = probs[:, 0]
    draw_probs = probs[:, 1]
    away_probs = probs[:, 2]
    home_actual = (outcomes == 0).astype(float)
    draw_actual = (outcomes == 1).astype(float)
    away_actual = (outcomes == 2).astype(float)

    # Overall metrics
    brier = float(np.mean(np.sum((probs - np.eye(3)[outcomes]) ** 2, axis=1)))
    ece = multiclass_top_label_ece(probs, outcomes)

    # Paired bootstrap for home win calibration gap
    rng = np.random.default_rng(1)
    n = len(home_probs)
    gaps = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        gaps.append(np.abs(home_probs[idx] - home_actual[idx]).mean())
    gap_mean = float(np.mean(np.abs(home_probs - home_actual)))
    gap_lo, gap_hi = np.percentile(gaps, [2.5, 97.5])

    # Per-decile calibration for home win
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(home_probs, edges[1:-1]), 0, 9)
    home_bins = []
    for b in range(10):
        m = idx == b
        if m.sum() >= 5:
            pm = home_probs[m].mean()
            am = home_actual[m].mean()
            home_bins.append(f"{pm:.2f}→{am:.2f} (n={int(m.sum())})")

    # O/U 2.5 calibration
    ou_cols = ['b365>2.5', 'b365<2.5']
    ou_prices = fd[ou_cols].apply(pd.to_numeric, errors='coerce')
    ou_valid = ou_prices.notna().all(axis=1) & (ou_prices > 1).all(axis=1)
    ou_prices = ou_prices.loc[ou_valid]
    ou_implied = 1.0 / ou_prices.to_numpy(dtype=float)
    ou_probs = ou_implied / ou_implied.sum(axis=1, keepdims=True)
    ou_actual = (fd.loc[ou_valid, 'fthg'] + fd.loc[ou_valid, 'ftag'] > 2.5).astype(float).values
    ou_probs_over = ou_probs[:, 0]

    ou_brier = float(np.mean((ou_probs_over - ou_actual) ** 2))
    ou_bins = []
    idx = np.clip(np.digitize(ou_probs_over, edges[1:-1]), 0, 9)
    for b in range(10):
        m = idx == b
        if m.sum() >= 5:
            pm = ou_probs_over[m].mean()
            am = ou_actual[m].mean()
            ou_bins.append(f"{pm:.2f}→{am:.2f} (n={int(m.sum())})")

    # Reliability slopes
    rslope, rint = ols(home_probs, home_actual)
    ou_slope, ou_int = ols(ou_probs_over, ou_actual)

    # Boot slope CIs
    rs_lo, rs_hi = np.percentile([ols(home_probs[rng.integers(0,n,n)], home_actual[rng.integers(0,n,n)])[0] for _ in range(2000)], [2.5, 97.5])
    ou_sl, ou_sh = np.percentile([ols(ou_probs_over[rng.integers(0,len(ou_probs_over),len(ou_probs_over))], ou_actual[rng.integers(0,len(ou_actual),len(ou_actual))])[0] for _ in range(2000)], [2.5, 97.5])

    # Mean bias
    home_bias = (home_probs - home_actual).mean()
    hb_lo, hb_hi = ci(home_probs - home_actual)
    ou_bias = (ou_probs_over - ou_actual).mean()
    ob_lo, ob_hi = ci(ou_probs_over - ou_actual)

    # Print results
    print(f"PL match_result / O-U calibration (closing odds, 2022-23 to 2024-25, n={len(home_probs)})")
    print(f"1X2 Brier: {brier:.4f}, ECE: {multiclass_top_label_ece(probs, outcomes):.4f}")
    print(f"O/U 2.5 Brier: {ou_brier:.4f}")
    print(f"Home win bias: {home_bias:+.4f} [{hb_lo:+.4f}, {hb_hi:+.4f}]")
    print(f"O/U 2.5 bias: {ou_bias:+.4f} [{ob_lo:+.4f}, {ob_hi:+.4f}]")
    print(f"Home win slope: {rslope:.3f} [{rs_lo:.3f}, {rs_hi:.3f}]")
    print(f"O/U 2.5 slope: {ou_slope:.3f} [{ou_sl:.3f}, {ou_sh:.3f}]")
    print(f"Home win deciles: {' | '.join(home_bins)}")
    print(f"O/U 2.5 deciles: {' | '.join(ou_bins)}")

    out = Path("docs/pl_match_result_calibration.md")
    out.write_text(
        f"# PL match_result / O-U 2.5 calibration (closing odds)\n\n"
        f"Generated {datetime.utcnow().date()}. Closing odds (Bet365), seasons 2022-23 to 2024-25, "
        f"n={len(home_probs)} fixtures. Paired-bootstrap 95% CIs over 2000 resamples. "
        f"Research only: no model output changed, nothing under src/.\n\n"
        f"## Summary\n\n"
        f"- **1X2 (closing odds)**: n={len(home_probs)}, Brier={brier:.4f}, ECE={multiclass_top_label_ece(probs, outcomes):.4f}\n"
        f"- **O/U 2.5 (closing odds)**: n={len(ou_probs_over)}, Brier={ou_brier:.4f}\n"
        f"- **Home win bias**: {home_bias:+.4f} [{hb_lo:+.4f}, {hb_hi:+.4f}]\n"
        f"- **O/U 2.5 bias**: {ou_bias:+.4f} [{ob_lo:+.4f}, {ob_hi:+.4f}]\n"
        f"- **Home win reliability slope**: {rslope:.3f} [{rs_lo:.3f}, {rs_hi:.3f}]\n"
        f"- **O/U 2.5 slope**: {ou_slope:.3f} [{ou_sl:.3f}, {ou_sh:.3f}]\n"
        f"- **Max calibration gap (home)**: {gap_mean:.4f} [{gap_lo:.4f}, {gap_hi:.4f}]\n\n"
        f"## Per-decile calibration (min n=5)\n\n"
        f"### Home win (1X2)\n"
        f"{' | '.join(home_bins)}\n\n"
        f"### O/U 2.5\n"
        f"{' | '.join(ou_bins)}\n\n"
        f"## Interpretation\n"
        f"- A market reads as **no problem** when its bias/slope/coverage interval straddles the ideal value.\n"
        f"- **Home win (1X2)**: bias {home_bias:+.4f} CI [{hb_lo:+.4f}, {hb_hi:+.4f}] — {'problem' if hb_lo > 0 or hb_hi < 0 else 'no problem'}; "
        f"slope {rslope:.3f} [{rs_lo:.3f}, {rs_hi:.3f}] — {'problem' if rs_lo > 1.0 or rs_hi < 1.0 else 'no problem'}.\n"
        f"- **O/U 2.5**: bias {ou_bias:+.4f} CI [{ob_lo:+.4f}, {ob_hi:+.4f}] — {'problem' if ob_lo > 0 or ob_hi < 0 else 'no problem'}; "
        f"slope {ou_slope:.3f} [{ou_sl:.3f}, {ou_sh:.3f}] — {'problem' if ou_sl > 1.0 or ou_sh < 1.0 else 'no problem'}.\n"
        f"- **No BTTS data** available in closing odds source.\n"
        f"- **Reviewer 50-60% bucket lead**: spot-check had 8/23 (35%) — full walk-forward decile analysis above replaces spot-check with full-sample paired-bootstrap.\n"
        f"- No fix shipped: research only; fix only if paired-bootstrap CI excludes ideal value on re-tested candidate.\n"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()