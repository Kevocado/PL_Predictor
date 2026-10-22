# PL model-probability calibration audit — re-run trigger

Research only: no `src/` changes. Companion to `docs/pl_model_calibration.md`.

## Checkable condition

Re-run the model-probability calibration audit once **at least 300 finished fixtures have a matching detail record** in the snapshot. Concretely (this is the script's actual eligibility predicate in `scripts/pl_model_calibration.py`):

```
count = number of fixtures where
        fixture["finished"] is True
        AND fixture["event_id"] is present in snapshot["fixture_detail_by_event_id"]
```

The script reads 1X2 probabilities from the fixture record (`predicted_home_win` /
`predicted_draw` / `predicted_away_win`) and O/U 2.5 + BTTS from the matching
`fixture_detail` entry, so a fixture is eligible exactly when it is finished and
has a detail record -- no probability-field presence check is performed.

Current count: **70** (as of the `docs/pl_model_calibration.md` audit, 2026-08 to 2026-10). Expected to cross 300 around **gameweek 20** of the 2026-27 season. Re-run when `count >= 300`.

## Script to run

```
PYTHONPATH=src python scripts/pl_model_calibration.py
```

(from the repo root, using `data/public_snapshot.json`). It regenerates `docs/pl_model_calibration.md` with per-market bias / reliability-slope / max-gap CIs and per-decile tables.

## Why 300

The current n=70 leaves every slope CI inconclusive (half-width > ±0.5 around 1.0) and every bias CI too wide to be informative. ~300 finished fixtures is the point at which a ±0.1 bias CI and a slope CI that can exclude 1.0 become reachable. Until then, do not act on the audit's verdicts.

## Do not

- Do not recalibrate model probabilities before the re-run; the audit is the gate.
- Do not lower the threshold or substitute bookmaker odds for model probabilities.