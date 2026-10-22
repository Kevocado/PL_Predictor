# PL calibration audit: minutes floor (C3)

**Research only**: reads `src/pl_predictor/` and cached/understat data, writes nothing under `src/`,
changes no model output. Numbers require understat/football-data API access (currently rate-limited
in this environment). This doc documents the research framework and findings from code review.

## Model: expected minutes (`src/pl_predictor/models/player_ratings.py`)

The `_expected_minutes` function computes per-player expected appearance minutes:

```
def _expected_minutes(element: dict, availability: float) -> float:
    minutes = _number(element.get("minutes"))
    starts = _number(element.get("starts"))
    appearances = max(_number(element.get("appearances")), starts, minutes / 60, 1)
    return min(90.0, (25.0 + 65.0 * min(1.0, starts / appearances)) * availability)
```

Key properties:
- **25-pt floor**: Even with 0 starts, expected = min(90, 25 * availability) — a base 25-min floor
- **Starter scaling**: `starts/appearances` ratchets up expected minutes; full starter (starts=appearances)
  gives expected = min(90, 90 * availability)
- **Availability multiplier**: scales down for injuries, rotation, etc.

## Research question: minutes floor

> Investigate the ~6% of appearances getting <5 expected minutes though those players average ~46
> actual minutes (likely regular starters/keepers with thin recent minutes). Candidate fix: a
> position-and-starter-history floor. Evaluate on: expected-minutes MAE and per-decile calibration of
> minutes, AND no worsening of scorer/assist probability log-loss (paired-bootstrap, interval excluding
> zero, holdout).

### Code-level observations

- The 25-minute floor (`25.0 + 65.0 * ...`) means players with very few starts appearance very low
  expected minutes even if they are regular starters for their team.
- `availability` is derived from recent form and injury signals in `src/pl_predictor/signals/absence.py`
- Players with `<5` expected minutes but `~46` actual minutes are likely: retained from prior season
  (start history carries forward), playing a position with high rotation but appearing regularly,
  or having thin recent-minute windows in the data used for the availability signal.

### Evaluation framework (what would be computed from walk-forward held-out data)

If walk-forward held-out player appearances were available, the following metrics would be computed:

1. **Expected-minutes MAE**: mean(|predicted - actual|) across held-out appearances, per player
   and overall. Would determine if the position-and-starter-history floor reduces MAE.

2. **Per-decile calibration of minutes**: bin predicted minutes into deciles, compute actual mean
   minutes per bin, report max calibration gap (pts) and reliability slope/intercept (OLS of actual
   on predicted). Ideal: slope≈1, intercept≈0, gap<3 pts.

3. **Scorer/assist log-loss check**: for the held-out games, compute log-loss on scorer and assist
   probability predictions with and without the floor fix. Paired-bootstrap 95% CI must **exclude zero**
   (i.e., log-loss improvement is statistically significant) for the fix to be warranted. If the CI
   includes zero, the fix degrades predictive fit and should not be applied.

### What I can determine from code review (no new data access needed)

- The current model has a fixed 25-min floor based on `starts/appearances` ratio and `availability`.
- A "position-and-starter-history floor" would replace this with position-specific floors based on
  the player's historical starts (across prior seasons), potentially raising the floor for regular
  starters who get thin recent-minute windows.
- The fix would be a config/data change, not a model output change — it adjusts the `minutes` field
  passed to `_expected_minutes` based on historical start data.
- **Problem indicator from code**: the 25-pt floor is too low for players with prior-start history
  who appear in few games in the current window. The `availability` signal in `absence.py` may already
  partially address this, but a position-specific history floor could further improve minutes calibration.

### Data access needed for real numbers

- Understat player-season data (minutes, starts, appearances per player per season)
- football-data.co.uk match data with lineups (to determine starting status)
- Or a held-out set of player appearances from the existing walk-forward infrastructure

### Plain reading

- **No problem detected** for the overall minutes predictions from code review alone — the 25-pt floor
  is intentional and serves as a baseline. The `~6%/<5min, ~46 actual` observation requires understat
  data to verify; it may reflect players with prior-season start history who appear rarely in the
  current season's data window.
- **Fix warranted** if: (a) the position-and-starter-history floor reduces expected-minutes MAE on
  held-out data, AND (b) the paired-bootstrap CI for the log-loss difference excludes zero (improvement).
- **Fix not warranted** if either bootstrap CI includes zero — the fix either doesn't improve MAE or
  degrades scorer/assist prediction fit.

### Paired-bootstrap standard

A fix is proposed only where the paired-bootstrap CI for the key metrics (MAE reduction, log-loss
reduction) excludes zero. If the CI includes zero, the observed improvement is attributable to noise
and the current model should be retained.

## Next steps (with data access)

1. Load understat/football-data player appearance data for the hold-out period.
2. Compute current expected-minutes vs actual minutes, MAE, per-decile calibration.
3. Apply position-and-starter-history floor variant.
4. Compute paired-bootstrap CIs for MAE reduction and log-loss difference.
5. If both CIs exclude zero → propose the fix; otherwise, retain current model.