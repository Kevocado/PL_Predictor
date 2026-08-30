# Role-Aware Player Ratings Design

## Purpose

Replace the current within-position percentile rating with a conservative,
evidence-based player assessment that distinguishes durable ability from
current form and match utility. The score is descriptive in Player Hub first;
its team-unit aggregate remains a research-only candidate for the scoreline
model.

## Score contract

Each player receives four distinct values.

- **Quality (0–100):** role-specific, multi-season assessment of durable
  ability. It is shrunk toward a role prior when evidence is sparse, so a
  player cannot reach an elite score merely by leading a small current pool.
- **Form (0–15):** short-to-medium-term lift earned only with sufficient
  starts/minutes. It blends recent underlying output, actual returns, and
  sustained opportunity (starts plus penalties/set pieces where available).
- **Overall (0–100):** `min(100, Quality + Form)`. 90+ means rare league-wide
  elite quality or an already strong player sustaining an exceptional breakout
  season; it is not assigned by percentile rank.
- **Impact (0–100):** Overall adjusted by current FPL availability and
  predicted minutes. It answers who is most useful in the upcoming gameweek,
  not who is intrinsically best.

Player Hub displays Overall first, then Quality, Form, Impact, expected
minutes, and the leading role-specific drivers. This makes players performing
above their durable level visible rather than hiding the distinction.

## Role targets and data

Train and validate separate historical role models, with only information
available before each player fixture. Historical FPL rows use shifted rolling
features; numerical FPL element IDs are never joined across seasons.

| Role | Durable target and drivers |
| --- | --- |
| GK | Shot prevention proxy (saves, clean sheets, goals conceded versus expected goals conceded where available), minutes, BPS |
| DEF | Defensive contribution/clean-sheet and xG-conceded signals, plus attacking xG/xA, minutes, BPS |
| MID | Expected goal involvement, chance creation, threat/creativity, goal contribution, minutes, set-piece opportunity |
| FWD | Expected goals, expected goal involvement, goal contribution, minutes, penalty/set-piece opportunity |

Each rich model competes with a role baseline using chronological seasons.
The rich model is selectable only if it improves its role's held-out MAE and
does not worsen RMSE. Otherwise that role retains its baseline. Current live
bootstrap data is mapped to the same feature contract and cached; no Player
Hub visit may trigger a per-player history fan-out or model fit.

Quality uses evidence shrinkage: current-season signal is blended with prior
season/role-prior evidence using minutes and starts as confidence weights.
Form is capped at zero until a minimum evidence threshold, then rises only
when underlying production, actual output, and opportunity agree. A finishing
spike without minutes or chance volume cannot create an elite Overall score.

## Team-unit experiment

For each historical fixture, choose a legal expected XI using only prior
starts and minutes, then aggregate expected-minutes-weighted **Quality +
Form** into eight candidate features: home/away GK, DEF, MID, and FWD. Do not
use realised starters, current-match minutes, or live availability in this
early-forecast dataset.

Evaluate against the unchanged production features with the existing
walk-forward folds. Record per-fold and aggregate RPS, Brier, ECE, scoreline
log loss, coverage, and feature importance. It must remain research-only until
manual review confirms lower mean RPS, no calibration regression, and a
non-regressing recent fold. Availability may be considered separately only in
the prospective confirmed-XI track.

## UI and validation

- Player Hub: sortable Overall, Quality, Form, and Impact columns; compact
  top-five role leaderboards by Overall; explicit source and freshness.
- Verify absolute calibration with distribution tests: no unconditional
  100-rated player; 90+ requires defined evidence/threshold conditions.
- Verify no target leakage, role normalisation, early-season shrinkage,
  availability affecting Impact only, and role-model promotion gates.
- Verify legal expected-XI construction, eight-unit aggregation, and that the
  candidate cannot be appended to the deployed feature list automatically.
