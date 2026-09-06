# Data-Led Player Ratings Design

## Purpose

Replace the hand-capped, last-season FPL rating with a multi-season,
role-aware assessment based on observed Premier League play. It must not use
FPL point projections or fixture-model outputs. The Player Hub result is
descriptive; any scoreline-model feature remains separately evaluated
research.

## Score contract

- **Overall (0-95):** durable Ability plus a small, evidence-gated current
  Form lift. It is the primary Player Hub ordering score.
- **Quality (0-95):** the durable, multi-season Ability component.
- **Live Form (0-100):** official FPL recent form, shrunk for minutes and
  starts. `vs Quality` identifies over- and under-performance.
- **Impact (0-95):** Overall adjusted by availability and expected minutes.
  Availability never changes Quality or Overall.

The fixed common display scale is anchored to historical qualifying Premier
League seasons, never the active player pool: 85-95 is rare sustained elite
performance; 70-84 is established good Premier League performance; below 70
is rotation/development/limited evidence. No display score can reach 100.

Players with fewer than 900 qualifying Premier League minutes in the latest
three completed seasons have `rating_status="provisional"`. Players with at
least 900 minutes but only one qualifying season have
`rating_status="limited"`. Player Hub shows **Provisional** or **Limited
evidence** instead of a misleading Overall rank. They remain visible with Live
Form and Impact but are excluded from Overall leaderboards and team-strength
aggregates.

## Data and role evidence

Serving reads only cached official FPL bootstrap data plus the local vaastav
FPL archive. The historical prior uses the latest three completed positioned
seasons, drops zero-minute rows, and aggregates a player by normalized name,
position, team, and season; FPL `element` IDs are never assumed stable across
seasons.

Role evidence is derived from observed per-90 outputs, then normalized against
fixed historical role anchors and recency-weighted 0.55, 0.30, and 0.15. It
never reads FPL `total_points` or projection fields. Goalkeeper shot prevention
uses observed xGC prevented (`expected_goals_conceded - goals_conceded` per
90), rather than comparing them with a reserve at the same club; defenders are
adjusted against their team-season defensive baseline. Raw saves and clean
sheets cannot alone make a busy keeper or a player on a strong team elite.
Midfielders and forwards use expected and actual goal/assist contribution per
90. An 1,800-minute effective sample confers full confidence.

## Team-strength boundary

Any future expected-XI feature uses a shifted latent role value, not rounded
UI ratings or a raw average. The eight GK/DEF/MID/FWD home/away unit fields
remain research-only and cannot reach live scoreline features unless a fresh
chronological experiment improves mean and recent-holdout RPS, Brier, and log
loss without a calibration regression.

## Operational constraints

- A Player Hub visit may not fetch per-player history, train a model, or call
  an external service; archive priors are memoized local-file work.
- The public deployment remains snapshot-only; updated hub ratings require a
  locally regenerated snapshot, never Render live computation.
- Tests cover provenance, context adjustment, fixed-scale calibration,
  Provisional exclusion, availability isolation, and no-lookahead.
