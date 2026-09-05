# Player Shots & Shots-on-Target Market Design

## Purpose

Add a real, measured player-level shots and shots-on-target prediction —
`expected_shots`, `expected_shots_on_target`, and
`anytime_shot_on_target_prob` — to the existing anytime goal/assist surface,
using Understat's per-shot data (already fetched and cached by
`data/understat_shots.py`, currently only aggregated to team-level set-piece
share). This must hold the same evidence bar the goal/assist model does: a
per-90 rate fit on real historical shot data, blended by current-season
sample size the same way `player_form.blended_current_form` already blends
goals/assists, not a proxy derived from an unrelated stat (ICT `threat`,
say) dressed up as a shots number.

## Scope

**In scope:** player-level shots and shots-on-target for the next fixture,
surfaced the same way `PlayerPrediction` already surfaces goals/assists — a
model projection, no live-odds edge (the Odds API's free bulk endpoint
carries no player-prop markets, same limitation BTTS/corners/cards/goals/
assists already live with).

**Out of scope for this design:** first-goalscorer timing, shots
conceded/faced, non-PL competitions, backfilling the crosswalk for retired/
transferred-out players who no longer have an FPL element.

## 1. Understat → FPL player crosswalk

A new cached artifact, `understat_element_id → fpl_element_id`, built once
per model retrain (same cadence as `models/manifest.py`'s other artifacts,
not per-request).

**Matching, validated against real data before writing this** (tested
directly against the live FPL bootstrap and the actual cached Understat
shot files, not assumed): a single normalized-name match — even reusing
`player_goals.py`'s existing `_normalise_name` — only clears ~66% of real
players, for three distinct, confirmed reasons: HTML entities surviving in
FPL's raw names (`O&#039;Nien`), letters NFKD genuinely cannot decompose
(`Ødegaard` does not become `Odegaard` — this is a pre-existing gap in
`_normalise_name` itself, not a new one), and — the dominant failure mode —
FPL's `first_name` is the formal form (`Benjamin`) where Understat records
the common one (`Ben`). Team-based disambiguation was tested and *rejected*:
requiring team-match alongside name actively loses matches, since a
transferred player's own historical shot rows carry his prior club, not his
current one.

Matching is therefore layered, applied in order, first hit wins:
1. Exact match on the fully-normalized name (HTML-unescaped, then the
   existing accent-stripping normalizer, plus a manual translit table for
   the handful of letters NFKD can't decompose: `ø→o, đ→d, ł→l, ß→ss, æ→ae,
   œ→oe, ð→d, þ→th`, both cases) against FPL's `first_name+" "+second_name`
   or `web_name` — measured 299/449 (66.6%) on the live current-season pool.
2. If the first name has a common-nickname expansion (a small built-in
   table — `ben→benjamin, josh→joshua, matt→matthew, mike→michael,
   tom→thomas, alex→alexander, dan(ny)→daniel, will→william, joe(y)→joseph,
   jim(my)/jamie→james, rob/bob(by)→robert, dave→david, chris→christopher,
   tony→anthony, andy→andrew, steve(vie)→stephen, pat(dy)→patrick,
   ron(nie)→ronald, fred(die)→frederick, gerry/jerry→gerald,
   greg→gregory, ken(ny)→kenneth, jack/johnny→john, jon(ny)→jonathan` — not
   exhaustive, extend as real misses surface), retry step 1 with the
   expanded first name — measured +2.
3. Fall back to surname-only match (against FPL's `second_name` or
   `web_name`) when it resolves to exactly one candidate — ambiguous
   (more than one same-surname candidate) is treated as unmatched, never
   guessed — measured +36.

Combined measured match rate: 337/449 (75.1%) of Understat's current
Understat-season player pool. The remaining ~25% is not purely a matching
gap: spot-checking a sample (Kyle Walker, Kieran Trippier, Nathan Aké,
James Ward-Prowse, Oleksandr Zinchenko) confirmed they are not present in
the current FPL bootstrap *under any name* — Understat's per-season shot
data spans players who have since left the league entirely, who correctly
have no valid crosswalk target at all. A future contributor extending the
nickname table or matching logic should re-measure against fresh data
rather than assume the current ~75% is the ceiling — it almost certainly
undercounts genuine coverage among *current* players specifically.

A player who still fails to match is logged and excluded from the shots
model for that retrain — they still get goal/assist predictions as before,
just no shots line. Never a hard failure: a crosswalk miss must degrade
the same way a `New-team fallback` or `is_fallback_prediction` already
does elsewhere in this codebase, not 501/crash the fixture-detail
response.

Stored as `models/understat_fpl_crosswalk.json` (git-tracked, alongside the
other `models/*.json` artifacts) so a fresh deploy has one immediately, same
reasoning as the existing trained models shipping in the repo rather than
requiring a first-run train.

## 2. Per-player shot extraction

New function(s) in `data/understat_shots.py` (not a new file — this stays
next to the existing per-match shot cache it reads, same one already
fetched for the set-piece-share feature, zero new network calls) that parse
the cached per-match shot CSVs into one row per player per match:
`element` (Understat id), `date`, `team`, `shots`, `shots_on_target`,
`goals`, `x_g`. `shots_on_target` is `result in {"Goal", "SavedShot"}` —
confirmed against real cached data: Understat's `result` values are
`Goal`, `SavedShot`, `MissedShots`, `BlockedShot`, `ShotOnPost`, `OwnGoal`;
`ShotOnPost` and `BlockedShot` are off-target by the standard betting
definition (only a shot that would have gone in without a save/post/block
counts). Joined onto `fpl_element_id` via the Section 1 crosswalk
immediately after extraction, so every downstream consumer stays in FPL
element-id space — the same id everything else in `player_goals.py`/
`player_form.py` already keys on.

## 3. Feature engineering

Reuse `player_form.py`'s rolling shift(1)-then-rolling-window machinery
unchanged: `shots` and `shots_on_target` become two more entries in
`RATE_STATS` (per-90, summed-then-divided like goals/assists/xG), computed
both in `build_historical_player_form` (training) and
`blended_current_form` (live serving) with no new code path — this is
exactly why Section 2's output is shaped like the existing FPL
gameweek-history frame instead of introducing a parallel feature-building
function.

## 4. Modeling — team-shot-volume scaling

`predict_player`'s existing scaling — `team_goal_expectation /
LEAGUE_AVERAGE_TEAM_GOALS` — is wrong for shots: a low-xG-per-shot team can
still take a high shot volume, and the scoreline model has no shot-count
output to anchor to. **Revised from this spec's original draft** (which
proposed building a new team-shot feature from Understat's match-dominance
aggregate): `features/rolling_form.py`'s `BASE_STATS` already includes
`shots_for`/`shots_against` — rolling team shot volume from
football-data.co.uk's `hs`/`as` columns, computed by the exact same
generic `shift(1).rolling(w).mean()` machinery goals-for already uses, and
already exposed for every live fixture as `FixtureFeatureContext.form`
(`self.form = rolling_form.latest_form(self.matches_df)`, `features/
build.py`), just never read downstream today. No new feature, no new data
source, no Understat/FPL player-id crosswalk needed at the *team* level —
this is genuinely free: `context.form.loc[home, "home_last_10_shots_for"]`
/ `context.form.loc[away, "away_last_10_shots_for"]` (home/away-split
variant, matching how home advantage affects shot volume) becomes the
scaling anchor for the shots targets specifically, alongside — not
replacing — `team_goal_expectation`, which keeps scaling goals/assists
exactly as today: `predict_player` computes a second `shots_scale`
alongside its existing `scale` (same `minutes_fraction * availability`
factors, different strength multiplier — `team_expected_shots /
LEAGUE_AVERAGE_TEAM_SHOTS`). `LEAGUE_AVERAGE_TEAM_SHOTS = 13.1`, measured
directly (`(df["hs"].sum() + df["as"].sum()) / (2 * len(df))` over
`football_data.load_training_data`'s last 3 completed seasons — 13.13,
rounded), the same "measured, not asserted" standard `FALLBACK_GOAL_
EXPECTANCY` was held to.

`context.form` only exists when the served model is the feature-driven ML
model (`hasattr(model, "context")` — Dixon-Coles/Bivariate-Poisson have no
per-fixture feature context at all, same condition `_data_confidence`
already gates on). When absent, `shots_scale` falls back to `1.0`
(`team_expected_shots` assumed at league average) rather than crashing or
guessing — the same degrade-gracefully convention as every other
confidence/fallback signal in this pipeline. `rank_team_players` currently
has no way to reach `context` at all (its signature takes `bootstrap`,
`current_event`, `position_priors`, etc., but not `models`/`context`) — it
gains an optional `context` parameter, threaded from `_rank_fixture_
players` in `api/routes.py` (which already holds `models = _get_models()`,
so `models["scoreline"].context` — `getattr(models["scoreline"], "context",
None)`, since Dixon-Coles/Bivariate-Poisson objects have no such
attribute — is available to pass through with no new plumbing beyond the
signature change itself).

`shots_scale` applies only to the two new targets — the existing `scale`
variable and its goals/assists callers are untouched. `fit_position_rate_models` extends its existing
`(("goals", "goals_scored"), ("assists", "assists"))` target loop with
`(("shots", "shots"), ("shots_on_target", "shots_on_target"))` — mechanically
the same Ridge-per-position fit, new targets only.

`anytime_shot_on_target_prob` is `1 - exp(-expected_shots_on_target)`, the
same Poisson-anytime conversion `anytime_probability` already applies to
goals/assists — no new probability model needed.

## 5. API surface

`PlayerPrediction` (schemas.py) gains three fields: `expected_shots`,
`expected_shots_on_target`, `anytime_shot_on_target_prob` — all optional/
nullable, `None` when the crosswalk has no match for that player rather
than a fabricated number. `rank_team_players` passes these through
`predict_player`'s output the same way `expected_goals`/`expected_assists`
already flow today. No new endpoint.

**Public deployment:** no extra work. `public_snapshot.py` already bakes
whatever `fixture_players` returns into `fixture_players_by_event_id` —
these three new fields ride along automatically the next time a snapshot
is generated, same as every other `PlayerPrediction` field.

## 6. Testing

- Crosswalk: unit tests on known hard cases — accented names (e.g.
  Rodrigo Bentancur-style diacritics), common-surname collisions broken by
  team, a deliberately unmatchable name asserting graceful exclusion (not
  a crash) — mirroring how `test_team_names.py` already tests
  `to_canonical`'s edge cases.
- Feature/model: extend `test_player_form.py` and `test_player_goals.py`
  the same way `test_rolling_player_form_never_uses_same_row_stat` already
  guards against off-by-one leakage — a shots-per-90 feature must fail the
  same leakage check goals/assists already pass. Explicitly re-run
  `test_blended_current_form_handles_player_with_no_current_season_history`
  -style coverage for the new `RATE_STATS` entries — this morning's
  `KeyError: 'minutes'` bug was exactly a missing-column guard gap, and
  shots/shots_on_target are new columns subject to the identical failure
  mode if a guard is missed.
- Crosswalk staleness: a test asserting every retrain rebuild fails loudly
  (not silently serves a stale mapping) if the crosswalk's `fpl_element_id`
  no longer resolves in the current `bootstrap.json` — a transferred/retired
  player must fall out, not silently misattribute shots to whoever now
  holds that id.

## 7. Evaluation gate before shipping to the live app

Before `expected_shots_on_target`/`anytime_shot_on_target_prob` reach
`rank_team_players`'s live output, run the walk-forward evaluation this
project already holds every model to (`evaluate/`'s pattern used for
corners/cards and the goal-contribution model): compare against a naive
position-average baseline on a genuinely held-out season. Ship only if it
beats that baseline — the same "measured, not assumed" bar
`docs/AI_CONTINUITY.md` already documents for every other market, and the
same standard the goal-contribution model met before replacing the prior
Poisson-union baseline.

## Operational notes

- Crosswalk rebuild cadence: every full retrain (`models/manifest.py`),
  not hourly auto-retrain — name/team matching is stable within a season
  and only needs updating when the squad changes (transfers), which a full
  retrain already tracks via `models/manifest_history.jsonl`.
- No new external API dependency: Understat is already scraped via
  `penaltyblog`; no new key, no new quota risk on Render.
- No effect on the free-tier memory story: shots/SoT ride the same
  `PlayerPrediction` payload the goal/assist prediction already produces
  per player — no new per-request computation shape.
