# PL Predictor: Research Findings

**Last updated:** 2026-08-29

This is a plain-language digest of the interesting things this project has
actually learned — what helped, what didn't, what surprised us, and why. It's
a companion to `docs/AI_CONTINUITY.md`, which is the dense, exhaustive
handoff log written for whoever (human or AI) next touches the code. This
file skips the protocol/methodology boilerplate and keeps only the findings
themselves, in the order they matter to someone asking "so what have we
actually figured out?"

Every claim below traces back to a numbered `EXP-YYYY-NN` or `OPS-YYYY-NN`
entry in `AI_CONTINUITY.md` — check there for the full protocol, per-fold
tables, and file-level detail behind any line here.

---

## 1. What actually made predictions better (shipped to production)

- **Squad continuity — how much of last season's team is still there.**
  (EXP-2026-18, 2026-08-29) The model's only sense of team strength was
  Elo/Pi ratings carried over from last season plus this season's own
  match results — which barely exist a few games in. So a team that
  overhauled its squad over the summer looks, to the model, identical to
  one that didn't, until results slowly prove otherwise. This started
  from a direct observation: Newcastle looked "hot" early in 2026-27 in a
  way the model hadn't priced in. The fix: a new feature computing what
  fraction of a team's total playing minutes last season came from
  players still registered to that club this season (using the FPL
  archive's gameweek-1 squad list, which is fixed before a ball is
  kicked). Tested specifically on the fixtures where teams' actual results
  diverged most from what the model expected in their first 8 games of a
  season — a "surprise" segment spanning 15 team-seasons across 8 years,
  not just Newcastle. It improved the model overall, improved more on the
  surprise segment specifically (the exact hypothesis being tested), and
  — the rare and important part — **held on the most recent season
  (2025-26) specifically**, not just on average. It did regress on the
  two oldest seasons in the test window. Promoted anyway: recent-season
  evidence is this project's strongest signal, and the regression was
  small and old.
- **Elo/Pi ratings and a win/loss streak feature.** (EXP-2026-09,
  EXP-2026-10) Confirmed to measurably help (RPS `0.20725 → 0.20656`
  from the streak feature alone) and confirmed *not* redundant with the
  Data Hub's own display ranking — removing them regressed the model.
  These were the first real "current form" signals the model had, and
  they earned their place with a direct before/after test, not
  intuition.
- **A wider training window for corners specifically — but only
  corners.** (EXP-2026-11, EXP-2026-12) Corners genuinely improves from
  12 seasons of history instead of 8 (MAE `2.69 → 2.67` on the real
  holdout). Scoreline and cards do *not* benefit from the same wider
  window — scoreline actually gets worse. This is why the project now
  trains each market on its own window rather than one shared one.
- **A from-scratch Poisson model beats Dixon-Coles/Bivariate-Poisson once
  it can see Elo/Pi.** (EXP-2026-15, EXP-2026-16) penaltyblog's built-in
  Dixon-Coles and Bivariate-Poisson models have no way to accept extra
  signals like team ratings — they only see goals and team identity. A
  custom-built Poisson model that *can* take Elo/Pi as an input beat both
  of them on every single fold, no exceptions, just by adding those two
  numbers. It still doesn't beat the main XGBoost model overall, but it
  now quietly powers the Over/Under 2.5 goals market specifically, where
  it has a real (if thin) edge.
- **A one-match-stale bug in xG-based rolling features.** (OPS-2026-03)
  Two separate no-lookahead safeguards were accidentally stacked on top
  of each other, so every rolling xG/set-piece feature was one match
  older than intended — for years, invisibly. Worse: this only affected
  *training* data, not live serving, meaning the model had been trained
  on stale features but served with fresh ones the whole time — a hidden
  train/serve mismatch. Fixed regardless of the metric moving slightly
  worse (see the "walk-forward average lies" pattern below for why that
  wasn't a reason to leave the bug in).
- **A correctness fix for cross-competition rest days.** (EXP-2026-17)
  Rest days were computed from Premier League matches only, so a team
  that actually played a midweek Champions League or cup fixture looked
  more rested than it really was. Fixed by pulling in Champions League/
  Europa League/Conference League/FA Cup/EFL Cup fixture dates. Flat on
  the metric, but it's a correctness fix (the true rest-day count), so it
  shipped regardless.

## 2. Things that looked promising and were rejected anyway — and why that's the point

The project measures a huge number of ideas that don't make the cut. Each
rejection here saved a future retest of the same idea, which is the entire
point of writing them down.

- **Retraining the model more often, or weighting recent matches more
  heavily, does not help.** Tested twice, independently, both ways:
  a full multi-cadence study (no retrain vs. every 10/19 matches,
  season-weighted, season-only — EXP-2026-05) and a from-scratch
  within-season-only model with no multi-season fit at all (EXP-2026-06).
  Both lost to the simple frozen multi-season model, and lost badly on
  the most recent season specifically. **This is the same question the
  user's "make it recursively improve over the season" idea was
  independently proposing** — it had already been tried twice and closed
  before that conversation happened, which is exactly why the follow-up
  research (squad continuity, above) went looking for a genuinely
  different kind of signal instead of a third variant of "retrain more."
- **A single global "which scoreline model wins" answer doesn't hide a
  better per-market or per-segment answer** — except for one specific,
  thin exception. Tested exhaustively (EXP-2026-14): the main XGBoost
  model wins every market outright, no exceptions, when compared
  head-to-head across all three model families. A "cold-start" segment
  that looked interesting on the surface turned out to be a measurement
  artifact (see below).
- **A newly-promoted team's first 10 matches is not a "cold-start
  segment" — it's just that one team.** (EXP-2026-14) Every season's
  "cold-start" fixture set, when actually inspected team-by-team, turned
  out to be a single newly-promoted club's own first 10 matches, re-run
  every year (2025-26's set was entirely Sunderland). The metric gap
  between models on this "segment" was noise from a sample of 10, not a
  real effect — the confidence intervals for every model overlapped
  heavily. Lesson carried forward into every later segment-based
  experiment: **check who's actually in a segment before trusting a
  number computed on it.**
- **Match "dominance" features (shots, shot quality, xG efficiency) and a
  wider historical window don't help the scoreline model**, even though
  the wider window *does* help corners specifically. (EXP-2026-11) Same
  data, same technique, opposite verdict depending on the market — a
  reminder that "more history" and "more features" aren't universal
  goods.
- **Blending or calibrating the final probabilities doesn't beat the raw
  model.** (EXP-2026-02) Platt scaling and a non-negative multi-model
  blend were both tested with a proper leakage-safe calibration split;
  both made things worse. The raw XGBoost probabilities, uncalibrated,
  are still what's best supported by evidence.
- **Projected player aggregates (rolling player-level rates rolled up to
  team level) regressed the model in every single walk-forward fold**,
  not just on average. (EXP-2026-03) The initial single-season test made
  it look like a razor-thin win; testing across all 5 folds made the
  rejection unambiguous. This is the project's clearest example of why a
  single holdout season is not enough evidence either way.
- **Shot-situation (set-piece share) features help corners but hurt
  cards**, and table-position "stakes" features hurt the main model
  outright despite being intuitively reasonable ("teams fighting
  relegation try harder"). Both are still computed but deliberately left
  out of the feature set that's actually used — cheap to keep computed,
  measured not to help.
- **A market-implied signal is not evidence of betting edge.** The
  project's own backtest of the model against real historical odds is
  flat-to-negative overall (`-4.6%` flat-stake yield, a 95% interval that
  spans zero) despite the model being genuinely better calibrated than
  the market on some markets. Being right about probabilities and having
  a profitable betting strategy are different claims, and this project
  is explicit that it has only ever established the first one.

## 3. The pattern that shows up again and again: a walk-forward average can lie

Three completely independent tuning attempts — retuning XGBoost's
hyperparameters after a bug fix (OPS-2026-03), widening Dixon-Coles/
Bivariate-Poisson's recency-decay window (EXP-2026-13), and testing a
12-season training window for the scoreline model (EXP-2026-11) — all
showed the *exact same failure shape*: the average across 5 historical
seasons improves, but the single most recent season (2025-26) gets worse.

Three unrelated experiments converging on the same failure mode is itself
evidence — it means 2025-26 is a genuinely different-shaped season for this
data, not a fluke in any one of these attempts. The project's response was
to make "does it also hold on the fixed, most-recent season specifically"
a non-negotiable second check for every promotion decision, not just a nice-
to-have — a rule that later correctly *passed* the squad-continuity feature
(finding #1 above) precisely because that one held up on both checks where
these three didn't.

## 4. Real bugs found along the way (not modelling questions — actual defects)

- **FPL's player ID is not stable across seasons.** Bruno Fernandes is
  player `333` in the 2022-23 archive and `373` in 2023-24 — same player,
  same club. Any code joining player data across a season boundary has to
  match by name, not by the archive's own numeric ID, or it silently
  merges different players' stats together. Found while building the
  squad-continuity feature, when several major clubs came back with
  nonsensical `0.0` continuity values.
- **A one-match-stale rolling xG bug** (described in section 1) that had
  been quietly training the model on data one match older than it should
  have been, for an unknown but non-trivial period, without ever showing
  up as an obvious error — just a slightly wrong number.
- **The Data Hub's team rankings could put a newly-promoted or returning
  club implausibly high** after just a handful of matches, because it was
  using an already-updated current-season rating rather than a stable
  pre-season baseline. Fixed by anchoring display rankings to the frozen
  pre-season rating (this is display-only — it never touched predictions).
- **A retrain could crash, or silently serve the wrong model, if the
  Over/Under-2.5-only covariate-Poisson model ever won the overall model
  comparison outright** (not just its usual per-market role) — two
  separate bugs in `models/manifest.py` that had simply never been
  reachable before, since the main XGBoost model has always won by a
  comfortable margin on real production data. Found the hard way: adding
  squad continuity to the training data was enough to flip that
  comparison on one test's deliberately small sample, and both bugs fired
  immediately. Fixed both regardless of how the trigger happened to arise,
  since a future feature or dataset change could reach the same edge case
  for real.

## 5. Data sources: what's usable, what's blocked, what's a dead end

- **Usable and already in production:** football-data.co.uk (deep
  historical results/odds), football-data.org (current season, and now
  Champions League fixture dates), Understat (team and shot-level xG),
  the official FPL API and the vaastav historical archive, ESPN (confirmed
  lineups, and now Europa League/Conference League/FA Cup/EFL Cup
  fixture dates), The Odds API (match result and totals only).
- **Blocked, not rejected:** ClubElo (`api.clubelo.com`) has never once
  responded across four separate attempts on different days, despite the
  main `clubelo.com` site working fine — looks like an issue with that
  specific API host. The code to use it (a promoted-team strength prior)
  is fully written and unit-tested against a synthetic response shape,
  but has never been evaluated against real data because it's never been
  reachable.
- **Confirmed dead ends, not worth revisiting:**
  - **FBref / Sports Reference** — their own published terms explicitly
    prohibit automated access, and they say outright they can't offer an
    API because they license their data from others who forbid
    redistribution. This is a licensing wall, not a technical one.
  - **Sofascore and FotMob** — both have deliberately built anti-scraping
    protection (FotMob's signed per-request token, Sofascore's
    Cloudflare bot management with TLS-fingerprint checks) specifically
    to stop this kind of access. Using either would mean building a
    bypass for an active technical control, which this project has
    decided not to do — a different, harder line than "no published
    terms."
  - **StatsBomb Open Data** — only two men's Premier League seasons exist
    in it at all (2015/16 and 2003/04), nowhere near continuous with the
    project's 8-season training window.
  - **football-data.org match statistics (possession, shots)** — checked
    directly against a real finished match; the free tier simply doesn't
    return this field at all, for any match.
- **The possession problem, specifically:** no source found so far has
  possession data that's both continuous across the full 8-season
  training window *and* available live for the current season on a
  usable licence. The project already has *some* current-season
  possession data (via the same undocumented feed used for live scores)
  but only shows it for display — adding it as a partial, mostly-missing
  feature would very likely repeat the same "asymmetric coverage adds
  noise, not signal" pattern already confirmed with shot-situation
  features.

## 6. Open threads worth revisiting

- **Cards features from EXP-2026-11's dominance study** won 3 of 5 folds
  outright but were inconclusive on the most-recent-season check — the
  most interesting unresolved thread in the project, worth another look
  once more seasons of data accumulate.
- **The covariate-Poisson model's Over/Under 2.5 edge is thin** (EXP-2026-16)
  — it wins on average and on the most recent season, but loses 2 of 5
  folds by a bigger margin than it wins by in any single fold. Kept live
  as a deliberate, disclosed judgment call, not a confident win; a future
  refinement (fitting the model's correlation term properly instead of
  fixing it at zero) could make this a much stronger candidate.
- **Squad continuity's own regressed folds** (the two oldest seasons in
  its test window) were not investigated further before promotion — worth
  checking whether they share a cause (e.g., thinner FPL archive coverage
  for their own "prior season" lookup) if a future retrain shows this
  feature underperforming live.
- **Manager changes and transfer spend** remain untested candidate
  leading-indicators for the same "off-season squad change" question
  squad continuity addresses — no dataset has been sourced or licensed
  for either yet.
