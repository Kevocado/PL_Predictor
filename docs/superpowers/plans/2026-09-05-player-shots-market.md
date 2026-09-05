# Player Shots & Shots-on-Target Market Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real, measured player-level shots and shots-on-target predictions (`expected_shots`, `expected_shots_on_target`, `anytime_shot_on_target_prob`) to the existing anytime goal/assist surface, built on Understat's per-shot data the project already fetches.

**Architecture:** A layered Understat→FPL player-id crosswalk (runtime-cached, not a git-tracked artifact) feeds a per-player-per-match shot extraction (same cache-per-season pattern `understat_shots.py` already uses for team aggregates). The extraction merges into the existing historical player-gameweek frame so it flows through `player_form.py`'s existing rolling-rate machinery unchanged. A new team-shot-volume scaling anchor comes from `rolling_form.py`'s already-computed (but currently unused) rolling `shots_for`/`shots_against` — no new team-level feature needed. `player_goals.py`'s existing Ridge-per-position rate model gains two new targets. A walk-forward gate against a naive baseline runs before this reaches live serving.

**Tech Stack:** Python 3.13, pandas, scikit-learn (Ridge), pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-player-shots-market-design.md` (three corrections made during planning, each already committed and re-verified against real data: the crosswalk's matching algorithm — Section 1 — was rewritten from an assumed simple lookup to a measured three-stage matcher after testing against live data showed the naive version only clears ~66%; the team-shot-volume scaling source — Section 4 — was corrected from a proposed new Understat feature to `rolling_form.py`'s already-computed rolling shots; the crosswalk's storage — Section 1 — was corrected from a git-tracked `manifest.py` artifact to a runtime-cached one, matching its sibling player-fitting functions).

## Global Constraints

- `shots_on_target` = Understat `result` in `{"Goal", "SavedShot"}` — `MissedShots`, `BlockedShot`, `ShotOnPost`, `OwnGoal` are off-target by the standard betting definition.
- Crosswalk matching, in order, first hit wins: (1) exact normalized-name match (HTML-unescape, then NFKD, then a manual translit table for `ø→o, đ→d, ł→l, ß→ss, æ→ae, œ→oe, ð→d, þ→th` both cases, then casefold+alnum-only), against FPL's `first_name+" "+second_name` or `web_name`; (2) if the first name has a nickname-table expansion, retry (1) with the expanded name; (3) surname-only match, only when it resolves to exactly one candidate (never guess on ambiguity). A player who never matches is logged and excluded — never a hard failure.
- `LEAGUE_AVERAGE_TEAM_SHOTS = 13.1` (measured: `(df["hs"].sum() + df["as"].sum()) / (2 * len(df))` over `football_data.load_training_data()`'s last 3 completed seasons — 13.13, rounded — same standard `FALLBACK_GOAL_EXPECTANCY` was held to).
- The crosswalk is runtime-cached (`ttl=24*3600`, same as `_get_position_priors`/`_get_lineup_model`) — never a git-tracked `models/*.json` artifact, never wired into `models/manifest.py`.
- No live-odds integration for this market (the Odds API's free bulk endpoint carries no player props) — this stays a model projection, same as goals/assists/corners/cards.
- A crosswalk miss, or a player with no shots data at all, degrades to `None`/absent fields — never crashes a fixture-detail response, matching this project's existing `is_fallback_prediction` convention.
- Before this reaches live serving (Task 9's gate), it must beat a naive position-average baseline on a genuinely held-out season — same "measured, not assumed" bar the goal-contribution model and every other market in this project were held to.

---

### Task 1: Understat → FPL crosswalk

**Files:**
- Modify: `src/pl_predictor/data/understat_shots.py` (add crosswalk builder)
- Modify: `src/pl_predictor/api/routes.py` (add cached wrapper, near `_get_position_priors` around line 251)
- Test: `tests/test_understat_shots.py`

**Interfaces:**
- Produces: `understat_shots.build_understat_fpl_crosswalk(shot_player_rows: pd.DataFrame, bootstrap: dict) -> dict[int, int]` — maps Understat `player_id` (int) to FPL `element_id` (int). `shot_player_rows` is a DataFrame with columns `player` (Understat display name, e.g. `"Ben White"`) and `player_id` (Understat's own id) — one row per unique Understat player, already deduplicated by the caller (this function does no season/file scanning itself, kept a pure function for testability).
- Consumes (Task 9's gate, and Task 6's live-serving wiring both need this): `routes._get_understat_fpl_crosswalk() -> dict[int, int]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_understat_shots.py` (new test cases; the file already exists with other `understat_shots` coverage per this project's test layout — if creating fresh, start with just these):

```python
import pandas as pd

from pl_predictor.data import understat_shots


def _bootstrap(elements):
    return {"elements": elements}


def _element(id_, first, second, web):
    return {"id": id_, "first_name": first, "second_name": second, "web_name": web}


def test_crosswalk_matches_exact_normalized_name():
    rows = pd.DataFrame([{"player": "Erling Haaland", "player_id": 100}])
    bootstrap = _bootstrap([_element(1, "Erling", "Haaland", "Haaland")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {100: 1}


def test_crosswalk_decodes_html_entities():
    rows = pd.DataFrame([{"player": "Luke O'Nien", "player_id": 101}])
    bootstrap = _bootstrap([_element(2, "Luke", "O&#039;Nien", "O'Nien")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {101: 2}


def test_crosswalk_transliterates_non_decomposable_letters():
    # NFKD alone cannot turn 'Ø' into 'O' -- confirmed against real FPL data
    # (Martin Ødegaard) during this plan's design.
    rows = pd.DataFrame([{"player": "Martin Odegaard", "player_id": 102}])
    bootstrap = _bootstrap([_element(3, "Martin", "Ødegaard", "Ødegaard")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {102: 3}


def test_crosswalk_expands_common_nicknames():
    rows = pd.DataFrame([{"player": "Ben White", "player_id": 103}])
    bootstrap = _bootstrap([_element(4, "Benjamin", "White", "White")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {103: 4}


def test_crosswalk_falls_back_to_unique_surname_match():
    # web_name is plain "Smith" (not "J.Smith", which would accidentally
    # exact-match at stage 1 via web_norm and never reach stage 3) --
    # "J Smith" matches neither the full name nor the web name, so this
    # only resolves through the surname-only fallback.
    rows = pd.DataFrame([{"player": "J Smith", "player_id": 104}])
    bootstrap = _bootstrap([_element(5, "Jordan", "Smith", "Smith")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {104: 5}


def test_crosswalk_never_guesses_on_ambiguous_surname():
    # Deliberately avoids an accidental exact web_name match at stage 1
    # (e.g. "A Murphy" vs a web_name of "A.Murphy" would exact-match and
    # never reach the ambiguous case this test means to exercise) --
    # "Alexander Murphy" matches neither candidate's full name (el7's
    # first_name is the nickname "Alex", not "Alexander") nor either
    # web_name, so it falls through to surname-only, where it hits both.
    rows = pd.DataFrame([{"player": "Alexander Murphy", "player_id": 105}])
    bootstrap = _bootstrap([
        _element(6, "Adam", "Murphy", "A.Murphy"),
        _element(7, "Alex", "Murphy", "Alex Murphy"),
    ])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert 105 not in result


def test_crosswalk_excludes_unmatchable_player_without_crashing():
    rows = pd.DataFrame([{"player": "Nobody Real", "player_id": 106}])
    bootstrap = _bootstrap([_element(8, "Someone", "Else", "Else")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_understat_shots.py -k crosswalk -v`
Expected: FAIL with `AttributeError: module 'pl_predictor.data.understat_shots' has no attribute 'build_understat_fpl_crosswalk'`

- [ ] **Step 3: Write the implementation**

Add to `src/pl_predictor/data/understat_shots.py` (near the top, after the existing `SET_PIECE_SITUATIONS` constant):

```python
import html as _html
import unicodedata

# Letters NFKD cannot decompose into their ASCII base form (confirmed
# directly against live FPL data: "Ødegaard" survives NFKD+casefold as
# "ødegaard", not "odegaard") -- both cases, since casefold happens after.
_EXTRA_TRANSLIT = str.maketrans({
    "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ß": "ss",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th",
})

# Common vs. formal first-name pairs -- confirmed the dominant real-world
# mismatch during this plan's design (FPL's first_name is the formal form,
# Understat records the common one, e.g. "Benjamin White" vs "Ben White").
# Not exhaustive; extend as real misses surface in production logs.
_NICKNAME_TO_FORMAL = {
    "ben": "benjamin", "josh": "joshua", "matt": "matthew", "mike": "michael",
    "tom": "thomas", "alex": "alexander", "nick": "nicholas", "dan": "daniel",
    "danny": "daniel", "sam": "samuel", "will": "william", "billy": "william",
    "joe": "joseph", "joey": "joseph", "jim": "james", "jimmy": "james",
    "jamie": "james", "rob": "robert", "bobby": "robert", "bob": "robert",
    "dave": "david", "davy": "david", "chris": "christopher",
    "harry": "harold", "charlie": "charles", "ed": "edward",
    "eddie": "edward", "ted": "edward", "tony": "anthony", "andy": "andrew",
    "steve": "stephen", "stevie": "stephen", "pat": "patrick",
    "paddy": "patrick", "ron": "ronald", "ronnie": "ronald",
    "fred": "frederick", "freddie": "frederick", "gerry": "gerald",
    "jerry": "gerald", "greg": "gregory", "ken": "kenneth",
    "kenny": "kenneth", "larry": "lawrence", "jack": "john",
    "johnny": "john", "jonny": "jonathan", "jon": "jonathan",
    "abdul": "abdullah",
}


def _normalise_crosswalk_name(value: str) -> str:
    """Same casefold+NFKD+alnum-only idiom as `models/player_goals.py::
    _normalise_name`, extended with HTML-entity decoding (FPL's raw
    `second_name` field can contain literal entities like `O&#039;Nien`)
    and `_EXTRA_TRANSLIT` for letters NFKD alone cannot decompose."""
    value = _html.unescape(str(value)).translate(_EXTRA_TRANSLIT)
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed.casefold() if char.isalnum())


def build_understat_fpl_crosswalk(shot_player_rows: pd.DataFrame, bootstrap: dict) -> dict[int, int]:
    """Map Understat `player_id` -> FPL `element_id`. `shot_player_rows` has
    one row per unique Understat player (`player`, `player_id` columns).
    Layered matching, first hit wins -- see this plan's Global Constraints
    for why each stage exists (measured against real data, not assumed).
    A player who matches nothing is simply absent from the returned dict,
    never a crash or a guess."""
    fpl_rows = []
    for element in bootstrap["elements"]:
        full_name = f"{element['first_name']} {element['second_name']}"
        fpl_rows.append((element["id"], element["first_name"], full_name, element["web_name"]))
    fpl_df = pd.DataFrame(fpl_rows, columns=["element_id", "first_name", "full_name", "web_name"])
    fpl_df["full_norm"] = fpl_df["full_name"].apply(_normalise_crosswalk_name)
    fpl_df["web_norm"] = fpl_df["web_name"].apply(_normalise_crosswalk_name)
    fpl_df["surname_norm"] = fpl_df["full_name"].apply(lambda n: _normalise_crosswalk_name(n.split()[-1]))

    crosswalk: dict[int, int] = {}
    for _, row in shot_player_rows.iterrows():
        parts = str(row["player"]).split()
        if not parts:
            continue
        first, surname = parts[0], parts[-1]
        full_norm = _normalise_crosswalk_name(row["player"])

        hit = fpl_df[(fpl_df["full_norm"] == full_norm) | (fpl_df["web_norm"] == full_norm)]
        if len(hit) == 1:
            crosswalk[int(row["player_id"])] = int(hit.iloc[0]["element_id"])
            continue

        expanded_first = _NICKNAME_TO_FORMAL.get(_normalise_crosswalk_name(first))
        if expanded_first:
            alt_norm = _normalise_crosswalk_name(f"{expanded_first} {surname}")
            hit = fpl_df[fpl_df["full_norm"] == alt_norm]
            if len(hit) == 1:
                crosswalk[int(row["player_id"])] = int(hit.iloc[0]["element_id"])
                continue

        surname_norm = _normalise_crosswalk_name(surname)
        hit = fpl_df[(fpl_df["surname_norm"] == surname_norm) | (fpl_df["web_norm"] == surname_norm)]
        if len(hit) == 1:
            crosswalk[int(row["player_id"])] = int(hit.iloc[0]["element_id"])
        # len(hit) == 0 or > 1: unmatched or ambiguous, excluded either way

    return crosswalk
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_understat_shots.py -k crosswalk -v`
Expected: PASS (all 7)

- [ ] **Step 5: Add the runtime-cached wrapper**

In `src/pl_predictor/api/routes.py`, add near `_get_position_priors` (around line 256):

```python
def _get_understat_fpl_crosswalk() -> dict:
    def build():
        shot_players = understat_shots.load_current_season_player_shot_rows()
        return understat_shots.build_understat_fpl_crosswalk(shot_players, _get_bootstrap())

    return _cached("understat_fpl_crosswalk", build, ttl=24 * 3600)
```

Add `understat_shots` to the existing `from ..data import fixtures as fixtures_mod` import block's neighbors — check the current imports at the top of `routes.py` first (`from ..data import espn, fpl_api, fpl_history`) and add `understat_shots` to that same import statement.

Note: `understat_shots.load_current_season_player_shot_rows()` doesn't exist yet — it's built in Task 2. This step only wires the cache; if Task 2 isn't done yet, leave this function uncalled (it's fine for it to reference a not-yet-existing function as long as nothing calls `_get_understat_fpl_crosswalk()` before Task 2 lands — no test in this task calls it, so this compiles-but-unused state is safe. Task 2's own tests will exercise the real function name.)

- [ ] **Step 6: Run the full test file**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_understat_shots.py -v`
Expected: all pass, including pre-existing tests in this file (if any) unaffected.

- [ ] **Step 7: Commit**

```bash
git add src/pl_predictor/data/understat_shots.py src/pl_predictor/api/routes.py tests/test_understat_shots.py
git commit -m "feat: add Understat-to-FPL player crosswalk"
```

---

### Task 2: Per-player-per-match shot extraction

**Files:**
- Modify: `src/pl_predictor/data/understat_shots.py`
- Test: `tests/test_understat_shots.py`

**Interfaces:**
- Consumes: nothing new from Task 1 directly (this task's own functions are what Task 1's `_get_understat_fpl_crosswalk` calls).
- Produces:
  - `understat_shots.load_current_season_player_shot_rows(season: str | None = None) -> pd.DataFrame` — one row per unique player (`player`, `player_id` columns) seen in the given season's (default: current) shot data. Feeds Task 1's crosswalk builder.
  - `understat_shots.load_player_shot_history(seasons: list[str] | None = None, force_refresh: bool = False) -> pd.DataFrame` — one row per (player, match): `player_id` (Understat id), `date`, `shots`, `shots_on_target`, `goals`, `x_g`. Feeds Task 3's merge into the historical training frame, and Task 6's live-serving lookup.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_understat_shots.py`:

```python
import numpy as np


def _shot_row(player, player_id, h_a, result, situation="OpenPlay", x_g="0.1"):
    return {
        "player": player, "player_id": player_id, "h_a": h_a, "result": result,
        "situation": situation, "x_g": x_g,
    }


def test_player_shot_extraction_counts_shots_and_shots_on_target(monkeypatch, tmp_path):
    monkeypatch.setattr(understat_shots, "UNDERSTAT_SHOTS_CACHE_DIR", tmp_path)
    fixtures = pd.DataFrame([{"understat_id": "111", "date": "2025-08-16", "team_home": "Arsenal", "team_away": "Chelsea"}])
    monkeypatch.setattr(understat_shots, "_fetch_season_fixtures_with_id", lambda *a, **k: fixtures)

    shots = pd.DataFrame([
        _shot_row("Bukayo Saka", 501, "h", "Goal"),
        _shot_row("Bukayo Saka", 501, "h", "MissedShots"),
        _shot_row("Bukayo Saka", 501, "h", "SavedShot"),
        _shot_row("Bukayo Saka", 501, "h", "BlockedShot"),
    ])
    scraper = object()
    monkeypatch.setattr(understat_shots, "fetch_match_shots", lambda _scraper, _id, force_refresh=False: shots)
    monkeypatch.setattr(understat_shots.pb.scrapers, "Understat", lambda *a, **k: scraper)

    df = understat_shots.load_player_shot_history(seasons=["2025-26"])
    row = df[df["player_id"] == 501].iloc[0]
    assert row["shots"] == 4
    assert row["shots_on_target"] == 2  # Goal + SavedShot only
    assert row["date"] == pd.Timestamp("2025-08-16")


def test_current_season_player_shot_rows_deduplicates_by_player_id(monkeypatch, tmp_path):
    monkeypatch.setattr(understat_shots, "UNDERSTAT_SHOTS_CACHE_DIR", tmp_path)
    history = pd.DataFrame([
        {"player": "Bukayo Saka", "player_id": 501, "date": pd.Timestamp("2025-08-16"), "shots": 4, "shots_on_target": 2, "goals": 1, "x_g": 0.4},
        {"player": "Bukayo Saka", "player_id": 501, "date": pd.Timestamp("2025-08-23"), "shots": 2, "shots_on_target": 1, "goals": 0, "x_g": 0.2},
    ])
    monkeypatch.setattr(understat_shots, "load_player_shot_history", lambda seasons=None, force_refresh=False: history)

    rows = understat_shots.load_current_season_player_shot_rows(season="2025-26")
    assert len(rows) == 1
    assert rows.iloc[0]["player_id"] == 501
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_understat_shots.py -k "player_shot_extraction or current_season_player_shot_rows" -v`
Expected: FAIL — functions don't exist yet.

- [ ] **Step 3: Write the implementation**

Add to `src/pl_predictor/data/understat_shots.py`, after `_aggregate_match_dominance`/`_load_season_match_dominance` (this mirrors that exact per-season-cache pattern, just at player granularity instead of team):

```python
_PLAYER_SHOT_COLS = ["player", "player_id", "date", "shots", "shots_on_target", "goals", "x_g"]


def _aggregate_player_shots(shots: pd.DataFrame, match_date) -> list[dict]:
    """One row per player who took at least one shot in this match."""
    shots = shots.copy()
    shots["x_g"] = shots["x_g"].astype(float)
    shots["is_on_target"] = shots["result"].isin({"Goal", "SavedShot"})
    shots["is_goal"] = shots["result"] == "Goal"

    rows = []
    for (player, player_id), group in shots.groupby(["player", "player_id"]):
        rows.append({
            "player": player,
            "player_id": int(player_id),
            "date": match_date,
            "shots": int(len(group)),
            "shots_on_target": int(group["is_on_target"].sum()),
            "goals": int(group["is_goal"].sum()),
            "x_g": float(group["x_g"].sum()),
        })
    return rows


def _load_season_player_shots(season: str, force_refresh: bool, request_delay: float) -> pd.DataFrame:
    """One row per (player, match) for a single season -- same one-
    aggregate-file-per-season cache layer as `_load_season_shot_situation`/
    `_load_season_match_dominance`, reusing the same per-match raw shot
    files (`{understat_id}.csv`) those already populate."""
    agg_cache_path = UNDERSTAT_SHOTS_CACHE_DIR / f"_player_shots_{season}.csv"
    if agg_cache_path.exists() and not force_refresh:
        return pd.read_csv(agg_cache_path, parse_dates=["date"])

    fixtures = _fetch_season_fixtures_with_id(season, force_refresh=force_refresh)
    scraper = pb.scrapers.Understat(COMPETITION, season)
    rows = []
    for _, fx in fixtures.iterrows():
        understat_id = str(fx["understat_id"])
        cache_path = UNDERSTAT_SHOTS_CACHE_DIR / f"{understat_id}.csv"
        was_cached = cache_path.exists()
        try:
            shots = fetch_match_shots(scraper, understat_id, force_refresh=force_refresh)
        except RuntimeError as exc:
            print(f"  ! Skipping player shots for match {understat_id}: {exc}")
            continue
        if not was_cached and request_delay:
            time.sleep(request_delay)
        if shots.empty:
            continue
        rows.extend(_aggregate_player_shots(shots, fx["date"]))

    df = pd.DataFrame(rows, columns=_PLAYER_SHOT_COLS)
    df.to_csv(agg_cache_path, index=False)
    return df


def load_player_shot_history(
    seasons: list[str] | None = None, force_refresh: bool = False, request_delay: float = 0.3
) -> pd.DataFrame:
    """One row per (player, match) across the given seasons (default: same
    completed-seasons window every other historical loader in this project
    uses). `player_id` is Understat's own id -- callers join onto FPL
    `element_id` via `build_understat_fpl_crosswalk` themselves; this
    function has no FPL dependency, matching `load_shot_situation_data`'s
    own separation of concerns."""
    from . import understat as understat_mod

    seasons = seasons or understat_mod.default_completed_seasons()
    frames = []
    for season in seasons:
        try:
            frames.append(_load_season_player_shots(season, force_refresh, request_delay))
        except RuntimeError as exc:
            print(f"  ! Skipping player shots {season}: {exc}")

    if not frames:
        return pd.DataFrame(columns=_PLAYER_SHOT_COLS)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_current_season_player_shot_rows(season: str | None = None) -> pd.DataFrame:
    """One row per unique player seen in the given (default: current)
    season's shot data -- exactly what `build_understat_fpl_crosswalk`
    needs, deduplicated so the crosswalk builder never sees the same
    player twice."""
    from . import understat as understat_mod

    season = season or understat_mod.default_completed_seasons(n=1)[-1]
    history = load_player_shot_history(seasons=[season])
    if history.empty:
        return pd.DataFrame(columns=["player", "player_id"])
    return history.drop_duplicates(subset=["player_id"])[["player", "player_id"]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_understat_shots.py -v`
Expected: all pass, including Task 1's crosswalk tests (unaffected) and both new tests.

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/data/understat_shots.py tests/test_understat_shots.py
git commit -m "feat: extract per-player shot/shots-on-target history from cached Understat data"
```

---

### Task 3: Merge shots into the historical training frame; extend RATE_STATS

**Files:**
- Modify: `src/pl_predictor/features/player_form.py` (extend `RATE_STATS`)
- Modify: `src/pl_predictor/models/player_goals.py` (add the merge helper)
- Test: `tests/test_player_form.py`, `tests/test_player_goals.py`

**Interfaces:**
- Consumes: `understat_shots.load_player_shot_history` (Task 2), `routes._get_understat_fpl_crosswalk` — actually this task needs the crosswalk as a plain dict, not tied to `routes.py`'s cache, so it calls `understat_shots.build_understat_fpl_crosswalk` + `understat_shots.load_current_season_player_shot_rows` directly (same two functions `routes._get_understat_fpl_crosswalk` wraps) — this keeps `player_goals.py`'s training code independent of the `routes.py` caching layer, consistent with how it already calls `fpl_history.load_player_gw_history` directly rather than through any cache.
- Produces: `player_goals._load_history_with_shots(seasons: list[str] | None = None) -> pd.DataFrame` — same shape as `fpl_history.load_player_gw_history`'s output, with two additional columns (`shots`, `shots_on_target`) merged in wherever a crosswalk match and a same-date shot record exist (NaN otherwise — `player_form.py`'s existing NaN-safe rolling logic already handles gaps, per this morning's fix to exactly this class of bug). Task 5 (`fit_position_rate_models`) calls this instead of `fpl_history.load_player_gw_history` directly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_player_form.py` (extends the file this session's earlier fix already touched):

```python
def test_shots_and_shots_on_target_are_new_rate_stats():
    assert "shots" in player_form.RATE_STATS
    assert "shots_on_target" in player_form.RATE_STATS


def test_build_historical_player_form_computes_shots_per90_when_present():
    df = pd.DataFrame([
        {"element": 1, "season": "2025-26", "kickoff_time": pd.Timestamp("2025-08-16"), "minutes": 90, "goals_scored": 1, "assists": 0, "shots": 4, "shots_on_target": 2},
        {"element": 1, "season": "2025-26", "kickoff_time": pd.Timestamp("2025-08-23"), "minutes": 90, "goals_scored": 0, "assists": 1, "shots": 2, "shots_on_target": 1},
    ])
    played, feature_cols = build_historical_player_form(df)
    assert "shots_per90_last3" in feature_cols
    assert "shots_on_target_per90_last3" in feature_cols
    # First appearance has no prior data -- NaN, same convention as goals_per90.
    first_row = played.sort_values("kickoff_time").iloc[0]
    assert pd.isna(first_row["shots_per90_last3"])
    second_row = played.sort_values("kickoff_time").iloc[1]
    assert second_row["shots_per90_last3"] == pytest.approx(4 / 90 * 90)  # only the first match is "prior"
```

Add to `tests/test_player_goals.py`:

```python
def test_load_history_with_shots_merges_on_crosswalk_and_date(monkeypatch):
    history = pd.DataFrame([
        {"element": 42, "season": "2025-26", "kickoff_time": "2025-08-16T14:00:00Z", "minutes": 90, "goals_scored": 1, "assists": 0},
    ])
    monkeypatch.setattr(player_goals.fpl_history, "load_player_gw_history", lambda seasons=None: history)

    shots = pd.DataFrame([
        {"player": "Test Player", "player_id": 900, "date": pd.Timestamp("2025-08-16"), "shots": 5, "shots_on_target": 3, "goals": 1, "x_g": 0.5},
    ])
    monkeypatch.setattr(player_goals.understat_shots, "load_player_shot_history", lambda seasons=None: shots)
    monkeypatch.setattr(player_goals.understat_shots, "load_current_season_player_shot_rows", lambda season=None: shots[["player", "player_id"]])
    monkeypatch.setattr(player_goals, "_get_bootstrap_for_crosswalk", lambda: {"elements": [{"id": 42, "first_name": "Test", "second_name": "Player", "web_name": "Player"}]})

    merged = player_goals._load_history_with_shots(seasons=["2025-26"])
    row = merged[merged["element"] == 42].iloc[0]
    assert row["shots"] == 5
    assert row["shots_on_target"] == 3


def test_load_history_with_shots_leaves_unmatched_rows_as_nan(monkeypatch):
    history = pd.DataFrame([
        {"element": 42, "season": "2025-26", "kickoff_time": "2025-08-16T14:00:00Z", "minutes": 90, "goals_scored": 1, "assists": 0},
    ])
    monkeypatch.setattr(player_goals.fpl_history, "load_player_gw_history", lambda seasons=None: history)
    monkeypatch.setattr(player_goals.understat_shots, "load_player_shot_history", lambda seasons=None: pd.DataFrame(columns=["player_id", "date", "shots", "shots_on_target"]))
    monkeypatch.setattr(player_goals.understat_shots, "load_current_season_player_shot_rows", lambda season=None: pd.DataFrame(columns=["player", "player_id"]))
    monkeypatch.setattr(player_goals, "_get_bootstrap_for_crosswalk", lambda: {"elements": []})

    merged = player_goals._load_history_with_shots(seasons=["2025-26"])
    assert pd.isna(merged.iloc[0]["shots"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_form.py tests/test_player_goals.py -k "shots" -v`
Expected: FAIL — `RATE_STATS` doesn't include shots yet, `_load_history_with_shots` doesn't exist.

- [ ] **Step 3: Extend RATE_STATS**

In `src/pl_predictor/features/player_form.py`, change:

```python
RATE_STATS = ["expected_goals", "expected_assists", "expected_goal_involvements", "clean_sheets", "saves"]
```

to:

```python
RATE_STATS = ["expected_goals", "expected_assists", "expected_goal_involvements", "clean_sheets", "saves", "shots", "shots_on_target"]
```

This one-line change is the entire feature-engineering step: `build_historical_player_form`, `blended_current_form`, and `position_rate_priors` all loop over `RATE_STATS` generically already (confirmed by reading all three functions this session) — `shots`/`shots_on_target` automatically get rolling per-90 columns wherever the source DataFrame has those columns present, with no new code path.

- [ ] **Step 4: Write `_load_history_with_shots`**

In `src/pl_predictor/models/player_goals.py`, add near the top-level functions (after the module-level constants, before `fit_reliability_coefficients`):

```python
def _get_bootstrap_for_crosswalk() -> dict:
    """Separate from `api/routes.py`'s own `_get_bootstrap()` cache on
    purpose -- this module's training code must not depend on the routes
    layer's cache, the same reason it calls `fpl_history.load_player_gw_
    history` directly rather than through any `routes.py` wrapper. A plain
    uncached call is fine here: this only runs on a 24-hour cache-miss
    inside `_load_history_with_shots` itself, not per-request."""
    return fpl_api.fetch_bootstrap()


def _load_history_with_shots(seasons: list[str] | None = None) -> pd.DataFrame:
    """`fpl_history.load_player_gw_history`'s output, with `shots`/
    `shots_on_target` merged in via the Understat crosswalk. Unmatched
    players or unmatched match-dates simply get NaN in the two new
    columns -- `player_form.py`'s existing per-`RATE_STATS`-column
    presence checks already degrade gracefully for exactly this shape
    (confirmed this session: the `KeyError: 'minutes'` fix earlier today
    was closing precisely this class of gap)."""
    history = fpl_history.load_player_gw_history(seasons=seasons)

    # fpl_history's season format is "2023-24"; Understat's own is just the
    # bare start year, "2023" (confirmed in data/understat.py::default_
    # completed_seasons's own docstring) -- a plain split, no helper needed.
    shot_seasons = [s.split("-")[0] for s in (seasons or fpl_history.default_completed_seasons())]
    shots = understat_shots.load_player_shot_history(seasons=shot_seasons)
    if shots.empty:
        history["shots"] = float("nan")
        history["shots_on_target"] = float("nan")
        return history

    current_season_players = understat_shots.load_current_season_player_shot_rows()
    crosswalk = understat_shots.build_understat_fpl_crosswalk(current_season_players, _get_bootstrap_for_crosswalk())

    shots = shots.copy()
    shots["element"] = shots["player_id"].map(crosswalk)
    shots = shots.dropna(subset=["element"])
    shots["element"] = shots["element"].astype(int)
    shots["date"] = pd.to_datetime(shots["date"]).dt.date

    history = history.copy()
    history["_match_date"] = pd.to_datetime(history["kickoff_time"]).dt.date
    merged = history.merge(
        shots[["element", "date", "shots", "shots_on_target"]],
        left_on=["element", "_match_date"],
        right_on=["element", "date"],
        how="left",
    ).drop(columns=["_match_date", "date"], errors="ignore")
    return merged
```

Add the two new imports this needs at the top of `player_goals.py` (alongside the existing `from ..data import fpl_api, fpl_history`):

```python
from ..data import fpl_api, fpl_history, understat_shots
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_form.py tests/test_player_goals.py -k "shots" -v`
Expected: PASS (all 4)

- [ ] **Step 6: Run the full two test files to check for regressions**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_form.py tests/test_player_goals.py -v`
Expected: all pass — `RATE_STATS`'s two new entries must not affect any existing goals/assists-only assertion (they only add new optional columns, never change existing ones).

- [ ] **Step 7: Commit**

```bash
git add src/pl_predictor/features/player_form.py src/pl_predictor/models/player_goals.py tests/test_player_form.py tests/test_player_goals.py
git commit -m "feat: merge Understat shots into the historical player training frame"
```

---

### Task 4: Team-shot-volume scaling in `predict_player`

**Files:**
- Modify: `src/pl_predictor/models/player_goals.py`
- Test: `tests/test_player_goals.py`

**Interfaces:**
- Consumes: nothing from earlier tasks in this plan — this is a standalone scaling change to `predict_player`.
- Produces: `predict_player(..., team_expected_shots: float | None = None, ...)` — one new optional keyword argument. `rank_team_players` (Task 6) is the only caller that needs to pass it; every existing caller (including all of today's tests) keeps working unchanged since it defaults to `None` (degrades to `shots_scale = 1.0`, per Global Constraints).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_player_goals.py`:

```python
def test_predict_player_scales_shots_by_team_expected_shots():
    rates = {"goals_per90": 0.3, "assists_per90": 0.1, "avg_minutes": 90, "shots_per90": 2.0, "shots_on_target_per90": 1.0}
    pred = player_goals.predict_player(
        rates, team_goal_expectation=1.5, availability=1.0,
        team_expected_shots=player_goals.LEAGUE_AVERAGE_TEAM_SHOTS * 2,
    )
    # team takes 2x league-average shots -> shots_scale = 2.0, minutes_fraction=1.0, availability=1.0
    assert pred["expected_shots"] == pytest.approx(2.0 * 2.0)
    assert pred["expected_shots_on_target"] == pytest.approx(1.0 * 2.0)
    assert pred["anytime_shot_on_target_prob"] == pytest.approx(1 - math.exp(-1.0 * 2.0))


def test_predict_player_defaults_shots_scale_to_one_without_team_expected_shots():
    """No team_expected_shots passed (e.g. a Dixon-Coles/Bivariate-Poisson
    fit with no per-fixture feature context) -- must not crash, must not
    guess: shots_scale defaults to 1.0 (league average assumed)."""
    rates = {"goals_per90": 0.3, "assists_per90": 0.1, "avg_minutes": 90, "shots_per90": 2.0, "shots_on_target_per90": 1.0}
    pred = player_goals.predict_player(rates, team_goal_expectation=1.5, availability=1.0)
    assert pred["expected_shots"] == pytest.approx(2.0 * 1.0)


def test_predict_player_omits_shots_fields_without_shots_rate_data():
    """A player with no shots_per90 in their rates dict (crosswalk miss,
    or genuinely no shot data yet) gets None, not a fabricated number."""
    rates = {"goals_per90": 0.3, "assists_per90": 0.1, "avg_minutes": 90}
    pred = player_goals.predict_player(rates, team_goal_expectation=1.5, availability=1.0)
    assert pred["expected_shots"] is None
    assert pred["expected_shots_on_target"] is None
    assert pred["anytime_shot_on_target_prob"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -k "shots_scale or shots_rate_data" -v`
Expected: FAIL — `predict_player` has no `team_expected_shots` parameter and returns no `expected_shots` key.

- [ ] **Step 3: Write the implementation**

In `src/pl_predictor/models/player_goals.py`, add the new constant near `LEAGUE_AVERAGE_TEAM_GOALS`:

```python
# Measured directly (`(df["hs"].sum() + df["as"].sum()) / (2 * len(df))`
# over `football_data.load_training_data()`'s last 3 completed seasons --
# 13.13, rounded), the same standard LEAGUE_AVERAGE_TEAM_GOALS was held to.
LEAGUE_AVERAGE_TEAM_SHOTS = 13.1
```

Modify `predict_player`'s signature and body. Change:

```python
def predict_player(
    rates: dict,
    team_goal_expectation: float,
    availability: float,
    reliability_coeffs: dict | None = None,
    expected_minutes: float | None = None,
    position: str | None = None,
    is_home: bool = False,
    position_rate_models: dict | None = None,
    is_penalty_taker: bool = False,
    is_set_piece_taker: bool = False,
) -> dict:
```

to:

```python
def predict_player(
    rates: dict,
    team_goal_expectation: float,
    availability: float,
    reliability_coeffs: dict | None = None,
    expected_minutes: float | None = None,
    position: str | None = None,
    is_home: bool = False,
    position_rate_models: dict | None = None,
    is_penalty_taker: bool = False,
    is_set_piece_taker: bool = False,
    team_expected_shots: float | None = None,
) -> dict:
```

Then, right after the existing `scale = strength_multiplier * minutes_fraction * availability` line, add:

```python
    shots_strength_multiplier = (team_expected_shots or LEAGUE_AVERAGE_TEAM_SHOTS) / LEAGUE_AVERAGE_TEAM_SHOTS
    shots_scale = shots_strength_multiplier * minutes_fraction * availability
```

Finally, in the function's `return {...}` block, add three new keys. The existing return statement is:

```python
    return {
        "expected_goals": lam_goals,
        "expected_assists": lam_assists,
        "anytime_goal_prob": anytime_probability(lam_goals),
        "anytime_assist_prob": anytime_probability(lam_assists),
        "anytime_goal_contribution_prob": anytime_probability(lam_goals + lam_assists),
    }
```

Replace it with:

```python
    shots_per90 = rates.get("shots_per90")
    shots_on_target_per90 = rates.get("shots_on_target_per90")
    lam_shots = shots_per90 * shots_scale if shots_per90 is not None else None
    lam_shots_on_target = shots_on_target_per90 * shots_scale if shots_on_target_per90 is not None else None

    return {
        "expected_goals": lam_goals,
        "expected_assists": lam_assists,
        "anytime_goal_prob": anytime_probability(lam_goals),
        "anytime_assist_prob": anytime_probability(lam_assists),
        "anytime_goal_contribution_prob": anytime_probability(lam_goals + lam_assists),
        "expected_shots": lam_shots,
        "expected_shots_on_target": lam_shots_on_target,
        "anytime_shot_on_target_prob": anytime_probability(lam_shots_on_target) if lam_shots_on_target is not None else None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -v`
Expected: all pass, including every pre-existing `predict_player` test (they don't pass `team_expected_shots` or shots rates, so they exercise the `None`-default paths this task adds).

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/models/player_goals.py tests/test_player_goals.py
git commit -m "feat: scale player shot predictions by team-level expected shot volume"
```

---

### Task 5: Extend `fit_position_rate_models` with shots targets

**Files:**
- Modify: `src/pl_predictor/models/player_goals.py`
- Test: `tests/test_player_goals.py`

**Interfaces:**
- Consumes: `player_goals._load_history_with_shots` (Task 3), `player_form.RATE_STATS` including shots (Task 3).
- Produces: `fit_position_rate_models`'s returned dict gains two new `(position, target)` keys: `(position, "shots")` and `(position, "shots_on_target")` for each of `DEF`/`MID`/`FWD` — same shape as the existing `(position, "goals")`/`(position, "assists")` keys, so `predict_player`'s existing `position_rate_models.get((position, "goals"))`-style lookups extend to shots with the same pattern (Task 7 wires this).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_player_goals.py`:

```python
def test_fit_position_rate_models_includes_shots_targets(monkeypatch):
    rows = []
    rng_goals = [0, 1, 0, 2, 1, 0, 1, 0, 0, 1] * 30
    for i in range(300):
        rows.append({
            "element": i % 20, "season": "2025-26", "position": "FWD", "was_home": i % 2 == 0,
            "goals_scored": rng_goals[i % len(rng_goals)], "assists": 0, "shots": 2 + (i % 4), "shots_on_target": 1 + (i % 2),
            "minutes": 90,
            **{f"{stat}_last{w}": 0.5 for stat in ["goals_per90", "assists_per90", "expected_goals_per90", "expected_assists_per90", "threat", "creativity", "shots_per90", "shots_on_target_per90"] for w in (3, 5, 10)},
        })
    df = pd.DataFrame(rows)
    monkeypatch.setattr(player_goals, "_load_history_with_shots", lambda seasons=None: df)
    monkeypatch.setattr(player_goals.player_form, "build_historical_player_form", lambda history: (history, []))

    models = player_goals.fit_position_rate_models()
    assert ("FWD", "shots") in models
    assert ("FWD", "shots_on_target") in models
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -k fit_position_rate_models_includes_shots -v`
Expected: FAIL — `fit_position_rate_models` still calls `fpl_history.load_player_gw_history` directly and only fits goals/assists.

- [ ] **Step 3: Write the implementation**

In `src/pl_predictor/models/player_goals.py`, modify `fit_position_rate_models`. Current body:

```python
def fit_position_rate_models(seasons: list[str] | None = None) -> dict:
    """Fit separate Ridge rate models for defenders, midfielders and forwards."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    seasons = seasons or fpl_history.default_completed_seasons()
    history = fpl_history.load_player_gw_history(seasons=seasons)
    played, _ = player_form.build_historical_player_form(history)
    models = {}
    for position in ("DEF", "MID", "FWD"):
        position_rows = played[played["position"] == position].copy()
        position_rows["was_home"] = position_rows["was_home"].astype(float)
        train = position_rows.dropna(subset=RATE_FEATURES)
        if len(train) < 250:
            continue
        for target, target_col in (("goals", "goals_scored"), ("assists", "assists")):
            target_rate = train[target_col] / train["minutes"] * 90
            model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
            model.fit(train[RATE_FEATURES], target_rate)
            models[(position, target)] = model
    return models
```

Change the `seasons`/`history` lines and the target loop:

```python
def fit_position_rate_models(seasons: list[str] | None = None) -> dict:
    """Fit separate Ridge rate models for defenders, midfielders and forwards."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    seasons = seasons or fpl_history.default_completed_seasons()
    history = _load_history_with_shots(seasons=seasons)
    played, _ = player_form.build_historical_player_form(history)
    models = {}
    for position in ("DEF", "MID", "FWD"):
        position_rows = played[played["position"] == position].copy()
        position_rows["was_home"] = position_rows["was_home"].astype(float)
        train = position_rows.dropna(subset=RATE_FEATURES)
        if len(train) < 250:
            continue
        for target, target_col in (("goals", "goals_scored"), ("assists", "assists"), ("shots", "shots"), ("shots_on_target", "shots_on_target")):
            target_train = train.dropna(subset=[target_col])
            if len(target_train) < 250:
                continue
            target_rate = target_train[target_col] / target_train["minutes"] * 90
            model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
            model.fit(target_train[RATE_FEATURES], target_rate)
            models[(position, target)] = model
    return models
```

The `target_train = train.dropna(subset=[target_col])` line matters: `goals_scored`/`assists` are always present in the base historical frame, but `shots`/`shots_on_target` are only present where Task 3's crosswalk actually matched — dropping NaNs per-target (rather than once for the whole loop) means a low shots-match-rate never shrinks the goals/assists training set, and a shots model with too few matched rows (`< 250`) is simply skipped for that position rather than fit on a tiny, unreliable sample.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -k fit_position_rate_models_includes_shots -v`
Expected: PASS

- [ ] **Step 5: Run the full test file**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -v`
Expected: all pass, including the pre-existing goals/assists-only `fit_position_rate_models` coverage (if any) — the per-target `dropna` change means goals/assists behavior is unchanged when `shots`/`shots_on_target` columns are entirely absent (mocked-away in older tests) since those two targets are just additional loop iterations, not a change to the existing ones.

- [ ] **Step 6: Commit**

```bash
git add src/pl_predictor/models/player_goals.py tests/test_player_goals.py
git commit -m "feat: fit position-level shots and shots-on-target rate models"
```

---

### Task 6: Wire team-shot-volume and live per-player shots into serving

**Files:**
- Modify: `src/pl_predictor/models/player_goals.py` (`rank_team_players`)
- Modify: `src/pl_predictor/api/routes.py` (`_rank_fixture_players`, `_get_current_season_player_shots`)
- Test: `tests/test_player_goals.py`

**Interfaces:**
- Consumes: Task 4's `predict_player(..., team_expected_shots=...)`, Task 1's `_get_understat_fpl_crosswalk`, Task 2's `load_player_shot_history`.
- Produces: `rank_team_players(..., team_expected_shots: float | None = None, player_shots_by_element: dict[int, dict] | None = None, ...)` — two new optional parameters. `routes._rank_fixture_players` is the only caller that needs to pass them.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_player_goals.py`:

```python
def test_rank_team_players_threads_team_expected_shots_and_live_shots_into_predict_player(monkeypatch):
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal"}],
        "elements": [{"id": 10, "team": 1, "element_type": 4, "web_name": "Striker", "status": "a", "first_name": "Test", "second_name": "Striker"}],
    }
    monkeypatch.setattr(player_goals.fpl_api, "fetch_player_summary", lambda *a, **k: (pd.DataFrame(), None))
    monkeypatch.setattr(player_goals.player_form, "blended_current_form", lambda *a, **k: ({"avg_minutes": 90.0, "shots_per90": 2.0, "shots_on_target_per90": 1.0}, "current"))
    monkeypatch.setattr(player_goals.player_form, "current_start_features", lambda *a, **k: {})
    monkeypatch.setattr(player_goals, "predict_lineup", lambda *a, **k: {"predicted_starter": True, "expected_minutes": 90.0})

    captured = {}
    real_predict_player = player_goals.predict_player

    def spy_predict_player(*args, **kwargs):
        captured["team_expected_shots"] = kwargs.get("team_expected_shots")
        return real_predict_player(*args, **kwargs)

    monkeypatch.setattr(player_goals, "predict_player", spy_predict_player)

    player_goals.rank_team_players(
        "Arsenal", team_goal_expectation=1.8, bootstrap=bootstrap, current_event=3,
        position_priors={}, is_home=True, team_expected_shots=20.0,
    )
    assert captured["team_expected_shots"] == 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -k threads_team_expected_shots -v`
Expected: FAIL — `rank_team_players` has no `team_expected_shots` parameter and never passes it to `predict_player`.

- [ ] **Step 3: Update `rank_team_players`**

In `src/pl_predictor/models/player_goals.py`, add `team_expected_shots: float | None = None` to `rank_team_players`'s signature (alongside the existing `is_home: bool = False,` line), and thread it into the `predict_player(...)` call inside the `for el, position, history, rates, confidence, start_features, lineup in player_data:` loop — change:

```python
        pred = predict_player(
            rates, team_goal_expectation, availability, reliability_coeffs,
            expected_minutes=lineup["expected_minutes"], position=position, is_home=is_home,
            position_rate_models=position_rate_models, is_penalty_taker=is_penalty_taker,
            is_set_piece_taker=is_set_piece_taker,
        )
```

to:

```python
        pred = predict_player(
            rates, team_goal_expectation, availability, reliability_coeffs,
            expected_minutes=lineup["expected_minutes"], position=position, is_home=is_home,
            position_rate_models=position_rate_models, is_penalty_taker=is_penalty_taker,
            is_set_piece_taker=is_set_piece_taker, team_expected_shots=team_expected_shots,
        )
```

Also add `expected_shots`, `expected_shots_on_target`, `anytime_shot_on_target_prob` to the `results.append({...})` dict via `**pred` — no change needed there, since `**pred` already spreads every key `predict_player` returns, including the three new ones from Task 4.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -k threads_team_expected_shots -v`
Expected: PASS

- [ ] **Step 5: Wire live per-player shots and team-expected-shots in `routes.py`**

In `src/pl_predictor/api/routes.py`, add a cached current-season player-shots lookup near `_get_odds_df` (around line 300):

```python
def _get_current_season_player_shots() -> pd.DataFrame:
    def build():
        return understat_shots.load_player_shot_history(seasons=[understat.default_completed_seasons(n=1)[-1]])

    return _cached("current_season_player_shots", build, ttl=_LIVE_CACHE_TTL_SECONDS)
```

Add `understat` to the existing data imports (alongside `understat_shots` added in Task 1's Step 5).

In `_rank_fixture_players` (around line 1067), the function currently ends its setup block with:

```python
    lineup_model = _get_lineup_model()
    position_rate_models = _get_position_rate_models()
    contribution_model = _get_ready_goal_contribution_model()
```

After that, add:

```python
    matches_df = _get_matches_df()
    team_expected_shots_by_side = {}
    if not matches_df.empty:
        from ..features import rolling_form
        form = rolling_form.latest_form(matches_df)
        for side_team, prefix in ((home, "home_"), (away, "away_")):
            col = f"{prefix}last_10_shots_for"
            if side_team in form.index and col in form.columns and pd.notna(form.loc[side_team, col]):
                team_expected_shots_by_side[side_team] = float(form.loc[side_team, col])
```

`_get_matches_df()` (defined at `routes.py:179`) is confirmed the right source: `football_data.load_training_data()` plus the current-season partial — the same football-data.co.uk-backed frame `_get_models()` itself uses, and the only matches frame in this codebase carrying real `hs`/`as` shot columns (`football_data_org` and the FPL-API fixture fallback don't). `_rank_fixture_players` doesn't already hold a local `matches_df` variable at this point in the function (only `bootstrap`/`current_event`/the model helpers) — this call adds one, reusing the same 5-minute-TTL cache every other `_get_matches_df()` call site in this file already shares, so it costs nothing extra beyond the first caller in that window.

Then update the `rank(...)` closure's `player_goals.rank_team_players(...)` call to pass `team_expected_shots=team_expected_shots_by_side.get(team)`:

```python
    def rank(team: str, team_goal_expectation: float, is_home: bool) -> list[PlayerPrediction]:
        ranked = player_goals.rank_team_players(
            team, team_goal_expectation, bootstrap, current_event, position_priors,
            reliability_coeffs=reliability_coeffs, lineup_model=lineup_model,
            position_rate_models=position_rate_models, contribution_model=contribution_model,
            is_home=is_home, confirmed_starters=confirmed_lineups.get(team),
            confirmed_starter_ids=confirmed_starter_ids,
            team_expected_shots=team_expected_shots_by_side.get(team),
        )
        return [PlayerPrediction(**{k: p[k] for k in PlayerPrediction.model_fields}) for p in ranked]
```

- [ ] **Step 6: Run the full player_goals test file**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/pl_predictor/models/player_goals.py src/pl_predictor/api/routes.py tests/test_player_goals.py
git commit -m "feat: wire team-level expected shots into live player-shots serving"
```

---

### Task 7: API surface — `PlayerPrediction` schema

**Files:**
- Modify: `src/pl_predictor/api/schemas.py`
- Test: `tests/test_player_goals.py` (schema field presence, not a new file)

**Interfaces:**
- Consumes: Task 4's `predict_player` output shape (already produces the three new dict keys), Task 6's `rank_team_players` (already spreads them via `**pred`).
- Produces: `PlayerPrediction` gains three new optional fields — the last piece connecting the already-flowing data to the actual HTTP response.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_player_goals.py`:

```python
def test_player_prediction_schema_carries_shots_fields():
    from pl_predictor.api.schemas import PlayerPrediction

    assert "expected_shots" in PlayerPrediction.model_fields
    assert "expected_shots_on_target" in PlayerPrediction.model_fields
    assert "anytime_shot_on_target_prob" in PlayerPrediction.model_fields
    # All three optional -- a crosswalk-miss player must validate with None.
    PlayerPrediction(
        player_id=1, name="Test", position="FWD", anytime_goal_prob=0.1,
        anytime_assist_prob=0.05, anytime_goal_contribution_prob=0.14,
        status="a", news="", confidence="current", predicted_starter=True,
        confirmed_starter=False, expected_minutes=90.0, is_penalty_taker=False,
        is_set_piece_taker=False,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -k schema_carries_shots -v`
Expected: FAIL — `PlayerPrediction` has no such fields yet (the bare-minimum constructor call in the test would already pass today since all *current* fields are required and supplied; the `model_fields` assertions are what fail).

- [ ] **Step 3: Write the implementation**

In `src/pl_predictor/api/schemas.py`, modify `PlayerPrediction`:

```python
class PlayerPrediction(BaseModel):
    player_id: int
    name: str
    position: str
    anytime_goal_prob: float
    anytime_assist_prob: float
    anytime_goal_contribution_prob: float
    status: str
    news: str
    confidence: str
    predicted_starter: bool
    confirmed_starter: bool
    expected_minutes: float
    is_penalty_taker: bool
    is_set_piece_taker: bool
```

to:

```python
class PlayerPrediction(BaseModel):
    player_id: int
    name: str
    position: str
    anytime_goal_prob: float
    anytime_assist_prob: float
    anytime_goal_contribution_prob: float
    status: str
    news: str
    confidence: str
    predicted_starter: bool
    confirmed_starter: bool
    expected_minutes: float
    is_penalty_taker: bool
    is_set_piece_taker: bool
    # None when the Understat crosswalk has no match for this player (or no
    # shots data exists yet) -- never a fabricated number. See
    # docs/superpowers/specs/2026-09-04-player-shots-market-design.md.
    expected_shots: float | None = None
    expected_shots_on_target: float | None = None
    anytime_shot_on_target_prob: float | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py -k schema_carries_shots -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite for the touched files**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_goals.py tests/test_player_form.py tests/test_understat_shots.py tests/test_fixture_detail.py tests/test_api_readiness.py -v`
Expected: all pass — `tests/test_fixture_detail.py`/`test_api_readiness.py` exercise the schema end-to-end and must not break from three new *optional* fields.

- [ ] **Step 6: Commit**

```bash
git add src/pl_predictor/api/schemas.py tests/test_player_goals.py
git commit -m "feat: surface player shots/shots-on-target predictions on the API"
```

---

### Task 8: Guard against the empty-history class of bug for the new columns

**Files:**
- Modify: `tests/test_player_form.py`

**Interfaces:** none new — this task is pure test coverage, closing the exact gap this session's own `KeyError: 'minutes'` incident revealed, now that `shots`/`shots_on_target` are new `RATE_STATS` columns subject to the identical failure mode.

- [ ] **Step 1: Write the test**

Add to `tests/test_player_form.py` (alongside `test_blended_current_form_handles_player_with_no_current_season_history`, added this session):

```python
def test_blended_current_form_handles_player_with_no_shots_data():
    """A player matched by the crosswalk but with a real (non-empty)
    current-season history that simply has no shots/shots_on_target
    columns at all (e.g. their Understat data hasn't been merged in yet)
    must not crash -- same empty-column-guard class of bug this session's
    `KeyError: 'minutes'` incident was."""
    history = pd.DataFrame([
        {"GW": 1, "minutes": 90, "goals_scored": 1, "assists": 0},
        {"GW": 2, "minutes": 90, "goals_scored": 0, "assists": 1},
    ])
    rates, confidence = blended_current_form(history, prior_season=None, position="FWD", position_priors={})
    assert "shots_per90" not in rates  # no source column -- absent, not fabricated
    assert confidence in ("current", "prior_season", "position_avg", "none")
```

- [ ] **Step 2: Run test to verify it passes immediately**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_form.py -k no_shots_data -v`
Expected: PASS already — `RATE_STATS`'s existing `if stat in played.columns` / `if stat in window_rows.columns` guards (present since before this plan, confirmed by reading `blended_current_form` in full this session) already handle a genuinely-absent column correctly. This test is a **regression guard**, not a bug fix — if it fails, something in Tasks 3-6 broke that existing guard, and that must be fixed before proceeding, not this test loosened.

- [ ] **Step 3: Commit**

```bash
git add tests/test_player_form.py
git commit -m "test: guard shots/shots_on_target against the empty-column class of bug"
```

---

### Task 9: Evaluation gate before shipping to the live app

Per the spec's Section 7: this market must beat a naive position-average baseline on a genuinely held-out season before Task 6's live-serving wiring is trusted. This mirrors `evaluate/player_stat_reliability.py`'s exact structure (train/test split by season, incremental comparison against the existing baseline this project already uses).

**Files:**
- Create: `src/pl_predictor/evaluate/player_shots_reliability.py`
- Test: `tests/test_player_shots_reliability.py`

**Interfaces:**
- Produces: `player_shots_reliability.evaluate_shots_model(seasons=None, test_season=None) -> dict` — one row of metrics per target (`shots`, `shots_on_target`): `mae_baseline` (naive position-average-per-90 predictor), `mae_model` (Task 5's Ridge rate model), and `mae_gain` (`mae_baseline - mae_model`; positive means the real model beats the naive baseline).

- [ ] **Step 1: Write the failing test**

Create `tests/test_player_shots_reliability.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pl_predictor.evaluate import player_shots_reliability


def test_evaluate_shots_model_reports_positive_gain_on_separable_data(monkeypatch):
    """Synthetic data where shots_per90_last10 is a near-perfect predictor
    of the held-out season's shots rate -- the fitted Ridge model must beat
    the naive constant-per-position baseline by a real, positive margin."""
    rng = np.random.default_rng(42)
    rows = []
    for season in ["2023-24", "2024-25"]:
        for i in range(400):
            rate = rng.uniform(0.5, 4.0)
            rows.append({
                "element": i, "season": season, "position": "FWD", "minutes": 90,
                "shots": rate + rng.normal(0, 0.05),
                "shots_per90_last10": rate,
            })
    df = pd.DataFrame(rows)
    monkeypatch.setattr(player_shots_reliability, "_prepare", lambda seasons=None: df)

    result = player_shots_reliability.evaluate_shots_model(test_season="2024-25")
    row = result[result["target"] == "shots"].iloc[0]
    assert row["mae_gain"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_shots_reliability.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Write the implementation**

Create `src/pl_predictor/evaluate/player_shots_reliability.py`:

```python
"""player_shots_reliability.py — does the fitted per-position shots/
shots-on-target rate model (`models/player_goals.py::fit_position_rate_
models`'s new targets, see docs/superpowers/specs/2026-09-04-player-shots-
market-design.md) actually beat a naive baseline on a genuinely held-out
season? Same train/test-by-season discipline as `player_stat_reliability.py`
-- a stat's apparent value must survive being measured on data its fit
never saw, not just correlate on the data it was fit on.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..models import player_goals
from ..features import player_form

TARGETS = ["shots", "shots_on_target"]


def _prepare(seasons: list[str] | None = None) -> pd.DataFrame:
    history = player_goals._load_history_with_shots(seasons=seasons)
    played, _ = player_form.build_historical_player_form(history)
    return played


def evaluate_shots_model(seasons: list[str] | None = None, test_season: str | None = None) -> pd.DataFrame:
    """One row per target: `mae_baseline` (predict every player at their
    position's held-out-season mean rate), `mae_model` (the real Ridge
    rate model, fit on train seasons only, scored on the held-out season),
    `mae_gain` (positive means the real model earns its complexity)."""
    played = _prepare(seasons)
    test_season = test_season or sorted(played["season"].unique())[-1]
    train = played[played["season"] != test_season]
    test = played[played["season"] == test_season]

    rows = []
    for target in TARGETS:
        feature_col = f"{target}_per90_last10"
        cols_needed = [feature_col, target, "minutes"]
        tr = train.dropna(subset=cols_needed)
        te = test.dropna(subset=cols_needed)
        if len(tr) < 100 or len(te) < 30:
            rows.append({"target": target, "n_train": len(tr), "n_test": len(te), "mae_baseline": None, "mae_model": None, "mae_gain": None})
            continue

        te_target_rate = te[target] / te["minutes"] * 90
        baseline_pred = pd.Series(tr[target].sum() / tr["minutes"].sum() * 90, index=te.index)
        mae_baseline = float(mean_absolute_error(te_target_rate, baseline_pred))

        model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
        train_target_rate = tr[target] / tr["minutes"] * 90
        model.fit(tr[[feature_col]], train_target_rate)
        model_pred = model.predict(te[[feature_col]])
        mae_model = float(mean_absolute_error(te_target_rate, model_pred))

        rows.append({
            "target": target, "n_train": len(tr), "n_test": len(te),
            "mae_baseline": mae_baseline, "mae_model": mae_model,
            "mae_gain": mae_baseline - mae_model,
        })

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_player_shots_reliability.py -v`
Expected: PASS

- [ ] **Step 5: Run the gate against real data and record the decision**

```bash
PYTHONPATH=src .venv/bin/python -c "
from pl_predictor.evaluate import player_shots_reliability
result = player_shots_reliability.evaluate_shots_model()
print(result.to_string())
"
```

Apply the decision rule from the spec's Section 7: ship Task 6's live-serving wiring only if `mae_gain > 0` for both targets on this real run. If a target shows `mae_gain <= 0` (the Ridge model doesn't beat the naive per-position baseline), do not remove that target from the live response — instead, in `models/player_goals.py::predict_player`, fall back to the naive per-position rate for that specific target only (mirroring how goals/assists already fall back to `fallback = prior_rate or position_prior` when no reliability coefficients exist) rather than serving an unvalidated model's number. Record whichever outcome occurred, and the real measured MAE numbers, as a new section at the end of `docs/superpowers/specs/2026-09-04-player-shots-market-design.md` (same "recorded after shipping" pattern the bet-recommendation spec now has) — do not silently ship without this record, per this project's own "measured, not assumed" standard.

- [ ] **Step 6: Commit**

```bash
git add src/pl_predictor/evaluate/player_shots_reliability.py tests/test_player_shots_reliability.py docs/superpowers/specs/2026-09-04-player-shots-market-design.md
git commit -m "feat: add walk-forward evaluation gate for the shots/SoT rate model"
```

---

## Self-Review Notes

- **Spec coverage:** Section 1 (crosswalk) → Task 1. Section 2 (extraction) → Task 2. Section 3 (feature engineering) → Task 3 (the `RATE_STATS` one-liner plus the historical merge that makes it meaningful). Section 4 (team-shot scaling) → Tasks 4 and 6. Section 5 (API surface) → Task 7. Section 6 (testing) → distributed across every task's own tests plus Task 8's dedicated empty-column regression guard. Section 7 (evaluation gate) → Task 9.
- **Placeholder scan:** none found — Task 6 Step 5's matches-frame source was verified directly against `routes.py:179` (`_get_matches_df`) before finalizing this plan, rather than left as an implementer-side guess.
- **Type consistency:** `predict_player`'s new `team_expected_shots` parameter (Task 4) matches `rank_team_players`'s new same-named parameter (Task 6) exactly. `PlayerPrediction`'s three new fields (Task 7) match the three new keys `predict_player` returns (Task 4) by name exactly (`expected_shots`, `expected_shots_on_target`, `anytime_shot_on_target_prob`). `_load_history_with_shots` (Task 3) is consumed identically by `fit_position_rate_models` (Task 5) and `player_shots_reliability._prepare` (Task 9).
