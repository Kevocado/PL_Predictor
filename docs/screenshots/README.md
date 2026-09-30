# Screenshots: fixtures with no `predicted_result`

Evidence for PR "fix: fixtures without predicted_result no longer blank the modal".

Captured with Playwright + chromium-headless-shell against the Vite dev server.
The API was stubbed **in the browser** from this repo's real
`data/public_snapshot.json`, using the same mappings the backend's `PUBLIC_MODE`
branch uses (`src/pl_predictor/api/routes.py`):

| request | served from |
| --- | --- |
| `GET /api/fixtures/gameweek` | `snapshot.fixtures_by_gameweek[str(gw)]` |
| `GET /api/fixtures/{id}` | `snapshot.fixture_detail_by_event_id[id]` |
| `GET /api/fixtures/{id}/players` | `snapshot.fixture_players_by_event_id[id]` |
| `GET /api/fixtures/{id}/player-review` | `snapshot.player_review_by_event_id[id]` |
| `GET /api/fixtures` | `[]` (routes.py returns `[]` in public mode) |

No data was invented. Nothing was typed in by hand.

## Fixtures shot

- **Target (the bug):** `backfill-Arsenal-Coventry-2026-08-21`, gameweek 1.
  Verified at capture time to be one of the 30 rows with **no** `predicted_result`
  key and no shot projections:
  `predicted_result`, `home_shots`, `away_shots`, `home_shots_on_target`,
  `away_shots_on_target`, `home_2plus_prob`, `away_2plus_prob`, `draw_signal`,
  `pre_match_value_bets` all **ABSENT**.
- **Control:** `560912` (Ipswich vs Everton), gameweek 38, which *does* carry all
  of those keys with real values.

## The images

| file | what it shows |
| --- | --- |
| `1-before-desktop-blank.png` | **Before the fix**, 1440px. Clicking the fixture throws during render, React unmounts the tree, and the page is blank. Captured with `document.body.innerText.length === 0` and a page error of exactly `TypeError: Cannot read properties of undefined (reading 'toFixed')`. |
| `2-before-390px-blank.png` | The same blank page at 390px. |
| `3-after-desktop-modal-top.png` | **After the fix**, 1440px. The modal renders in full: final score, prediction review (3/4 match calls correct), player call review, reported match statistics. |
| `4-after-desktop-goals-panel.png` | 1440px, scrolled to "Match result &amp; goals" — the panel the crash came from. The `… to score 2+`, "Total goals", "Predicted margin", "Predicted shots" and "Predicted shots on target" rows are **absent** because this fixture has no data for them. No `NaN`, no invented `0`. |
| `5-after-390px-modal-top.png` | The same fixed modal at 390px. |
| `6-after-390px-goals-panel.png` | 390px, the honest empty state in the goals panel. |
| `7-control-desktop-full-rows.png` | **Control.** A fixture that *does* have the data: all the rows above are present and populated. Proves the fix suppresses rows only when there is genuinely nothing to show. |
| `8-control-390px-full-rows.png` | The control at 390px. |
| `9-control-desktop-modal-top.png`, `10-control-390px-modal-top.png` | Control modal, top of panel. |

## How "honest empty state" was checked, not just eyeballed

For each of the four "after"/"control" captures the script also read the live
`document.body.innerText` and asserted:

- zero uncaught page errors,
- no `NaN` anywhere,
- for the target: no "Predicted shots" row and no "to score 2+" row present,
- for the control: both **are** present.