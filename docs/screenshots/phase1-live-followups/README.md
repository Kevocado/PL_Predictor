# Phase 1 live follow-ups — PL

Real Chromium (`chromium-headless-shell` via `playwright-core`) against this
branch's Vite dev server, **driving the real path**. PL's modal is opened by a
`click()` on an unplayed fixture card, because `FixturesPage` keeps `selected` in
state and `CurrentGameweekCard` calls `onSelect` — there is no deep link, so
nothing here is a component rendered in isolation. The gameweek is reached with
the site's own "Next gameweek" arrow, from the current gameweek (5, already
played) to gameweek 6.

The API is a stand-in harness process serving the repo's own
`data/public_snapshot.json` over the paths the Vite `/api` proxy forwards to.
Nothing in the page is mocked; the harness is not site code and is not in the
diff. Desktop is 1280×1000, mobile is 390×844.

| file | what it shows |
|---|---|
| `desktop-fixtures-list.png` | the real gameweek list, 10 unplayed cards |
| `desktop-modal.png` | the modal opened by clicking one |
| `desktop-BEFORE-bare-heading.png` | **the defect**: `Arsenal vs Leeds` as a bold heading above the AI button, nothing under it |
| `desktop-after-no-bare-heading.png` | **the fix**: the AI button follows the record strip directly |
| `mobile-390-*.png` | the same four at 390px |
| `measurement.json` | the raw DOM measurements, both viewports |

The BEFORE shots were taken with `FixtureModal.tsx` stashed on this branch, on
the same fixture, in the same browser — so the pair is a controlled comparison
and not two unrelated captures.

## What the live DOM said

Measured inside the opened dialog, at **both** viewports:

```
before:  flowRows:    ["H4: Arsenal vs Leeds"]
         flowHeadings:["H4: Arsenal vs Leeds"]
         flowText:    "Arsenal vs Leeds"

after:   flowRows:    []
         flowHeadings:[]
         flowText:    ""
```

The block is untouched by either: `Made before kickoff`, `Arsenal is the pick.`,
`71% win · Arsenal`, `3.3 total goals`, `54% both score`, the three-way bar
`Arsenal 71% / Draw 17% / Leeds 12%`, and the record strip.

## Figures: each one once

`measurement.json` records the sweep. The only figure reported in both the block
and the site is `17%`, and it is a rounding coincidence between two different
facts, not a duplicated figure — traced in the underlying data for this exact
fixture (event `560593`):

- the bar's draw segment: `detail.draw.prob = 0.168` → `17%`
- Stach's anytime assist: `anytime_assist_prob = 0.1654` → `17%`

Two different facts, in two different sections, about two different things. The
committed test asserts duplication by **field identity** rather than by number
for exactly this reason, and notes the limit of a numeric sweep in its comment.

Figures are compared as **whole tokens**, never as substrings: `"7%"` occurs
inside `"17%"` and `"4%"` inside `"54%"`, and a substring sweep on this page
reports three phantom duplications. The first pass of this audit made that
mistake.

## Regenerating

The harness scripts are not committed (they live in `/tmp`). With the Vite dev
server up and the stand-in API on `127.0.0.1:8000`:

```
node /tmp/pl-shoot.mjs http://127.0.0.1:<vite-port> <out-dir>
```
