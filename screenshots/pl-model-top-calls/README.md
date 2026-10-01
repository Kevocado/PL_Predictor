# PL — Model's top calls (phase 2, task 3)

Six screenshots of the fixture modal's **Model's top calls** list, taken in a
real browser (Playwright, chromium-headless-shell) against the real backend in
`PUBLIC_MODE`, serving the committed `data/public_snapshot.json`.

The modal is opened the way a reader opens it — **by clicking a fixture card on
the gameweek page**. There is no deep link into the modal, and nothing here
renders the modal in isolation.

Fixture: gameweek 5, Brentford v Chelsea. The card clicked was
`"Fri 18 Sep · 2:00 PM Rebuilt after kickoff Brentford Final 3–0 Chelsea …"`.

| file | what it shows |
|---|---|
| `picks-desktop.png` | the whole list, 1440px: both categories, three rows each, the out line absent because nobody in this snapshot is out |
| `picks-390px-top.png` | 390px, scrolled to the top of the list |
| `picks-390px-end.png` | 390px, scrolled to the bottom of the list |
| `picks-out-desktop.png` | 1440px, the **out-player** state — see the caveat below |
| `picks-out-390px-top.png` | 390px, top of the list in the out-player state |
| `picks-out-390px-out.png` | 390px, the out line itself |

## The caveat on the three `picks-out-*` files, read this before quoting them

**Those three show a SYNTHETIC status, not live data.**

Measured on the committed snapshot: gameweek 5 has 160 players, of which
**0** have an FPL status in `{i, s, u}` — and across all 380 fixtures and 6096
player rows in the snapshot there are still **0**. So the out line cannot be
photographed from this data at all, and a screenshot implying otherwise would be
a fabricated claim.

What those three files actually are: the real page, opened by the same real
click, with the `/api/fixtures/*/players` response intercepted and the top goal
scorer's `status` set to `"i"`. The UI, the ranking, the click and the rendering
are real; **one field of the payload is not.** The script labels it
`"Knock (synthetic)"` in the player's news.

What that run measured, from the live DOM:

- Thiago, who was the **top** goal call at 47%, appears in **0** rows of **0**
  rankings.
- He is named **exactly once** in the out line, below both lists, with no bar and
  no rank position.
- The freed slot was backfilled from the available pool (Schade 43%, Anthony
  30%, Damsgaard 25%), so an out player can neither hold a rank nor leave a gap.
- The line reads: `Out: Thiago · Brentford — not ranked. FPL squad status
  (Injured) · as read for the Fri, Sep 18 gameweek`

## Duplicate-figure audit

A previous audit in this repo reported three phantom duplicates by searching
rendered text for the substring `"7%"`, which is contained in `"17%"`. These
figures were re-audited by **parsing** the rendered number, anchored at the end
of the string, and grouping by category:

```
Anytime goal  | Thiago      | 47%
Anytime goal  | Schade      | 43%
Anytime goal  | Anthony     | 30%
Anytime assist| Schade      | 43%
Anytime assist| Janelt      | 36%
Anytime assist| Damsgaard   | 31%
```

**0 duplicated figures.** Schade's 43% appears in both lists, which is not a
duplicate: the two figures belong to two different categories, and the audit
keys on the category.

The audit is proven non-vacuous by a control, not asserted to be: the same
function is fed a planted pair and does report it —
`[{category: "Anytime goal", percent: "7", names: ["Planted B", "Planted C"]}]`.
A matcher that cannot fail would also have "passed" this list.
