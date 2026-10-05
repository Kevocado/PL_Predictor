"""PL's `absence` signal, and why PL needed a different one from NFL's.

Spec `2026-10-01-fixture-signals-design.md` §4 gives the example as *"Out: J.
Jacobs, our #2 rush projection (78 yds)"* — NFL, in YARDS. PL's own units are a
SHARE: the model's per-player claim is `anytime_goal_prob`, so the row reads
*"Out: Saka, our #2 midfielder, 62% to score"*. Both are the same
`figures.projection` magnitude in the sport's own units, which is exactly why the
shared component draws a plain unsigned number and leaves the unit in the
headline: only the adapter knows which it is holding.

Three decisions here that are not obvious, and each of them is a way to be wrong:

**1. DOUBTFUL IS NOT OUT.** `fpl_api.availability_multiplier` returns `0.0` for
`i`/`s`/`u` and `chance/100` for `d`. A doubtful player therefore has a NONZERO
availability and is still expected to take some part, so listing them under "Out"
would assert something the data contradicts. Standing decision, and Phase 2's own
test: doubtful stays a FLAG, not a number. It is excluded here, and `news` is
carried in the row's source line rather than promoted into the headline.

**2. THE RANK IS OVER EVERY PLAYER, INCLUDING THE ONES WHO ARE OUT.** This is the
one that is easy to get backwards. `facts._players_out` ranks the fixture's
players and a naive absence row reads the rank off that list — but if the list is
ranked with the out players REMOVED, then removing one promotes everyone below
them, and "#2" quietly becomes "#1". The rank stated must be the rank the player
HELD, so it is read from the full ranked pool with out players still in it. The
pool arrives already ranked by `anytime_goal_prob` (`player_goals.predict_*`
sorts every result it returns, before any truncation), so the index in that list
IS the rank over everyone.

**3. `n` COUNTS INJURED PLAYERS, NOT GRADEd OBSERVATIONS.** That is what makes
the spec's `n >= 30` rate floor inapplicable, and the shared component agrees:
`DRAWS_A_RATE.absence_strip === false` for the same reason — a row about a
handful of injured players is not a rate drawn from a handful of games. So this
adapter emits no floor of its own, and that is deliberate rather than an omission.

The whole-percent rounding in `absence_signal` is also load-bearing, and is the
same rounding CFB's trust adapter uses: `SignalRows` throws
`HeadlineFigureMismatchError` when a row's words do not state the figure its
marker draws, so the figure is rounded ONCE here and both the marker and the
headline are built from that one rounded number.
"""
from __future__ import annotations

from ..api.facts import _num

#: FPL's per-player status codes. `fpl_api` documents them: a=available,
#: i=injured, d=doubtful, s=suspended, u=unavailable. Only the three with a
#: `0.0` availability multiplier are "out"; `d` is deliberately absent (see the
#: module docstring, decision 1) and `a` is the ordinary case.
OUT_STATUSES = frozenset({"i", "s", "u"})

VISUAL = "absence_strip"
PROJECTION_FIGURE = "projection"


def _is_out(row: dict) -> bool:
    """Whether this row is a player who is out.

    An unknown or missing `status` is NOT "out". The failure this avoids is a
    snapshot from an older schema, where the field is absent, quietly rendering
    as "every player with a projection is out" — which is not a smaller claim
    than the truth, it is the opposite one.
    """
    status = row.get("status")
    return isinstance(status, str) and status.strip().lower() in OUT_STATUSES


def _ranked(rows: list[dict]) -> list[tuple[int, float, dict]]:
    """`(rank, prob, row)` for every player with a usable projection.

    RANK IS 1-BASED over ALL of `rows`, out players included, and `rank` counts
    only players that HAVE a projection — because a player with no projection is
    not in the model's ranking at all, and calling them "#4" would state a place
    in a table they are not in.

    Rows are sorted here rather than trusted to arrive sorted. `player_goals`
    happens to sort before it truncates, but the list also comes out of a cached
    public snapshot that was written by an older build, and a stated rank is a
    claim about an order: deriving it from the list's own order is what makes it
    true regardless of which of those two it came from.
    """
    usable: list[tuple[float, dict]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        prob = _num(row.get("anytime_goal_prob"))
        if prob is None:
            continue
        usable.append((float(prob), row))
    usable.sort(key=lambda item: item[0], reverse=True)
    return [(i + 1, prob, row) for i, (prob, row) in enumerate(usable)]


def headline(name: str, rank: int, stated_pct: int) -> str:
    """The one line, at most 12 words, stating the figure the marker draws.

    The stated figure must EQUAL the marker: the shared component compares
    numbers and throws `HeadlineFigureMismatchError` otherwise, so it is passed in
    already rounded rather than recomputed here.

    No sportsbook vocabulary and no certainty: "62% to score" is what the model
    expected of a player, and it is worth nothing now that they are not playing.
    """
    return f"Out: {name}, our #{rank} scorer, {stated_pct}% to score"


def absence_signal(
    game_id: str,
    rows: list[dict],
    as_of: str = "",
) -> dict | None:
    """The single `absence` signal this fixture can honestly carry, or None.

    ONE row, the out player the model expected most from. A gameweek has no
    "match" to lose and a fixture page has room for two or three signals total
    (the endpoint's own `MAX_SIGNALS`), so "the biggest absence" is the claim a
    reader can act on; listing five injured players is a squad note, not a
    signal, and the endpoint would keep three of them at best.

    Returns None — rather than a row with a thin sample — when there is no out
    player, when no player carries a usable projection, or when the pool is
    empty. None of those is an error: most gameweeks have nobody out, and spec §2
    is explicit that a fixture with nothing to say renders no rows at all.
    """
    ranked = _ranked(rows)
    out = [(rank, prob, row) for rank, prob, row in ranked if _is_out(row)]
    if not out:
        return None

    # `max` by PROBABILITY, not by rank: the two orderings agree today because
    # rank is assigned by sorting on probability, and this states the intent
    # rather than relying on that. The row is about how much was expected.
    rank, prob, row = max(out, key=lambda item: item[1])

    name = row.get("name")
    if not isinstance(name, str) or not name.strip():
        # A player the model cannot NAME is not a player a reader can be told
        # about, and "Out: , our #2 scorer" is not a degraded version of the
        # claim — it is a different one.
        return None

    stated_pct = round(prob * 100)
    return {
        "kind": "absence",
        "sport": "pl",
        "game_id": str(game_id),
        "headline": {
            "text": headline(name, rank, stated_pct),
            "figures": {PROJECTION_FIGURE: stated_pct},
        },
        # Injured PLAYERS, not graded games. See the module docstring, decision 3.
        "n": len(out),
        "source": f"FPL status · {len(out)} player{'s' if len(out) != 1 else ''} out",
        "as_of": as_of or "",
        # Rises with the projection the row itself draws: a player out at 62% is a
        # bigger absence than one out at 8%, and the figure is already on the
        # page, so this adds no claim — it only orders rows the endpoint keeps.
        "strength": stated_pct / 100.0,
        "pre_kickoff_only": True,
        "visual": VISUAL,
    }
