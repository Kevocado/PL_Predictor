"""Shared 1x2 outcome/draw-agreement logic.

Draws are structurally suppressed in the marginal 1x2 probabilities --
summed over the whole scoreline grid, there are always far more non-draw
cells than draw ones, so draw's marginal probability rarely tops the
argmax even when the single most likely individual scoreline genuinely is
a draw (e.g. "1-1" at 11% can still beat every other single cell while
marginal draw sits under 30%, below both marginal home_win and away_win).

`draw_agreement`/`predicted_result` capture this: the scoreline model (top
scoreline) and the 1x2 percentage model "agree" on a draw when the top
scoreline is a draw AND marginal draw probability clears a threshold --
i.e. the percentage model gave a draw real, non-trivial weight, not just
a coincidental near-tie. This is the single definition of that agreement,
shared by pre-match display (`api/schemas.py`) and post-match accuracy
(`tracking/store.py`) so both surfaces always agree with each other.
"""

from __future__ import annotations

# Chosen from the live tracked-fixture distribution: marginal draw_prob
# across resolved fixtures ranged ~15%-28%, and every fixture whose top
# scoreline was itself a draw had marginal draw_prob >= 22%. 20% sits at
# the low end of that band -- a real, above-baseline (1/3) show of
# confidence, not a rubber stamp on every near-tie top scoreline.
DRAW_AGREEMENT_THRESHOLD = 0.20


def scoreline_outcome(scoreline: str | None) -> str | None:
    """"2-1" -> "home_win", "1-1" -> "draw", "0-2" -> "away_win". None for
    anything unparseable (no fabricated guess)."""
    if not scoreline or "-" not in scoreline:
        return None
    try:
        home, away = (int(part) for part in scoreline.split("-", 1))
    except ValueError:
        return None
    return "home_win" if home > away else "away_win" if away > home else "draw"


def draw_agreement(scoreline: str | None, draw_prob: float, threshold: float = DRAW_AGREEMENT_THRESHOLD) -> bool:
    """True when the top scoreline is a draw AND marginal draw probability
    clears `threshold` -- the scoreline model and the percentage model
    "agree" a draw is the real call, not just a technicality."""
    return scoreline_outcome(scoreline) == "draw" and draw_prob >= threshold


def predicted_result(
    scoreline: str | None,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    threshold: float = DRAW_AGREEMENT_THRESHOLD,
) -> str:
    """The single-label 1x2 pick: marginal argmax, promoted to "draw" when
    `draw_agreement` holds even though a narrow home/away edge won the
    argmax. Never demotes an already-argmax draw."""
    marginal_pick = max(
        (("home_win", home_win_prob), ("draw", draw_prob), ("away_win", away_win_prob)),
        key=lambda pair: pair[1],
    )[0]
    if marginal_pick != "draw" and draw_agreement(scoreline, draw_prob, threshold):
        return "draw"
    return marginal_pick
