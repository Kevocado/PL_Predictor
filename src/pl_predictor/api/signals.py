"""signals.py — `GET /api/signals/{event_id}`, this gameweek's signal payloads.

Spec `2026-10-01-fixture-signals-design.md` §3 (the contract) and §2: *"no data,
no signal"* — a gameweek with nothing to say returns `{"signals": []}`, which the
client renders as no rows at all, not as an empty state or a filler card.

This is PL's first signal, and it is the first `absence_strip` any sport ships.
The visual itself lives in the shared component (`predictor-ui`'s
`SignalRows`), which refused `absence_strip` outright until the hub PR that
builds it; the payload here is deliberately independent of that, because an
adapter that could not be written and tested before its renderer existed would
mean the two halves could not be checked separately.

`MAX_SIGNALS` is the spec's own rule — "a fixed rule (not the model) ranks them
by `strength` and keeps the top 2-3" — so it lives here, at the endpoint, and not
in a component. Today one adapter returns at most one signal, so nothing is
truncated; it is in place because the rule belongs to the endpoint.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from . import facts
from ..signals import absence

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

#: Spec §2. One adapter, so this never truncates anything today.
MAX_SIGNALS = 3


def signals_for_game(event_id: str) -> list[dict]:
    """Every signal this gameweek can honestly carry, strongest first.

    Each adapter is asked independently and may return nothing. An adapter that
    raises is treated as "no signal" rather than being allowed to take down the
    endpoint: a signal is an enhancement on a gameweek page, and the page has to
    survive its absence.

    A STARTED gameweek carries none, and that is the rule rather than an
    optimisation — see `facts.game_context`, which is where the started flag and
    the empty player pool come from, and `get_facts`, which already renders no
    players for the same case. An absence is only information before the players
    are known, so a payload quoting one afterwards would be hindsight wearing a
    prediction's clothes.
    """
    ctx = facts.game_context(event_id)
    if ctx["started"]:
        return []

    try:
        signal = absence.absence_signal(ctx["id"], ctx["rows"])
    except Exception:
        logger.exception("absence signal unavailable for %s", event_id)
        signal = None

    found = [signal] if signal is not None else []
    return sorted(found, key=lambda s: s["strength"], reverse=True)[:MAX_SIGNALS]


@router.get("/signals/{event_id}")
def get_signals(event_id: str) -> dict:
    """The signals for one gameweek. `{"signals": []}` is a valid, complete answer.

    An unknown gameweek is a 404, matching `/facts/{event_id}`: the id grammar
    belongs to `facts` and this router does not own a second one.
    """
    try:
        signals = signals_for_game(event_id)
    except HTTPException:
        raise
    except Exception:
        # The gameweek itself is unreadable, or the player pipeline is down.
        logger.exception("signals unavailable for %s", event_id)
        # The SAME shape as the success path, so a client reading `sport` or `id`
        # does not get a different object only when the backend is failing — the
        # one moment a client is least able to cope with a special case.
        return {"sport": "pl", "id": event_id, "signals": []}

    return {"sport": "pl", "id": event_id, "signals": signals}
