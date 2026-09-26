"""facts.py — the read-only PL /facts bundle the match explainer consumes.

Read-only and suggest-only: this router never writes, never retrains and
never places anything.

Two rules shape this module, both learned the hard way in Tasks 8-10:

1. **A started fixture is described only by its stored pre-kickoff record.**
   The snapshot's fixture DETAIL is recomputed from the current model and
   genuinely disagrees with the stored card once a match has been played (a
   real example: the card said 0.749, the rebuilt detail said 0.708). Quoting
   the detail would be hindsight dressed as a prediction, so for a fixture
   that has started the pick, its timing and its ``pick_won`` all come from
   the card — the pre-kickoff record the tracking store actually captured.
   A backfilled card is ``rebuilt``: shown, never judged.

2. **No per-request state in module globals.** These are sync endpoints and
   FastAPI runs them in a thread pool, so every id is passed explicitly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import PUBLIC_MODE
from . import routes

router = APIRouter()
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _snapshot() -> dict:
    return routes._public_snapshot()


# --- matchPick, ported from frontend/src/lib/pick.ts --------------------

def match_pick(home: float, draw: float, away: float, team_home: str, team_away: str) -> dict:
    """The one pick a fixture is judged on: the most likely of home/draw/away.

    A port of the frontend's matchPick, and deliberately the same tie order as
    tracking/store.py's max() over (home, draw, away) — the card's verdict and
    this bundle's pick can never disagree."""
    side, prob = "home_win", home
    if draw > prob:
        side, prob = "draw", draw
    if away > prob:
        side, prob = "away_win", away
    if side == "home_win":
        label = f"{team_home} win"
    elif side == "away_win":
        label = f"{team_away} win"
    else:
        label = "Draw"
    return {"side": side, "label": label, "prob": prob}


# --- helpers ------------------------------------------------------------

def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("prob")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _prob_field(detail: dict, key: str) -> float | None:
    """Detail numbers arrive as bare floats or as {prob, implied, edge}."""
    return _num(detail.get(key))


def _iso_utc(value: Any) -> str:
    stamp = _as_utc(value)
    return "" if stamp is None else stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


# --- snapshot access ----------------------------------------------------

def _card(event_id: str) -> dict | None:
    """The stored pre-kickoff record, straight from the tracking store's own
    view of the fixture. This is the honesty anchor for a started match."""
    if PUBLIC_MODE:
        snap = _snapshot()
        for week in (snap.get("fixtures_by_gameweek") or {}).values():
            for card in (week or {}).get("fixtures", []):
                if str(card.get("event_id")) == str(event_id):
                    return card
        return None
    return routes.tracking_store.get_fixture_prediction(event_id)


def _detail(event_id: str) -> dict | None:
    if PUBLIC_MODE:
        return (_snapshot().get("fixture_detail_by_event_id") or {}).get(str(event_id))
    try:
        return routes.fixture_detail(str(event_id), read_only=True)
    except Exception:
        logger.info("fixture detail unavailable for %s", event_id)
        return None


def _players(event_id: str) -> list[dict]:
    if PUBLIC_MODE:
        block = (_snapshot().get("fixture_players_by_event_id") or {}).get(str(event_id)) or {}
    else:
        try:
            block = routes.fixture_players(str(event_id))
        except Exception:
            logger.info("fixture players unavailable for %s", event_id)
            return []
    out = []
    for side, team_key in (("home_players", "team_home"), ("away_players", "team_away")):
        for player in block.get(side) or []:
            out.append({**player, "_team": team_key})
    return out


# --- bundle assembly ----------------------------------------------------

def _status(card: dict | None, detail: dict | None, now: datetime) -> str:
    commence = _as_utc((card or {}).get("commence_time") or (detail or {}).get("commence_time"))
    home_goals = _num((card or {}).get("actual_goals_home"))
    away_goals = _num((card or {}).get("actual_goals_away"))
    if (card or {}).get("finished") or (home_goals is not None and away_goals is not None):
        return "final"
    return "live" if commence is not None and commence <= now else "upcoming"


def _card_probs(card: dict | None) -> tuple[float, float, float] | None:
    if not card:
        return None
    home = _num(card.get("predicted_home_win"))
    draw = _num(card.get("predicted_draw"))
    away = _num(card.get("predicted_away_win"))
    if home is None or draw is None or away is None:
        return None
    return home, draw, away


def _markets(detail: dict | None, probs: tuple[float, float, float] | None, team_home: str, team_away: str) -> list[dict]:
    out: list[dict] = []
    if probs is not None:
        home, draw, away = probs
        market: dict[str, Any] = {
            "market": "result",
            "model": {"home_win": home, "draw": draw, "away_win": away},
        }
        if detail and detail.get("has_live_odds"):
            implied = {
                team_home: _implied(detail, "home_win"),
                team_away: _implied(detail, "away_win"),
            }
            edge = {
                team_home: _edge(detail, "home_win"),
                team_away: _edge(detail, "away_win"),
            }
            if implied[team_home] is not None or implied[team_away] is not None:
                market["implied"] = implied
            if edge[team_home] is not None or edge[team_away] is not None:
                market["edge"] = edge
        out.append(market)

    total = _prob_field(detail or {}, "predicted_total_goals")
    if total is not None:
        over = _prob_field(detail or {}, "over_2_5")
        under = _prob_field(detail or {}, "under_2_5")
        market = {"market": "total_goals", "model_total": total}
        if over is not None:
            market["over_2_5"] = over
        if under is not None:
            market["under_2_5"] = under
        out.append(market)

    btts = _prob_field(detail or {}, "btts_yes_prob")
    if btts is not None:
        out.append({"market": "btts", "yes_prob": btts})
    return out


def _implied(detail: dict, key: str) -> float | None:
    value = (detail.get(key) or {}) if isinstance(detail.get(key), dict) else {}
    return _num(value.get("implied"))


def _edge(detail: dict, key: str) -> float | None:
    value = (detail.get(key) or {}) if isinstance(detail.get(key), dict) else {}
    return _num(value.get("edge"))


def _drivers(detail: dict | None) -> list[dict]:
    """Recent form, the one explanatory signal the fixture detail actually
    carries. No feature-contribution block exists on this endpoint, so
    nothing is invented here."""
    drivers: list[dict] = []
    if not detail:
        return drivers
    home_form = detail.get("home_recent_form")
    away_form = detail.get("away_recent_form")
    if home_form and away_form:
        points = {"W": 3, "D": 1, "L": 0}
        home_pts = sum(points.get(str(r).upper()[:1], 0) for r in list(home_form)[:5])
        away_pts = sum(points.get(str(r).upper()[:1], 0) for r in list(away_form)[:5])
        leaders = detail.get("team_home") if home_pts >= away_pts else detail.get("team_away")
        drivers.append({
            "name": "Recent form",
            "value": f"{''.join(str(r).upper()[:1] for r in list(home_form)[:5])} v {''.join(str(r).upper()[:1] for r in list(away_form)[:5])}",
            "direction": leaders or "both",
        })
    return drivers


def _players_out(rows: list[dict], team_home: str, team_away: str) -> list[dict]:
    ranked = []
    for row in rows:
        prob = _num(row.get("anytime_goal_prob"))
        if prob is None:
            continue
        ranked.append((prob, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    out = []
    for prob, row in ranked[:3]:
        team = team_home if row.get("_team") == "team_home" else team_away
        out.append({"name": row.get("name"), "team": team, "projection": f"{prob * 100:.0f}% to score"})
    return out


def _record() -> dict | None:
    """Pre-kick-off accuracy. get_track_record() already excludes backfilled
    fixtures, so these counts are honest."""
    data = routes.tracking_store.get_track_record() or {}
    settled = int(data.get("n_resolved_fixtures") or 0)
    if settled <= 0:
        return None
    pct = _num(data.get("pct_correct_overall"))
    return {
        "label": "Picks made before kick-off",
        "hits": None if pct is None else int(round(pct * settled)),
        "settled": settled,
    }


def _result(card: dict | None, status: str, pick_timing: str, pick_side: str | None) -> dict | None:
    if status != "final" or not card:
        return None
    home_goals = _num(card.get("actual_goals_home"))
    away_goals = _num(card.get("actual_goals_away"))
    if home_goals is None or away_goals is None:
        return None
    result: dict[str, Any] = {"score": f"{card.get('team_home')} {home_goals:.0f}-{away_goals:.0f}"}
    if pick_timing != "pre_kickoff" or pick_side is None:
        return result
    actual = "home_win" if home_goals > away_goals else ("away_win" if away_goals > home_goals else "draw")
    result["pick_won"] = bool(pick_side == actual)
    return result


@router.get("/facts/upcoming")
def get_facts_upcoming(hours: int = 72) -> dict:
    """Ids of fixtures kicking off within the window, for pre-generation."""
    if hours < 0:
        raise HTTPException(status_code=422, detail="hours must be >= 0")
    now = _now()
    cutoff = now + timedelta(hours=hours)
    ids = []
    if PUBLIC_MODE:
        snap = _snapshot()
        for week in (snap.get("fixtures_by_gameweek") or {}).values():
            for card in (week or {}).get("fixtures", []):
                if card.get("finished"):
                    continue
                commence = _as_utc(card.get("commence_time"))
                if commence is None or commence <= now or commence > cutoff:
                    continue
                event_id = card.get("event_id")
                if event_id is not None and str(event_id) not in ids:
                    ids.append(str(event_id))
    else:
        for event_id, commence in _live_upcoming():
            if commence <= now or commence > cutoff:
                continue
            if str(event_id) not in ids:
                ids.append(str(event_id))
    return {"ids": ids}


def _live_upcoming() -> list[tuple[str, datetime]]:
    """Live mode: the site's own current-gameweek fixtures, read-only."""
    try:
        payload = routes.current_gameweek_fixtures()
    except Exception:
        logger.info("live fixtures unavailable for /facts/upcoming")
        return []
    out = []
    for card in payload.get("fixtures", []) or []:
        if card.get("finished"):
            continue
        commence = _as_utc(card.get("commence_time"))
        if card.get("event_id") is not None and commence is not None:
            out.append((str(card["event_id"]), commence))
    return out


@router.get("/facts/{event_id}")
def get_facts(event_id: str) -> dict:
    now = _now()
    card = _card(event_id)
    detail = _detail(event_id)
    if card is None and detail is None:
        raise HTTPException(status_code=404, detail=f"No fixture with event_id={event_id}")

    team_home = (card or detail).get("team_home")
    team_away = (card or detail).get("team_away")
    status = _status(card, detail, now)
    started = status in ("live", "final")

    # THE RULE: once a fixture has started, only the stored pre-kickoff card
    # may supply probabilities. The detail is today's model and is discarded
    # for the pick entirely.
    if started:
        probs = _card_probs(card)
    else:
        probs = _card_probs(card) or (
            (_prob_field(detail, "home_win"), _prob_field(detail, "draw"), _prob_field(detail, "away_win"))
            if detail and all(_prob_field(detail, k) is not None for k in ("home_win", "draw", "away_win"))
            else None
        )

    pick = None
    if probs is not None and all(p is not None for p in probs):
        pick = match_pick(probs[0], probs[1], probs[2], team_home, team_away)

    if pick is None:
        pick_timing = "none"
    elif (card or {}).get("backfilled"):
        pick_timing = "rebuilt"
    else:
        pick_timing = "pre_kickoff"

    # Once a fixture has started, the detail (totals, BTTS, odds, form) and
    # the player block are post-kickoff rebuilds — form can even include this
    # match. Only the stored card's result probabilities may be quoted.
    pre_start = None if started else detail
    markets = [] if (started and card is None) else _markets(pre_start, probs, team_home, team_away)

    return {
        "sport": "pl",
        "id": str(event_id),
        "title": f"{team_away} at {team_home}",
        "starts_at": _iso_utc((card or detail).get("commence_time")),
        "status": status,
        "pick_timing": pick_timing,
        "pick": pick,
        "markets": markets,
        "drivers": _drivers(pre_start),
        "context": {"gameweek": _current_gameweek()} if _current_gameweek() else {},
        "players": [] if started else _players_out(_players(event_id), team_home, team_away),
        "record": _record(),
        "result": _result(card, status, pick_timing, pick["side"] if pick else None),
    }


def _current_gameweek():
    if PUBLIC_MODE:
        return _snapshot().get("current_gameweek")
    try:
        return routes._resolve_current_gameweek(None, None)
    except Exception:
        return None
