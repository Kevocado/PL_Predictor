"""PL offence-versus-defence duels for one match, from xG efficiency known BEFORE the match.

Each duel is one side's attack (xG created) against the other side's matching defence (xG conceded),
expressed as league ranks. Only matches of `season` strictly before `as_of`, and each team's last
`WINDOW` of them, are used: early in a season there are fewer than `MIN_GAMES` and the answer is
no duels, which is better than ranking last year's roster.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .duel import Duel, make_duel, ranks

WINDOW = 8
MIN_GAMES = 3
#: (duel id, attack column, defence column, attack noun, defence noun). Defence columns are xG ALLOWED: lower is better.
DUELS = [
    ("attack_vs_defence", "xg_for", "xg_against", "chance creation", "chance prevention"),
]


def load_history_gaps(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Past absolute rank gaps per duel type, from `data/duel_gaps.json` (written by validation
    from the walk-forward matches). An absent file or an absent type yields {}, which sends
    `edge_strength` down the gap-scaled fallback instead of a percentile."""
    path = Path(path) if path is not None else Path(__file__).resolve().parents[3] / "data" / "duel_gaps.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: np.asarray(v, dtype=float) for k, v in raw.items() if isinstance(v, list) and v}


def _team_perspective(understat_df: pd.DataFrame) -> pd.DataFrame:
    """Convert from match format (team_home, team_away, xg_home, xg_away) to per-team format."""
    home = pd.DataFrame({
        "date": understat_df["date"],
        "team": understat_df["team_home"],
        "xg_for": understat_df["xg_home"],
        "xg_against": understat_df["xg_away"],
    })
    away = pd.DataFrame({
        "date": understat_df["date"],
        "team": understat_df["team_away"],
        "xg_for": understat_df["xg_away"],
        "xg_against": understat_df["xg_home"],
    })
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])


def _recent_means(xg_data: pd.DataFrame, matches_df: pd.DataFrame, as_of: pd.Timestamp, season: int) -> pd.DataFrame:
    """Last WINDOW matches per team, from `season` strictly before `as_of`, with at least MIN_GAMES of data."""
    if xg_data.empty or matches_df.empty:
        return pd.DataFrame()
    long_df = _team_perspective(xg_data)
    meta = matches_df[["date", "season"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    xg_idxed = long_df.merge(meta, on="date", how="left")
    xg = xg_idxed.assign(
        date=pd.to_datetime(xg_idxed["date"]),
        season=xg_idxed["season"],
    )
    xg = xg[(xg["date"] < as_of) & (xg["season"] == season)].sort_values("date")
    last = xg.groupby("team").tail(WINDOW)
    counts = last.groupby("team").size()
    means = last.groupby("team").mean(numeric_only=True)
    return means[counts.reindex(means.index) >= MIN_GAMES]


def matchups_for_game(
    home: str, away: str, matches_df: pd.DataFrame, xg_data: pd.DataFrame, as_of, season: int,
    history_gaps: dict[str, np.ndarray] | None = None, min_gap: int = 5,
) -> list[Duel]:
    """Up to two duels (one kind x two directions), strongest first. Empty when either team lacks data."""
    means = _recent_means(xg_data, matches_df, pd.Timestamp(as_of), season)
    if home not in means.index or away not in means.index:
        return []
    history_gaps = history_gaps or {}
    out: list[Duel] = []
    for duel_id, attack_col, defence_col, attack_noun, defence_noun in DUELS:
        attack_ranks = ranks(means[attack_col].to_dict(), higher_is_better=True)
        defence_ranks = ranks(means[defence_col].to_dict(), higher_is_better=False)
        for attacker_side in ("home", "away"):
            d = make_duel(
                f"{duel_id}:{attacker_side}",
                home=home, away=away, attacker_side=attacker_side,
                attack_ranks=attack_ranks, defence_ranks=defence_ranks,
                history_gaps=history_gaps.get(duel_id, np.array([])), min_gap=min_gap,
                stat=attack_noun, foil=defence_noun,
            )
            if d is not None:
                out.append(d)
    out.sort(key=lambda d: d.strength, reverse=True)
    return out


def to_context(duels: list[Duel], pick_side: str | None, limit: int = 4,
               lift_gate: dict[str, bool] | None = None) -> list[dict]:
    """Facts-bundle form. `toward_pick` is True when the duel favours the pick
    AND the residual-lift gate has proven the duel's TYPE: `lift_gate`
    maps a duel type ("attack_vs_defence", id without the :side suffix) to
    whether it passes. Unproven (missing/failing) types ship `toward_pick` None
    -- neutral context, never Edge or Risk.

    The gate is FAIL CLOSED: `lift_gate=None` behaves exactly like `{}` (nothing
    is proven), so there is no "ungated" mode -- a caller that forgets to wire
    the gate can never emit Edge/Risk-capable rows before a type is proven.
    """
    out: list[dict] = []
    directed_types: set[str] = set()
    for d in duels[:limit]:
        duel_type = d.id.split(":")[0]
        proven = (lift_gate or {}).get(duel_type, False)
        # The lift test validates the STRONGEST duel of a type per game, so only that one (the first of its type in
        # the strongest-first order `matchups_for_game` returns) may carry a direction. The opposite-direction
        # duel of the same type was never tested and stays neutral context.
        eligible = proven and duel_type not in directed_types
        toward_pick = None if (pick_side is None or not eligible) else (d.toward == pick_side)
        if eligible and pick_side is not None:
            directed_types.add(duel_type)
        out.append({
            "id": d.id,
            "attacker": d.attacker,
            "defender": d.defender,
            "stat": d.stat,
            "foil": d.foil,
            "attacker_rank": d.attacker_rank,
            "defender_rank": d.defender_rank,
            "n_teams": d.n_teams,
            "toward_pick": toward_pick,
        })
    return out
