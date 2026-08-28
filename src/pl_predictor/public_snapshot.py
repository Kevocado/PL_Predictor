"""public_snapshot.py — precomputes everything the public, password-gated
deployment needs to serve, so that deployment never has to run this
project's live-serving feature pipeline itself (Elo/Pi replay, rolling
form, xG, shot-situation, the whole `FixtureFeatureContext` build).
Confirmed live: doing that computation on Render's free tier climbs to its
512MB memory ceiling within minutes and gets OOM-killed — this app's live
state just doesn't fit in a resource-constrained host's budget.

Run this locally (full resources, your own machine, same as retraining)
whenever you want to refresh what the public site shows:

    python -m pl_predictor.public_snapshot

Then commit + push `data/public_snapshot.json` — Render's next deploy picks
it up automatically. See `api/routes.py`'s `PUBLIC_MODE` branches (each
guarded by a `if PUBLIC_MODE:` at the top of the function) for where it's
read back instead of computing live.

Deliberately scoped to only what the public frontend actually calls
(Fixtures, Data Hub) — not a snapshot of the whole app's API surface. The
Model page's `/api/manifest` already just reads `models/manifest.json`
directly (no heavy computation involved), so it needs no snapshot entry;
same for the admin-only endpoints, which are hard-blocked in PUBLIC_MODE
regardless (see `routes.py::_admin_only`).
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi.encoders import jsonable_encoder

from .api import routes
from .config import PUBLIC_SNAPSHOT_PATH


def build_snapshot() -> dict:
    print("Loading live-serving state (models, matches, fixtures)...")
    fd_org_matches = routes._get_fd_org_matches()
    has_matchday = not fd_org_matches.empty and fd_org_matches["matchday"].notna().any()
    min_gameweek = int(fd_org_matches["matchday"].min()) if has_matchday else 1
    max_gameweek = int(fd_org_matches["matchday"].max()) if has_matchday else 38

    track_summary = routes.tracking_store.get_track_record()
    current_gameweek = track_summary["current_gameweek"]
    if current_gameweek is None and not fd_org_matches.empty:
        unfinished = fd_org_matches[~fd_org_matches["finished"]]
        current_gameweek = int(unfinished["matchday"].min()) if not unfinished.empty else min_gameweek
    current_gameweek = routes._resolve_current_gameweek(current_gameweek, fd_org_matches) or min_gameweek

    print(f"Building fixtures for gameweeks {min_gameweek}-{max_gameweek} (current: {current_gameweek})...")
    fixtures_by_gameweek = {}
    for gw in range(min_gameweek, max_gameweek + 1):
        print(f"  gameweek {gw}")
        fixtures_by_gameweek[str(gw)] = routes.current_gameweek_fixtures(gameweek=gw)

    print("Building Data Hub snapshot...")
    hub = {
        "rankings": routes.get_power_rankings(),
        "table": routes.get_projected_table(),
        "track_record": routes.get_hub_track_record(),
        "teams": routes.get_team_hub(),
        "players": routes.get_player_hub(),
    }

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "current_gameweek": current_gameweek,
        "min_gameweek": min_gameweek,
        "max_gameweek": max_gameweek,
        "fixtures_by_gameweek": fixtures_by_gameweek,
        "hub": hub,
    }


def main() -> None:
    snapshot = jsonable_encoder(build_snapshot())
    PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {PUBLIC_SNAPSHOT_PATH} ({PUBLIC_SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
