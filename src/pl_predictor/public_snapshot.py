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
(Fixtures list + detail, Data Hub, FPL) — not a snapshot of the whole app's API
surface. The Model page's `/api/manifest` already just reads
`models/manifest.json` directly (no heavy computation involved), so it
needs no snapshot entry; same for the admin-only endpoints, which are
hard-blocked in PUBLIC_MODE regardless (see `routes.py::_admin_only`).
`post_match` and player-review both depend on tracking_store history,
which the public deployment never accumulates itself (background tracking
is skipped entirely in PUBLIC_MODE) — but this script runs locally, where
that history *does* exist, so both get computed here and baked into the
snapshot rather than left out.
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi.encoders import jsonable_encoder

from .api import routes
from .config import PUBLIC_SNAPSHOT_PATH


# How many gameweeks past the current one still get freshly rebuilt every
# run. Everything further out reuses the previous snapshot verbatim until
# it enters this window — a full-season rebuild (all 38 gameweeks) was
# confirmed live to take ~9 minutes and touch 380 fixtures every single
# run, the large majority of them months away and unchanged since the
# last run. 5 covers over a month of lookahead, which is what actually
# gets browsed/planned against; a real model retrain still forces a full
# rebuild regardless (see `model_changed` below), since a retrain can
# move every prediction at once, not just the near-term ones.
REBUILD_GAMEWEEKS_AHEAD = 5


def build_snapshot(previous: dict | None = None) -> dict:
    """Build the public read-only data bundle from local live-serving state.

    Finished details are normally reused from the prior snapshot.  A detail
    with no reported statistics is retried so delayed official box-score data
    can still appear in the public app. Newer details also include their
    immutable value-bet calls; legacy snapshot entries remain safely usable.

    Only gameweeks within `REBUILD_GAMEWEEKS_AHEAD` of the current one are
    freshly computed each run — everything else (fully-finished past
    gameweeks, and far-future ones nobody's browsing yet) is carried
    forward from `previous` unchanged, unless the model itself was
    retrained since the last snapshot (`model_fingerprint` mismatch),
    in which case every gameweek rebuilds since a retrain can change any
    prediction, not just near-term ones.
    """
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

    previous = previous or {}
    previous_fixtures_by_gameweek = previous.get("fixtures_by_gameweek", {})
    model_fingerprint = routes.manifest_lib.manifest_fingerprint()
    model_changed = previous.get("model_fingerprint") != model_fingerprint or not previous_fixtures_by_gameweek

    # One gameweek of look-back margin (not just >= current_gameweek) so a
    # fixture whose stats arrived late is still retried at least once after
    # its gameweek finishes, not frozen the instant it rolls out of window.
    rebuild_from = max(min_gameweek, current_gameweek - 1)
    rebuild_to = min(max_gameweek, current_gameweek + REBUILD_GAMEWEEKS_AHEAD)
    rebuild_window = set(range(rebuild_from, rebuild_to + 1))

    print(
        f"Building fixtures for gameweeks {min_gameweek}-{max_gameweek} (current: {current_gameweek}); "
        f"{'model changed, rebuilding all' if model_changed else f'rebuilding {rebuild_from}-{rebuild_to}, reusing the rest'}..."
    )
    fixtures_by_gameweek = {}
    reused_gameweeks: set[int] = set()
    for gw in range(min_gameweek, max_gameweek + 1):
        if not model_changed and gw not in rebuild_window and str(gw) in previous_fixtures_by_gameweek:
            fixtures_by_gameweek[str(gw)] = previous_fixtures_by_gameweek[str(gw)]
            reused_gameweeks.add(gw)
            continue
        print(f"  gameweek {gw}")
        fixtures_by_gameweek[str(gw)] = routes.current_gameweek_fixtures(gameweek=gw)

    event_id_to_gw = {
        fixture["event_id"]: int(gw)
        for gw, gw_data in fixtures_by_gameweek.items()
        for fixture in gw_data.get("fixtures", [])
        if fixture.get("event_id")
    }
    event_ids = sorted(event_id_to_gw)
    finished_event_ids = {
        fixture["event_id"]
        for gw_data in fixtures_by_gameweek.values()
        for fixture in gw_data.get("fixtures", [])
        if fixture.get("event_id") and fixture.get("finished")
    }
    previous_detail = previous.get("fixture_detail_by_event_id", {})
    previous_players = previous.get("fixture_players_by_event_id", {})
    previous_review = previous.get("player_review_by_event_id", {})
    previous_finished_ids = {
        fixture["event_id"]
        for gw_data in previous_fixtures_by_gameweek.values()
        for fixture in gw_data.get("fixtures", [])
        if fixture.get("event_id") and fixture.get("finished")
    }
    def detail_needs_refresh(detail: object) -> bool:
        # Older/synthetic partial snapshots can omit fixture-detail fields;
        # retain their old reuse behaviour. Real fixture details include
        # actual_stats, so a null value tells us the final box score had not
        # reached the feed at the last snapshot run.
        if not isinstance(detail, dict) or "actual_stats" not in detail:
            return False
        return detail.get("actual_stats") is None

    reusable_ids = {
        event_id
        for event_id in event_ids
        if event_id in previous_detail
        and (
            # Whole gameweek carried forward untouched above — its detail
            # can't have changed either, finished or not.
            event_id_to_gw.get(event_id) in reused_gameweeks
            or (
                event_id in finished_event_ids
                and event_id in previous_finished_ids
                and not detail_needs_refresh(previous_detail[event_id])
            )
        )
    }

    print(
        f"Building fixture detail + players for {len(event_ids)} fixtures "
        f"({len(reusable_ids)} reused from the previous snapshot)..."
    )
    fixture_detail_by_event_id = {}
    fixture_players_by_event_id = {}
    player_review_by_event_id = {}
    for i, event_id in enumerate(event_ids, 1):
        reused_detail = event_id in reusable_ids
        if reused_detail:
            fixture_detail_by_event_id[event_id] = previous_detail[event_id]
        # Reuse players/review independently of detail — a fixture can be
        # finished (detail reusable) while one of these is still missing
        # from a prior partial failure; that must still get one real
        # attempt here rather than staying permanently empty forever after.
        reused_players = reused_detail and event_id in previous_players
        if reused_players:
            fixture_players_by_event_id[event_id] = previous_players[event_id]
        # The frontend only ever requests player-review for a finished
        # fixture (it's gated on fixture_detail.post_match, which is only
        # set once a match is over) — computing it for a still-upcoming
        # fixture is pure waste, and calling it ~369 times unconditionally
        # is exactly what turned this into a multi-hour run the first time.
        needs_review = event_id in finished_event_ids
        reused_review = reused_detail and (not needs_review or event_id in previous_review)
        if needs_review and event_id in previous_review:
            player_review_by_event_id[event_id] = previous_review[event_id]
        if reused_detail and reused_players and reused_review:
            continue

        print(f"  [{i}/{len(event_ids)}] {event_id}")
        if not reused_detail:
            try:
                # read_only=True: skips tracking_store writes/reconciliation
                # entirely (see routes.py::_build_fixture_detail's
                # docstring) — that reconcile call re-scans the *entire*
                # matches_df against every unresolved prediction on every
                # invocation, which is the difference between this loop
                # taking seconds and taking minutes across a full season's
                # worth of fixtures.
                fixture_detail_by_event_id[event_id] = routes.fixture_detail(event_id, read_only=True)
            except Exception as exc:  # noqa: BLE001 - one bad fixture shouldn't kill the whole snapshot
                print(f"    ! skipped detail: {exc}")
                continue
        if not reused_players:
            try:
                fixture_players_by_event_id[event_id] = routes.fixture_players(event_id, read_only=True)
            except Exception as exc:  # noqa: BLE001 - players are a nice-to-have, not core detail
                print(f"    ! skipped players: {exc}")
        if needs_review and not reused_review:
            try:
                # No read_only here: this is the one write we want at
                # build time — it's what backfills real player-outcome
                # tracking history locally, same as browsing the admin
                # app would to do.
                player_review_by_event_id[event_id] = routes.fixture_player_review(event_id)
            except Exception as exc:  # noqa: BLE001 - review is a nice-to-have, not core detail
                print(f"    ! skipped player review: {exc}")

    print("Building Data Hub snapshot...")
    hub = {
        "rankings": routes.get_power_rankings(),
        "table": routes.get_projected_table(),
        "track_record": routes.get_hub_track_record(),
        "teams": routes.get_team_hub(),
        "players": routes.get_player_hub(),
    }

    print("Building FPL snapshot...")
    try:
        fpl = {
            "projections": routes.fpl_projections(),
            "optimal_xi": routes.fpl_optimal_xi(),
            "squad": routes.fpl_squad(),
        }
    except Exception as exc:  # noqa: BLE001 - FPL data should not block a fixture deploy
        print(f"  ! skipped FPL snapshot: {exc}")
        fpl = {"projections": {"players": []}, "optimal_xi": {}, "squad": {}}

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_fingerprint": model_fingerprint,
        "current_gameweek": current_gameweek,
        "min_gameweek": min_gameweek,
        "max_gameweek": max_gameweek,
        "fixtures_by_gameweek": fixtures_by_gameweek,
        "fixture_detail_by_event_id": fixture_detail_by_event_id,
        "fixture_players_by_event_id": fixture_players_by_event_id,
        "player_review_by_event_id": player_review_by_event_id,
        "hub": hub,
        "fpl": fpl,
    }


def main() -> None:
    previous = json.loads(PUBLIC_SNAPSHOT_PATH.read_text()) if PUBLIC_SNAPSHOT_PATH.exists() else None
    snapshot = jsonable_encoder(build_snapshot(previous))
    PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {PUBLIC_SNAPSHOT_PATH} ({PUBLIC_SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
