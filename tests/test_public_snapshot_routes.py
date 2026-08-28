"""Public deployment routes must serve entirely from the precomputed
public_snapshot.json (see public_snapshot.py) — never call the live-serving
pipeline (_get_models/_get_matches_df/hub_analytics) at all. That's the
whole point of the snapshot: confirmed live that running the real
computation on Render's free tier OOMs it. These tests assert the routes
return exactly what's in a fake snapshot, and that they never touch a
live-computation function that would raise if called (monkeypatched to
explode) — same lightweight monkeypatch style as the rest of this suite,
no HTTP mocking library."""

from pl_predictor.api import routes


def _explode(*_a, **_k):
    raise AssertionError("must not call the live-serving pipeline in PUBLIC_MODE")


def test_current_gameweek_fixtures_serves_from_snapshot(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(
        routes,
        "_public_snapshot_cache",
        {"current_gameweek": 3, "fixtures_by_gameweek": {"3": {"gameweek": 3, "fixtures": ["fake"]}}},
    )
    monkeypatch.setattr(routes, "_value_bet_table", _explode)
    monkeypatch.setattr(routes, "_get_models", _explode)

    assert routes.current_gameweek_fixtures() == {"gameweek": 3, "fixtures": ["fake"]}
    assert routes.current_gameweek_fixtures(gameweek=3) == {"gameweek": 3, "fixtures": ["fake"]}


def test_current_gameweek_fixtures_degrades_gracefully_when_missing(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot_cache", {"current_gameweek": 3, "fixtures_by_gameweek": {}})

    result = routes.current_gameweek_fixtures(gameweek=99)
    assert result["gameweek"] == 99
    assert result["fixtures"] == []


def test_hub_routes_serve_from_snapshot(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    fake_hub = {
        "rankings": {"rankings": ["fake-rankings"]},
        "table": {"table": ["fake-table"]},
        "track_record": {"summary": "fake-summary"},
        "teams": {"teams": "fake-teams"},
        "players": {"players": "fake-players"},
    }
    monkeypatch.setattr(routes, "_public_snapshot_cache", {"hub": fake_hub})
    monkeypatch.setattr(routes, "_get_matches_df", _explode)
    monkeypatch.setattr(routes, "_get_models", _explode)
    monkeypatch.setattr(routes.hub_analytics, "build_team_hub", _explode)
    monkeypatch.setattr(routes.hub_analytics, "build_player_hub", _explode)

    assert routes.get_power_rankings() == fake_hub["rankings"]
    assert routes.get_projected_table() == fake_hub["table"]
    assert routes.get_hub_track_record() == fake_hub["track_record"]
    assert routes.get_team_hub() == fake_hub["teams"]
    assert routes.get_player_hub() == fake_hub["players"]


def test_hub_routes_degrade_gracefully_when_no_snapshot_exists(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot_cache", {})

    assert routes.get_power_rankings() == {"rankings": [], "ratings_history": {}, "season": None}
    assert routes.get_projected_table() == {"table": [], "season": None}
    assert routes.get_hub_track_record() == {"summary": {}, "biggest_upsets": [], "gameweeks": []}
    assert routes.get_team_hub() == {}
    assert routes.get_player_hub() == {}


def test_private_mode_unaffected(monkeypatch):
    """PUBLIC_MODE=False (the default everywhere else) must never touch
    the snapshot at all — regression guard for the private/local app."""
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)
    monkeypatch.setattr(routes, "_public_snapshot", _explode)

    # None of these should reach _public_snapshot() when PUBLIC_MODE is
    # False; they'll fail later trying to hit real data sources instead,
    # which is fine — this test only cares that the snapshot short-circuit
    # itself is skipped.
    for fn in (routes.get_power_rankings, routes.get_projected_table, routes.get_hub_track_record):
        try:
            fn()
        except AssertionError as exc:
            raise
        except Exception:
            pass
