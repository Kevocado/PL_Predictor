import pandas as pd

from pl_predictor.api import routes


def test_health_reports_background_cache_warmup_state(monkeypatch):
    monkeypatch.setattr(
        routes,
        "_warmup_status",
        {"state": "warming", "started_at": "2026-08-29T12:00:00+00:00", "completed_at": None, "failures": ["odds_df"]},
    )

    assert routes.get_health() == {
        "status": "ok",
        "cache_warmup": {"state": "warming", "started_at": "2026-08-29T12:00:00+00:00", "completed_at": None, "failures": ["odds_df"]},
    }


def test_warm_caches_prioritises_calibration_before_noncritical_payloads(monkeypatch):
    """A first dashboard visit must not wait behind fixtures, odds, and rankings."""
    calls = []

    def cached(name):
        return lambda: calls.append(name)

    monkeypatch.setattr(routes, "_get_matches_df", cached("matches"))
    monkeypatch.setattr(routes, "_get_models", cached("models"))
    monkeypatch.setattr(routes, "get_calibration", cached("calibration"))
    monkeypatch.setattr(routes, "_get_fixtures_df", cached("fixtures"))
    monkeypatch.setattr(routes, "_get_remaining_fixtures_df", cached("remaining"))
    monkeypatch.setattr(routes, "_get_bootstrap", cached("bootstrap"))
    monkeypatch.setattr(routes, "_get_odds_df", cached("odds"))
    monkeypatch.setattr(routes, "get_power_rankings", cached("rankings"))
    monkeypatch.setattr(routes, "get_projected_table", cached("table"))

    routes.warm_caches()

    assert calls[:3] == ["matches", "models", "calibration"]


def test_manifest_reports_live_result_coverage_separately_from_training_coverage(monkeypatch):
    monkeypatch.setattr(routes.manifest_lib, "load_manifest", lambda: {"n_current_season_matches": 10})
    monkeypatch.setattr(
        routes,
        "_get_live_current_results_df",
        lambda: pd.DataFrame([{"event_id": str(index)} for index in range(14)]),
        raising=False,
    )

    manifest = routes.get_manifest()

    assert manifest["n_current_season_matches"] == 10
    assert manifest["live_current_season_matches"] == 14
    assert manifest["live_results_source"] == "official_fpl"


def test_manifest_stays_available_when_the_live_result_feed_is_temporarily_down(monkeypatch):
    monkeypatch.setattr(routes.manifest_lib, "load_manifest", lambda: {"n_current_season_matches": 10})
    monkeypatch.setattr(routes, "_get_live_current_results_df", lambda: (_ for _ in ()).throw(ConnectionError("offline")))

    manifest = routes.get_manifest()

    assert manifest == {"n_current_season_matches": 10, "live_results_source": "unavailable"}


def test_live_standings_fall_back_to_official_fpl_results_when_paid_key_is_absent(monkeypatch):
    results = pd.DataFrame(
        [
            {"date": "2026-08-21", "team_home": "Arsenal", "team_away": "Chelsea", "goals_home": 1, "goals_away": 0},
            {"date": "2026-08-22", "team_home": "Chelsea", "team_away": "Burnley", "goals_home": 2, "goals_away": 0},
        ]
    )
    monkeypatch.setattr(routes, "_get_fd_org_standings", lambda: pd.DataFrame())
    monkeypatch.setattr(routes, "_get_live_current_results_df", lambda: results, raising=False)

    standings, source = routes._get_live_current_standings("2026-2027")

    assert source == "official_fpl"
    assert standings["team"].tolist()[:2] == ["Chelsea", "Arsenal"]
