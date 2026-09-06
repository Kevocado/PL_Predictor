import pandas as pd

from pl_predictor.api import routes


def test_live_fpl_fixture_id_resolves_for_detail_and_player_lookup(monkeypatch):
    """A current-gameweek card must use its FPL id after kickoff."""
    live_fixture = pd.DataFrame(
        [{
            "event_id": 16,
            "team_home": "Chelsea",
            "team_away": "Brighton",
            "commence_time": pd.Timestamp("2026-08-30T13:00:00Z"),
            "gameweek": 2,
            "has_odds": False,
        }]
    )
    monkeypatch.setattr(routes, "_value_bet_table", lambda: pd.DataFrame())
    monkeypatch.setattr(routes, "_get_remaining_fixtures_df", lambda: pd.DataFrame())
    monkeypatch.setattr(routes.fixtures_mod, "_fixtures_from_fpl_api", lambda: live_fixture)
    monkeypatch.setattr(routes, "_get_models", lambda: {"scoreline": object()})
    monkeypatch.setattr(routes.scoreline, "predict_fixture", lambda *_args, **_kwargs: {"home_win": 0.5})
    monkeypatch.setattr(routes, "_team_fixture_to_summary", lambda fixture, _prediction: fixture)
    monkeypatch.setattr(
        routes,
        "_build_fixture_detail",
        lambda _summary, home, away, read_only=False: {"home": home, "away": away},
    )

    assert routes.fixture_detail("16") == {"home": "Chelsea", "away": "Brighton"}
    assert routes._resolve_fixture_teams("16") == ("Chelsea", "Brighton")
