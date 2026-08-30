import pandas as pd

from pl_predictor.api import hub_analytics


def test_team_hub_combines_form_xg_and_style(monkeypatch):
    matches = pd.DataFrame(
        [
            {
                "season": "2026-2027", "date": pd.Timestamp("2026-08-15"), "team_home": "Arsenal", "team_away": "Chelsea",
                "goals_home": 2, "goals_away": 1, "hs": 12, "as": 8, "hst": 5, "ast": 3,
                "hc": 6, "ac": 4, "hf": 9, "af": 11, "hy": 2, "ay": 3, "hr": 0, "ar": 0,
            }
        ]
    )
    xg = pd.DataFrame(
        [{"date": pd.Timestamp("2026-08-15"), "team_home": "Arsenal", "team_away": "Chelsea", "xg_home": 1.6, "xg_away": 0.9, "goals_home": 2, "goals_away": 1}]
    )
    situations = pd.DataFrame(
        [{"date": pd.Timestamp("2026-08-15"), "team_home": "Arsenal", "team_away": "Chelsea", "home_set_piece_xg_share": 0.25, "away_set_piece_xg_share": 0.1}]
    )
    monkeypatch.setattr(hub_analytics.understat, "load_xg_data", lambda **_: xg)
    monkeypatch.setattr(hub_analytics.understat_shots, "load_shot_situation_data", lambda **_: situations)

    report = hub_analytics.build_team_hub(matches, "2026-2027")
    arsenal = next(team for team in report["teams"] if team["team"] == "Arsenal")

    assert arsenal["points"] == 3
    assert arsenal["xg_for"] == 1.6
    assert arsenal["set_piece_xg_share"] == 0.25
    assert arsenal["recent_matches"][0]["result"] == "W"
    assert arsenal["form_points_per_match"] == 3.0
    assert arsenal["form_trend"] == "new"
    # No bootstrap supplied — assists/xa degrade to a neutral default
    # rather than crashing (matches_df alone has no assist data at all).
    assert arsenal["assists"] == 0
    assert arsenal["xa"] is None


def test_team_hub_sums_assists_and_xa_from_fpl_bootstrap():
    matches = pd.DataFrame(
        [
            {
                "season": "2026-2027", "date": pd.Timestamp("2026-08-15"), "team_home": "Arsenal", "team_away": "Chelsea",
                "goals_home": 2, "goals_away": 1, "hs": 12, "as": 8, "hst": 5, "ast": 3,
                "hc": 6, "ac": 4, "hf": 9, "af": 11, "hy": 2, "ay": 3, "hr": 0, "ar": 0,
            }
        ]
    )
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}],
        "elements": [
            {"id": 1, "team": 1, "assists": 5, "expected_assists": "3.2"},
            {"id": 2, "team": 1, "assists": 2, "expected_assists": "1.1"},
            {"id": 3, "team": 2, "assists": 4, "expected_assists": None},
        ],
    }

    report = hub_analytics.build_team_hub(matches, "2026-2027", bootstrap=bootstrap)
    arsenal = next(team for team in report["teams"] if team["team"] == "Arsenal")
    chelsea = next(team for team in report["teams"] if team["team"] == "Chelsea")

    assert arsenal["assists"] == 7
    assert arsenal["xa"] == 4.3
    assert chelsea["assists"] == 4
    assert chelsea["xa"] is None


def test_team_hub_uses_official_live_results_for_the_current_table(monkeypatch):
    """A provisional FPL final score must move the visible Team Hub table."""
    matches = pd.DataFrame(
        [
            {
                "season": "2026-2027", "date": pd.Timestamp("2026-08-15"), "team_home": "Man City", "team_away": "Bournemouth",
                "goals_home": 2, "goals_away": 1, "hs": 12, "as": 8, "hst": 5, "ast": 3,
                "hc": 6, "ac": 4, "hf": 9, "af": 11, "hy": 2, "ay": 3, "hr": 0, "ar": 0,
            }
        ]
    )
    live_results = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-08-15"), "team_home": "Man City", "team_away": "Bournemouth", "goals_home": 2, "goals_away": 1},
            {"date": pd.Timestamp("2026-08-28"), "team_home": "Crystal Palace", "team_away": "Man City", "goals_home": 1, "goals_away": 4},
        ]
    )
    monkeypatch.setattr(hub_analytics.understat, "load_xg_data", lambda **_: pd.DataFrame())
    monkeypatch.setattr(hub_analytics.understat_shots, "load_shot_situation_data", lambda **_: pd.DataFrame())

    report = hub_analytics.build_team_hub(matches, "2026-2027", live_results=live_results)

    city = next(team for team in report["teams"] if team["team"] == "Man City")
    assert city["played"] == 2
    assert city["points"] == 6
    assert city["goals_for"] == 6
    assert city["streak"] == 2


def test_player_hub_exposes_form_table_stats_without_squad_health():
    report = hub_analytics.build_player_hub(
        {
            "teams": [{"id": 1, "name": "Arsenal"}],
            "element_types": [{"id": 1, "singular_name_short": "GKP"}, {"id": 3, "singular_name_short": "MID"}],
            "elements": [
                {
                    "id": 1, "web_name": "Raya", "team": 1, "element_type": 1, "status": "a",
                    "minutes": 900, "starts": 10, "appearances": 10, "goals_scored": 0, "assists": 0,
                    "expected_goals": "0", "expected_assists": "0", "expected_goal_involvements": "0",
                    "clean_sheets": 4, "saves": 25, "threat": "0", "creativity": "0", "ict_index": "0", "bps": 120, "bonus": 3, "news": "",
                },
                {
                    "id": 7, "web_name": "Saka", "team": 1, "element_type": 3, "status": "d",
                    "chance_of_playing_next_round": 50, "minutes": 900, "starts": 10, "appearances": 10, "goals_scored": 1, "assists": 1,
                    "expected_goals": "0.8", "expected_assists": "0.4", "expected_goal_involvements": "1.2",
                    "threat": "44.0", "creativity": "32.0", "ict_index": "7.6", "bps": 25, "bonus": 3, "news": "Knock",
                }
            ],
        }
    )

    saka = next(player for player in report["players"] if player["name"] == "Saka")
    assert saka["xgi"] == 1.2
    assert saka["status"] == "d"
    assert {"quality_rating", "form_rating", "live_form_rating", "live_form_vs_quality", "overall_rating", "current_impact_rating"} <= saka.keys()
    assert report["rating_model_source"] == "role_aware_evidence_baseline"
    assert report["leaderboards"]["MID"][0]["overall_rating"] == saka["overall_rating"]
    assert report["leaderboards"]["GK"][0]["name"] == "Raya"
    assert next(player for player in report["players"] if player["name"] == "Raya")["position"] == "GK"
    assert "squad_health" not in report
