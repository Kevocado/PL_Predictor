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


def test_player_hub_exposes_form_table_stats_without_squad_health():
    report = hub_analytics.build_player_hub(
        {
            "teams": [{"id": 1, "name": "Arsenal"}],
            "element_types": [{"id": 3, "singular_name_short": "MID"}],
            "elements": [
                {
                    "id": 7, "web_name": "Saka", "team": 1, "element_type": 3, "status": "d",
                    "chance_of_playing_next_round": 50, "minutes": 90, "goals_scored": 1, "assists": 1,
                    "expected_goals": "0.8", "expected_assists": "0.4", "expected_goal_involvements": "1.2",
                    "threat": "44.0", "creativity": "32.0", "ict_index": "7.6", "bps": 25, "bonus": 3, "news": "Knock",
                }
            ],
        }
    )

    assert report["players"][0]["xgi"] == 1.2
    assert report["players"][0]["status"] == "d"
    assert "squad_health" not in report
