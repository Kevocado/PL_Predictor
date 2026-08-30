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
    assert {"quality_rating", "form_rating", "live_form_rating", "live_form_vs_quality", "overall_rating", "current_impact_rating", "rating_status", "rating_evidence_minutes"} <= saka.keys()
    assert report["rating_model_source"] == "data_led_multiseason_role_evidence"
    assert report["leaderboards"]["MID"][0]["overall_rating"] == saka["overall_rating"]
    assert report["leaderboards"]["GK"][0]["name"] == "Raya"
    assert next(player for player in report["players"] if player["name"] == "Raya")["position"] == "GK"
    assert "squad_health" not in report


def test_player_hub_leaderboards_sort_by_live_form_not_overall(monkeypatch):
    """Quality is a role-aware prior blended with historical evidence — early
    in a season it's still mostly last season's number, not something the
    player has earned yet this year. The leaderboards must surface who's
    actually in form right now, so they sort by Live Form even when that
    disagrees with who has the higher Overall/Quality rating."""
    from pl_predictor.api import hub_analytics as hub_analytics_module

    monkeypatch.setattr(
        hub_analytics_module.player_ratings,
        "rate_bootstrap_elements",
        lambda elements, positions, historical_priors=None: {
            # Established star: high Quality/Overall from prior seasons, but
            # cold so far this season.
            1: {"quality_rating": 85.0, "form_rating": 2.0, "live_form_rating": 20.0, "live_form_vs_quality": -65.0, "overall_rating": 87.0, "current_impact_rating": 30.0},
            # In-form breakout: lower Quality/Overall, but the hottest form
            # in the position right now.
            2: {"quality_rating": 55.0, "form_rating": 12.0, "live_form_rating": 90.0, "live_form_vs_quality": 35.0, "overall_rating": 67.0, "current_impact_rating": 80.0},
        },
    )

    report = hub_analytics.build_player_hub(
        {
            "teams": [{"id": 1, "name": "Arsenal"}],
            "element_types": [{"id": 3, "singular_name_short": "MID"}],
            "elements": [
                {
                    "id": 1, "web_name": "Star", "team": 1, "element_type": 3, "status": "a",
                    "minutes": 900, "starts": 10, "appearances": 10, "goals_scored": 1, "assists": 0,
                    "expected_goals": "0", "expected_assists": "0", "expected_goal_involvements": "0",
                    "threat": "0", "creativity": "0", "ict_index": "0", "bps": 20, "bonus": 0, "news": "",
                },
                {
                    "id": 2, "web_name": "Breakout", "team": 1, "element_type": 3, "status": "a",
                    "minutes": 900, "starts": 10, "appearances": 10, "goals_scored": 6, "assists": 3,
                    "expected_goals": "4", "expected_assists": "2", "expected_goal_involvements": "6",
                    "threat": "80.0", "creativity": "60.0", "ict_index": "20.0", "bps": 60, "bonus": 9, "news": "",
                },
            ],
        }
    )

    leaders = report["leaderboards"]["MID"]
    assert [player["name"] for player in leaders] == ["Breakout", "Star"]
    assert leaders[0]["quality_rating"] == 55.0
    assert leaders[0]["live_form_rating"] == 90.0


def test_player_hub_marks_provisional_players_without_giving_them_a_numeric_overall(monkeypatch):
    """A regression to numerical fallback ratings would rank unknown players by Overall."""
    monkeypatch.setattr(
        hub_analytics.player_ratings,
        "rate_bootstrap_elements",
        lambda elements, positions, historical_priors=None: {
            1: {
                "quality_rating": None, "form_rating": 0.0, "live_form_rating": 45.0,
                "live_form_vs_quality": None, "overall_rating": None, "current_impact_rating": 40.0,
                "rating_status": "provisional", "rating_evidence_minutes": 0.0,
            },
            2: {
                "quality_rating": 82.0, "form_rating": 2.0, "live_form_rating": 60.0,
                "live_form_vs_quality": -22.0, "overall_rating": 84.0, "current_impact_rating": 80.0,
                "rating_status": "established", "rating_evidence_minutes": 4_000.0,
            },
        },
    )
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal"}],
        "element_types": [{"id": 3, "singular_name_short": "MID"}],
        "elements": [
            {"id": 1, "web_name": "Newcomer", "team": 1, "element_type": 3, "status": "a", "minutes": 90, "starts": 1, "appearances": 1, "goals_scored": 1, "assists": 0, "expected_goals": "0.2", "expected_assists": "0", "expected_goal_involvements": "0.2", "threat": "20", "creativity": "0", "ict_index": "3", "bps": 15, "bonus": 0, "news": ""},
            {"id": 2, "web_name": "Established", "team": 1, "element_type": 3, "status": "a", "minutes": 900, "starts": 10, "appearances": 10, "goals_scored": 5, "assists": 3, "expected_goals": "4", "expected_assists": "2", "expected_goal_involvements": "6", "threat": "80", "creativity": "60", "ict_index": "20", "bps": 100, "bonus": 8, "news": ""},
        ],
    }

    report = hub_analytics.build_player_hub(bootstrap)
    newcomer = next(player for player in report["players"] if player["name"] == "Newcomer")

    assert newcomer["rating_status"] == "provisional"
    assert newcomer["overall_rating"] is None
    assert newcomer["rating_evidence_minutes"] == 0.0
    # The existing role cards are intentionally *Live Form* leaderboards, so
    # known and provisional players can both appear when their current form
    # is earned. They are not Overall rankings.
    assert [player["name"] for player in report["leaderboards"]["MID"]] == ["Established", "Newcomer"]
