from pl_predictor.models import power_rankings
import pandas as pd


class FakeModel:
    def get_params(self):
        return {
            "attack_Hull": 2.0,
            "defence_Hull": 0.2,
            "attack_Arsenal": 1.2,
            "defence_Arsenal": 0.8,
        }


class StrongArsenalPrior:
    def get_params(self):
        return {
            "attack_Hull": 1.0,
            "defence_Hull": 0.9,
            "attack_Arsenal": 2.0,
            "defence_Arsenal": 0.2,
        }


class Ratings:
    def __init__(self, values):
        self.values = values

    def get_team_rating(self, team):
        return self.values[team]


def test_limited_current_season_data_is_shrunk_toward_league_mean():
    rankings = power_rankings.power_rankings(
        FakeModel(), current_teams={"Hull", "Arsenal"}, games_played={"Hull": 2, "Arsenal": 20}
    )
    hull = next(row for row in rankings if row["team"] == "Hull")

    assert hull["attack"] < 2.0
    assert hull["defence"] > 0.2
    assert hull["confidence"] == "limited"


def test_dominance_ranking_keeps_preseason_strength_after_one_upset(monkeypatch):
    monkeypatch.setattr(
        power_rankings.scoreline,
        "predict_fixture",
        lambda *_: {"home_goal_expectation": 1.8, "away_goal_expectation": 0.8},
    )
    matches = pd.DataFrame(
        [{"date": pd.Timestamp("2026-08-20"), "team_home": "Hull", "team_away": "Arsenal", "goals_home": 2, "goals_away": 0, "hst": 4, "ast": 5}]
    )

    rankings = power_rankings.dominance_power_rankings(StrongArsenalPrior(), matches, current_teams={"Hull", "Arsenal"})

    assert rankings[0]["team"] == "Arsenal"
    hull = next(row for row in rankings if row["team"] == "Hull")
    assert abs(hull["form_adjustment"]) < power_rankings._FORM_MAX_ADJUSTMENT


def test_preseason_ranking_is_not_labeled_as_a_new_team():
    rankings = power_rankings.dominance_power_rankings(
        StrongArsenalPrior(), pd.DataFrame(), current_teams={"Hull", "Arsenal"}
    )

    assert {row["confidence"] for row in rankings} == {"preseason"}


def test_elo_pi_ranking_requires_strength_in_both_history_seeded_ratings():
    rankings = power_rankings.elo_pi_power_rankings(
        Ratings({"Arsenal": 1770.0, "Hull": 1510.0}),
        Ratings({"Arsenal": 1.4, "Hull": 0.1}),
        current_teams={"Arsenal", "Hull"},
        games_played={"Arsenal": 1, "Hull": 1},
    )

    assert rankings[0]["team"] == "Arsenal"
    assert rankings[0]["ranking_method"] == "elo_pi"
    assert rankings[0]["confidence"] == "limited"


def test_blended_form_ranking_preserves_a_preseason_prior_with_early_form():
    rankings = power_rankings.blended_form_power_rankings(
        StrongArsenalPrior(),
        Ratings({"Arsenal": 1770.0, "Hull": 1510.0}),
        Ratings({"Arsenal": 1.4, "Hull": 0.1}),
        current_teams={"Arsenal", "Hull"},
        form_weight=0.25,
        games_played={"Arsenal": 1, "Hull": 1},
    )

    assert rankings[0]["team"] == "Arsenal"
    assert rankings[0]["ranking_method"] == "preseason_elo_pi_blend"
