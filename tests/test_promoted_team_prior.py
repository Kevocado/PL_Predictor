"""Unit tests for features/promoted_team_prior.py (EXP-2026-05 slot).

Uses a small synthetic match history plus a monkeypatched
`data.clubelo.team_rating_asof` — no real ClubElo fetch, since the live API
has never responded from this project's development environment (see
data/clubelo.py's status note). These tests cover the pure bridging math,
cold-start detection, and no-lookahead discipline; they cannot yet confirm
this improves the model, which requires a real fetch and a walk-forward
evaluation per docs/AI_CONTINUITY.md's promotion gate.
"""

import pandas as pd
import pytest

from pl_predictor.features import promoted_team_prior
from pl_predictor.features.promoted_team_prior import clubelo_elo_prior, cold_start_teams, zscore_bridge


def test_zscore_bridge_maps_source_mean_to_target_mean():
    source = pd.Series({"A": 1000.0, "B": 1200.0, "C": 1400.0})
    target = pd.Series({"A": 1500.0, "B": 1600.0, "C": 1700.0})
    # C's source rating equals the source mean -> should map to target mean
    assert zscore_bridge(source, target, source.mean()) == pytest.approx(target.mean())


def test_zscore_bridge_preserves_relative_ranking():
    source = pd.Series({"A": 1000.0, "B": 1200.0, "C": 1400.0})
    target = pd.Series({"A": 1500.0, "B": 1600.0, "C": 1700.0})
    weak = zscore_bridge(source, target, 900.0)
    strong = zscore_bridge(source, target, 1500.0)
    assert weak < strong


def test_zscore_bridge_falls_back_to_target_mean_on_zero_variance():
    source = pd.Series({"A": 1000.0, "B": 1000.0})
    target = pd.Series({"A": 1500.0, "B": 1700.0})
    assert zscore_bridge(source, target, 1234.0) == pytest.approx(target.mean())


def _synthetic_matches():
    dates = pd.date_range("2024-08-01", periods=6, freq="7D")
    rows = [
        {"date": dates[0], "team_home": "A", "team_away": "B", "goals_home": 2, "goals_away": 1, "ftr": "H"},
        {"date": dates[1], "team_home": "B", "team_away": "C", "goals_home": 0, "goals_away": 0, "ftr": "D"},
        {"date": dates[2], "team_home": "C", "team_away": "A", "goals_home": 1, "goals_away": 3, "ftr": "A"},
        {"date": dates[3], "team_home": "A", "team_away": "C", "goals_home": 2, "goals_away": 0, "ftr": "H"},
        # NewTeam's first-ever match, after A/B/C already have history.
        {"date": dates[4], "team_home": "NewTeam", "team_away": "A", "goals_home": 0, "goals_away": 1, "ftr": "A"},
        {"date": dates[5], "team_home": "B", "team_away": "NewTeam", "goals_home": 2, "goals_away": 2, "ftr": "D"},
    ]
    df = pd.DataFrame(rows)
    df["season"] = "2024-2025"
    return df


def test_cold_start_teams_flags_only_teams_with_zero_history():
    matches = _synthetic_matches()
    assert cold_start_teams(matches) == {"A", "B", "C", "NewTeam"}
    # Once matches before NewTeam's debut are excluded, only A/B/C have
    # ever played, and *they* were each cold-start once too — but as of
    # right before NewTeam's debut, only NewTeam has zero history.
    before_debut = matches[matches["date"] < matches["date"].iloc[4]]
    assert cold_start_teams(before_debut) == {"A", "B", "C"}


def test_clubelo_elo_prior_uses_only_data_strictly_before_kickoff(monkeypatch):
    matches = _synthetic_matches()
    debut_date = matches["date"].iloc[4]

    calls = []

    def fake_rating(team, date):
        calls.append((team, date))
        ratings = {"A": 1900.0, "B": 1700.0, "C": 1600.0, "NewTeam": 1500.0}
        return ratings.get(team)

    monkeypatch.setattr(promoted_team_prior.clubelo, "team_rating_asof", fake_rating)

    prior = clubelo_elo_prior(matches, "NewTeam", debut_date)

    assert prior is not None
    # Every ClubElo lookup must be dated strictly before the fixture, never
    # the fixture's own date or later.
    lookup_date = pd.Timestamp(debut_date) - pd.Timedelta(days=1)
    assert all(pd.Timestamp(date) == lookup_date for _, date in calls)
    # NewTeam is the weakest of the four ClubElo ratings, so its bridged
    # Elo prior must be below the established teams' local Elo mean.
    established_local = {
        team: promoted_team_prior.ratings.fit_elo(
            matches[matches["date"] < debut_date]
        ).get_team_rating(team)
        for team in ("A", "B", "C")
    }
    assert prior < sum(established_local.values()) / len(established_local)


def test_clubelo_elo_prior_returns_none_when_team_unmapped(monkeypatch):
    matches = _synthetic_matches()
    debut_date = matches["date"].iloc[4]
    monkeypatch.setattr(promoted_team_prior.clubelo, "team_rating_asof", lambda team, date: None)

    assert clubelo_elo_prior(matches, "NewTeam", debut_date) is None
