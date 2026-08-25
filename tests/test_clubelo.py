"""Unit tests for data/clubelo.py against a mocked HTTP response.

The live API has never responded from this project's development
environment (see config.py::CLUBELO_BASE_URL) — these tests verify the CSV
parsing, ENG-level filtering, team-name mapping, and caching logic against
a synthetic response shaped like ClubElo's documented CSV format. They do
not, and cannot yet, confirm that shape matches a real response.
"""

from pl_predictor.data import clubelo

_FAKE_CSV = """Rank,Club,Country,Level,Elo,From,To
1,Man City,ENG,1,2010.5,2026-08-01,2026-08-10
2,Arsenal,ENG,1,1980.2,2026-08-01,2026-08-10
15,Nottingham,ENG,1,1550.1,2026-08-01,2026-08-10
21,Leeds,ENG,2,1600.0,2026-08-01,2026-08-10
1,Real Madrid,ESP,1,2100.0,2026-08-01,2026-08-10
"""


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


def test_fetch_ratings_asof_filters_to_england_level_1_and_2(monkeypatch, tmp_path):
    monkeypatch.setattr(clubelo, "CLUBELO_CACHE_DIR", tmp_path)
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeResponse(_FAKE_CSV)

    monkeypatch.setattr(clubelo.requests, "get", fake_get)

    ratings = clubelo.fetch_ratings_asof("2026-08-05")

    assert set(ratings["team"]) == {"Man City", "Arsenal", "Nott'm Forest", "Leeds"}
    assert "Real Madrid" not in ratings["team"].tolist()
    assert len(calls) == 1


def test_fetch_ratings_asof_caches_and_does_not_refetch(monkeypatch, tmp_path):
    monkeypatch.setattr(clubelo, "CLUBELO_CACHE_DIR", tmp_path)
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeResponse(_FAKE_CSV)

    monkeypatch.setattr(clubelo.requests, "get", fake_get)

    clubelo.fetch_ratings_asof("2026-08-05")
    clubelo.fetch_ratings_asof("2026-08-05")

    assert len(calls) == 1


def test_team_rating_asof_returns_none_for_unmapped_team(monkeypatch, tmp_path):
    monkeypatch.setattr(clubelo, "CLUBELO_CACHE_DIR", tmp_path)
    monkeypatch.setattr(clubelo.requests, "get", lambda url, timeout: _FakeResponse(_FAKE_CSV))

    assert clubelo.team_rating_asof("Man City", "2026-08-05") == 2010.5
    assert clubelo.team_rating_asof("Real Madrid", "2026-08-05") is None
