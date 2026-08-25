import math

import pandas as pd
import pytest

from pl_predictor.data import understat_shots


def _shots():
    return pd.DataFrame(
        [
            {"situation": "OpenPlay", "h_a": "h", "x_g": "0.10", "x": "0.90", "y": "0.50"},
            {"situation": "OpenPlay", "h_a": "h", "x_g": "0.30", "x": "0.95", "y": "0.40"},
            {"situation": "Penalty", "h_a": "h", "x_g": "0.76", "x": "0.89", "y": "0.50"},
            {"situation": "FromCorner", "h_a": "h", "x_g": "0.05", "x": "0.85", "y": "0.60"},
            {"situation": "OpenPlay", "h_a": "a", "x_g": "0.20", "x": "0.80", "y": "0.50"},
        ]
    )


def test_aggregate_match_dominance_home_side():
    home, _ = understat_shots._aggregate_match_dominance(_shots())

    assert home["total_xg"] == pytest.approx(1.21)
    assert home["non_penalty_xg"] == pytest.approx(0.45)  # excludes the 0.76 penalty
    assert home["shots"] == 4
    assert home["xg_per_shot"] == pytest.approx(1.21 / 4)
    assert home["open_play_xg_share"] == pytest.approx(0.40 / 1.21)
    assert home["set_piece_xg_share"] == pytest.approx(0.81 / 1.21)  # penalty + corner

    expected_distances = [
        math.sqrt((1 - 0.90) ** 2 + (0.50 - 0.5) ** 2),
        math.sqrt((1 - 0.95) ** 2 + (0.40 - 0.5) ** 2),
        math.sqrt((1 - 0.89) ** 2 + (0.50 - 0.5) ** 2),
        math.sqrt((1 - 0.85) ** 2 + (0.60 - 0.5) ** 2),
    ]
    assert home["avg_shot_distance"] == pytest.approx(sum(expected_distances) / 4)


def test_aggregate_match_dominance_away_side_single_shot():
    _, away = understat_shots._aggregate_match_dominance(_shots())

    assert away["total_xg"] == pytest.approx(0.20)
    assert away["non_penalty_xg"] == pytest.approx(0.20)
    assert away["shots"] == 1
    assert away["open_play_xg_share"] == pytest.approx(1.0)
    assert away["set_piece_xg_share"] == pytest.approx(0.0)


def test_aggregate_match_dominance_side_with_no_shots_returns_none_shares():
    empty_side = pd.DataFrame(columns=["situation", "h_a", "x_g", "x", "y"])
    home, _ = understat_shots._aggregate_match_dominance(empty_side)

    assert home["shots"] == 0
    assert home["total_xg"] == 0.0
    assert home["xg_per_shot"] is None
    assert home["open_play_xg_share"] is None
    assert home["avg_shot_distance"] is None


def test_load_season_match_dominance_reuses_cached_raw_shot_files_no_network(monkeypatch, tmp_path):
    """The per-match raw shot cache already exists (e.g. from
    load_shot_situation_data's earlier fetch) — building the v2 dominance
    aggregate over it must not call the network at all."""
    monkeypatch.setattr(understat_shots, "UNDERSTAT_SHOTS_CACHE_DIR", tmp_path)

    fixtures = pd.DataFrame(
        {
            "understat_id": [111],
            "date": ["2019-08-09"],
            "team_home": ["Liverpool"],
            "team_away": ["Norwich"],
        }
    )
    fixtures.to_csv(tmp_path / "_fixtures_2019.csv", index=False)
    _shots().to_csv(tmp_path / "111.csv", index=False)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not fetch over the network when raw shots are already cached")

    monkeypatch.setattr(understat_shots.pb.scrapers, "Understat", lambda *a, **k: object())
    monkeypatch.setattr(understat_shots, "_fetch_with_retry", _fail_if_called)

    df = understat_shots._load_season_match_dominance("2019", force_refresh=False, request_delay=0)

    assert len(df) == 1
    assert df.iloc[0]["team_home"] == "Liverpool"
    assert df.iloc[0]["home_total_xg"] == pytest.approx(1.21)
    assert (tmp_path / "_aggregate_v2_2019.csv").exists()
