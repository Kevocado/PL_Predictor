import pandas as pd
import pytest

from pl_predictor.evaluate import data_coverage_audit as audit


def test_audit_football_data_reports_missing_field_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "FOOTBALL_DATA_CACHE_DIR", tmp_path)
    df = pd.DataFrame(
        {
            "goals_home": [1, 2, None],
            "goals_away": [0, 1, 2],
            "hs": [10, 12, 8],
            "as": [5, 6, 7],
            "hc": [4, 5, 3],
            "ac": [2, 3, 1],
            "hy": [1, 2, 0],
            "ay": [0, 1, 1],
        }
    )
    df.to_csv(tmp_path / "2020-2021.csv", index=False)

    result = audit.audit_football_data(["2020-2021", "2021-2022"])

    cached_row = result[result["season"] == "2020-2021"].iloc[0]
    assert bool(cached_row["cached"]) is True
    assert cached_row["n_fixtures"] == 3
    assert cached_row["missing_field_rate"] == pytest.approx(1 / (3 * 8))

    missing_row = result[result["season"] == "2021-2022"].iloc[0]
    assert bool(missing_row["cached"]) is False
    assert pd.isna(missing_row["n_fixtures"])


def test_audit_understat_shots_flags_seasons_needing_a_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "UNDERSTAT_SHOTS_CACHE_DIR", tmp_path)

    # 2019: fixture list cached, but only one of two matches has shots cached.
    pd.DataFrame({"understat_id": [111, 222]}).to_csv(tmp_path / "_fixtures_2019.csv", index=False)
    pd.DataFrame({"situation": ["OpenPlay"], "x_g": [0.3]}).to_csv(tmp_path / "111.csv", index=False)

    result = audit.audit_understat_shots(["2018", "2019"])

    row_2018 = result[result["season"] == "2018"].iloc[0]
    assert bool(row_2018["fixtures_cached"]) is False
    assert pd.isna(row_2018["n_matches_expected"])

    row_2019 = result[result["season"] == "2019"].iloc[0]
    assert bool(row_2019["fixtures_cached"]) is True
    assert row_2019["n_matches_expected"] == 2
    assert row_2019["n_matches_shot_cached"] == 1
