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


def _bootstrap(elements):
    return {"elements": elements}


def _element(id_, first, second, web):
    return {"id": id_, "first_name": first, "second_name": second, "web_name": web}


def test_crosswalk_matches_exact_normalized_name():
    rows = pd.DataFrame([{"player": "Erling Haaland", "player_id": 100}])
    bootstrap = _bootstrap([_element(1, "Erling", "Haaland", "Haaland")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {100: 1}


def test_crosswalk_decodes_html_entities():
    rows = pd.DataFrame([{"player": "Luke O'Nien", "player_id": 101}])
    bootstrap = _bootstrap([_element(2, "Luke", "O&#039;Nien", "O'Nien")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {101: 2}


def test_crosswalk_transliterates_non_decomposable_letters():
    # NFKD alone cannot turn 'Ø' into 'O' -- confirmed against real FPL data
    # (Martin Ødegaard) during this plan's design.
    rows = pd.DataFrame([{"player": "Martin Odegaard", "player_id": 102}])
    bootstrap = _bootstrap([_element(3, "Martin", "Ødegaard", "Ødegaard")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {102: 3}


def test_crosswalk_expands_common_nicknames():
    rows = pd.DataFrame([{"player": "Ben White", "player_id": 103}])
    bootstrap = _bootstrap([_element(4, "Benjamin", "White", "White")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {103: 4}


def test_crosswalk_falls_back_to_unique_surname_match():
    # web_name is plain "Smith" (not "J.Smith", which would accidentally
    # exact-match at stage 1 via web_norm and never reach stage 3) --
    # "J Smith" matches neither the full name nor the web name, so this
    # only resolves through the surname-only fallback.
    rows = pd.DataFrame([{"player": "J Smith", "player_id": 104}])
    bootstrap = _bootstrap([_element(5, "Jordan", "Smith", "Smith")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {104: 5}


def test_crosswalk_never_guesses_on_ambiguous_surname():
    # Deliberately avoids an accidental exact web_name match at stage 1
    # (e.g. "A Murphy" vs a web_name of "A.Murphy" would exact-match and
    # never reach the ambiguous case this test means to exercise) --
    # "Alexander Murphy" matches neither candidate's full name (el7's
    # first_name is the nickname "Alex", not "Alexander") nor either
    # web_name, so it falls through to surname-only, where it hits both.
    rows = pd.DataFrame([{"player": "Alexander Murphy", "player_id": 105}])
    bootstrap = _bootstrap([
        _element(6, "Adam", "Murphy", "A.Murphy"),
        _element(7, "Alex", "Murphy", "Alex Murphy"),
    ])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert 105 not in result


def test_crosswalk_excludes_unmatchable_player_without_crashing():
    rows = pd.DataFrame([{"player": "Nobody Real", "player_id": 106}])
    bootstrap = _bootstrap([_element(8, "Someone", "Else", "Else")])
    result = understat_shots.build_understat_fpl_crosswalk(rows, bootstrap)
    assert result == {}


def _shot_row(player, player_id, h_a, result, situation="OpenPlay", x_g="0.1"):
    return {
        "player": player, "player_id": player_id, "h_a": h_a, "result": result,
        "situation": situation, "x_g": x_g,
    }


def test_player_shot_extraction_counts_shots_and_shots_on_target(monkeypatch, tmp_path):
    monkeypatch.setattr(understat_shots, "UNDERSTAT_SHOTS_CACHE_DIR", tmp_path)
    fixtures = pd.DataFrame([{"understat_id": "111", "date": "2025-08-16", "team_home": "Arsenal", "team_away": "Chelsea"}])
    monkeypatch.setattr(understat_shots, "_fetch_season_fixtures_with_id", lambda *a, **k: fixtures)

    shots = pd.DataFrame([
        _shot_row("Bukayo Saka", 501, "h", "Goal"),
        _shot_row("Bukayo Saka", 501, "h", "MissedShots"),
        _shot_row("Bukayo Saka", 501, "h", "SavedShot"),
        _shot_row("Bukayo Saka", 501, "h", "BlockedShot"),
    ])
    scraper = object()
    monkeypatch.setattr(understat_shots, "fetch_match_shots", lambda _scraper, _id, force_refresh=False: shots)
    monkeypatch.setattr(understat_shots.pb.scrapers, "Understat", lambda *a, **k: scraper)

    df = understat_shots.load_player_shot_history(seasons=["2025-26"])
    row = df[df["player_id"] == 501].iloc[0]
    assert row["shots"] == 4
    assert row["shots_on_target"] == 2  # Goal + SavedShot only
    assert row["date"] == pd.Timestamp("2025-08-16")


def test_current_season_player_shot_rows_deduplicates_by_player_id(monkeypatch, tmp_path):
    monkeypatch.setattr(understat_shots, "UNDERSTAT_SHOTS_CACHE_DIR", tmp_path)
    history = pd.DataFrame([
        {"player": "Bukayo Saka", "player_id": 501, "date": pd.Timestamp("2025-08-16"), "shots": 4, "shots_on_target": 2, "goals": 1, "x_g": 0.4},
        {"player": "Bukayo Saka", "player_id": 501, "date": pd.Timestamp("2025-08-23"), "shots": 2, "shots_on_target": 1, "goals": 0, "x_g": 0.2},
    ])
    monkeypatch.setattr(understat_shots, "load_player_shot_history", lambda seasons=None, force_refresh=False: history)

    rows = understat_shots.load_current_season_player_shot_rows(season="2025-26")
    assert len(rows) == 1
    assert rows.iloc[0]["player_id"] == 501
