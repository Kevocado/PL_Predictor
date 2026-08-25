from pl_predictor.data.team_names import CANONICAL_TEAMS, to_canonical


def test_queens_park_rangers_maps_to_qpr():
    """QPR's last Premier League season (2014-15) sits outside the old
    8-season default window, so this alias gap stayed dormant until the
    12-season match-dominance research window reached back far enough to
    hit it (found via evaluate/data_coverage_audit.py)."""
    assert to_canonical("Queens Park Rangers", source="understat") == "QPR"
    assert to_canonical("Queens Park Rangers", source="odds_api") == "QPR"
    assert "QPR" in CANONICAL_TEAMS


def test_unmapped_name_falls_back_to_input_rather_than_raising():
    assert to_canonical("Definitely Not A Real Club", source="understat") == "Definitely Not A Real Club"
