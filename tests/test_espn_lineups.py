from pl_predictor.data.espn import _confirmed_starters


def test_confirmed_starters_requires_a_full_starting_eleven():
    rows = [{"starter": True, "athlete": {"displayName": f"Player {number}"}} for number in range(11)]
    payload = {
        "rosters": [
            {"team": {"displayName": "Fulham"}, "roster": rows},
            {"team": {"displayName": "Chelsea"}, "roster": rows[:-1]},
        ]
    }

    assert _confirmed_starters(payload, "Fulham", "Chelsea") == {"Fulham": [f"Player {number}" for number in range(11)]}
