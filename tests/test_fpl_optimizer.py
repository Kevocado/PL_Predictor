from pl_predictor.models import fpl


def _player(player_id, position, team_id, price, points, fixtures=1):
    return {
        "player_id": player_id, "web_name": f"P{player_id}", "name": f"Player {player_id}", "position": position,
        "team_id": team_id, "team": f"T{team_id}", "price": price, "projected_points": points,
        "fixture_count": fixtures, "availability": 1.0, "expected_minutes": 80.0,
    }


def _pool():
    result, i = [], 1
    for position, count in (("GK", 4), ("DEF", 9), ("MID", 9), ("FWD", 6)):
        for offset in range(count):
            result.append(_player(i, position, (i % 10) + 1, 4.0 + (offset % 3), 10.0 - offset / 5))
            i += 1
    return result


def test_optimal_xi_is_legal_and_marks_captain():
    result = fpl.optimal_xi(_pool())
    positions = [player["position"] for player in result["starting_xi"]]
    assert len(result["starting_xi"]) == 11
    assert positions.count("GK") == 1
    assert 3 <= positions.count("DEF") <= 5
    assert 2 <= positions.count("MID") <= 5
    assert 1 <= positions.count("FWD") <= 3
    assert result["captain"] is not None and result["vice_captain"] is not None


def test_optimal_xi_honours_selected_formation():
    result = fpl.optimal_xi(_pool(), formation="3-4-3")
    positions = [player["position"] for player in result["starting_xi"]]
    assert positions.count("DEF") == 3
    assert positions.count("MID") == 4
    assert positions.count("FWD") == 3


def test_squad_respects_shape_budget_and_club_limit():
    result = fpl.build_squad(_pool(), budget=100.0)
    squad = result["squad"]
    assert len(squad) == 15
    assert result["spent"] <= 100.0
    assert {position: sum(p["position"] == position for p in squad) for position in fpl.SQUAD_SHAPE} == fpl.SQUAD_SHAPE
    assert max(sum(p["team_id"] == team for p in squad) for team in {p["team_id"] for p in squad}) <= 3


def test_transfer_recommendations_are_like_for_like():
    players = _pool()
    current = [p["player_id"] for p in players[:15]]
    result = fpl.transfer_recommendations(players, current)
    assert all(idea["out"]["position"] == idea["in"]["position"] for idea in result["recommendations"])


def test_transfer_cost_is_a_signed_price_delta_not_floored_at_zero():
    """A downgrade in price (buying someone cheaper) must show as a
    negative cost (money back), not a misleading £0.0m that looks
    identical to a same-priced swap — the exact bug a real transfer
    planner run surfaced: every downgrade showed as £0.0m regardless of
    how large the real price gap actually was."""
    out_player = _player(101, "DEF", team_id=1, price=8.0, points=2.0)
    in_player = _player(102, "DEF", team_id=2, price=4.0, points=10.0)
    current = [p["player_id"] for p in _pool()[:14]] + [out_player["player_id"]]
    players = _pool() + [out_player, in_player]

    result = fpl.transfer_recommendations(players, current, bank=0.0, free_transfers=1)

    idea = next(i for i in result["recommendations"] if i["out"]["player_id"] == 101)
    assert idea["cost"] == -4.0


def test_transfer_recommendations_are_deduplicated_and_capped_to_free_transfers():
    """Every player must appear as an 'out' or 'in' at most once across the
    returned list, and the list must be no longer than `free_transfers` —
    otherwise a single very-high-projection player floods the whole list
    as the answer for every other outgoing player, which reads as a
    redundant, arbitrary bag of ideas rather than an actual N-transfer plan."""
    players = _pool()
    current = [p["player_id"] for p in players[:15]]

    result = fpl.transfer_recommendations(players, current, bank=100.0, free_transfers=2)

    assert len(result["recommendations"]) <= 2
    out_ids = [idea["out"]["player_id"] for idea in result["recommendations"]]
    in_ids = [idea["in"]["player_id"] for idea in result["recommendations"]]
    assert len(out_ids) == len(set(out_ids))
    assert len(in_ids) == len(set(in_ids))


def test_transfer_recommendations_exclude_unaffordable_swaps():
    out_player = _player(201, "DEF", team_id=1, price=4.0, points=2.0)
    in_player = _player(202, "DEF", team_id=2, price=9.0, points=10.0)
    current = [p["player_id"] for p in _pool()[:14]] + [out_player["player_id"]]
    players = _pool() + [out_player, in_player]

    result = fpl.transfer_recommendations(players, current, bank=1.0, free_transfers=1)

    assert all(idea["in"]["player_id"] != 202 for idea in result["recommendations"])


def test_projection_sums_double_gameweek_and_marks_blank():
    bootstrap = {
        "teams": [{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}, {"id": 3, "name": "Gamma"}],
        "elements": [
            {"id": 1, "first_name": "A", "second_name": "One", "web_name": "A1", "team": 1, "element_type": 3, "status": "a", "now_cost": 80, "starts": 2, "appearances": 2, "minutes": 180, "expected_goals": "1.0", "expected_assists": "0.4", "form": "5"},
            {"id": 2, "first_name": "G", "second_name": "Blank", "web_name": "GB", "team": 3, "element_type": 3, "status": "a", "now_cost": 50, "starts": 2, "appearances": 2, "minutes": 180, "expected_goals": "0", "expected_assists": "0", "form": "1"},
        ],
    }
    fixtures = [
        {"event": 4, "finished": False, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4},
        {"event": 4, "finished": False, "team_h": 2, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3},
    ]
    result = fpl.build_projections(bootstrap, fixtures, 4, lambda h, a: {"home_goal_expectation": 2.0, "away_goal_expectation": 1.0})
    alpha, blank = result["players"]
    assert alpha["fixture_count"] == 2
    assert blank["fixture_count"] == 0
