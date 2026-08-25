from pl_predictor.evaluate.goal_contribution_research import build_goal_contribution_frame


def test_goal_contribution_frame_has_only_prior_form_features():
    frame, _ = build_goal_contribution_frame(seasons=["2023-24"])
    first_rows = frame.sort_values(["season", "element", "kickoff_time"]).groupby(["season", "element"], sort=False).head(1)
    assert (first_rows["goals_per90_last3"] == 0).all()
    assert (first_rows["assists_per90_last3"] == 0).all()
    assert (first_rows["expected_minutes_pre_match"] == 0).all()
