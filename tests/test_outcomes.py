"""Unit tests for the shared 1x2 outcome/draw-agreement helpers."""

from pl_predictor.outcomes import draw_agreement, predicted_result, scoreline_outcome


def test_scoreline_outcome_parses_home_draw_away():
    assert scoreline_outcome("2-1") == "home_win"
    assert scoreline_outcome("1-1") == "draw"
    assert scoreline_outcome("0-2") == "away_win"


def test_scoreline_outcome_returns_none_for_unparseable_input():
    assert scoreline_outcome(None) is None
    assert scoreline_outcome("") is None
    assert scoreline_outcome("not-a-score") is None


def test_draw_agreement_requires_both_top_scoreline_and_probability_floor():
    assert draw_agreement("1-1", 0.25) is True
    assert draw_agreement("1-1", 0.10) is False  # top scoreline agrees, but probability too low
    assert draw_agreement("2-1", 0.30) is False  # probability high, but top scoreline isn't a draw


def test_predicted_result_is_plain_argmax_when_no_agreement():
    assert predicted_result("2-1", 0.50, 0.25, 0.25) == "home_win"


def test_predicted_result_promotes_draw_on_agreement():
    assert predicted_result("1-1", 0.40, 0.30, 0.30) == "draw"


def test_predicted_result_does_not_promote_below_threshold():
    assert predicted_result("1-1", 0.40, 0.15, 0.45) == "away_win"


def test_predicted_result_never_demotes_an_already_argmax_draw():
    assert predicted_result("2-1", 0.20, 0.50, 0.30) == "draw"
