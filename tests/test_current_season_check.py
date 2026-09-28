import pandas as pd
import pytest

from pl_predictor.evaluate import current_season_check


# Marked `network`, and the only test in the repo that is. Measured 2026-09-28:
# with every other cache directory warm this is the sole test that still fails,
# and the sole reason the gate cannot be green from a cache alone. It reaches
# `evaluate_count_market_arms_on_current_season`, which loops over `ARM_SPECS`,
# and the `with_dominance` specs call `_dominance_extra_frame` over
# `default_completed_seasons(n=12)`. That is the understat shot loader, one
# file per match, and it is the reason a cold clone is ~4,600 requests:
#
#     cold + network   4926.07s (1:22:06)   4,723 cache files, 87MB
#     warm             669.66s (0:11:09)   all pass
#     warm, no understat 1269.45s (0:21:09) 1 failed -- this one
#
# The 21 minutes for the third row is worth noting: missing the understat cache
# does not just fail this test, it doubles the suite, because the loaders retry
# with backoff before giving up. A gate in that state is both red and slow.
#
# This is a classification, not a fix. The test's own doctrine -- in
# tests/conftest.py -- is that a test reading real data should patch the loader
# with monkeypatch instead, and it already patches the *current* season. What it
# cannot patch is the twelve historical seasons the arm asks for, and narrowing
# `ARM_SPECS` to 11 seasons would change what the arm asserts. Both are real
# work and neither belongs in a CI fix, so the test is marked honestly and the
# gate stops pretending it is offline.
@pytest.mark.network
def test_evaluate_count_market_arms_on_current_season_flags_low_power(monkeypatch):
    """With fewer fixtures than MIN_FIXTURES_FOR_A_DECISION, the result must
    say so explicitly (has_enough_power=False) rather than silently
    returning a table that looks as authoritative as the full walk-forward
    comparison."""
    small_current_season = pd.DataFrame(
        {
            "date": pd.date_range("2026-08-21", periods=10, freq="1D"),
            "team_home": ["Arsenal"] * 10,
            "team_away": ["Chelsea"] * 10,
            "goals_home": [1] * 10,
            "goals_away": [1] * 10,
            "ftr": ["D"] * 10,
            "hc": [5] * 10,
            "ac": [4] * 10,
            "hy": [1] * 10,
            "ay": [1] * 10,
            "hr": [0] * 10,
            "ar": [0] * 10,
            "season": ["2026-2027"] * 10,
        }
    )
    monkeypatch.setattr(current_season_check.football_data, "fetch_current_season_partial", lambda: small_current_season)
    monkeypatch.setattr(current_season_check.football_data, "CURRENT_SEASON_START_YEAR", 2026)

    result = current_season_check.evaluate_count_market_arms_on_current_season("total_corners")

    assert (result["n_fixtures"] == 10).all()
    assert result.attrs["has_enough_power"] is False


def test_evaluate_count_market_arms_on_current_season_raises_when_nothing_played_yet(monkeypatch):
    monkeypatch.setattr(current_season_check.football_data, "fetch_current_season_partial", lambda: pd.DataFrame())

    with pytest.raises(RuntimeError, match="No completed current-season fixtures"):
        current_season_check.evaluate_count_market_arms_on_current_season("total_corners")
