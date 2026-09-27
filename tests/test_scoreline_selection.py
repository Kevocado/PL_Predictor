"""Tests for the scoreline-selection-bias measurement.

The numbers here are the justification for changing what PL publishes, so they
are tested against hand-built tables where the right answer is known by
inspection. A bug in this module would not change the model; it would change the
justification for a decision about what the project claims, which is worse.
"""

import numpy as np
import pandas as pd
import pytest

from pl_predictor.evaluate.scoreline_selection import (
    nested_selection_rps,
    per_fold_rps,
    summarise,
)

A, B, C = "a_candidate", "b_candidate", "c_candidate"


def _metrics(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"val_season": season, "candidate": candidate, "rps": rps, "brier": rps, "n_val": 380, "n_train": 1000}
         for season, candidate, rps in rows]
    )


def test_per_fold_pivots_to_season_by_candidate():
    table = per_fold_rps(_metrics([("s1", A, 0.2), ("s1", B, 0.1), ("s2", A, 0.3), ("s2", B, 0.15)]))
    assert list(table.index) == ["s1", "s2"]
    assert list(table.columns) == [A, B]
    assert table.loc["s2", A] == pytest.approx(0.3)


def test_nested_selection_picks_the_leader_of_earlier_folds_only():
    """`a` leads after s1 and s2, so s3 must be scored with `a` even though `b`
    happens to win s3 outright. Choosing on the fold being scored is the exact
    bias being measured."""
    per_fold = pd.DataFrame(
        {A: [0.10, 0.10, 0.30], B: [0.20, 0.20, 0.05]},
        index=["s1", "s2", "s3"],
    )
    mean, chosen = nested_selection_rps(per_fold)
    assert chosen == ["s2:a_candidate", "s3:a_candidate"]
    assert mean == pytest.approx((0.10 + 0.30) / 2), (
        "s2 is scored with the s1 leader and s3 with the s1-s2 leader")


def test_nested_selection_never_scores_the_first_fold():
    """The opening fold has no earlier evidence. Scoring it would mean selecting
    on the fold it is reported on."""
    per_fold = pd.DataFrame({A: [0.05, 0.20, 0.20], B: [0.30, 0.20, 0.20]}, index=["s1", "s2", "s3"])
    mean, chosen = nested_selection_rps(per_fold)
    assert len(chosen) == 2 and "s1" not in " ".join(chosen)
    assert mean == pytest.approx(0.20)


def test_nested_selection_is_undefined_with_a_single_fold():
    mean, chosen = nested_selection_rps(pd.DataFrame({A: [0.2]}, index=["s1"]))
    assert chosen == []
    assert np.isnan(mean)


def test_the_optimism_gap_opens_when_selection_flatters_the_result():
    """A candidate that is briefly brilliant then consistently mediocre wins
    `min()` on its good fold and drags the published number down. Nested
    selection cannot see that in advance, so the honest estimate must be worse."""
    metrics = _metrics([
        ("s1", A, 0.10), ("s1", B, 0.20),
        ("s2", A, 0.40), ("s2", B, 0.21),
        ("s3", A, 0.40), ("s3", B, 0.22),
    ])
    result = summarise(metrics)
    assert result["incumbent_min_on_fold"] == pytest.approx((0.10 + 0.21 + 0.22) / 3)
    # s2 is scored with A, the s1 leader. A's s2 result then hands the lead to B
    # for s3, which is the point: nested selection switches when the evidence
    # changes, where min() keeps whichever candidate flattered that one fold.
    assert result["nested_choices"] == ["s2:a_candidate", "s3:b_candidate"]
    assert result["nested_selection"] == pytest.approx((0.40 + 0.22) / 2)
    # On the folds both cover (s2, s3) the incumbent takes 0.21 and 0.22.
    assert result["incumbent_min_on_fold_shared"] == pytest.approx(0.215)
    assert result["optimism"] == pytest.approx(0.31 - 0.215)
    assert result["optimism"] > 0, "selecting on the scored fold must flatter the result"


def test_the_optimism_gap_is_zero_when_one_candidate_dominates():
    """With no ambiguity there is nothing to select, so the published number and
    the honest number coincide. Guards against the gap being an artefact of the
    metric rather than of the selection."""
    metrics = _metrics([
        ("s1", A, 0.10), ("s1", B, 0.30),
        ("s2", A, 0.11), ("s2", B, 0.31),
        ("s3", A, 0.12), ("s3", B, 0.32),
    ])
    result = summarise(metrics)
    assert result["optimism"] == pytest.approx(0.0, abs=1e-12)
    assert result["best_by_walk_forward_mean"] == A


def test_a_single_fold_leaves_the_optimism_figure_undefined():
    """With one fold there is nothing to nest, so the comparison must be `nan`
    rather than a confident-looking zero."""
    result = summarise(_metrics([("s1", A, 0.10), ("s1", B, 0.20)]))
    assert result["n_folds"] == 1
    assert result["n_scored_by_nested"] == 0
    assert np.isnan(result["optimism"])


def test_summarise_reports_which_candidate_the_walk_forward_mean_picks():
    metrics = _metrics([
        ("s1", A, 0.20), ("s1", B, 0.10), ("s1", C, 0.30),
        ("s2", A, 0.21), ("s2", B, 0.11), ("s2", C, 0.31),
    ])
    result = summarise(metrics)
    assert result["best_by_walk_forward_mean"] == B
    assert result["walk_forward_mean"][B] == pytest.approx(0.105)
    assert result["n_folds"] == 2
    assert result["n_scored_by_nested"] == 1


# --- the manifest block ----------------------------------------------------
#
# The point of all of the above is that the number the project publishes is
# self-describing. These pin that, including the failure mode that matters most:
# a missing diagnostics file must produce "not computed", never a stale figure
# presented as current.


def test_the_manifest_block_is_absent_when_no_diagnostics_have_been_computed(tmp_path, monkeypatch):
    import json

    from pl_predictor.models import manifest as manifest_lib

    monkeypatch.setattr(manifest_lib, "SELECTION_WALK_FORWARD_PATH", tmp_path / "absent.json")
    selection = manifest_lib._selection_block({}, {"rps_ci_low": 0.1, "rps_ci_high": 0.3})
    assert selection["walk_forward_status"] == "not_computed"
    assert selection["walk_forward"] is None
    assert "headline_rps" not in selection, "no headline may be invented without a walk-forward"


def test_a_corrupt_diagnostics_file_does_not_fail_a_retrain(tmp_path, monkeypatch):
    from pl_predictor.models import manifest as manifest_lib

    corrupt = tmp_path / "selection_walk_forward.json"
    corrupt.write_text("{not json")
    monkeypatch.setattr(manifest_lib, "SELECTION_WALK_FORWARD_PATH", corrupt)
    selection = manifest_lib._selection_block({}, {})
    assert selection["walk_forward_status"] == "not_computed"


def test_the_manifest_block_says_the_holdout_was_selected_on_it(tmp_path, monkeypatch):
    from pl_predictor.models import manifest as manifest_lib

    monkeypatch.setattr(manifest_lib, "SELECTION_WALK_FORWARD_PATH", tmp_path / "absent.json")
    selection = manifest_lib._selection_block({"ml_scoreline": 0.20}, {"rps_ci_low": 0.18, "rps_ci_high": 0.22})
    assert selection["holdout_selected_on_this_fold"] is True
    assert selection["candidates_on_this_holdout"] == {"ml_scoreline": 0.20}
    assert selection["rps_ci"] == [0.18, 0.22]
    assert selection["recommended_headline"] == "walk_forward_mean_rps"


def test_the_manifest_headline_is_the_walk_forward_mean_of_the_named_model(tmp_path, monkeypatch):
    from pl_predictor.models import manifest as manifest_lib

    cache = tmp_path / "selection_walk_forward.json"
    cache.write_text(
        '{"walk_forward_mean": {"ml_scoreline": 0.199654, "covariate_poisson": 0.202646},'
        ' "best_by_walk_forward_mean": "ml_scoreline", "optimism": 0.002242}'
    )
    monkeypatch.setattr(manifest_lib, "SELECTION_WALK_FORWARD_PATH", cache)
    selection = manifest_lib._selection_block({}, {})
    assert selection["walk_forward_status"] == "cached"
    assert selection["headline_rps"] == pytest.approx(0.199654)
    assert selection["walk_forward"]["optimism"] == pytest.approx(0.002242)
