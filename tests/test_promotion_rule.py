"""Tests for the amended promotion rule.

The rule itself is the thing under test, because it is the thing every future
model change is judged by. These are written against the real evidence in the
ledger: the bivariate_poisson override that this amendment exists to catch, and
the opponent-defence promotion that has to keep passing.
"""

import pytest

from pl_predictor.evaluate.promotion_rule import DEFAULT_MAJORITY, two_gate_verdict

# The shape the majority-of-folds rule is for: the mean improves, the most recent
# fold improves, but only 2 of 5 folds do. Loss magnitudes are small so the mean
# genuinely improves and the test isolates gate 1b rather than tripping gate 1.
#
# NB this is *not* the bivariate_poisson override, which won 3 of 5 and would
# clear this rule -- see `test_gate_1b_would_not_have_caught_the_rejected_override`.
BP = {
    "2021-22": {"incumbent": 0.68890, "candidate": 0.68800},
    "2022-23": {"incumbent": 0.68910, "candidate": 0.68915},
    "2023-24": {"incumbent": 0.69020, "candidate": 0.69025},
    "2024-25": {"incumbent": 0.68950, "candidate": 0.68955},
    "2025-26": {"incumbent": 0.68970, "candidate": 0.68900},
}

# EXP-2026-25: opponent_defence, 4 of 4 folds, most-recent fold improved.
OPPONENT = {
    "2022-23": {"incumbent": 0.192208, "candidate": 0.191584},
    "2023-24": {"incumbent": 0.188242, "candidate": 0.186827},
    "2024-25": {"incumbent": 0.175250, "candidate": 0.174551},
    "2025-26": {"incumbent": 0.163033, "candidate": 0.162624},
}


# A small measured noise figure so the older tests exercise the gate they are
# about. The "no noise supplied" path is covered separately, by
# `test_a_verdict_without_a_noise_figure_is_provisional_not_a_pass`.
NOISE = 0.0005


def _judge(table, **kwargs):
    kwargs.setdefault("noise", NOISE)
    return two_gate_verdict(
        {fold: v["incumbent"] for fold, v in table.items()},
        {fold: v["candidate"] for fold, v in table.items()},
        **kwargs,
    )


def test_the_amendment_keeps_the_promotion_that_earned_it():
    verdict = _judge(OPPONENT)
    assert verdict.promoted and verdict.complete
    assert verdict.folds_won == 4 and verdict.folds_compared == 4
    assert all(gate.passed for gate in verdict.gates)


def test_a_majority_must_be_strict():
    """3 of 5 passes, 2 of 5 does not. Rounding the threshold *down* would
    reintroduce exactly the leniency the amendment removes. Both cases win the
    most recent fold and improve the mean, so gate 1 and gate 2 pass either way
    and the assertion is about gate 1b alone."""
    def case(**candidates):
        return {f"s{i}": {"incumbent": 1.0, "candidate": candidates.get(f"s{i}", 1.0)} for i in range(1, 6)}

    three_of_five = case(s1=0.9, s2=0.9, s3=1.05, s4=1.05, s5=0.9)
    two_of_five = case(s1=0.9, s2=1.05, s3=1.05, s4=1.05, s5=0.9)

    passing = _judge(three_of_five)
    gates = {gate.name: gate for gate in passing.gates}
    assert passing.promoted and passing.folds_won == 3
    assert all(gate.passed for gate in passing.gates)

    failing = _judge(two_of_five)
    gates = {gate.name: gate for gate in failing.gates}
    assert gates["gate 1 - walk-forward mean improves"].passed
    assert gates["gate 2 - most recent fold improves"].passed
    assert not gates["gate 1b - majority of folds improve"].passed
    assert not failing.promoted


def test_a_tie_is_not_a_win():
    """Counting ties as wins would let a change that moved nothing satisfy a
    majority-of-folds gate."""
    verdict = _judge({
        "s1": {"incumbent": 0.2, "candidate": 0.2},
        "s2": {"incumbent": 0.2, "candidate": 0.2},
        "s3": {"incumbent": 0.2, "candidate": 0.19},
    })
    assert verdict.folds_won == 1
    assert not verdict.promoted, "a change that only ever ties is not an improvement"


def test_gate_2_still_guards_the_newest_season_independently():
    """A change can win most folds and still break on the newest season, which is
    the failure this project has hit three times."""
    verdict = _judge({
        "2023-24": {"incumbent": 0.20, "candidate": 0.19},
        "2024-25": {"incumbent": 0.20, "candidate": 0.19},
        "2025-26": {"incumbent": 0.20, "candidate": 0.21},
    })
    gates = {gate.name: gate for gate in verdict.gates}
    assert gates["gate 1 - walk-forward mean improves"].passed
    assert gates["gate 1b - majority of folds improve"].passed
    assert not gates["gate 2 - most recent fold improves"].passed
    assert not verdict.promoted


def test_only_shared_folds_are_compared():
    """A fold the candidate could not be scored on is not evidence for it."""
    verdict = two_gate_verdict(
        {"s1": 0.2, "s2": 0.2, "s3": 0.2},
        {"s1": 0.19, "s2": 0.19},
    )
    assert verdict.folds_compared == 2
    assert "s3" not in verdict.summary


def test_the_most_recent_fold_is_the_latest_by_label_not_by_dict_order():
    """Seasons sort chronologically as strings, so gate 2 must not depend on the
    order the caller happened to build the dict in."""
    shuffled = {"2025-26": BP["2025-26"], "2021-22": BP["2021-22"], "2024-25": BP["2024-25"], "2023-24": BP["2023-24"], "2022-23": BP["2022-23"]}
    gate = {g.name: g for g in _judge(shuffled).gates}["gate 2 - most recent fold improves"]
    assert gate.detail.startswith("2025-26"), gate.detail


def test_no_shared_folds_is_not_promoted_and_says_so():
    verdict = two_gate_verdict({"s1": 0.2}, {"s9": 0.1})
    assert not verdict.promoted
    assert verdict.folds_compared == 0
    assert "no comparable folds" in verdict.summary


def test_higher_is_better_is_supported():
    """RPS and log loss want lower; accuracy wants higher. A rule that silently
    assumed one direction would be a trap."""
    verdict = two_gate_verdict(
        {"s1": 0.60, "s2": 0.60, "s3": 0.60},
        {"s1": 0.62, "s2": 0.61, "s3": 0.61},
        metric="accuracy", lower_is_better=False, noise=0.005,
    )
    assert verdict.promoted
    assert "accuracy" in verdict.summary


def test_the_default_majority_is_a_strict_half():
    assert DEFAULT_MAJORITY == 0.5
    verdict = _judge(OPPONENT)
    assert verdict.majority_required == 0.5


# --- gate 1c: the noise margin, which is the gate that does the work ---------


def test_a_verdict_without_a_noise_figure_is_provisional_not_a_pass():
    """`we did not check` and `it passed` must not look alike. A caller that has
    not measured the noise gets `complete=False` and a summary that says gate 1c
    was not evaluated -- it does not fail open, and it does not claim a pass."""
    verdict = two_gate_verdict(
        {fold: v["incumbent"] for fold, v in OPPONENT.items()},
        {fold: v["candidate"] for fold, v in OPPONENT.items()},
    )
    assert not verdict.complete
    assert not verdict.promoted
    assert "PROMOTED PROVISIONALLY" not in verdict.summary
    gate = {g.name: g for g in verdict.gates}["gate 1c - improvement exceeds measured noise"]
    assert not gate.passed and "NOT EVALUATED" in gate.detail


def test_an_improvement_inside_the_noise_band_does_not_promote():
    """The shape that motivated gate 1c: the mean moves the right way, the mean
    moves by less than the metric's own noise."""
    verdict = two_gate_verdict(
        {"s1": 0.200, "s2": 0.200, "s3": 0.200},
        {"s1": 0.1999, "s2": 0.1999, "s3": 0.1999},
        noise=0.0005,
    )
    gate = {g.name: g for g in verdict.gates}["gate 1c - improvement exceeds measured noise"]
    assert not gate.passed
    assert not verdict.promoted
    assert verdict.complete, "the noise figure was supplied, so this is a real rejection"


def test_an_improvement_well_inside_the_noise_does_not_promote():
    verdict = two_gate_verdict(
        {"s1": 0.200, "s2": 0.200, "s3": 0.200},
        {"s1": 0.1990, "s2": 0.1990, "s3": 0.1990},
        noise=0.0020,
    )
    assert not verdict.promoted


def test_a_real_promotion_passes_once_the_noise_is_measured():
    """EXP-2026-25 re-adjudicated with its own measured paired half-width of
    0.000161. The mean improvement is +0.000787, about 4.9x the noise, and it
    wins every fold."""
    verdict = two_gate_verdict(
        {"2022-23": 0.192208, "2023-24": 0.188242, "2024-25": 0.175250, "2025-26": 0.163033},
        {"2022-23": 0.191584, "2023-24": 0.186827, "2024-25": 0.174551, "2025-26": 0.162624},
        metric="log_loss",
        noise=0.000161,
    )
    assert verdict.promoted and verdict.complete
    assert verdict.folds_won == 4
    gate = {g.name: g for g in verdict.gates}["gate 1c - improvement exceeds measured noise"]
    assert gate.passed
    assert "PROMOTED" in verdict.summary and "PROVISIONALLY" not in verdict.summary


def test_gate_1b_would_not_have_caught_the_rejected_override():
    """The retraction, pinned as a test so it cannot be reintroduced.

    EXP-2026-16's covariate_poisson won 3 of 5 folds and improved the mean, so it
    clears a majority-of-folds rule. An earlier note in this project claimed it
    "won only 2 of 5" and that the majority rule would have caught it. Both are
    wrong, and a majority rule resting on that story would have been justified by
    a misreading of its own evidence.
    """
    verdict = two_gate_verdict(
        {"2021-22": 0.700873, "2022-23": 0.682485, "2023-24": 0.666202, "2024-25": 0.673814, "2025-26": 0.691073},
        {"2021-22": 0.686906, "2022-23": 0.679133, "2023-24": 0.672151, "2024-25": 0.684834, "2025-26": 0.690040},
        metric="over_2_5_log_loss",
        noise=0.018736,
    )
    gates = {g.name: g for g in verdict.gates}
    assert verdict.folds_won == 3
    assert gates["gate 1 - walk-forward mean improves"].passed
    assert gates["gate 1b - majority of folds improve"].passed, "3 of 5 clears a majority"
    assert gates["gate 2 - most recent fold improves"].passed
    # Only the noise margin rejects it.
    assert not gates["gate 1c - improvement exceeds measured noise"].passed
    assert not verdict.promoted


def test_a_strict_majority_at_an_even_fold_count():
    """`test_a_majority_must_be_strict` only used odd fold counts, where `>` and
    `>=` are indistinguishable -- 2.5 admits no integer. At n=4 they differ, and
    `>=` would admit a 2-2 split."""
    def case(**candidates):
        return {f"s{i}": {"incumbent": 1.0, "candidate": candidates.get(f"s{i}", 1.0)} for i in range(1, 5)}

    split = _judge(case(s1=0.9, s2=0.9, s3=1.05, s4=1.05))
    gates = {g.name: g for g in split.gates}
    assert split.folds_won == 2 and split.folds_compared == 4
    assert gates["gate 1 - walk-forward mean improves"].passed
    assert not gates["gate 1b - majority of folds improve"].passed, "2 of 4 is not a majority"
    assert not split.promoted

    # s4 must win: gate 2 is the most recent fold, and a loss there fails the
    # verdict for a different reason than the majority rule under test.
    three_of_four = _judge(case(s1=1.05, s2=0.9, s3=0.9, s4=0.9))
    assert three_of_four.promoted and three_of_four.folds_won == 3


def test_an_improvement_exactly_equal_to_the_noise_does_not_pass():
    """`improvement > threshold` is strict. `>=` would admit a change whose entire
    measured margin is inside the noise."""
    verdict = two_gate_verdict(
        {"s1": 0.2, "s2": 0.2, "s3": 0.2},
        {"s1": 0.2, "s2": 0.2, "s3": 0.19},
        noise=0.01, noise_basis="fold_mean",
    )
    gate = {g.name: g for g in verdict.gates}["gate 1c - improvement exceeds measured noise"]
    assert not gate.passed
    assert not verdict.promoted


def test_a_per_fold_noise_figure_is_divided_by_sqrt_folds():
    """The gated quantity is a mean over folds; a per-fold half-width is therefore
    too strict for it by about sqrt(F). Comparing them unscaled rejects real
    improvements and reads as evidence."""
    import math

    inc = {f"s{i}": 0.2 for i in range(1, 5)}
    cand = {f"s{i}": 0.2 for i in range(1, 5)}
    for fold in ("s1", "s2", "s3", "s4"):
        cand[fold] = 0.199
    verdict = two_gate_verdict(inc, cand, noise=0.002, noise_basis="per_fold")
    gate = {g.name: g for g in verdict.gates}["gate 1c - improvement exceeds measured noise"]
    assert verdict.promoted, f"0.001 against a threshold of {0.002 / 2:.6f} should pass: {gate.detail}"
    assert f"{0.002 / math.sqrt(4):.6f}" in gate.detail


def test_an_unknown_noise_basis_is_rejected():
    with pytest.raises(ValueError, match="noise_basis"):
        two_gate_verdict({"s1": 0.2}, {"s1": 0.1}, noise=0.01, noise_basis="per_season")


def test_the_provisional_summary_is_reachable_and_states_itself():
    """`promoted and not complete` cannot happen -- the noise gate is hard-coded
    not-passed when no noise is supplied. So `complete` is reported on its own
    line rather than via an unreachable "PROVISIONALLY" headline."""
    verdict = two_gate_verdict(
        {"2022-23": 0.192208, "2023-24": 0.188242},
        {"2022-23": 0.191584, "2023-24": 0.186827},
    )
    assert not verdict.promoted and not verdict.complete
    assert "INCOMPLETE" in verdict.summary
    assert "PROVISIONALLY" not in verdict.summary
