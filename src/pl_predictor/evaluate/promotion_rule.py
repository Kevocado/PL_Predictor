"""The project's promotion rule, as code rather than as a paragraph.

PL has always judged a model change by a two-gate rule: improve the walk-forward
mean **and** hold on the most recent completed season. It lived only in
`docs/AI_CONTINUITY.md`, and that turned out to be load-bearing in a bad way —
two experiments read the same paragraph and reached different conclusions about
the same evidence.

**The amendment (2026-09-27, EXP-2026-27) adds two gates, and the second one is
the important one.**

*Gate 1b — the improvement must hold in a majority of shared folds.* Gate 1 was a
mean, and a mean can be carried by a single large win. This stops the
one-lucky-fold shape. It is a real but *weak* guard, and it is worth being exact
about how weak.

*Gate 1c — the improvement must exceed the metric's own measured noise.* This is
the gate that does the work, and it exists because of a correction. An earlier
note cited "the bivariate_poisson override won only 2 of 5 folds" as the example
a majority rule would have caught, and the override behind that argument was
**covariate_poisson** — which won **3 of 5** (2021-22, 2022-23, 2025-26; losing
2023-24 and 2024-25) and therefore *clears* a majority rule comfortably.

To be precise about what was and was not wrong: the **2-of-5 count was true of
bivariate_poisson** (Task 7b's fresh 5-fold evaluation, 2 of 5). The error was
conflating the two overrides. So the majority gate has no demonstrated near-miss
to point at; it is kept as a cheap guard against a single large win carrying a
mean, not because it caught something. What the reverted override actually failed
on was its **margin**: a mean gain of 0.000277 log-loss against a measured noise
half-width of 0.018736. A 3-of-5 record with margins of 0.0003 is roughly what
noise produces half the time, so the fold count was never the tell.

Hence a noise margin, and hence `noise` being a required input rather than a
constant: **noise is per-metric and per-n**. EXP-2026-26's 0.018736 is an RPS
figure for the scoreline candidates and does not transfer to G+A log loss at a
different n. A rule with a hardcoded noise floor would be confidently wrong on
the next metric, which is the failure mode this whole module exists to remove.

When `noise` is not supplied the verdict is **provisional**: gate 1c is recorded
as not evaluated, `complete` is False, and the summary says so. It does not fail
open and it does not pretend to have passed.

Putting the rule here rather than in prose means a future experiment is judged by
the same code as this one, and cannot quietly adopt a friendlier reading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# A change must win at least this fraction of shared folds, not merely the mean.
DEFAULT_MAJORITY = 0.5

# EXP-2026-26 measured a per-fold RPS bootstrap half-width of 0.018736 at n=380 for
# the *scoreline* candidates. Recorded as a provenance-bearing number, NOT used as
# a default: it is specific to that metric and that n, and reusing it elsewhere
# would be a confidently-wrong constant rather than an absent one.
MEASURED_NOISE_EXAMPLE = {"metric": "rps", "n_val": 380, "half_width": 0.018736}


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class TwoGateVerdict:
    promoted: bool
    metric: str
    incumbent_mean: float | None
    candidate_mean: float | None
    folds_compared: int
    folds_won: int
    majority_required: float
    # False when a gate could not be evaluated (no noise figure supplied). A
    # provisional verdict is not a pass, so this carries no default: a caller
    # that forgets it gets a loud error rather than a silent "complete".
    complete: bool
    gates: tuple[Gate, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        if self.incumbent_mean is None or self.candidate_mean is None:
            return "NOT PROMOTED - no comparable folds."
        # No "provisional" branch: `promoted` requires every gate to have passed,
        # and the noise gate is hard-coded not-passed when no noise was supplied,
        # so `promoted and not complete` is unreachable. `complete` is reported on
        # its own line instead, where it is actually informative.
        lines = [f"{'PROMOTED' if self.promoted else 'NOT PROMOTED'} on {self.metric}:"]
        lines.append(
            f"  mean {self.incumbent_mean:.6f} -> {self.candidate_mean:.6f} "
            f"({self.candidate_mean - self.incumbent_mean:+.6f})"
        )
        lines.append(f"  folds won {self.folds_won} of {self.folds_compared}")
        if not self.complete:
            lines.append("  INCOMPLETE - a required gate could not be evaluated (see below)")
        lines.extend(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}" for gate in self.gates)
        return "\n".join(lines)


def _is_better(candidate: float, incumbent: float, lower_is_better: bool) -> bool:
    """Strictly better. A tie is not a win — counting ties as wins would let a
    change that moved nothing satisfy a majority-of-folds gate."""
    return candidate < incumbent if lower_is_better else candidate > incumbent


def two_gate_verdict(
    incumbent: dict[str, float],
    candidate: dict[str, float],
    *,
    metric: str = "rps",
    lower_is_better: bool = True,
    majority: float = DEFAULT_MAJORITY,
    noise: float | None = None,
    noise_basis: str = "per_fold",
) -> TwoGateVerdict:
    """Judge a change by the project's amended two-gate rule.

    `incumbent` and `candidate` map fold label -> score. Fold labels must sort
    chronologically; season strings do, which is why they are used. Only folds
    present in **both** are compared, and only those count toward the mean, the
    majority, and the most-recent-fold gate — a fold the candidate could not be
    scored on cannot be evidence for it either way.

    The four gates:

    1. the walk-forward mean improves;
    2. a majority of shared folds improve;
    3. the mean improvement exceeds `noise` (the 2026-09-27 amendment's real gate);
    4. the most recent shared fold improves.

    All evaluated gates must pass. Gate 4 stays separate from gate 2 because a
    change can be broadly better and still break on the newest season, which is
    the specific failure this project has been bitten by three times.

    `noise` is the measured noise half-width **for this metric at this n** — a
    bootstrap CI half-width, or the half-width of a paired per-fold difference
    interval, which is tighter and the right choice when both arms are scored on
    the same fixtures. It is deliberately not defaulted.

    `noise_basis` says which scale `noise` is on, because the thing being gated is
    a mean over folds and the two are not interchangeable: a per-fold half-width
    must be divided by sqrt(folds) to be compared against a fold mean. Default
    `"per_fold"`, which is what every measurement in this project has produced. A caller that does not
    measure it gets `complete=False` and a verdict that says gate 1c was not
    evaluated, because "we did not check" and "it passed" must not look alike.
    """
    shared = sorted(set(incumbent) & set(candidate))
    if not shared:
        return TwoGateVerdict(
            promoted=False, complete=False, metric=metric,
            incumbent_mean=None, candidate_mean=None,
            folds_compared=0, folds_won=0, majority_required=majority,
            gates=(Gate("comparable folds", False,
                        f"none shared between incumbent ({len(incumbent)}) and candidate ({len(candidate)})"),),
        )

    won = [fold for fold in shared if _is_better(candidate[fold], incumbent[fold], lower_is_better)]
    incumbent_mean = sum(incumbent[fold] for fold in shared) / len(shared)
    candidate_mean = sum(candidate[fold] for fold in shared) / len(shared)

    required = len(shared) * majority
    # An odd fold count cannot give an exact half, so a majority is a strict
    # majority: 3 of 5 passes, 2 of 5 does not. Rounding the threshold *down*
    # would reintroduce precisely the leniency the amendment removes.
    majority_ok = len(won) > required

    improvement = (incumbent_mean - candidate_mean) if lower_is_better else (candidate_mean - incumbent_mean)
    if noise is None:
        noise_gate = Gate("gate 1c - improvement exceeds measured noise", False,
                          "NOT EVALUATED - no noise figure supplied for this metric and n")
        threshold = float("nan")
    else:
        # The quantity gated is a mean over `len(shared)` folds, whose standard
        # error is roughly the per-fold SE over sqrt(F). Comparing a per-fold
        # half-width against a fold mean is therefore about sqrt(F) too strict --
        # conservative, but it would reject a real improvement and read as
        # evidence, which is the failure this module exists to prevent. The caller
        # declares which scale it measured.
        if noise_basis == "per_fold":
            threshold = noise / math.sqrt(len(shared))
        elif noise_basis == "fold_mean":
            threshold = noise
        else:
            raise ValueError(f"noise_basis must be 'per_fold' or 'fold_mean', got {noise_basis!r}")
        noise_gate = Gate("gate 1c - improvement exceeds measured noise",
                          improvement > threshold,
                          f"improvement {improvement:+.6f} vs threshold {threshold:.6f} "
                          f"({noise:.6f} {noise_basis}, {len(shared)} folds)")

    gates = (
        Gate("gate 1 - walk-forward mean improves",
             _is_better(candidate_mean, incumbent_mean, lower_is_better),
             f"{incumbent_mean:.6f} -> {candidate_mean:.6f}"),
        Gate("gate 1b - majority of folds improve",
             majority_ok,
             f"won {len(won)} of {len(shared)}; needs more than {majority:.0%} "
             f"({required:.1f} folds)"),
        noise_gate,
        Gate("gate 2 - most recent fold improves",
             _is_better(candidate[shared[-1]], incumbent[shared[-1]], lower_is_better),
             f"{shared[-1]}: {incumbent[shared[-1]]:.6f} -> {candidate[shared[-1]]:.6f}"),
    )

    return TwoGateVerdict(
        promoted=all(gate.passed for gate in gates),
        complete=noise is not None,
        metric=metric, incumbent_mean=incumbent_mean, candidate_mean=candidate_mean,
        folds_compared=len(shared), folds_won=len(won), majority_required=majority,
        gates=gates,
    )
