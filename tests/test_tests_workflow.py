"""A gate whose failures depend on a cache is not a gate.

`tests.yml` landed on 2026-09-28 (PR #7) and went red on its first run against
`main`: **20 failed, 492 passed, 27 skipped** in 231.64s. Not bugs in the change
under test -- tests reaching for a `data/cache/` that a fresh clone does not
have, because the directory is gitignored and only `.gitkeep` is tracked.

This file exists because the fix for that was obvious and the *next* failure
was not. Once the gate was given a cache-warming step, the remaining way to
break it is subtle: set the timeout below the cold cost, pipe a step's output
so its exit status stops meaning anything, or quietly let the gate depend on a
third party. Each of those has shipped in this file's own history.

So the checks below are about **properties**, not about the current shape of the
workflow. Three of them are calibrated against a real number rather than a
string, because the actual bug was a value that had never been measured:

* `check_the_timeouts_clear_the_measured_costs` parses each job's
  `timeout-minutes` and the duration the workflow's own comment claims, and
  fails if either is below its cost. The warm-cache job must clear the cold
  run; the gate job must clear the gate run. A check that read only the first
  timeout would pass a generous warm beside a strangled gate.
* `check_warming_and_gating_are_separate_jobs` requires a step that may fetch
  and a *different job* that gates with the network blocked, with a tighter
  budget. Merging them would let a test that reaches the network pass -- which
  is the whole thing `tests/conftest.py`'s guard is for -- and would leave the
  gate with no budget of its own.
* `check_the_warm_step_reports_a_real_exit_status` rejects `| tail` and similar.
  A pipe makes the step's status the *last* command's, so a cold-cache failure
  reports as a step that succeeded.

And the standing finding from `pl-deploy-and-hang-report.md` applies here too:
every guard in `test_deploy_workflow.py` is pinned against `deploy.yml` and
none can see the README, which is how the README came to describe the wrong
deploy path for so long. The same shape: until this file, **no test read
`tests.yml` at all**, and the README did not mention it in 0 places.
`check_readme_names_the_test_gate` closes both directions, including the
command that settles whether the check is actually required -- a fact about the
server that no test in this repository can see.

Every check is also run against a deliberately broken copy to prove it can
fail, following `test_deploy_workflow.py`. A guard that passes vacuously is
worse than no guard, because it is trusted.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO / ".github" / "workflows"
WORKFLOW = WORKFLOW_DIR / "tests.yml"
SNAPSHOT_WORKFLOW = WORKFLOW_DIR / "refresh-public-snapshot.yml"
README = REPO / "README.md"

# Matches the deploy guard's window. The README wraps at 80 columns, so a
# line-local rule fires on a true sentence that merely wrapped mid-claim, and
# a guard that cries wolf gets deleted rather than fixed.
SENTENCE_CHARS = 200

# Measured on this machine against origin/main, 2026-09-28, and on the CI
# runner the same day. These are the numbers the workflow's comments quote, and
# the timeout checks hold the settings to them.
#
#   cold + network, whole suite        4926.07s (1:22:06)
#   cold + network, gate half only     2594.15s (0:43:14)
#   warm + guard                        669.66s (0:11:09)
#   gate half, CI runner               1336s (22m16s)
#
# The first is what the warm-cache job pays when no cache was restored, and its
# timeout has to clear it or the runner is killed before actions/cache's post
# step runs. The fourth is what the *gate* job pays, and it is the reason the
# gate's timeout is 30 and not 20: 20 minutes is 1200s against a 1336s gate.
COLD_RUN_SECONDS = 4926.07
GATE_RUN_SECONDS = 1336

# The three upstream services the suite fetches when allowed to.
THIRD_PARTIES = ("FPL", "ClubElo", "football-data.co.uk")


def workflow_text() -> str:
    assert WORKFLOW.exists(), f"{WORKFLOW} is missing; a commit is ungated"
    return WORKFLOW.read_text()


def readme_text() -> str:
    return README.read_text()


def steps(text: str) -> list[dict]:
    """Every step in every job, as `{job, name, run, env, with, if, continue_on_error}`.

    Read structurally rather than by regex over the file. The assertions that
    matter are about a *specific step* having an env var or not; a file-wide
    `in` check is satisfied by the token appearing anywhere, which is the
    vacuous version of the same question. The `job` key is load-bearing: warm
    and gate live in different jobs, and several checks below assert that.
    """
    doc = yaml.safe_load(text)
    out: list[dict] = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            out.append(
                {
                    "job": job_name,
                    "job_timeout": job.get("timeout-minutes"),
                    "name": step.get("name") or step.get("uses") or "",
                    "run": step.get("run") or "",
                    "env": step.get("env") or {},
                    "with": step.get("with") or {},
                    "if": step.get("if") or "",
                    "continue_on_error": step.get("continue-on-error"),
                }
            )
    return out


def job_timeouts(text: str) -> dict[str, int]:
    """Each job's `timeout-minutes`, by job name.

    The two jobs have different budgets for different reasons -- the warm job
    must clear the cold-run cost, the gate job must clear the gate cost -- so a
    check that reads only the first timeout in the file would pass a workflow
    whose gate is strangled. Both are asserted separately.
    """
    doc = yaml.safe_load(text)
    return {
        name: int(job["timeout-minutes"])
        for name, job in (doc.get("jobs") or {}).items()
        if job.get("timeout-minutes") is not None
    }


# --------------------------------------------------------------- the checks


def check_yaml_parses(text: str) -> None:
    """A merge should not be where this is first discovered."""
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise AssertionError(f"tests.yml does not parse as YAML: {exc}") from exc


def check_the_timeouts_clear_the_measured_costs(text: str) -> None:
    """Each job's `timeout-minutes` must exceed what that job actually costs.

    This is the check that would have caught the live bug, and the bug was a
    number nobody had measured. This file carried `timeout-minutes: 60` with a
    comment explaining that 30 was too small and 60 was "generous"; the cold
    run is 4926.07s, so 60 minutes was *below* the cost. The job was killed
    before `actions/cache`'s post step ran, so it saved no cache, so the next
    run was cold too -- the exact deadlock `c22a310` diagnosed, still live with
    a value that looked safe because it was bigger than the previous one.

    There are two budgets because there are two jobs, and they fail
    independently. The warm-cache job must clear the cold-run floor
    (`COLD_RUN_SECONDS`); the gate job must clear the gate floor
    (`GATE_RUN_SECONDS`). A check that reads only the first timeout in the file
    would pass a workflow whose warm job is generous and whose gate is
    strangled -- which is exactly what `timeout-minutes: 20` on the gate was,
    at 1200s against a 1336s measured gate. That is the same bug in miniature,
    and this check exists so it cannot be reintroduced as a "tightening".

    The floors are measured numbers, not round ones, because the whole failure
    was that a round number felt safe. Each is a floor and not an equality so a
    future, faster suite can still be accommodated.
    """
    timeouts = job_timeouts(text)
    assert "warm-cache" in timeouts, (
        "tests.yml has no `warm-cache` job with a timeout. Without one, a hung "
        "upstream leaves the job pending for GitHub's 6-hour default, which reads "
        "as 'in progress' rather than 'broken'."
    )
    assert "pytest" in timeouts, (
        "tests.yml has no `pytest` gate job with a timeout."
    )
    warm_floor = int(COLD_RUN_SECONDS / 60) + 1
    assert timeouts["warm-cache"] >= warm_floor, (
        f"warm-cache timeout is {timeouts['warm-cache']}, but a cold run costs "
        f"{COLD_RUN_SECONDS}s (~{warm_floor} minutes). A timeout below the cold "
        "cost is not a safety improvement: the job is killed before "
        "actions/cache's post step runs, so it saves no cache, so the next run "
        "is cold too and times out too."
    )
    gate_floor = int(GATE_RUN_SECONDS / 60) + 1
    assert timeouts["pytest"] >= gate_floor, (
        f"gate timeout is {timeouts['pytest']}, but the gate half measured "
        f"{GATE_RUN_SECONDS}s (~{gate_floor} minutes) on the CI runner. A gate "
        "timeout below the gate cost kills green runs for slowness, which reads "
        "exactly like a broken change."
    )
    # The comments have to claim the same numbers the settings are held to, or
    # the two drift and the next reader trusts whichever one they find first.
    for needle in ("4926.07s", "1336s"):
        assert needle in text, (
            f"the timeout comments quote no {needle} duration. The bug these "
            "checks exist for was a value that had never been measured, so the "
            "number has to be in the file, next to the setting it justifies."
        )


def check_warming_and_gating_are_separate_jobs(text: str) -> None:
    """A job may fetch; a *different job* must gate with the network blocked.

    This is the property that makes the warm step safe rather than a hole.
    `tests/conftest.py` blocks outbound connections so that a test needing data
    it does not have fails fast instead of hanging; the warm step disables that
    block to fetch the data. If one step did both, a test that reaches the
    network would pass -- and the guard would be checking nothing, on a run
    that looked green.

    So: exactly one step sets `PL_ALLOW_NETWORK`, the step that runs the gated
    suite must not, and -- structurally -- the two must live in different jobs.
    The job split is what gives the gate its own tight budget: a gate that
    shares the warm job's 150-minute ceiling has no budget at all, and a warm
    that shares the gate's 30 minutes cannot survive a cold cache. Merging the
    jobs reintroduces the deadlock under a different name.
    """
    may_fetch = [s for s in steps(text) if str(s["env"].get("PL_ALLOW_NETWORK", "")) == "1"]
    assert may_fetch, (
        "no step in tests.yml sets PL_ALLOW_NETWORK=1. The suite genuinely needs the "
        "network -- data/cache/ is gitignored, so a fresh clone has no FPL "
        "player-gameweek archive and no football-data.co.uk CSVs -- and with the "
        "network blocked 20 of 540 tests fail on a cold runner. That is what made "
        "the gate red on arrival (run 36446591387: 20 failed in 231.64s)."
    )
    assert len(may_fetch) == 1, (
        f"{len(may_fetch)} steps set PL_ALLOW_NETWORK=1 "
        f"({[s['name'] for s in may_fetch]}). Exactly one -- the warm step. Every "
        "extra one widens the window in which a test that reaches the network is "
        "reported as passing."
    )
    gating = [s for s in steps(text) if re.search(r"pytest tests/ -q", s["run"])]
    assert gating, "tests.yml has no step running `pytest tests/ -q`"
    # The gated step and the warm step must be different steps. The warm step
    # also runs `pytest tests/`, so identity is by name, not by run body.
    for step in gating:
        if step["name"] == may_fetch[0]["name"]:
            continue
        assert str(step["env"].get("PL_ALLOW_NETWORK", "")) != "1", (
            f"the gating step {step['name']!r} sets PL_ALLOW_NETWORK=1. Warming and "
            "gating in one step means a test that reaches the network passes, and the "
            "guard in tests/conftest.py checks nothing on a run that looks green."
        )
    # And different jobs: the split is what lets the gate carry a 30-minute
    # budget while the warm carries 150. A check that only asserted separate
    # steps would pass a single-job workflow whose gate has no budget of its
    # own -- the structural unit that matters is the job, not the step.
    warm_job = may_fetch[0]["job"]
    gate_jobs = {s["job"] for s in gating if s["name"] != may_fetch[0]["name"]}
    assert gate_jobs, "no gating step distinct from the warm step was found"
    assert warm_job not in gate_jobs, (
        f"the warm step and the gating step share job {warm_job!r}. Warming and "
        "gating in one job means one timeout covers both, so either the gate can "
        "hang for the warm budget or the warm cannot survive a cold cache."
    )
    timeouts = job_timeouts(text)
    for gate_job in gate_jobs:
        assert timeouts[gate_job] < timeouts[warm_job], (
            f"gate job {gate_job!r} has timeout {timeouts[gate_job]}, not below warm "
            f"job {warm_job!r}'s {timeouts[warm_job]}. The gate is fast by design -- "
            "it never pays the cold cost -- and its budget must say so, or a future "
            "edit can inflate it back to a single shared ceiling without failing "
            "anything."
        )


def check_the_warm_step_reports_a_real_exit_status(text: str) -> None:
    """A piped step's exit status is the last command's, not pytest's.

    This file shipped `run: pytest tests/ -q 2>&1 | tail -20`. The step's status
    became `tail`'s, which is always zero, so the `continue-on-error: true`
    sitting directly above it was unreachable and a cold-cache failure was
    reported as a step that succeeded. The pipe was there to keep the log
    short; `continue-on-error` is what was actually wanted.
    """
    for step in steps(text):
        if not re.search(r"pytest[^\n|]*\|", step["run"]):
            continue
        assert False, (
            f"step {step['name']!r} pipes pytest into another command "
            f"({step['run'].strip()[:60]!r}). A pipeline's exit status is the last "
            "command's, so the step always succeeds and a failure is reported as a "
            "pass. Use `continue-on-error: true` for a best-effort step -- it is "
            "already set on the warm step and the pipe made it unreachable."
        )


def check_the_warm_step_does_not_run_the_guard_s_own_tests(text: str) -> None:
    """`test_network_guard.py` fails when the guard is disabled; that is the design.

    Those three tests assert the network block is in place. Run with
    `PL_ALLOW_NETWORK=1` they fail -- measured: 3 failed, all in
    `test_network_guard.py`. In the warm step that is the guard working, not a
    warm failure, and the gated step below runs them properly with the block
    on. So the warm step must exclude them.
    """
    warm = _the_step_with(text, "PL_ALLOW_NETWORK", "1")
    # A positive requirement, not "the filename appears". The pre-fix warm step
    # was `pytest tests/ -q 2>&1 | tail -20` -- no mention of the guard's own
    # tests, and it ran all 540 of them with the block disabled, so three
    # failed. A check that only noticed the filename would have passed that.
    assert "test_network_guard.py" in warm["run"] and "--ignore" in warm["run"], (
        f"the warm step does not exclude tests/test_network_guard.py (it runs "
        f"{warm['run'].strip()[:70]!r}). Those three tests assert the network block "
        "is in place, so they FAIL when PL_ALLOW_NETWORK=1 disables it -- measured: "
        "3 failed, all in that file. That is the guard working, not a warm failure. "
        "The warm step must exclude them; the gated step runs them with the block on."
    )


def check_the_gate_is_advisory_because_it_reads_third_parties(text: str) -> None:
    """A required check must not be red-able by an upstream outage.

    The gate reads FPL, ClubElo and football-data.co.uk. An upstream outage
    turns it red for a reason that has nothing to do with the change under
    review, and a required check that is intermittently red for reasons outside
    the diff is a check people learn to re-run rather than read.

    `main` has no branch protection and no rulesets today, so the check is
    advisory in fact as well as in intent. That is a fact about the *server*,
    which nothing in this repository can see -- so the README carries the
    command that settles it, the way `test_deploy_workflow.py` carries
    `gh api /user/packages/container/pl-predictor` for the GHCR linkage.
    """
    lowered = text.lower()
    for word in ("required:", "required_status_checks"):
        assert word not in lowered, (
            f"tests.yml contains {word!r}. The gate reads {', '.join(THIRD_PARTIES)} "
            "and can be made red by any of them going down, so it is advisory. "
            "Promoting it to a required status check is a deliberate decision about "
            "branch protection, not a workflow edit."
        )
    assert any(s["continue_on_error"] is True for s in steps(text)), (
        "no step in tests.yml is marked `continue-on-error: true`. The warm step and "
        "the network-marked half both read live upstreams and are advisory; that has "
        "to be expressed in the file rather than in a reader's memory."
    )


def check_the_gate_does_not_assume_a_clean_tree(text: str) -> None:
    """The suite writes to four tracked `models/*.json` on every run.

    Running it rewrites the XGBoost models. That is a real wart -- it is why
    `git status` is dirty after a local run and it would trip a cleanliness
    check in any other repo that ran this suite. It means a
    `git diff --exit-code` step added here later would fail on an untouched
    tree, for a reason unrelated to the change. Asserted so whoever adds one
    reads this first.
    """
    for needle in ("git diff --exit-code", "git status --porcelain"):
        assert needle not in text, (
            f"tests.yml contains {needle!r}. The suite writes to four tracked "
            "models/*.json on every run, so this fails on a clean tree for a reason "
            "that has nothing to do with the change under review. Either stop the "
            "suite writing to tracked files, or scope the check to the paths it does "
            "not touch."
        )


def check_the_cache_key_is_not_shared_with_the_snapshot_job(
    text: str, snapshot: str
) -> None:
    """Two jobs caching `data/cache` must not share a key family.

    They cache the same path and want different things from it:
    `refresh-public-snapshot.yml` warms only what `public_snapshot.py` reads
    (5.4MB, 74 files) while this suite needs 4,723 files and 87MB. A shared
    family would have each job restore the other's smaller view and overwrite
    it, and the symptom would be a gate that is mysteriously cold on a repo
    whose cache list looks full.

    In fact the two families were *already* different and unrecognised as
    related: `data-cache-` in the snapshot job, `pl-upstream-caches-` here,
    with zero entries ever written under the second. This check is what makes
    the separation deliberate rather than a coincidence nobody can see.
    """
    def family(text: str) -> set[str]:
        return set(re.findall(r"^\s*key:\s*([A-Za-z0-9_-]+?)-\$\{\{", text, re.M))

    mine, theirs = family(text), family(snapshot)
    assert mine and theirs, (
        "could not read a cache key out of one of the workflows; this check is "
        "vacuous if it cannot see a key to compare"
    )
    assert not (mine & theirs), (
        f"tests.yml and refresh-public-snapshot.yml share the cache key family "
        f"{mine & theirs} for the same path. They want different contents -- the "
        "snapshot job warms 74 files, this suite needs 4,723 -- so a shared key "
        "makes each job overwrite the other's view of data/cache."
    )


def check_readme_names_the_test_gate(readme: str) -> None:
    """The README is where a reader goes when CI is red.

    It said nothing about `tests.yml` -- 0 mentions -- while a sibling section
    went to some length about which deploy workflow is authoritative, and while
    the gate was red on arrival. A reader told this repo is careful about
    deploying, and told nothing about what runs its tests, will assume there is
    no gate, or find the red and not know what it means.

    Both directions, because either alone is satisfied by a README that says
    nothing: the README must name `tests.yml` and must say whether it gates.
    The window is a sentence rather than a line, and centred on the mention,
    because the README wraps at 80 columns -- a line-local rule reported a true
    sentence that merely wrapped mid-claim, and a guard that cries wolf gets
    deleted rather than fixed.
    """
    mentions = list(re.finditer(r"tests\.yml", readme))
    assert mentions, (
        "the README never mentions tests.yml. It documents which deploy workflow is "
        "authoritative in detail and says nothing about the one that gates a commit, "
        "which is the question a reader has when CI is red."
    )
    qualified = [
        m
        for m in mentions
        if re.search(
            r"advisory|not required|required status",
            readme[max(0, m.start() - SENTENCE_CHARS): m.end() + SENTENCE_CHARS],
            re.I,
        )
    ]
    assert qualified, (
        "the README names tests.yml but nowhere says whether it gates a merge. It is "
        "advisory, and that is a decision with a reason behind it -- the suite reads "
        "three third parties -- so a reader should not have to infer it from a red run."
    )
    # A claim nobody can check is how the deploy section rotted. This one is a
    # fact about the server, not about the repository.
    assert "gh api" in readme, (
        "the README's test-gate section asserts a fact it does not let the reader "
        "check. Whether `tests` is a required status check lives in branch protection "
        "on the server; the command that settles it belongs next to the claim, the way "
        "`gh api /user/packages/container/pl-predictor` does for the GHCR linkage."
    )


# ------------------------------------------------------------ the plumbing


def _the_step_with(text: str, env_key: str, env_value: str) -> dict:
    """The one step whose `env` sets `env_key` to `env_value`.

    Asserts there is exactly one. Zero means the workflow does not do the
    thing at all; more than one means the caller is inspecting whichever
    happened to be first.
    """
    matches = [s for s in steps(text) if str(s["env"].get(env_key, "")) == env_value]
    assert len(matches) == 1, (
        f"expected exactly one step with {env_key}={env_value!r}, found {len(matches)}: "
        f"{[s['name'] for s in matches]}"
    )
    return matches[0]


# Workflow checks only. `check_readme_names_the_test_gate` takes the *README*,
# not this file, so putting it in this dict would let `test_each_check_can_fail`
# hand it a mutated workflow and call the result a mutation proof. It has its
# own, in `test_the_readme_check_can_fail`.
CHECKS = {
    "parses": check_yaml_parses,
    "warm_timeout": check_the_timeouts_clear_the_measured_costs,
    # Same check, second leg: the mutation loop breaks one setting at a time,
    # so the gate leg gets its own entry. A single "timeout" entry could only
    # break the warm setting and would leave a strangled gate unproven.
    "gate_timeout": check_the_timeouts_clear_the_measured_costs,
    # Same check, third leg: the job budgets must stay ordered. Inflating the
    # gate to the warm budget re-merges the two costs without touching the job
    # boundary, and neither timeout leg alone would catch it.
    "job_budgets": check_warming_and_gating_are_separate_jobs,
    "separate_jobs": check_warming_and_gating_are_separate_jobs,
    "exit_status": check_the_warm_step_reports_a_real_exit_status,
    "guard_tests": check_the_warm_step_does_not_run_the_guard_s_own_tests,
    "advisory": check_the_gate_is_advisory_because_it_reads_third_parties,
    "clean_tree": check_the_gate_does_not_assume_a_clean_tree,
}


# ---------------------------------------------------------------- the tests


def test_the_workflow_file_exists():
    assert WORKFLOW.exists(), f"{WORKFLOW} is missing; a commit is ungated"
    assert len(workflow_text()) > 400, "the workflow looks like a stub"


def test_each_check_can_fail():
    """A guard that cannot fail is not a guard.

    Each check runs against the real file, and against a copy with one thing
    broken. A broken copy that passes means the check is vacuous.
    """
    good = workflow_text()
    for check in CHECKS.values():
        check(good)
    check_the_cache_key_is_not_shared_with_the_snapshot_job(
        good, SNAPSHOT_WORKFLOW.read_text()
    )

    broken = {
        # Unclosed flow sequence at the top level: a genuine parse error, so it
        # exercises `parses`.
        "parses": ("name: tests\n", "name: [tests\n"),
        # Back to the values that shipped and made runs unsurvivable: the warm
        # job at 60 against a 4926s cold run, and the gate at 20 against a
        # 1336s gate. Either one reintroduces its own deadlock.
        "warm_timeout": ("timeout-minutes: 150", "timeout-minutes: 60"),
        "gate_timeout": ("timeout-minutes: 45", "timeout-minutes: 20"),
        # Same 30 inflated to the warm budget: the jobs stay split but the
        # budgets merge, so the gate no longer says it is fast by design.
        "job_budgets": ("timeout-minutes: 45", "timeout-minutes: 150"),
        # The pipe this file shipped, whose exit status is `tail`'s.
        "exit_status": ("run: >-\n          pytest tests/", "run: pytest tests/ -q | tail -20\n          pytest tests/"),
        # The exact violation the check exists for: the *gating* step allowed
        # to fetch, so a test that reaches the network passes the gate.
        "separate_jobs": (
            '          PYTHONPATH: ${{ github.workspace }}/src\n'
            '        run: pytest tests/ -q -m "not network"',
            '          PYTHONPATH: ${{ github.workspace }}/src\n'
            '          PL_ALLOW_NETWORK: "1"\n'
            '        run: pytest tests/ -q -m "not network"',
        ),
        "clean_tree": ("ruff check src tests", "git diff --exit-code\n          ruff check src tests"),
        # The pre-fix warm step: ran the guard's own tests with the block off.
        "guard_tests": (
            "--ignore=tests/test_network_guard.py",
            "--ignore=tests/test_walk_forward.py",
        ),
    }
    for name, (old, new) in broken.items():
        assert good.count(old) >= 1, f"the {name!r} mutation anchor {old!r} is absent"
        try:
            CHECKS[name](good.replace(old, new))
        except AssertionError:
            continue
        raise AssertionError(f"the {name!r} check passed a workflow with {old!r} broken")


def test_the_timeouts_clear_the_measured_costs():
    check_the_timeouts_clear_the_measured_costs(workflow_text())


def test_the_cold_run_floor_is_the_measured_number_not_a_round_one():
    """The floor is 4926.07s, not 60 minutes.

    Pinned because the failure was a value that felt safe by comparison with
    the one before it. If someone "tidies" `COLD_RUN_SECONDS` to 3600 the guard
    silently stops guarding.
    """
    assert COLD_RUN_SECONDS == 4926.07, (
        f"COLD_RUN_SECONDS is {COLD_RUN_SECONDS}, not the measured 4926.07. A round "
        "number here would re-create the exact bug: a timeout below the cold cost "
        "that looks generous because it is bigger than the previous value."
    )


def test_the_gate_floor_is_the_measured_number_not_a_round_one():
    """The floor is 1336s, not 20 minutes.

    Pinned for the same reason as the cold floor, and this one is the more
    likely regression: 20 minutes *feels* like a generous gate budget, and it
    is 136 seconds short of what the gate measured on the CI runner. If someone
    "tightens" `GATE_RUN_SECONDS` to 1200 the guard silently stops guarding the
    leg it exists for.
    """
    assert GATE_RUN_SECONDS == 1336, (
        f"GATE_RUN_SECONDS is {GATE_RUN_SECONDS}, not the measured 1336. A round "
        "number here would re-create the exact bug in miniature: a gate timeout "
        "below the gate cost that looks tight-but-safe because it is smaller "
        "than the warm budget."
    )


def test_warming_and_gating_are_separate_jobs():
    check_warming_and_gating_are_separate_jobs(workflow_text())


def test_the_warm_step_reports_a_real_exit_status():
    check_the_warm_step_reports_a_real_exit_status(workflow_text())


def test_the_warm_step_does_not_run_the_guards_own_tests():
    check_the_warm_step_does_not_run_the_guard_s_own_tests(workflow_text())


def test_the_gate_is_advisory_because_it_reads_third_parties():
    check_the_gate_is_advisory_because_it_reads_third_parties(workflow_text())


def test_the_gate_does_not_assume_a_clean_tree():
    check_the_gate_does_not_assume_a_clean_tree(workflow_text())


def test_the_cache_key_is_not_shared_with_the_snapshot_job():
    check_the_cache_key_is_not_shared_with_the_snapshot_job(
        workflow_text(), SNAPSHOT_WORKFLOW.read_text()
    )


def test_readme_points_at_the_test_gate():
    check_readme_names_the_test_gate(readme_text())


def test_the_advisory_check_can_fail():
    """Broken both ways, because it is a two-part condition.

    A single-token mutation cannot break it, which is exactly why it has its
    own proof rather than living in the table above.
    """
    good = workflow_text()
    check_the_gate_is_advisory_because_it_reads_third_parties(good)

    no_advisory_step = re.sub(
        r"^\s*continue-on-error: true\n", "", good, flags=re.M
    )
    try:
        check_the_gate_is_advisory_because_it_reads_third_parties(no_advisory_step)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "the advisory check passed a workflow with no `continue-on-error` step"
        )

    claims_required = good + "\n# required: true\n"
    try:
        check_the_gate_is_advisory_because_it_reads_third_parties(claims_required)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "the advisory check passed a workflow that claims to be required"
        )


def test_the_readme_check_can_fail():
    """The README guard, broken the two ways it could plausibly rot.

    A README that stops naming the gate, and one that names it while dropping
    the command that settles whether it is actually required. The second is
    the interesting one: it is the shape the deploy section had -- a correct
    table and a false claim in the note below it.
    """
    real = readme_text()
    check_readme_names_the_test_gate(real)

    without_gate = re.sub(r"tests\.yml", "the CI job", real)
    try:
        check_readme_names_the_test_gate(without_gate)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "the README check passed a README that never names tests.yml"
        )

    without_command = re.sub(r"gh api[^\n]*", "", real)
    try:
        check_readme_names_the_test_gate(without_command)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "the README check passed a README that states the gate's required-ness "
            "without the command that settles it"
        )
