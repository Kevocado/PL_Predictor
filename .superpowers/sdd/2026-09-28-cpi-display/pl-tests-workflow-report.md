# PL: the test gate — was it missing, and should it be required

**Date:** 2026-09-28
**PR:** [#14](https://github.com/Kevocado/PL_Predictor/pull/14) — **open, not merged**

---

## 1. The headline: it was not missing, and it was already red

**`tests.yml` is in `origin/main`, and the content is recoverable verbatim. It
was not absent when this task was written, and by the time this work finished it
had already been run, and it had already failed.**

The brief said the workflow was "registered on GitHub but absent from
`origin/main`". That was true of the `origin/main` the hang report was written
against, and false by the time I looked. PR #7 (`4975f22`, merged 10:50 today)
added the file. Verified three ways:

```
$ gh api repos/Kevocado/PL_Predictor/actions/workflows
{"id": 368696374, "name": "tests", "path": ".github/workflows/tests.yml", "state": "active"}

$ git ls-tree origin/main .github/workflows/
100644 blob 288f8145…  .github/workflows/tests.yml      # the c22a310 version, as of 4975f22

$ sha256(gh api …/git/blobs/288f8145… | base64 -d) == sha256(the file in origin/main)
290aa66dbc5bac1df0eea8abe88a062b2541fb7e8154bb3811f5dba3bb3550b2  (both)
```

So: **verbatim, byte-for-byte, not reconstructed.** Nothing needed guessing.

Two consequences worth stating because they change what the task was:

- **The prior agent's "unmerged" commit `c22a310` was in fact merged**, via PR
  #7. Its blob is `288f8145…`, which is exactly what `origin/main` carried. The
  brief's instruction to read it for a better approach was sound advice that had
  already been acted on — its 60-minute timeout and its date-stamped cache key
  are both in the shipped file.
- **The gate was already enabled, and already red on arrival.** It is not a
  hypothetical about whether a restored workflow would pass:

```
run 36446591387   push main   4975f22c   2026-09-28T15:50:10Z
  pytest    failure
  frontend  success
  20 failed, 492 passed, 27 skipped, 1 deselected in 231.64s
```

That is precisely the "a check that lands red teaches everyone to ignore red"
outcome the brief warned about, and it had already happened.

**And `origin/main` moved again underneath me mid-task.** By the time my work
was rebased, `main` was at `6dcdb95` and `tests.yml` was a *different* file
(blob `44e50f8`), which had gained a "Warm the upstream cache" step. So the
deliverable is not "restore the workflow" but "the workflow now exists and has
three live defects, one of them fatal" — below.

---

## 2. Required, or advisory? — **advisory, and it already is**

`main` has **no branch protection and no rulesets**:

```
$ gh api repos/Kevocado/PL_Predictor/branches/main/protection
{"message": "Branch not protected", "status": "404"}
$ gh api repos/Kevocado/PL_Predictor/rulesets
[]
```

So `tests` is advisory in fact as well as in intent, whatever the YAML says. I
did **not** add branch protection. That is a deliberate act of governance on a
repo where a merge to `main` is a production deploy, and it is not mine to make.

The argument for keeping it advisory even once it is green:

1. **It reads three third parties.** FPL, ClubElo, football-data.co.uk. The
   gating step runs with the network *blocked* against a cache that a prior step
   fetched, so an upstream outage lands on the warm step and the gate has
   nothing to read. A required check that intermittently goes red for reasons
   outside the diff is a check people learn to re-run rather than read.
2. **The warm cost is 43 minutes and it is not always paid.** With the fix in
   this PR the warm step is skipped whenever a cache was restored, but the
   first run on any new branch ref is cold, because Actions caches are
   branch-scoped. A required check with a 43-minute cold path and a 15-minute
   warm path is a check people route around.
3. **It writes to tracked files.** The suite rewrites four tracked
   `models/*.json` on every run. That is not a reason not to gate, but it is a
   reason the tree is not a clean-room input, and it is the kind of thing that
   makes a future "just add `git diff --exit-code`" step red for no reason.

The honest comparison the brief asked for — *a suite whose failures depend on
cache state, run in CI, versus no check at all* — resolves the same way it did
for the prior agent, but for a narrower reason. It is no longer "the suite
hangs". It is "the suite is fine, but the gate is only as good as a cache and
three upstreams". That is worth having, provided it is not load-bearing for a
merge.

---

## 3. The three live defects, all now measured

### a. `timeout-minutes: 60` was **below** the cold cost. This is fatal.

The shipped comment explained the value: *"a warm run is ~13 minutes, the first
run is cold and does not finish in 30, so 30 was too tight and 60 is
generous."* The cold duration had never been measured. I measured it:

```
$ PL_ALLOW_NETWORK=1 PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
3 failed, 510 passed, 27 skipped, 8555 warnings in 4926.07s (1:22:06)
```

`4926.07s` is **82 minutes**. `timeout-minutes: 60` is 3600s. So on a cold
runner the job is killed at 60 minutes, *before* `actions/cache`'s post step
runs — so it saves no cache, so the next run is cold too, so it times out
again. The `c22a310` deadlock, still live, with a value that read as safe
because it was larger than the previous one.

This is also the number `c22a310` said it could not get: *"the cold duration is
not known precisely because it was killed at 30."* It is 4926.07s.

### b. The warm step's exit status was `tail`'s

```yaml
run: pytest tests/ -q 2>&1 | tail -20
continue-on-error: true
```

A pipeline's exit status is the last command's. `tail` always succeeds, so the
step was **always** reported as passing and the `continue-on-error: true` on the
line above it was unreachable. A cold-cache failure was reported as a step that
"succeeded".

### c. The warm step ran the guard's own tests with the guard disabled

Those three tests assert the network block is in place, so they fail when
`PL_ALLOW_NETWORK=1` disables it — measured, `3 failed`, all in
`test_network_guard.py`. They were being run anyway, and `tail` hid it.

---

## 4. What this PR changes

- **`timeout-minutes: 60 → 90`**, with the measured numbers in the comment
  instead of the guess.
- **The warm step stops piping to `tail`**, so it has a real exit status and
  `continue-on-error` does what its comment says.
- **The warm step excludes `test_network_guard.py`** and selects
  `-m "not network"`. Both are correctness, not tidiness: the guard's own tests
  cannot pass with the guard off, and the one `network`-marked test is 70 of
  the 82 minutes above for a test the gate does not run.
- **The warm step is skipped when a cache was restored** (`id: cache` +
  `cache-matched-key`). This is the answer to "a CI runner is a cold clone" —
  it is a cold clone on its *first* run, not on every one.
- **`test_evaluate_count_market_arms_on_current_season_flags_low_power` is
  marked `@pytest.mark.network`.** It is the only test that needs the per-match
  understat archive, and marking it is what makes `-m "not network"` select
  something: it selected 0 of 540 before, so the split the workflow described
  was decorative. It now selects 2 of 567.
- **A nightly `schedule`**, so the `-m network` half is not verified only when a
  human remembers to dispatch it.
- **`tests/test_tests_workflow.py` — 13 tests, and the standing finding closed.**
  See below.
- **A README section**, which the README did not have.

---

## 5. The blind spot, and the guard for it

The standing finding was that *every* guard in `test_deploy_workflow.py` is
pinned against `deploy.yml` and none could see the README. `tests.yml` had the
same blind spot and one step further: **no test read `tests.yml` at all**, and
the README mentioned it in **0 places**, while a sibling section went to some
length about which deploy workflow is authoritative.

`tests/test_tests_workflow.py` closes both directions. It pins *properties*,
not the current shape of the file, and three of them are calibrated against a
measured number rather than a string — because the actual bug was a value that
had never been measured.

### Mutation proof — against the real `origin/main` workflow and README

Not a synthetic mutation. The real artifacts, as they were:

```
Against the REAL origin/main tests.yml (the one that went red on run 36446591387):
  passes   [parses]
  CAUGHT   [timeout]    timeout-minutes is 60, but a cold run costs 4926.07s (~83 minutes)
  passes   [separate_steps]
  CAUGHT   [exit_status] step 'Warm the upstream cache' pipes pytest into another command
  CAUGHT   [guard_tests]  the warm step does not exclude tests/test_network_guard.py
  passes   [advisory]
  passes   [clean_tree]
  CAUGHT   [readme]      the README never mentions tests.yml
```

Four of the four defects in §3, plus the README blind spot.

`separate_steps`, `advisory`, `clean_tree` and `cache_key` *pass* on the
historical file. That is correct, not a gap: they are forward-looking invariants
for failure modes that have not happened yet. Claiming otherwise would be
padding.

### A guard that fires on the thing it guards

Writing the README section broke `test_deploy_workflow.py`. That guard mutates
`gh run list` in the README to prove its own check can fail, and asserts the
string appears **exactly once**; my section used the same command and the
deploy guard failed with *"the 'checkable' mutation 'gh run list' appears 2
times, so it may not be breaking the thing it claims to"*. Two rounds, because
my explanatory note about the collision contained the string it was explaining.

That is the deploy guard doing exactly its job, on a different file, and it is
the best evidence I have that the guard is load-bearing rather than decorative.
The README section now uses `gh api`, and says why.

---

## 6. The measurements

Interpreter confirmed to exist before any of these were believed:
`.venv/bin/python -> 3.13.0`, `python -c "print('interpreter OK')"`.

| | wall | result |
|---|---|---|
| warm cache, network blocked | **669.66s** (11m09s) | 513 passed, 27 skipped |
| cold cache, network blocked | 148.21s (2m28s) | 20 failed, 493 passed |
| cold cache, network allowed | 4926.07s (1h22m) | 3 failed*, 510 passed, 4,723 files / 87MB |
| cold, network allowed, gate half only | 2594.15s (43m14s) | 3 failed*, 508 passed, 2 deselected |
| warm minus understat_shots | 1269.45s (21m09s) | **1 failed**, 512 passed |
| **final gating step, current `main`** | **925.26s** (15m25s) | **538 passed, 27 skipped, 2 deselected** |

\* all three in `test_network_guard.py`, because `PL_ALLOW_NETWORK=1` disables
the guard they assert. That is the guard working.

**Two caveats, both real:**

1. **This machine was not idle.** Load average ran 5–12 throughout, with other
   agents' `pytest` processes on the same box. The 669.66s warm figure was taken
   at load ~5.7; the final 925.26s at load ~10.2. The gap is contention, not
   regression — the two runs are the same command on the same tree, and the
   intervening change added 13 tests. **Quote 669.66s as the warm number and
   treat it as a floor, not a promise.**
2. **The cold numbers are from the 540-test tree at `4975f22`.** The suite is
   567 tests as of `6dcdb95` (PR #13). The 20 cache-dependent failures are the
   same 20, but the passed/failed counts in those rows are against 540. I did
   not re-run a 43-minute cold measurement against the current tree; the README
   and the workflow comments both say which commit each number came from.

The final row is the one that matters operationally, and it is against the
current `main`: **green.**

---

## 7. Flagged, not fixed

- **The suite dirties tracked `models/*.json`** — four of them, on every run.
  Confirmed and left alone. It is flagged in the README and pinned by
  `check_the_gate_does_not_assume_a_clean_tree` so that whoever adds a
  `git diff --exit-code` step reads the warning first.
- **19 tests still need a warm cache.** Fixing them properly means patching
  their loaders with `monkeypatch`, which is the guard's own stated doctrine and
  is a much larger change than a CI fix. Until then they are cache-dependent,
  and that is the honest reason the gate is advisory.
- **`ARM_SPECS`' 12-season arm** is why one test needs 4,649 files. Narrowing it
  to 11 would change what the arm asserts. Not a CI decision.

---

## 8. Constraints honoured

- **`.env` was never read.** No `cat`, `grep` or read touched it.
- **The stale local checkout is byte-for-byte unchanged.** All work was in a
  worktree at `/private/var/folders/…/opencode/pl-tests` off `origin/main`.
  `git worktree list` in the original tree is unchanged; nothing was written
  under `PL_Predictor/` except via `git worktree` metadata.
- **Never merged.** One PR, opened, not merged — a merge to `main` here is a
  production deploy, and the branch protection question is deliberately left
  open.
- **No `rg`** (not installed; used `grep`). No browser. `frontend/` untouched —
  its job in the same workflow passed on every run and is not my business.
