"""The snapshot workflow must persist `data/tracking.db`, and the two paths must not drift.

The track record was dishonest for a reason that had nothing to do with the
scoring code: the scheduled workflow restored `data/cache` but not the tracking
database, so every run started from an empty store. A pre-kickoff pick is
written once and never rewritten, so "empty every run" means no in-time pick
was ever kept — which is exactly what the live payload showed (n_rebuilt 50 of
50).

This is a workflow assertion, which is unusual and is here because the failure
is invisible in every other test: the scoring code is correct, the tests pass,
and the number on the page is still wrong. `config.py` resolving the database to
`data/tracking.db` while the workflow caches `data/cache` is a two-line
disagreement between a Python constant and a YAML string, and nothing else in
the repo would notice it.
"""
import re
from pathlib import Path

import pytest

import pl_predictor.config as config

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "refresh-public-snapshot.yml"


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text()


def _cached_paths(text: str) -> list[str]:
    return re.findall(r"^\s*path:\s*(\S+)\s*$", text, re.M)


def test_the_tracking_database_path_is_the_one_config_resolves():
    """The premise of the whole file.

    Written as a test rather than a comment because the two can drift: a rename
    in `config.py` would leave the workflow caching a path nothing writes, and
    the symptom would be the same silent empty store.
    """
    resolved = Path(config.TRACKING_DB_PATH)
    assert resolved.name == "tracking.db"
    assert resolved.parent.name == "data", (
        f"TRACKING_DB_PATH moved to {resolved}; the workflow's cached path must follow it"
    )


def test_the_workflow_caches_the_tracking_database(text):
    cached = _cached_paths(text)
    assert "data/tracking.db" in cached, (
        f"the workflow caches {cached} and not data/tracking.db. Every scheduled run then "
        f"starts with an empty tracking store, so no pre-kickoff pick is ever kept and the "
        f"headline is computed entirely from rebuilt picks."
    )


def test_the_tracking_cache_key_is_not_scoped_to_one_run(text):
    """A per-run key on a file that must persist is the same bug one level down.

    `data/cache` is keyed `data-cache-${{ github.run_id }}` on purpose — its
    contents are disposable and TTL'd. The tracking database is the accumulating
    record; keyed the same way, every run would restore only a cache that run
    saved, which is always empty on restore.
    """
    block = re.search(
        r"name: Restore tracking database(.*?)(?=\n      - name: |\Z)", text, re.S
    )
    assert block, "no 'Restore tracking database' step found"
    step = block.group(1)
    assert "restore-keys" in step, (
        "the tracking database is restored with no restore-keys, so it only ever matches "
        "its own run's cache — which does not exist yet on a cold run"
    )
    assert "tracking-db-" in step, "the tracking cache key is not distinguishable from the data cache's"


def test_the_data_cache_step_is_untouched(text):
    """The speed fix must survive the correctness fix.

    Restoring `data/cache` is what took the build from ~11 minutes to ~5. It is
    a separate step with a separate key, and this asserts the two have not been
    merged into one — which would restore the tracking database on the
    disposable cache's per-run key and reintroduce the original failure.
    """
    assert "path: data/cache" in text
    assert text.count("path: data/tracking.db") == 1
