"""Site-local copy must not tell a reader that a counted pick was "not counted".

predictor-hub#67 removes "not counted" from the SHARED package (`predictor-ui`,
which PL vendors under `frontend/src/predictor-ui/` and must not edit by hand —
`predictor-ui.sync.test.ts` pins those file hashes). What it cannot reach is
SITE-LOCAL copy, and that is this file's whole subject.

Swept from Python rather than from vitest on purpose: the strings live in `.tsx`
files spread across `frontend/src`, and the rule is a repo-wide ban on two
phrases rather than a behaviour of one component. A vitest test would have to
enumerate the components to sweep; this walks the tree.

The rendered-panel half of the same rule — a pick made after the start must
never be *presented* as one made before it — lives in
`frontend/src/components/TrackRecordPanel.countedPicks.test.tsx`, because that
is a statement about what a reader sees and only a render can check it.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "frontend" / "src"

# Files that arrive wholesale from predictor-hub on the next re-vendor. Listed
# so the sweep can exclude them without passing vacuously: the assertions
# below prove they are still full of the old wording, so the exclusion is real.
VENDORED_PARTS = {"predictor-ui"}

FORBIDDEN = [
    (re.compile(r"not\s+counted", re.I), '"not counted"'),
    (re.compile(r"Rebuilt after", re.I), '"Rebuilt after ..."'),
]


def _text_files(*, include_tests: bool):
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in {".ts", ".tsx"} or path.name.endswith(".d.ts"):
            continue
        is_test = ".test." in path.name or ".spec." in path.name
        if is_test != include_tests:
            continue
        if any(part in VENDORED_PARTS for part in path.parts):
            continue
        yield path


_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(r"^\s*(//|\*)")
#: A line that ASSERTS THE ABSENCE of the wording is this rule's own guard, not
#: a violation of it — `expect(...).not.toMatch(/not counted/i)` has to name the
#: words to forbid them. So these lines are excluded from the sweep, which is
#: what lets a test pin the ban without tripping it.
_NEGATIVE_ASSERTION = re.compile(
    r"\.not\.|queryBy|queryAllBy|toBeNull\(\)|not\.toHaveTextContent"
)


def _scannable(line: str) -> bool:
    return not (_COMMENT_LINE.match(line) or _NEGATIVE_ASSERTION.search(line))


def _hits(pattern):
    out = []
    for path in _text_files(include_tests=False):
        # Comments document the rule and quote the old words to say what changed;
        # they are not copy a reader sees. Strips both `/* */` blocks and `//`.
        body = _COMMENT_BLOCK.sub("", path.read_text())
        for number, line in enumerate(body.splitlines(), start=1):
            if pattern.search(line) and _scannable(line):
                out.append(f"{path.relative_to(SRC)}:{number}: {line.strip()[:120]}")
    return out


@pytest.mark.parametrize("pattern,words", FORBIDDEN, ids=[w for _, w in FORBIDDEN])
def test_no_site_local_copy_holds_the_old_wording(pattern, words):
    """Every hit is listed with its file and line, so a reviewer can see what
    changed rather than trusting a summary."""
    hits = _hits(pattern)
    assert not hits, (
        f"site-local copy still contains {words}, in {len(hits)} place(s):\n  "
        + "\n  ".join(hits)
        + "\n\nA pick made after kickoff is COUNTED now (spec "
          "2026-10-01-track-record-counts-every-pick). predictor-hub#67 fixes the shared "
          "package; these are the strings it cannot reach."
    )


def test_no_site_local_test_asserts_the_old_wording():
    """A test asserting the old wording pins a rule this reversal removed, and
    it would also fail the moment the re-vendor lands — from the wrong side.

    Excludes negative assertions, which have to name the words in order to forbid
    them, and comments, which quote them to document the change.
    """
    offenders = []
    for path in _text_files(include_tests=True):
        body = _COMMENT_BLOCK.sub("", path.read_text())
        for number, line in enumerate(body.splitlines(), start=1):
            if not _scannable(line):
                continue
            for pattern, words in FORBIDDEN:
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(SRC)}:{number} ({words})")
    assert not offenders, (
        "site-local tests still assert copy this reversal removed:\n  "
        + "\n  ".join(offenders)
    )


def test_the_vendored_package_is_the_only_place_the_old_words_still_live():
    """So the sweep above is not passing because it cannot see the vendored copy.

    predictor-hub#67 is open and unmerged, so `frontend/src/predictor-ui/`
    legitimately still carries "not counted" today. This asserts that is TRUE and
    that it is confined there — after the re-vendor lands, whatever remains is
    site-local by definition and must be fixed by hand.
    """
    vendored_dir = SRC / "predictor-ui"
    assert vendored_dir.is_dir(), "the vendored package moved; update VENDORED_PARTS and this test"

    still = []
    for path in sorted(vendored_dir.rglob("*.ts*")):
        if ".test." in path.name:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for pattern, _ in FORBIDDEN:
                if pattern.search(line):
                    still.append(f"{path.relative_to(SRC)}:{number}")
    assert still, (
        "the vendored package no longer holds the old words, so the re-vendor has landed. Drop "
        "the site-local copies for good and update this test to assert that tree is clean."
    )


def test_the_vendored_tree_is_not_edited_by_hand():
    """Editing a vendored file is silently undone by the next re-vendor, and the
    edit is invisible until then.

    `predictor-ui.sync.test.ts` pins the content hashes; this is the Python-side
    statement of the same rule, and it names the reason: a site-local fix must
    live in a site-local file.
    """
    sync = SRC / "predictor-ui" / "SYNC.json"
    assert sync.is_file(), "SYNC.json is gone; the vendoring contract needs re-establishing"
    assert not re.search(
        r"made_before_kickoff|pre_kickoff", (SRC / "predictor-ui" / "components" / "InstantBlock.tsx").read_text()
    ), "a vendored file was edited by hand. Change predictor-hub and re-run scripts/sync-ui.mjs."
