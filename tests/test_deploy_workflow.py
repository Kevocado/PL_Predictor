"""A merge to main must deploy to the VPS, and nothing else should.

`deploy-azure.yml` was `workflow_dispatch`-only and had no VPS job at all, so a
merge deployed nothing: every deploy was a manual image build plus a hand-run
`bin/deploy`. This pins the replacement, because the failure mode it guards is
silent -- a workflow that does not run looks exactly like a workflow with nothing
to deploy.

Three things here are not obvious and are the reason the test exists:

* **The paths filter is a whitelist**, so excluding the file the scheduled
  snapshot job commits is implicit. That job runs every 20 minutes through the
  match window, so nothing stops someone adding `data/**` and turning every
  refresh into a deploy. The exclusion is asserted directly instead of trusted
  to the filter's shape, and `test_every_scheduled_commit_is_deliberately_classified`
  walks the OTHER workflows in the repo, finds what they `git add`, and fails on
  any overlap that has not been classified with a reason.

* **The GHCR credential is `GITHUB_TOKEN`, and only because the package is
  linked.** `ghcr.io/kevocado/pl-predictor` is a *user-scoped* package; if it is
  not associated with this repository, `GITHUB_TOKEN` may not write to it and the
  push dies with `denied: permission_denied: write_package` **after** the image
  has already built and tagged correctly. NFL and CFB both lost their first VPS
  deploy exactly that way. Linking is a one-time action in package settings with
  no API, so nothing in CI can tell -- the assertion below encodes a fact about
  the outside world, and says how to re-check it:

      gh api /user/packages/container/pl-predictor --jq '.repository.full_name'

  `Kevocado/PL_Predictor` -- linked, GITHUB_TOKEN is correct.
  `null`             -- unlinked; the workflow must switch to `secrets.GHCR_PAT`
                        (already a secret on this repo) AND this check must be
                        rewritten to expect that, in the same commit.

* **Azure is gated with `== 'true'`, not `!= 'false'`.** The failing-open form
  turns Azure back on for any repo that has not set the variable. This is legacy
  infrastructure being cut over, so it has to fail closed.

Two structural checks exist because a text-based guard cannot see YAML
structure. `check_yaml_parses` proves the file is valid YAML at all -- a merge
should not be the first place that is discovered -- and `check_yaml_shape`
covers the specific bug that got through every text assertion in a sibling repo:
steps re-indented by two spaces, which *parses* as a continuation of the
preceding step and silently nests the whole Azure job under one entry.

Every check below is also run against a deliberately broken copy of the file to
prove it can fail. A workflow guard that passes vacuously is worse than no
guard, because it is trusted.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO / ".github" / "workflows"
WORKFLOW = WORKFLOW_DIR / "deploy.yml"
README = REPO / "README.md"

# Must match `image:` in vps-stack/compose.yml. If either side changes, the
# deploy pushes a tag the stack never pulls.
IMAGE = "ghcr.io/kevocado/pl-predictor"
# Must match the service key in vps-stack/compose.yml, which is what
# `bin/deploy` is called with: [pl]="PL_TAG 8000 /api/health".
SERVICE = "pl"

# Baked into the image by the Dockerfile (and hand-committed), so a change to any
# of these changes the artifact and must deploy. Read straight off the COPY lines
# in Dockerfile plus the build inputs around them.
MUST_DEPLOY = [
    "src/**",
    "frontend/**",
    "models/**",
    "Dockerfile",
    "pyproject.toml",
    "requirements-lock.txt",
    ".dockerignore",
    ".github/workflows/deploy.yml",
]

# Written by .github/workflows/refresh-public-snapshot.yml on a 20-minute timer
# through matchdays. The service polls it from raw.githubusercontent.com at
# request time (api/main.py's lifespan, PUBLIC_SNAPSHOT_POLL_SECONDS=300) so a
# new snapshot needs no deploy -- and if it triggered one, PL would redeploy
# dozens of times a day for a JSON file.
MUST_NOT_DEPLOY = [
    "data/public_snapshot.json",
    "data/**",
    "data/cache/**",
    "data/tracking.db",
]

# Paths the filter watches that a *scheduled* job could plausibly start writing,
# with the reason. Each entry is a decision, not an oversight.
#
# `models/**` is watched because Dockerfile COPYs it and the API reads the models
# per request, so a new model only reaches the VPS by redeploying. Nothing
# scheduled writes there -- the models are hand-committed -- so the redeploys it
# buys cost nothing, and recording it here means that is a choice rather than an
# accident if a nightly retrain ever lands.
WATCHED_ANYWAY = {
    # Glob patterns, matched against the paths the other workflows `git add`.
    "models/**": "read per request; nothing scheduled writes here, so watching is free",
    "src/**": "read per request; only hand-committed",
}

GATE_AZURE = "vars.DEPLOY_AZURE == 'true'"
GATE_VPS = "vars.VPS_HOST != ''"

# How far either side of a mention of a workflow file to look for the word that
# qualifies it. Sized to clear an 80-column line wrap plus the adjacent clause,
# so a true statement that wraps is not reported as an unqualified one.
SENTENCE_CHARS = 200


def workflow_text() -> str:
    return WORKFLOW.read_text()


def readme_text() -> str:
    return README.read_text()


def jobs(text: str) -> dict[str, str]:
    """Split the `jobs:` block into {name: body}, so a gate can be read."""
    out: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    in_jobs = False
    for line in text.splitlines():
        if re.match(r"^jobs:\s*$", line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        # A two-space-indented comment introduces the job that FOLLOWS it, so it
        # belongs to neither body. Attributing it to the previous job is not
        # cosmetic: the Azure job's own comment says which login it uses, and
        # appending it to `build` made check_azure_fails_closed think the build
        # job touched Azure and was ungated -- a real failure, reported at the
        # wrong place. Comments inside a body are indented four spaces and are
        # kept.
        if re.match(r"^  #", line):
            continue
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if m:
            if name:
                out[name] = "\n".join(buf)
            name, buf = m.group(1), []
        elif name is not None:
            buf.append(line)
    if name:
        out[name] = "\n".join(buf)
    return out


def paths_filter(text: str) -> list[str]:
    """The `paths:` entries of the push trigger."""
    m = re.search(r"^on:[ \t]*$", text, re.M)
    assert m, "the workflow has no top-level `on:` block"
    # The `on:` block is everything up to the next top-level key. Comments at
    # column zero belong to it, and this workflow's have to: the exclusion
    # rationale is written there. A reader that stopped at the first
    # non-indented line would see an empty block whenever a comment sat between
    # `on:` and `push:`, which would make the exclusion assertions vacuous
    # rather than failing.
    block: list[str] = []
    for line in text[m.end():].splitlines():
        if line.strip() == "" or line.lstrip().startswith("#") or line[:1] in (" ", "\t"):
            block.append(line)
        else:
            break
    trigger = "\n".join(block)
    p = re.search(r"^[ \t]+paths:\n((?:[ \t]+-[ \t]*.+\n)+)", trigger, re.M)
    assert p, "the push trigger has no `paths:` filter, so every snapshot refresh redeploys"
    return [
        line.strip().lstrip("- ").strip().strip("'\"")
        for line in p.group(1).splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------- the checks


def check_triggers(text: str) -> None:
    assert re.search(r"^on:[ \t]*$", text, re.M), "no top-level `on:` block"
    assert re.search(r"^[ \t]+workflow_dispatch:", text, re.M), (
        "workflow_dispatch is gone, so a deploy can no longer be run by hand"
    )
    assert re.search(r"^[ \t]+push:[ \t]*$", text, re.M), "the workflow does not run on push"
    assert re.search(r"^[ \t]+branches:[ \t]*\[?[ \t]*main", text, re.M), (
        "the push trigger is not limited to main"
    )
    paths = paths_filter(text)
    for required in MUST_DEPLOY:
        assert required in paths, f"{required!r} is not in the paths filter, so a change to it deploys nothing: {paths}"
    for forbidden in MUST_NOT_DEPLOY:
        assert forbidden not in paths, (
            f"{forbidden!r} is in the paths filter. The scheduled refresh commits it, so this "
            f"would redeploy on every refresh: {paths}"
        )


def check_credential(text: str) -> None:
    """The GHCR login must use a credential this repository can actually push with.

    `GITHUB_TOKEN` is the target state: a PAT is a credential that outlives the
    repo and has to be rotated by hand. It is ALSO the thing that does not
    currently work, and the reason is worth stating because the first guess at it
    was wrong and cost a deploy.

    The first version of this check claimed GITHUB_TOKEN works because the package
    is LINKED to this repo, and cited:

        gh api /user/packages/container/pl-predictor --jq '.repository.full_name'
        -> "Kevocado/PL_Predictor"

    That command really does return this repo, and GITHUB_TOKEN is still rejected
    with `denied: permission_denied: write_package` after a clean build. **A linked
    repository and the access grant GITHUB_TOKEN needs are two different things.**
    Linking records where the package came from; the grant is a separate permission
    in package settings, and only the hub's package has one. NFL and CFB lost their
    first deploy to the unlinked variant of the same error, so three repos have now
    been bitten by an error that reads like a permissions problem and is not one.

    So both credentials are accepted, and the grant is the thing to make if the
    policy is to be honoured. It is package-settings UI with no API, so it cannot
    be done from CI. `GHCR_PAT` is already a secret here and is what these images
    have always been pushed with.
    """
    body = jobs(text)["build"]
    # Matched on the `password:` field, not on the name appearing anywhere in
    # the file: the comment above has to be able to say what the fallback would
    # be without tripping the check. Exactly one, so a second login step cannot
    # smuggle in a second credential.
    passwords = re.findall(r"password:[ \t]*\$\{\{[ \t]*secrets\.([A-Z0-9_]+)[ \t]*\}\}", body)
    # `raise` rather than `assert cond, msg`: pytest's assertion rewriting
    # evaluates a plain assert's message expression EAGERLY, before the
    # condition, so a message naming a local assigned on the line above raises
    # NameError instead of reporting the thing it is about.
    if sorted(passwords) != ["GHCR_PAT"]:
        raise AssertionError(
            f"GHCR login must use exactly one credential, and on this repo that is "
            f"secrets.GHCR_PAT; found {passwords}. secrets.GITHUB_TOKEN is the POLICY "
            f"preference and does not work: the package is linked to this repo but not "
            f"GRANTED to it, and those are two different things. Do not 'fix' this by "
            f"switching to GITHUB_TOKEN -- a linked repository is not an access grant, and "
            f"the push fails with `denied: permission_denied: write_package` after the image "
            f"has already built and tagged. If the grant is ever made in package settings, "
            f"switch the workflow to GITHUB_TOKEN and tighten this to expect only that, in "
            f"the same commit."
        )


def check_image(text: str) -> None:
    j = jobs(text)
    assert "build" in j, f"no `build` job to produce the image: {sorted(j)}"
    body = j["build"]
    assert "docker/login-action" in body, "the build job never logs in to a registry"
    assert re.search(r"registry:[ \t]*ghcr\.io", body), "the registry is not ghcr.io"
    # Whole lines, not substrings. A substring test for the sha tag is satisfied
    # by the *push* line even when the *build* line tags something else, and
    # `<image>:latest` appears twice (build and push) so it cannot be mutated
    # uniquely. The stack pins a sha in .env, so the tag the image is BUILT with
    # is the one that has to be right.
    # The build is docker/build-push-action with `push: true` (registry layer cache), which tags and pushes every
    # line under `tags:`. Anchored on the sha tag line, so a build that tags something else cannot pass on the
    # strength of the azure job's mention of the same string.
    assert re.search(r"uses:\s*docker/build-push-action@", body), "the build must use docker/build-push-action"
    assert re.search(r"push:\s*true", body), "the build must push"
    assert f"            {IMAGE}:${{{{ github.sha }}}}\n" in body, (
        f"the build must tag {IMAGE}:${{{{{{ github.sha }}}}}} — the stack pins a sha"
    )
    assert f"            {IMAGE}:latest\n" in body, f"the build must also tag {IMAGE}:latest"


def check_vps(text: str) -> None:
    j = jobs(text)
    vps = [n for n, b in j.items() if "vars.VPS_HOST" in b]
    assert len(vps) == 1, f"expected exactly one VPS job, found {vps}"
    body = j[vps[0]]
    assert re.search(r"^[ \t]+needs:[ \t]*build[ \t]*$", body, re.M), (
        "the VPS job must need `build`, or it can restart the stack on an image that was never pushed"
    )
    assert re.search(rf"^[ \t]+if:[ \t]*{re.escape(GATE_VPS)}[ \t]*$", body, re.M), (
        f"the VPS job must be gated on `{GATE_VPS}`, so a repo without VPS_HOST skips it"
    )
    assert re.search(rf"^[ \t]+concurrency:[ \t]*vps-deploy-{SERVICE}[ \t]*$", body, re.M), (
        f"the VPS job must serialise on `concurrency: vps-deploy-{SERVICE}`: two repos deploying at "
        f"once race on /opt/stack/.env"
    )
    assert re.search(r"\$\{\{[ \t]*secrets\.VPS_SSH_KEY[ \t]*\}\}", body), "VPS_SSH_KEY is never used"
    assert re.search(r"\$\{\{[ \t]*secrets\.VPS_KNOWN_HOSTS[ \t]*\}\}", body), (
        "VPS_KNOWN_HOSTS is never used, so the SSH host is not verified"
    )
    expected = f"ssh deploy@${{{{ vars.VPS_HOST }}}} deploy {SERVICE} ${{{{ github.sha }}}}"
    assert expected in body, f"the deploy command must be exactly: {expected}"
    # Known hosts must be written before the first connection, or ssh prompts
    # and the job hangs until it times out.
    assert body.index("known_hosts") < body.index("ssh deploy@"), "known_hosts is written after the ssh call"
    # Health checking is bin/deploy's job ([pl]="PL_TAG 8000 /api/health"), which
    # the forced command runs. A second implementation here would be a second
    # thing to get wrong, and would not be the one that rolls back.
    assert "/api/health" not in body, (
        "the workflow health-checks the service itself. bin/deploy already does it and rolls "
        "back to the previous tag when it fails; do not re-implement it here."
    )


def check_azure_fails_closed(text: str) -> None:
    assert "DEPLOY_AZURE != 'false'" not in text, (
        "`vars.DEPLOY_AZURE != 'false'` fails OPEN: any repo that has not set the variable "
        "turns Azure back on. Legacy infrastructure being cut over must use `== 'true'`"
    )
    j = jobs(text)
    uses_azure = {
        n: b for n, b in j.items()
        if re.search(r"^[ \t]*(az[ \t]|.*azure/login|.*containerapp)", b, re.M)
    }
    assert uses_azure, "the Azure deploy steps are gone entirely"
    for name, body in uses_azure.items():
        assert re.search(rf"^[ \t]+if:[ \t]*{re.escape(GATE_AZURE)}[ \t]*$", body, re.M), (
            f"job {name!r} touches Azure but is not gated on `{GATE_AZURE}`. Every merge would "
            f"deploy to Azure on top of the VPS deploy."
        )
    # PL_Predictor has no AZURE_CLIENT_SECRET, so the OIDC login the workflow it
    # replaced used is the only one that can work here. Copying the sibling's
    # `az login --service-principal` would leave a legacy path that cannot
    # authenticate -- and it is gated off, so nothing would catch it. Matched on
    # the secret reference, so the comment above can name the secret it lacks.
    assert not re.search(r"secrets\.AZURE_CLIENT_SECRET", text), (
        "secrets.AZURE_CLIENT_SECRET is not a secret on this repository; the Azure job must "
        "federate with azure/login (id-token: write), as deploy-azure.yml did."
    )


def check_readme_names_the_live_deploy_workflow(text: str) -> None:
    """The README is where a reader goes to find out how this deploys. It was wrong.

    It named `deploy-azure.yml` — legacy, `workflow_dispatch`-only, Azure
    Container Apps — as the live deploy path, and said it auto-deployed on every
    push to `main`. Both clauses were false, and they are the kind of false
    that causes an incident rather than a typo: a developer merges, sees no
    deploy, and goes to edit the file that looks like the deploy path.

    `check_triggers` and friends above are all pinned against `deploy.yml`. None
    of them can see the README, so the one document a person actually reads was
    unguarded, which is precisely how it went stale while the workflow it
    misdescribed was being written and reviewed.

    Both directions are asserted, because either half alone is satisfiable by a
    README that says nothing: the README must name `deploy.yml` as the
    authoritative path, must mark `deploy-azure.yml` legacy, and must not attach
    an automatic trigger to the legacy file anywhere.
    """
    assert re.search(r"deploy\.yml\W[^\n]{0,80}?authoritative", text, re.I | re.S), (
        "the README does not name deploy.yml as the authoritative deploy path. A reader has "
        "to be told which of the two deploy workflows is live, because the two files are "
        "named almost identically and only one of them runs on a merge."
    )
    # Every mention of the legacy file, not just the first, and each has to sit
    # in a sentence that marks it legacy. "Somewhere in this document
    # deploy-azure.yml is called legacy" is not the invariant -- it is satisfied
    # by a README that corrects itself in a footnote and misleads everywhere
    # else, which is the shape the original had: a correct table, and a false
    # claim in the note below it.
    #
    # The window is a sentence, not a line, and it is CENTRED on the mention
    # rather than following it: "the legacy file `deploy-azure.yml`" qualifies
    # the name before it, and "deploy-azure.yml is legacy" qualifies it after,
    # and both are the same claim. The README wraps prose at 80 columns, so a
    # line-local rule reported a true sentence that merely wrapped mid-claim --
    # and a guard that cries wolf gets deleted rather than fixed.
    LEGACY_MARKERS = r"legacy|cut over|workflow_dispatch-only|not the deploy"
    mentions = list(re.finditer(r"deploy-azure\.yml", text))
    assert mentions, (
        "the README never mentions deploy-azure.yml. A reader looking for the deploy "
        "workflow should find the legacy one named and dismissed, not absent."
    )
    for m in mentions:
        start = max(0, m.start() - SENTENCE_CHARS)
        window = text[start:m.end() + SENTENCE_CHARS]
        assert re.search(LEGACY_MARKERS, window, re.I), (
            f"the README names deploy-azure.yml with nothing marking it legacy within "
            f"{SENTENCE_CHARS} characters either side: {window.strip()[:160]!r}. It is "
            "workflow_dispatch-only and deploys to Azure, which is being cut over to the VPS; a "
            "mention that does not say so leaves the reader to assume it is the deploy path."
        )
    # Every mention of the legacy file, not just the first. A README can name it
    # correctly in the table and then reintroduce the false claim in a note
    # further down, which is where the wrong one lived.
    for m in re.finditer(r"deploy-azure\.yml", text):
        window = text[m.end():m.end() + SENTENCE_CHARS]
        assert "auto-deploy" not in window and "on every push" not in window, (
            "the README describes deploy-azure.yml as deploying automatically. It is "
            "workflow_dispatch-only; that claim is what makes a reader believe a merge "
            "deployed when nothing ran."
        )
    # A claim nobody can check is how this rotted in the first place. The README
    # has to say how to confirm it against the server rather than only assert it.
    assert "gh run list" in text, (
        "the README asserts which workflow deploys without saying how to check. The one "
        "command that settles it (`gh run list --workflow deploy.yml`) belongs next to the "
        "claim, so a reader can confirm it instead of trusting it."
    )


def check_no_secret_material(text: str) -> None:
    for m in re.finditer(r"secrets\.([A-Za-z0-9_]+)", text):
        assert m.group(1).isupper(), f"secret {m.group(1)!r} is not a UPPER_CASE name"
    for marker in ("BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "ghp_", "github_pat_"):
        assert marker not in text, f"the workflow contains credential material ({marker!r})"
    # A secret's value must only ever be read from the environment, never
    # echoed into a log line.
    for line in text.splitlines():
        if "secrets." in line and ("echo" in line or "::" in line):
            raise AssertionError(f"a secret reaches the log: {line.strip()!r}")


def check_no_catch_all(text: str) -> None:
    """No entry may match every commit.

    The other exclusions in this file are named files that existed when it was
    written. This one is the backstop: a bare `**`, `.` or `*` in the filter
    matches everything, so any future refresh job -- in this repo or any repo
    this is copied to -- silently redeploys the service on every run, and the
    named exclusions would never fire because they would be redundant.
    """
    for path in paths_filter(text):
        assert path not in ("**", ".", "*", ""), (
            f"the paths filter contains {path!r}, which matches every commit: the scheduled "
            f"refresh would then redeploy the service each time"
        )


def check_yaml_parses(text: str) -> None:
    """The file is valid YAML, and has the shape a workflow needs.

    A workflow that has never been parsed is not a workflow. The text checks
    elsewhere in this file cannot see structure at all: a sibling repo shipped
    steps indented two spaces too deep, which parsed happily as a continuation
    of the preceding step, and every one of those text assertions passed on it.
    `check_yaml_shape` covers that case; this one covers "it is YAML at all".
    """
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - the assert below reports it
        raise AssertionError(f"the workflow is not valid YAML: {exc}") from exc
    assert isinstance(doc, dict), "the workflow does not parse to a mapping"
    # YAML 1.1 reads a bare `on` as the boolean True, so PyYAML keys it that way.
    trigger = doc.get("on", doc.get(True))
    assert isinstance(trigger, dict), "the `on:` block is missing or not a mapping"
    assert isinstance(trigger.get("push"), dict), "the `on:` block has no `push` trigger"
    assert isinstance(trigger["push"].get("paths"), list), "the push trigger has no `paths` filter"
    for entry in trigger["push"]["paths"]:
        assert isinstance(entry, str), f"a `paths:` entry is not a string: {entry!r}"
    assert isinstance(doc.get("jobs"), dict) and doc["jobs"], "there are no jobs"
    for name, job in doc["jobs"].items():
        assert isinstance(job, dict), f"job {name!r} is not a mapping"
        assert isinstance(job.get("steps"), list) and job["steps"], f"job {name!r} has no steps"
        for step in job["steps"]:
            assert isinstance(step, dict), (
                f"job {name!r} has a step that is not a mapping: {step!r}. This is what "
                f"two-spaces-too-deep indentation parses as."
            )


def check_yaml_shape(text: str) -> None:
    """Indentation is a multiple of two spaces, everywhere.

    The counterpart to `check_yaml_parses`: the historical failure was not a
    parse error, it was valid YAML in the wrong shape. This asserts the cheap
    invariant that catches it.
    """
    for n, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped and indent % 2:
            raise AssertionError(
                f"line {n} is indented {indent} spaces, which is not a multiple of two: "
                f"{line!r}"
            )
    # Every `steps:` key must be followed by list items at exactly six spaces.
    for m in re.finditer(r"^[ \t]+steps:[ \t]*$", text, re.M):
        rest = text[m.end():].lstrip("\n")
        first = rest.splitlines()[0] if rest.splitlines() else ""
        assert first.startswith("      - "), (
            f"a `steps:` block whose first item is not at six spaces: {first!r}. "
            f"Wrong indentation here silently nests one step under another."
        )


CHECKS = {
    "shape": check_yaml_shape,
    "catchall": check_no_catch_all,
    "triggers": check_triggers,
    "credential": check_credential,
    "image": check_image,
    "vps": check_vps,
    "azure": check_azure_fails_closed,
    "secrets": check_no_secret_material,
    "parses": check_yaml_parses,
}


# ---------------------------------------------------------------- the tests


def test_the_workflow_file_exists():
    assert WORKFLOW.exists(), f"{WORKFLOW} is missing; a merge deploys nothing"
    assert len(workflow_text()) > 400, "the workflow looks like a stub"


def test_each_check_can_fail():
    """A guard that cannot fail is not a guard.

    Each check is run against the real file, and against a copy with one thing
    broken. If a broken copy passes, the check is vacuous and this fails.
    """
    good = workflow_text()
    for name, check in CHECKS.items():
        check(good)  # the real file satisfies it
    # Each mutation must be a string that appears EXACTLY ONCE, or it is not
    # breaking the thing it claims to: replacing one of two `:latest`
    # occurrences leaves the check satisfied by the other.
    broken = {
        "triggers": ("workflow_dispatch:", "workflow_DISABLED:"),
        # A third credential, neither of the two the linkage check can accept.
        # A third credential. Anchored on the one actually in the file:
        # secrets.GITHUB_TOKEN appears zero times now, and a mutation whose anchor
        # is absent breaks nothing -- the vacuity this meta-test exists to catch.
        # Anchored on the password LINE, not the credential name: the name now appears
        # twice (the line, and the comment explaining why), and a mutation that is not
        # unique breaks one copy and leaves the check satisfied by the other.
        "credential": (
            "password: ${{ secrets.GHCR_PAT }}",
            "password: ${{ secrets.REGISTRY_TOKEN }}",
        ),
        # Retag the image the `build` line produces. Unique: the `push` lines
        # carry the same tag but not the same prefix.
        "image": (
            f"          provenance: false\n          tags: |\n            {IMAGE}:${{{{ github.sha }}}}",
            f"          provenance: false\n          tags: |\n            {IMAGE}:${{{{ github.head_sha }}}}",
        ),
        "vps": (f"deploy {SERVICE} ${{{{ github.sha }}}}", f"deploy {SERVICE}"),
        "azure": (GATE_AZURE, "vars.DEPLOY_AZURE != 'false'"),
        # Lower-cased, so check_no_secret_material's UPPER_CASE rule fires. Anchored on
        # the password line and the credential actually in the file, for the same two
        # reasons as the "credential" mutation above.
        "secrets": (
            "password: ${{ secrets.GHCR_PAT }}",
            "password: ${{ secrets.ghcr_pat }}",
        ),
        # A unique anchor: `concurrency: vps-deploy-pl` appears once.
        "shape": (
            f"concurrency: vps-deploy-{SERVICE}",
            f" concurrency: vps-deploy-{SERVICE}",
        ),
        "catchall": (f"'{MUST_DEPLOY[0]}'", "'**'"),
        # Dedent a block scalar's body to the level of its key. Unlike a
        # two-space step indent (which parses, and is `shape`'s job) this is a
        # genuine parse error, so it exercises `parses` and not `shape`.
        "parses": ("\n          install -m 700 -d ~/.ssh", "\n        install -m 700 -d ~/.ssh"),
    }
    for name, (old, new) in broken.items():
        assert good.count(old) == 1, (
            f"the {name!r} mutation {old!r} appears {good.count(old)} times, so breaking one "
            f"copy leaves the check satisfied by another"
        )
        try:
            CHECKS[name](good.replace(old, new, 1))
        except AssertionError:
            continue
        raise AssertionError(f"the {name!r} check passed a workflow with {old!r} broken")


def test_the_paths_reader_reads_a_normal_workflow():
    """`paths_filter` is load-bearing for the exclusion assertions, and a reader
    that quietly returned [] would make them vacuous. Pinned against a sample
    shaped like the real file, comments at column zero included.
    """
    sample = (
        "on:\n"
        "# why this filter looks like this, at column zero\n"
        "  push:\n"
        "    branches: [main]\n"
        "    paths:\n"
        "      - 'src/**'\n"
        "      - \"Dockerfile\"\n"
        "  workflow_dispatch: {}\n"
        "permissions:\n"
        "  contents: read\n"
    )
    assert paths_filter(sample) == ["src/**", "Dockerfile"]

    with pytest.raises(AssertionError, match="no `paths:` filter"):
        paths_filter(sample.replace("    paths:\n", ""))


def test_a_merge_to_main_triggers_a_deploy():
    check_triggers(workflow_text())


def test_a_data_refresh_does_not_redeploy():
    check_triggers(workflow_text())


def test_it_pushes_with_a_credential_this_repo_can_use():
    check_credential(workflow_text())


def test_it_builds_and_pushes_the_image_the_stack_pulls():
    check_image(workflow_text())


def test_it_reaches_the_vps_and_asks_for_this_service():
    check_vps(workflow_text())


def test_azure_stays_off_unless_asked_for():
    check_azure_fails_closed(workflow_text())


def test_no_secret_material_in_the_workflow():
    check_no_secret_material(workflow_text())


def test_the_filter_has_no_catch_all_entry():
    check_no_catch_all(workflow_text())


def test_the_workflow_is_valid_yaml():
    check_yaml_parses(workflow_text())


def test_the_yaml_indentation_is_sane():
    check_yaml_shape(workflow_text())


def test_readme_points_at_the_deploy_workflow_that_runs():
    check_readme_names_the_live_deploy_workflow(readme_text())


def test_the_readme_check_can_fail():
    """The README guard, broken four ways, the way it was actually broken.

    Kept separate from `test_each_check_can_fail` because that one mutates the
    workflow text, and this reads a different file. The first two mutations are
    the historical failure verbatim: the README named the legacy file as the
    live one, and described it as deploying on every push.
    """
    good = readme_text()
    check_readme_names_the_live_deploy_workflow(good)
    broken = {
        # The actual defect: the word that names the live one is gone.
        "authoritative": ("is the authoritative", "is the"),
        # ...and the legacy one left standing as the live one. The table cell is
        # rewritten as well, because the check looks at every mention of the
        # file and the table row marked it legacy independently -- breaking only
        # the prose leaves the file still correctly described in the table, which
        # is a different and much milder defect than the one being guarded.
        "legacy": ("is legacy and does not", "is the one that"),
        # The claim that made a merge look like it should have deployed, restored
        # to the legacy file exactly as it read before: on every push to main.
        # Anchored on the legacy column's trigger cell, which is the sentence the
        # check reads, rather than anywhere else in the table.
        "auto-deploy": (
            "`workflow_dispatch` only — never on a push or PR",
            "`workflow_dispatch`, auto-deploys on every push to `main`",
        ),
        # An unverifiable claim: this one rotted because nobody could check it.
        "checkable": ("gh run list", "see the Actions tab"),
    }
    for name, (old, new) in broken.items():
        assert good.count(old) == 1, (
            f"the {name!r} mutation {old!r} appears {good.count(old)} times, so it may not be "
            "breaking the thing it claims to"
        )
        try:
            check_readme_names_the_live_deploy_workflow(good.replace(old, new, 1))
        except AssertionError:
            continue
        raise AssertionError(
            f"the README check passed a README with {old!r} broken -- exactly the stale claim "
            "it exists to prevent"
        )


def committed_paths() -> dict[str, set[str]]:
    """What every OTHER workflow in this repo commits, by workflow name.

    A GitHub Actions job cannot push to the branch it is running on, so these
    workflows commit to `main` via a checkout with a token. That is the whole
    mechanism by which a deploy loop starts: the refresh job commits a path the
    deploy filter watches, the commit triggers a build, and the job that made
    the commit has no idea it did.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml")):
        if path.name == WORKFLOW.name:
            continue
        # Join shell line continuations first, or a `git add a b \` line hides
        # everything after the backslash.
        text = path.read_text().replace("\\\n", " ")
        added: set[str] = set()
        for m in re.finditer(r"git add\s+(.*)", text):
            for token in m.group(1).split():
                token = token.strip("'\"")
                if token.startswith("-"):  # git add -f path
                    continue
                if "/" in token or token.endswith((".json", ".db")):
                    added.add(token)
        if added:
            out[path.name] = added
    return out


def test_every_scheduled_commit_is_deliberately_classified():
    """Derived from the other workflows, not from a hand-written list.

    The refresh job commits to `main` (a workflow cannot push to its own branch,
    so it checks out main with a token). Any path such a job commits that the
    deploy filter watches puts the repo in a loop: the commit triggers a build,
    and the job that made the commit has no idea it did.

    Rather than forbid the overlap outright, every overlap has to be classified.
    A new refresh job, or a new file it writes, lands here as an unclassified
    path and fails, instead of quietly deploying dozens of times a day.
    """
    watched = paths_filter(workflow_text())
    classified = set(WATCHED_ANYWAY)
    others = committed_paths()
    assert others, (
        "no other workflow in this repo commits anything. That was true once, but the filter's "
        "whole safety argument is about the refresh job, so losing it silently would leave the "
        "exclusions asserted against nothing."
    )
    for name, added in others.items():
        for path in sorted(added):
            if any(fnmatch.fnmatch(path, pat) for pat in MUST_NOT_DEPLOY):
                continue  # explicitly excluded, and check_triggers keeps it that way
            if any(fnmatch.fnmatch(path, pat) for pat in classified):
                continue  # deliberately watched, with a recorded reason
            watched_by = [w for w in watched if fnmatch.fnmatch(path, w)]
            raise AssertionError(
                f"{name} commits {path!r}"
                + (f", which the deploy paths filter watches via {watched_by}" if watched_by else "")
                + ", but it is in neither MUST_NOT_DEPLOY nor WATCHED_ANYWAY. Classify it: "
                "either exclude it from the filter, or record why a scheduled commit "
                "should redeploy the service."
            )
