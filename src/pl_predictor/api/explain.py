"""Proxy for the plain-English explainer service.

PL's FastAPI serves the site, so Caddy only ever reverse-proxies this app. The
browser therefore cannot reach the explainer directly: it calls
``/api/explain/{sport}/{id}`` here, and this route forwards to the service's
own ``/explain/{sport}/{id}``.

The proxy is deliberately thin and deliberately unforgiving. A summary is a
nice-to-have on top of a fixture page, so a missing, slow or broken explainer
becomes a 502 with a fixed message and the site's own error state takes it from
there -- the fixture itself never depends on this route answering.
"""
from __future__ import annotations

import os
import sys
from urllib.parse import quote

import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()

#: Ceiling on the accepted timeout, so a fat-fingered `600` cannot hold a worker
#: thread for ten minutes. Well above the browser's own 20 s read timeout, which is
#: the point at which the client has given up anyway.
_MAX_TIMEOUT_S = 60.0
_DEFAULT_TIMEOUT_S = 15.0


def _positive_float(raw: str, name: str) -> float:
    """Parse a duration that must be a positive number, or fall back to the default.

    `float(os.getenv(...))` at import raises on a typo, and `app.py` imports this
    module -- so `EXPLAINER_TIMEOUT_S=` (set but empty, a realistic compose
    accident) or `=abc` would take the WHOLE API down, not just this route. And
    `=0` parses fine, then makes `requests` raise a bare `ValueError`, which is
    not a `RequestException` and escapes the handler below as a 500: a fourth
    failure shape on a route that promises the site has exactly one.

    Falls back rather than raising, because a bad env var should cost one panel,
    not the whole site. The operator gets a line on stderr at boot.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(f"warning: {name}={raw!r} is not a number; using the default", file=sys.stderr)
        return _DEFAULT_TIMEOUT_S
    if not 0 < value <= _MAX_TIMEOUT_S:
        print(f"warning: {name}={raw!r} is outside (0, {_MAX_TIMEOUT_S}]; "
              f"using the default", file=sys.stderr)
        return _DEFAULT_TIMEOUT_S
    return value


# The explainer is reached over the internal compose network by service name.
EXPLAINER_URL = os.getenv("EXPLAINER_URL", "http://predictor-explainer:8090").rstrip("/")
# Deliberately well below the browser's 20 s read timeout for this site. This
# route is a sync def, so an in-flight request occupies one of Starlette's worker
# threads for its whole duration -- and it shares that pool with every other route
# in this app, all 22 of which are sync too. Forty concurrent explainer requests
# would hold the pool for the whole timeout and stall /health and /facts/* with
# it, so the reason to be fast here is the shared pool at least as much as the
# browser. The explainer's own 25 s model timeout is longer on purpose: a first
# uncached match usually lands here and the panel shows its Try again state,
# which pre-generation exists to avoid.
EXPLAINER_TIMEOUT_S = _positive_float(os.getenv("EXPLAINER_TIMEOUT_S", "15"), "EXPLAINER_TIMEOUT_S")

#: Every failure is this message. An upstream error can carry key material or an
#: internal path, and the browser is not the place to find out.
_UNAVAILABLE = "The summary service is not available."


@router.get("/api/explain/{sport}/{explainer_id:path}/context")
def explain_context(sport: str, explainer_id: str):
    """Forward to the explainer's no-model `/context` route, under every rule of `explain`.

    Declared BEFORE the catch-all: `{explainer_id:path}` would otherwise swallow
    `<id>/context` and forward it to the summary route as an id.
    """
    return _forward(sport, explainer_id, "/context")


@router.get("/api/explain/{sport}/{explainer_id:path}")
def explain(sport: str, explainer_id: str):
    """Forward to the explainer's summary route; see `_forward` for the guarantees."""
    return _forward(sport, explainer_id, "")


def _forward(sport: str, explainer_id: str, suffix: str):
    """Forward to the explainer, or say plainly that it could not be reached.

    No upstream body is ever returned, on any status: a 2xx is the only thing
    passed through. Redirects are not followed, and neither path segment is
    allowed to walk out of the explainer's own route.
    """
    # What is enforced, precisely -- a walk-out check on its own does not get all
    # of this:
    #
    # * `..` anywhere in either segment, INTERIOR included. A prefix check
    #   (`startswith("..")`) satisfies every case the tests used to carry while
    #   letting `secrets/../x` through, and a request for that does leave.
    # * a leading `/` in the id, or a `/` in the sport. Neither contains a single
    #   `..`, so no amount of walk-out checking sees them -- yet `quote()` puts
    #   `/etc/passwd` on the wire as `%2Fetc%2Fpasswd` and the explainer's own
    #   ASGI layer decodes it back, so the request lands on a deeper path than the
    #   caller named. An id is one relative segment; anything else is refused.
    #
    # All of it is refused BEFORE any request goes out, so a rejected input never
    # becomes a request. 502 like every other failure, so the site's error state
    # takes it and no upstream (and no request) is involved.
    if ".." in sport or ".." in explainer_id or "/" in sport or not explainer_id or explainer_id.startswith("/"):
        raise HTTPException(status_code=502, detail=_UNAVAILABLE)
    url = f"{EXPLAINER_URL}/explain/{quote(sport, safe='')}/{quote(explainer_id, safe='')}{suffix}"
    try:
        response = requests.get(url, timeout=EXPLAINER_TIMEOUT_S, allow_redirects=False)
        if not (200 <= response.status_code < 300):
            # A 404 from the explainer means "no summary for this fixture", which
            # the site renders as a fixture with no panel. Anything else is the
            # service being absent, slow or broken. Both are 502 here so the site
            # has exactly one failure shape to handle.
            raise requests.HTTPError(f"upstream {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            # Not annotated `-> dict` on purpose. The annotation makes FastAPI infer
            # `response_model=dict`, and a 2xx whose JSON is a list, string, number
            # or null raises `ResponseValidationError`, which has no registered
            # handler -- so the browser got a bare 500 text/plain: a third failure
            # shape on a route whose whole design is that the site has exactly
            # one, and the offending upstream value lands in the server traceback.
            # Checking here makes it the same 502 as every other unusable answer.
            raise requests.HTTPError(f"upstream 2xx body is a {type(body).__name__}")
        return body
    except requests.RequestException as exc:
        # This also covers "a 2xx that is not JSON", which is the case a separate
        # `except ValueError` used to be written for. It was unreachable:
        # `requests.exceptions.JSONDecodeError` subclasses
        # `requests.exceptions.InvalidJSONError`, which subclasses
        # `RequestException`, so the handler above catches it first. Verified
        # against the pinned requests (2.34.2) rather than assumed -- a mutation
        # check that replaced this branch with `except KeyError` still passed
        # every test, which is what a dead branch looks like from the outside.
        #
        # Both cases are the same failure from the browser's point of view: the
        # explainer is misconfigured, absent or slow, so the site shows its retry
        # state. One handler, one message.
        raise HTTPException(status_code=502, detail=_UNAVAILABLE) from exc