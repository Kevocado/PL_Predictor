"""Proxy for the plain-English explainer service.

PL's FastAPI serves the site, so Caddy only ever reverse-proxies this app. The
browser therefore cannot reach the explainer directly: it calls
``/api/explain/{sport}/{id}`` here, and this route forwards to the service's
own ``/explain/{sport}/{id}``.

The proxy is deliberately thin and deliberately forgiving. A summary is a
nice-to-have on top of a fixture page, so a missing or slow explainer becomes a
502 with a plain message, and the site's own error state takes it from there --
the fixture itself never depends on this route answering.
"""
from __future__ import annotations

import os
from urllib.parse import quote

import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()

# The explainer is reached over the internal compose network by service name.
EXPLAINER_URL = os.getenv("EXPLAINER_URL", "http://predictor-explainer:8090").rstrip("/")
# Deliberately BELOW the browser's 20 s read timeout for this site. This route
# is a sync def, so an in-flight request occupies one of Starlette's worker
# threads for its whole duration: a proxy that waits longer than the client
# spends a thread on an answer nobody is waiting for. The explainer's own
# 25 s model timeout is longer than this on purpose — a first uncached match
# usually lands here and the panel shows its Try again state, which pre-
# generation exists to avoid.
EXPLAINER_TIMEOUT_S = float(os.getenv("EXPLAINER_TIMEOUT_S", "15"))


_UNAVAILABLE = "The summary service is not available."


@router.get("/api/explain/{sport}/{explainer_id:path}/context")
def explain_context(sport: str, explainer_id: str):
    """Forward to the explainer's no-model `/context` route.

    Declared BEFORE the catch-all, which would otherwise swallow `<id>/context`.
    Stricter than `explain` below (that route is left untouched): `..` anywhere,
    a `/` in the sport, and an empty or leading-`/` id are refused before any
    request goes out; a non-dict 2xx body is a 502 rather than a bare 500. Every
    failure is one fixed 502 and no upstream body is ever returned.
    """
    if ".." in sport or ".." in explainer_id or "/" in sport or not explainer_id or explainer_id.startswith("/"):
        raise HTTPException(status_code=502, detail=_UNAVAILABLE)
    url = f"{EXPLAINER_URL}/explain/{quote(sport, safe='')}/{quote(explainer_id, safe='')}/context"
    try:
        response = requests.get(url, timeout=EXPLAINER_TIMEOUT_S, allow_redirects=False)
        if not (200 <= response.status_code < 300):
            raise requests.HTTPError(f"upstream {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            raise requests.HTTPError(f"upstream 2xx body is a {type(body).__name__}")
        return body
    except requests.RequestException as exc:
        # Also covers a non-JSON 2xx: `JSONDecodeError` subclasses `RequestException`.
        raise HTTPException(status_code=502, detail=_UNAVAILABLE) from exc


@router.get("/api/explain/{sport}/{explainer_id:path}")
def explain(sport: str, explainer_id: str) -> dict:
    """Forward to the explainer, or say plainly that it could not be reached.

    No upstream body is ever returned: an explainer error can carry key
    material or an internal path, so every non-2xx becomes a fixed message.
    Redirects are not followed, and neither path segment is allowed to walk out
    of the explainer's own route.
    """
    if ".." in sport or ".." in explainer_id:
        # Rejected before any request goes out: a path segment that walks out
        # of the explainer's own route would make this a forwarder to any path
        # on that host. 502 like every other failure, so the site's error state
        # takes it and no upstream (or no request) is involved.
        raise HTTPException(status_code=502, detail="The summary service is not available.")
    url = f"{EXPLAINER_URL}/explain/{quote(sport, safe='')}/{quote(explainer_id, safe='')}"
    try:
        response = requests.get(url, timeout=EXPLAINER_TIMEOUT_S, allow_redirects=False)
        if not (200 <= response.status_code < 300):
            raise requests.HTTPError(f"upstream {response.status_code}")
        return response.json()
    except requests.RequestException as exc:
        # A 404 from the explainer means "no summary for this fixture", which
        # the site renders as a fixture with no panel. Anything else is the
        # service being absent, slow or broken, and the site's error state takes
        # it from there. Neither echoes an upstream body.
        raise HTTPException(status_code=502, detail="The summary service is not available.") from exc
    except ValueError as exc:
        # A 200 that is not JSON: the explainer is misconfigured or something
        # else is answering on its port. A 502, so the site shows its retry
        # state rather than a bare 500 it cannot interpret.
        raise HTTPException(status_code=502, detail="The summary service is not available.") from exc
