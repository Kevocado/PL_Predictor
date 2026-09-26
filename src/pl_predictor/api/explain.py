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
# A model can be slow. Long enough for the service's own 25 s call plus its
# fallback retry, short enough that a hung explainer does not hold the page.
EXPLAINER_TIMEOUT_S = float(os.getenv("EXPLAINER_TIMEOUT_S", "30"))


@router.get("/api/explain/{sport}/{explainer_id:path}")
def explain(sport: str, explainer_id: str) -> dict:
    """Forward to the explainer, or say plainly that it could not be reached."""
    url = f"{EXPLAINER_URL}/explain/{sport}/{quote(explainer_id, safe='/')}"
    try:
        response = requests.get(url, timeout=EXPLAINER_TIMEOUT_S)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="The summary service is not available.") from exc

    if response.status_code == 404:
        # No summary for this fixture is a normal answer, not a failure; the
        # site renders the fixture without the panel.
        raise HTTPException(status_code=404, detail="No summary for this fixture.")
    if response.status_code >= 500:
        raise HTTPException(status_code=502, detail="The summary service is not available.")
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail="The summary could not be read.")
    return response.json()
