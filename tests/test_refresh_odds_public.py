"""routes.py::refresh_odds_public -- the public "Refresh odds" button.
Unlike /refresh-odds (admin-only, runs the real pipeline), this route is
reachable in PUBLIC_MODE and must never run that pipeline itself -- it only
asks GitHub Actions to run the snapshot refresh sooner (see the route's own
docstring for why: the free-tier public host OOMs on the real compute)."""

import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pl_predictor.api import routes


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router)
    return app


def _reset_cooldown():
    routes._last_public_refresh_trigger = 0.0


def test_public_mode_without_token_returns_503(monkeypatch):
    _reset_cooldown()
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "GITHUB_ACTIONS_TOKEN", None)
    client = TestClient(_app())
    resp = client.post("/api/refresh-odds/public")
    assert resp.status_code == 503


def test_public_mode_with_token_dispatches_workflow(monkeypatch):
    _reset_cooldown()
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "GITHUB_ACTIONS_TOKEN", "fake-token")

    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers, json))
        return FakeResponse()

    monkeypatch.setattr(routes.requests, "post", fake_post)
    client = TestClient(_app())
    resp = client.post("/api/refresh-odds/public")
    assert resp.status_code == 200
    assert resp.json()["status"] == "requested"
    assert len(calls) == 1
    url, headers, body = calls[0]
    assert "refresh-public-snapshot.yml/dispatches" in url
    assert headers["Authorization"] == "Bearer fake-token"
    assert body == {"ref": "main"}


def test_private_mode_runs_real_refresh_directly(monkeypatch):
    _reset_cooldown()
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)
    called = {}
    monkeypatch.setattr(routes, "_get_odds_df", lambda force=False: called.setdefault("force", force))
    monkeypatch.setattr(routes, "_clear_cache", lambda *keys: called.setdefault("cleared", keys))
    client = TestClient(_app())
    resp = client.post("/api/refresh-odds/public")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert called["force"] is True


def test_get_odds_df_falls_back_to_empty_on_any_request_failure(monkeypatch):
    """Real incident: The Odds API returns 401 once its monthly free-tier
    credit quota is exhausted (not 429) -- indistinguishable, from here, from
    any other HTTP failure. Only OddsAPIKeyMissing was caught before this;
    any other requests failure used to bubble up as an unhandled 500 from
    every endpoint that touches odds (`/refresh-odds`, `/fixtures`, the
    value-bet table) instead of degrading to "no live odds", the same way a
    missing key already does."""
    routes._clear_cache("odds_df")

    def _raise(force_refresh=False):
        raise requests.HTTPError("401 quota exhausted")

    monkeypatch.setattr(routes, "fetch_epl_odds", _raise)

    result = routes._get_odds_df(force=True)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_cooldown_blocks_rapid_repeat_requests(monkeypatch):
    _reset_cooldown()
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)
    monkeypatch.setattr(routes, "_get_odds_df", lambda force=False: None)
    monkeypatch.setattr(routes, "_clear_cache", lambda *keys: None)
    client = TestClient(_app())
    first = client.post("/api/refresh-odds/public")
    second = client.post("/api/refresh-odds/public")
    assert first.status_code == 200
    assert second.status_code == 429
