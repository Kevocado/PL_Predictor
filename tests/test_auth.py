"""api/routes.py's _admin_only dependency — the public deployment's only
access control now that it has no login gate: the four write endpoints
(retrain/refresh-odds/refresh-fixtures/backtest) 404 unconditionally under
PUBLIC_MODE regardless of who's asking, since the site is otherwise a
plain public read-only page with nothing a password would protect."""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from pl_predictor.api import routes


def _admin_app() -> FastAPI:
    app = FastAPI()

    @app.post("/admin-thing", dependencies=[Depends(routes._admin_only)])
    def admin_thing():
        return {"ok": True}

    return app


def test_admin_only_blocks_in_public_mode(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    client = TestClient(_admin_app())
    assert client.post("/admin-thing").status_code == 404


def test_admin_only_allows_when_not_public(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)
    client = TestClient(_admin_app())
    assert client.post("/admin-thing").status_code == 200
