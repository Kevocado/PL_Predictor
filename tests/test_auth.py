"""Guest-password gate for the public deployment — api/auth.py's
GuestAuthMiddleware and api/routes.py's _admin_only dependency. Uses a
minimal standalone FastAPI app rather than the real `main.app` (whose
lifespan does real network calls via warm_caches) — same "test the piece in
isolation" style as the rest of this suite."""

import base64

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from pl_predictor.api import auth as auth_mod
from pl_predictor.api import routes


def _basic(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _auth_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(auth_mod.GuestAuthMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


def test_no_auth_required_when_public_mode_off(monkeypatch):
    monkeypatch.setattr(auth_mod, "PUBLIC_MODE", False)
    client = TestClient(_auth_app())
    assert client.get("/ping").status_code == 200


def test_rejects_missing_credentials_in_public_mode(monkeypatch):
    monkeypatch.setattr(auth_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(auth_mod, "GUEST_PASSWORD", "guest")
    client = TestClient(_auth_app())
    resp = client.get("/ping")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"].startswith("Basic")


def test_rejects_wrong_password_in_public_mode(monkeypatch):
    monkeypatch.setattr(auth_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(auth_mod, "GUEST_PASSWORD", "guest")
    client = TestClient(_auth_app())
    resp = client.get("/ping", headers=_basic("guest", "wrong"))
    assert resp.status_code == 401


def test_accepts_correct_password_any_username(monkeypatch):
    monkeypatch.setattr(auth_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(auth_mod, "GUEST_PASSWORD", "guest")
    client = TestClient(_auth_app())
    resp = client.get("/ping", headers=_basic("anyone", "guest"))
    assert resp.status_code == 200


def test_no_guest_password_configured_always_rejects_in_public_mode(monkeypatch):
    """A misconfigured deployment (PUBLIC_MODE on, no GUEST_PASSWORD set)
    must fail closed, not accept every password."""
    monkeypatch.setattr(auth_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(auth_mod, "GUEST_PASSWORD", None)
    client = TestClient(_auth_app())
    resp = client.get("/ping", headers=_basic("anyone", "anything"))
    assert resp.status_code == 401


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
