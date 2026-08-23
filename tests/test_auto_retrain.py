"""maybe_auto_retrain's growing-match-count trigger — this is what makes
the weekly refresh actually happen without a manual click: once
`football_data.fetch_current_season_partial()` (now pulselive.com-backed
when football-data.co.uk hasn't published yet — see test_football_data.py)
reports more matches than the last training run saw, a retrain should
fire; otherwise it shouldn't. Monkeypatches `manifest_lib.train_all` itself
(a real fit costs ~10-30s) so this only tests the trigger logic, not
training."""

import pandas as pd
import pytest

from pl_predictor.api import routes
from pl_predictor.models import manifest as manifest_lib


@pytest.fixture
def manifest_path_exists(tmp_path, monkeypatch):
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text("{}")
    monkeypatch.setattr(manifest_lib, "MANIFEST_PATH", fake_manifest)
    return fake_manifest


def test_retrains_when_more_current_season_matches_available(manifest_path_exists, monkeypatch):
    monkeypatch.setattr(manifest_lib, "load_manifest", lambda: {"n_current_season_matches": 6})
    monkeypatch.setattr(routes.football_data, "fetch_current_season_partial", lambda: pd.DataFrame({"team_home": ["X"] * 12}))

    calls = {"trained": False}
    monkeypatch.setattr(manifest_lib, "train_all", lambda: calls.update(trained=True))
    monkeypatch.setattr(routes, "_clear_cache", lambda *keys: None)

    routes.maybe_auto_retrain()
    assert calls["trained"] is True


def test_skips_retrain_when_match_count_unchanged(manifest_path_exists, monkeypatch):
    monkeypatch.setattr(manifest_lib, "load_manifest", lambda: {"n_current_season_matches": 12})
    monkeypatch.setattr(routes.football_data, "fetch_current_season_partial", lambda: pd.DataFrame({"team_home": ["X"] * 12}))

    calls = {"trained": False}
    monkeypatch.setattr(manifest_lib, "train_all", lambda: calls.update(trained=True))

    routes.maybe_auto_retrain()
    assert calls["trained"] is False


def test_skips_retrain_when_source_returns_none(manifest_path_exists, monkeypatch):
    """Both football-data.co.uk and pulselive.com came up empty (see
    test_football_data.py) — fetch_current_season_partial returns None."""
    monkeypatch.setattr(manifest_lib, "load_manifest", lambda: {"n_current_season_matches": 0})
    monkeypatch.setattr(routes.football_data, "fetch_current_season_partial", lambda: None)

    calls = {"trained": False}
    monkeypatch.setattr(manifest_lib, "train_all", lambda: calls.update(trained=True))

    routes.maybe_auto_retrain()
    assert calls["trained"] is False


def test_does_nothing_without_a_manifest_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_lib, "MANIFEST_PATH", tmp_path / "does_not_exist.json")

    calls = {"trained": False}
    monkeypatch.setattr(manifest_lib, "train_all", lambda: calls.update(trained=True))

    routes.maybe_auto_retrain()
    assert calls["trained"] is False
