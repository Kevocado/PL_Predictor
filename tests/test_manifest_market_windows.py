from pl_predictor.models import manifest


def test_train_all_reuses_the_same_frame_when_windows_match(monkeypatch, tmp_path):
    """When an explicit `seasons` override is passed, it applies uniformly
    to every market (MARKET_TRAINING_WINDOWS is bypassed) — the corners
    frame must be the exact same object as the default frame, not a
    second, redundant build."""
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "MANIFEST_HISTORY_PATH", tmp_path / "manifest_history.jsonl")
    monkeypatch.setattr(manifest, "DIXON_COLES_PATH", tmp_path / "dixon_coles.pkl")
    monkeypatch.setattr(manifest, "BIVARIATE_POISSON_PATH", tmp_path / "bivariate_poisson.pkl")
    monkeypatch.setattr(manifest, "ML_HOME_MODEL_PATH", tmp_path / "ml_home.json")
    monkeypatch.setattr(manifest, "ML_AWAY_MODEL_PATH", tmp_path / "ml_away.json")
    monkeypatch.setattr(manifest, "CORNERS_MODEL_PATH", tmp_path / "corners.json")
    monkeypatch.setattr(manifest, "CARDS_MODEL_PATH", tmp_path / "cards.json")

    build_calls = []
    real_build_frame = manifest._build_frame

    def counting_build_frame(seasons, current_partial):
        build_calls.append(tuple(seasons))
        return real_build_frame(seasons, current_partial)

    monkeypatch.setattr(manifest, "_build_frame", counting_build_frame)

    result = manifest.train_all(seasons=["2023-2024", "2024-2025", "2025-2026"], include_current_season=False)

    assert len(build_calls) == 1, "an explicit uniform `seasons` override must not trigger a second corners-only build"
    assert result["corners"]["seasons"] == result["seasons"]
    assert result["market_training_windows"] == {"scoreline": 8, "corners": 12, "cards": 8}


def test_train_all_gives_corners_its_own_window_by_default(monkeypatch, tmp_path):
    """With no explicit `seasons` override, corners must train on its own
    (larger) MARKET_TRAINING_WINDOWS-driven window even though scoreline/
    cards stay on the default — verified with small substitute window
    sizes so the test stays fast."""
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "MANIFEST_HISTORY_PATH", tmp_path / "manifest_history.jsonl")
    monkeypatch.setattr(manifest, "DIXON_COLES_PATH", tmp_path / "dixon_coles.pkl")
    monkeypatch.setattr(manifest, "BIVARIATE_POISSON_PATH", tmp_path / "bivariate_poisson.pkl")
    monkeypatch.setattr(manifest, "ML_HOME_MODEL_PATH", tmp_path / "ml_home.json")
    monkeypatch.setattr(manifest, "ML_AWAY_MODEL_PATH", tmp_path / "ml_away.json")
    monkeypatch.setattr(manifest, "CORNERS_MODEL_PATH", tmp_path / "corners.json")
    monkeypatch.setattr(manifest, "CARDS_MODEL_PATH", tmp_path / "cards.json")
    monkeypatch.setattr(manifest, "MARKET_TRAINING_WINDOWS", {"scoreline": 3, "corners": 4, "cards": 3})

    build_calls = []
    real_build_frame = manifest._build_frame

    def counting_build_frame(seasons, current_partial):
        build_calls.append(tuple(seasons))
        return real_build_frame(seasons, current_partial)

    monkeypatch.setattr(manifest, "_build_frame", counting_build_frame)

    result = manifest.train_all(seasons=None, include_current_season=False)

    assert len(build_calls) == 2, "differing windows must trigger exactly two builds (default + corners)"
    assert len(result["corners"]["seasons"]) > len(result["seasons"])
    assert result["corners"]["seasons"][0] < result["seasons"][0]  # corners' window reaches further back
    # Serving shares one feature_cols list across corners/cards -- the
    # manifest.py::train_all assertion already guards this at train time;
    # confirm here too that it didn't silently diverge.
    assert result["corners"]["importance"]["gain"].keys() == set(result["features"])
