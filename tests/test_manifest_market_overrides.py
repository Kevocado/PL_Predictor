from pl_predictor.data import football_data
from pl_predictor.models import manifest


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "MANIFEST_HISTORY_PATH", tmp_path / "manifest_history.jsonl")
    monkeypatch.setattr(manifest, "DIXON_COLES_PATH", tmp_path / "dixon_coles.pkl")
    monkeypatch.setattr(manifest, "BIVARIATE_POISSON_PATH", tmp_path / "bivariate_poisson.pkl")
    monkeypatch.setattr(manifest, "ML_HOME_MODEL_PATH", tmp_path / "ml_home.json")
    monkeypatch.setattr(manifest, "ML_AWAY_MODEL_PATH", tmp_path / "ml_away.json")
    monkeypatch.setattr(manifest, "CORNERS_MODEL_PATH", tmp_path / "corners.json")
    monkeypatch.setattr(manifest, "CARDS_MODEL_PATH", tmp_path / "cards.json")
    monkeypatch.setattr(manifest, "COVARIATE_POISSON_PATH", tmp_path / "covariate_poisson.pkl")


def test_train_all_includes_covariate_poisson_as_a_fourth_candidate(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    result = manifest.train_all(seasons=["2023-2024", "2024-2025", "2025-2026"], include_current_season=False)

    assert "covariate_poisson" in result["scoreline"]
    assert "rps" in result["scoreline"]["covariate_poisson"]["metrics"]
    assert set(result["scoreline"]["market_metrics"].keys()) == {
        "dixon_coles",
        "bivariate_poisson",
        "ml_scoreline",
        "covariate_poisson",
    }
    # ml_scoreline is expected to keep winning 1X2 on this data -- if this
    # ever flips, market_overrides' *meaning* changes (an override only
    # matters relative to whichever model IS chosen), so this assertion is
    # deliberately here to catch that silently changing.
    assert result["scoreline"]["chosen_model"] == "ml_scoreline"
    assert result["scoreline"]["market_overrides"] == manifest.MARKET_MODEL_OVERRIDES


def test_load_models_resolves_market_override_to_a_loaded_model_with_context(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    manifest.train_all(seasons=["2023-2024", "2024-2025", "2025-2026"], include_current_season=False)

    matches_df = football_data.load_training_data(seasons=["2023-2024", "2024-2025", "2025-2026"])
    models = manifest.load_models(matches_df=matches_df)

    assert "over_2_5" in models["scoreline_market_overrides"]
    override_model = models["scoreline_market_overrides"]["over_2_5"]
    assert override_model.context is not None
    # The override model must actually be usable for a live prediction.
    grid = override_model.predict("Arsenal", "Chelsea")
    assert grid.home_win + grid.draw + grid.away_win > 0.99


def test_load_models_requires_matches_df_when_an_override_needs_context(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    manifest.train_all(seasons=["2023-2024", "2024-2025", "2025-2026"], include_current_season=False)

    try:
        manifest.load_models(matches_df=None)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "matches_df" in str(exc)
