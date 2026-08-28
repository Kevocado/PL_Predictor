import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ManifestModelMetrics, ManifestResponse } from "../types";
import { InfoTooltip } from "../components/InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

const MODEL_LABELS: Record<string, string> = {
  dixon_coles: "Dixon-Coles",
  bivariate_poisson: "Bivariate Poisson",
  ml_scoreline: "XGBoost (machine learning)",
  covariate_poisson: "Covariate Poisson",
};

// Public read-only counterpart to CalibrationPage — headline numbers and a
// plain-English description only, no retrain control, feature-importance
// internals, or betting/backtest panels (those stay on the private app).
export function ModelSummaryPage() {
  const [manifest, setManifest] = useState<ManifestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.manifest().then(setManifest).catch((e) => setError(e.message));
  }, []);

  if (error) {
    return <div className="rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>;
  }
  if (!manifest) {
    return <div className="py-16 text-center text-pl-text-faint">Loading model info…</div>;
  }

  const chosenLabel = MODEL_LABELS[manifest.scoreline.chosen_model] ?? manifest.scoreline.chosen_model;
  const scorelineModels: Record<string, ManifestModelMetrics | undefined> = {
    dixon_coles: manifest.scoreline.dixon_coles,
    bivariate_poisson: manifest.scoreline.bivariate_poisson,
    ml_scoreline: manifest.scoreline.ml_scoreline,
  };
  const chosenMetrics = scorelineModels[manifest.scoreline.chosen_model]?.metrics;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-pl-text">How the predictions are made</h2>
        <p className="mt-2 max-w-2xl text-sm text-pl-text-dim">
          PL Predictor fits several statistical and machine-learning models on{" "}
          {manifest.seasons.length} seasons of Premier League results ({manifest.seasons[0]}–
          {manifest.seasons[manifest.seasons.length - 1]}), tests each one on a held-out season it
          never trained on, and automatically serves whichever scores best. Right now that's{" "}
          <span className="font-semibold text-pl-text">{chosenLabel}</span> for match outcomes and
          scorelines, with separate models for corners and cards.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
          <span className="text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
            Match outcome accuracy
          </span>
          <p className="mt-2 text-sm text-pl-text-dim">
            RPS <InfoTooltip text={GLOSSARY.rps} align="left" />{" "}
            <span className="font-semibold text-pl-text">{chosenMetrics?.rps?.toFixed(4) ?? "—"}</span>
            <br />
            Brier <InfoTooltip text={GLOSSARY.brier} align="left" />{" "}
            <span className="font-semibold text-pl-text">{chosenMetrics?.brier?.toFixed(4) ?? "—"}</span>
          </p>
        </div>
        <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
          <span className="text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Corners model</span>
          <p className="mt-2 text-sm text-pl-text-dim">
            MAE <InfoTooltip text={GLOSSARY.mae} align="left" />{" "}
            <span className="font-semibold text-pl-text">{manifest.corners.metrics.mae?.toFixed(2) ?? "—"}</span>
          </p>
        </div>
        <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
          <span className="text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Cards model</span>
          <p className="mt-2 text-sm text-pl-text-dim">
            MAE <InfoTooltip text={GLOSSARY.mae} align="left" />{" "}
            <span className="font-semibold text-pl-text">{manifest.cards.metrics.mae?.toFixed(2) ?? "—"}</span>
          </p>
        </div>
      </div>

      <p className="text-xs text-pl-text-faint">
        Last trained {new Date(manifest.trained_at).toLocaleString()} on {manifest.n_train} matches.
      </p>
    </div>
  );
}
