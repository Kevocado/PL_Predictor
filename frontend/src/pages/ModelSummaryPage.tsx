import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ManifestModelMetrics, ManifestResponse } from "../types";
import { InfoTooltip } from "../components/InfoTooltip";
import { FeatureImportanceChart } from "../components/FeatureImportanceChart";
import { GLOSSARY } from "../lib/glossary";

const MODEL_LABELS: Record<string, string> = {
  dixon_coles: "Dixon-Coles",
  bivariate_poisson: "Bivariate Poisson",
  ml_scoreline: "XGBoost (machine learning)",
  covariate_poisson: "Covariate Poisson",
};

// Public read-only counterpart to CalibrationPage — headline numbers, a
// fuller plain-English explanation, and the same feature-importance charts,
// but still no retrain control or betting/backtest panels (those stay on
// the private app).
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
          PL Predictor fits three candidate models for match outcomes and scorelines — two
          classical statistical models (Dixon-Coles and Bivariate Poisson, which estimate each
          team's attacking/defensive strength directly) and one XGBoost machine-learning model
          (which learns from rolling form, Elo/Pi ratings, expected goals, rest days, and head-to-head
          history) — on {manifest.seasons.length} seasons of Premier League results (
          {manifest.seasons[0]}–{manifest.seasons[manifest.seasons.length - 1]}). Every retrain
          tests all three on a held-out season none of them trained on, and automatically serves
          whichever scores best on that test — right now that's{" "}
          <span className="font-semibold text-pl-text">{chosenLabel}</span>. Corners and cards get
          their own separate models trained the same way.
        </p>
        <p className="mt-3 max-w-2xl text-sm text-pl-text-dim">
          Every feature the models see is computed strictly from information available{" "}
          <em>before</em> kickoff — no result ever leaks into its own prediction. New candidate
          features are only added if they measurably improve held-out accuracy first; several have
          been tried and rejected for not clearing that bar.
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

      {manifest.scoreline.chosen_model === "ml_scoreline" &&
        manifest.scoreline.ml_scoreline.importance_home &&
        manifest.scoreline.ml_scoreline.importance_away && (
          <div>
            <h2 className="mb-3 text-lg font-semibold text-pl-text">What drives the scoreline predictions</h2>
            <p className="mb-3 max-w-2xl text-xs text-pl-text-faint">
              The live scoreline model is two XGBoost regressors — one predicts the home team's
              expected goals, one the away team's. Toggle between Permutation (how much held-out
              accuracy drops when a feature is shuffled — more trustworthy) and Gain (XGBoost's own
              internal split-usefulness score — can overstate a feature the model happens to split
              on often).
            </p>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <FeatureImportanceChart title="Home goals model" importance={manifest.scoreline.ml_scoreline.importance_home} />
              <FeatureImportanceChart title="Away goals model" importance={manifest.scoreline.ml_scoreline.importance_away} />
            </div>
          </div>
        )}

      <p className="text-xs text-pl-text-faint">
        Last trained {new Date(manifest.trained_at).toLocaleString()} on {manifest.n_train} matches.
      </p>
    </div>
  );
}
