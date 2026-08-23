import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { CalibrationResponse, ManifestResponse } from "../types";
import { CalibrationPanel } from "../components/CalibrationPanel";
import { BacktestPanel } from "../components/BacktestPanel";
import { InfoTooltip } from "../components/InfoTooltip";
import { FeatureImportanceChart } from "../components/FeatureImportanceChart";
import { ModelFreshnessPanel } from "../components/ModelFreshnessPanel";
import { GLOSSARY } from "../lib/glossary";

export function CalibrationPage() {
  const [calibration, setCalibration] = useState<CalibrationResponse | null>(null);
  const [manifest, setManifest] = useState<ManifestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);

  const load = () => {
    setError(null);
    return Promise.all([api.calibration(), api.manifest()])
      .then(([cal, man]) => {
        setCalibration(cal);
        setManifest(man);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const busy = loading || retraining;

  const retrain = async () => {
    setRetraining(true);
    setError(null);
    try {
      await api.retrain();
      // Keep the busy state through the post-retrain refetch too — this is
      // what "Retrain" actually pulling the latest result means end to
      // end, not just the training step. Without this the button briefly
      // looked clickable again while stale numbers were still on screen.
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-pl-text">Model calibration</h2>
          {manifest && (
            <p className="mt-0.5 text-xs text-pl-text-faint">
              Trained {new Date(manifest.trained_at).toLocaleString()} on {manifest.seasons[0]}–
              {manifest.seasons[manifest.seasons.length - 1]} · {manifest.n_train} matches · scoreline model:{" "}
              {manifest.scoreline.chosen_model.replace("_", " ")}
            </p>
          )}
        </div>
        <button
          onClick={retrain}
          disabled={busy}
          title={loading && !retraining ? "Still loading the current numbers — hang on a moment" : undefined}
          className="rounded-lg bg-pl-pink px-4 py-2 text-sm font-semibold text-white transition hover:bg-pl-pink-soft disabled:opacity-50"
        >
          {retraining ? "Retraining… (~1-2 min)" : loading ? "Loading…" : "Retrain models"}
        </button>
      </div>

      {error && <div className="rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>}

      {loading && !calibration && !error && (
        <div className="py-16 text-center text-pl-text-faint">Loading calibration data…</div>
      )}

      {calibration && <CalibrationPanel data={calibration} />}

      {manifest && <ModelFreshnessPanel manifest={manifest} />}

      {manifest && (
        <div>
          <h2 className="mb-3 text-lg font-semibold text-pl-text">Scoreline model comparison</h2>
          <p className="mb-3 max-w-2xl text-xs text-pl-text-faint">
            Every retrain fits three candidate scoreline models on the same training data and picks whichever scores
            best (lowest RPS <InfoTooltip text={GLOSSARY.rps} align="left" />) on a held-out season it never trained
            on. The winner is what actually powers every prediction in the app right now.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[420px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-pl-border text-left text-xs uppercase tracking-wide text-pl-text-faint">
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2 pr-4">
                    RPS <InfoTooltip text={GLOSSARY.rps} align="left" />
                  </th>
                  <th className="py-2 pr-4">
                    Brier <InfoTooltip text={GLOSSARY.brier} align="left" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    { key: "dixon_coles", label: "Dixon-Coles", data: manifest.scoreline.dixon_coles },
                    { key: "bivariate_poisson", label: "Bivariate Poisson", data: manifest.scoreline.bivariate_poisson },
                    { key: "ml_scoreline", label: "ML scoreline (XGBoost)", data: manifest.scoreline.ml_scoreline },
                  ] as const
                ).map(({ key, label, data }) => {
                  const chosen = manifest.scoreline.chosen_model === key;
                  return (
                    <tr
                      key={key}
                      className={`border-b border-pl-border/50 ${chosen ? "bg-pl-pink/10" : ""}`}
                    >
                      <td className="py-2 pr-4 font-medium text-pl-text">
                        {label}
                        {chosen && (
                          <span className="ml-2 rounded bg-pl-pink px-1.5 py-0.5 text-[10px] font-semibold uppercase text-white">
                            in use
                          </span>
                        )}
                      </td>
                      <td className="py-2 pr-4 text-pl-text-dim">{data.metrics.rps?.toFixed(4)}</td>
                      <td className="py-2 pr-4 text-pl-text-dim">{data.metrics.brier?.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {manifest && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
            <span className="text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Corners model</span>
            <p className="mt-2 text-sm text-pl-text-dim">
              MAE <InfoTooltip text={GLOSSARY.mae} align="left" />{" "}
              <span className="font-semibold text-pl-text">{manifest.corners.metrics.mae?.toFixed(2)}</span>{" "}
              &middot; mean actual{" "}
              <span className="font-semibold text-pl-text">{manifest.corners.metrics.mean_actual?.toFixed(1)}</span>{" "}
              &middot; dispersion <InfoTooltip text={GLOSSARY.dispersion} align="left" />{" "}
              <span className="font-semibold text-pl-text">{manifest.corners.dispersion?.toFixed(2)}</span>
            </p>
          </div>
          <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
            <span className="text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Cards model</span>
            <p className="mt-2 text-sm text-pl-text-dim">
              MAE <InfoTooltip text={GLOSSARY.mae} align="left" />{" "}
              <span className="font-semibold text-pl-text">{manifest.cards.metrics.mae?.toFixed(2)}</span>{" "}
              &middot; mean actual{" "}
              <span className="font-semibold text-pl-text">{manifest.cards.metrics.mean_actual?.toFixed(1)}</span>{" "}
              &middot; dispersion <InfoTooltip text={GLOSSARY.dispersion} align="left" />{" "}
              <span className="font-semibold text-pl-text">{manifest.cards.dispersion?.toFixed(2)}</span>
            </p>
          </div>
        </div>
      )}

      {manifest?.scoreline.chosen_model === "ml_scoreline" &&
        manifest.scoreline.ml_scoreline.importance_home &&
        manifest.scoreline.ml_scoreline.importance_away && (
          <div>
            <h2 className="mb-3 text-lg font-semibold text-pl-text">What drives the scoreline predictions</h2>
            <p className="mb-3 max-w-2xl text-xs text-pl-text-faint">
              The live scoreline model is two XGBoost regressors — one predicts the home team's expected goals, one
              the away team's — each ranked separately below. Toggle between Permutation (how much held-out accuracy
              drops when a feature is shuffled — more trustworthy) and Gain (XGBoost's own internal split-usefulness
              score — can overstate a feature the model happens to split on often).
            </p>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <FeatureImportanceChart title="Home goals model" importance={manifest.scoreline.ml_scoreline.importance_home} />
              <FeatureImportanceChart title="Away goals model" importance={manifest.scoreline.ml_scoreline.importance_away} />
            </div>
          </div>
        )}

      {manifest?.corners.importance && manifest?.cards.importance && (
        <div>
          <h2 className="mb-3 text-lg font-semibold text-pl-text">What drives the corners/cards predictions</h2>
          <p className="mb-3 max-w-2xl text-xs text-pl-text-faint">
            {manifest.scoreline.chosen_model === "ml_scoreline"
              ? "Corners and cards are separate XGBoost models sharing the same rolling-form/Elo/xG features as the scoreline model above."
              : "The scoreline (1X2) model doesn't have \"features\" to rank this way right now — it's a Dixon-Coles/Bivariate-Poisson model that fits each team's attack/defence strength directly (see Power Rankings in the Data Hub for that model's own transparency view). Corners and cards are separate XGBoost models trained on rolling-form features, so their importance can be ranked directly."}
          </p>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <FeatureImportanceChart title="Corners" importance={manifest.corners.importance} />
            <FeatureImportanceChart title="Cards" importance={manifest.cards.importance} />
          </div>
        </div>
      )}

      <div>
        <h2 className="mb-3 text-lg font-semibold text-pl-text">Value-bet backtest</h2>
        <BacktestPanel />
      </div>
    </div>
  );
}
