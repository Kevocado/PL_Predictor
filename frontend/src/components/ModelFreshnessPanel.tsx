import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { ManifestHistoryEntry, ManifestResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";

interface Props {
  manifest: ManifestResponse;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

export function ModelFreshnessPanel({ manifest }: Props) {
  const [history, setHistory] = useState<ManifestHistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .manifestHistory()
      .then((r) => setHistory(r.history))
      .catch((e) => setError(e.message));
  }, [manifest.trained_at]); // re-fetch after a retrain, since that's what appends a new point

  const chartData =
    history?.map((h) => ({
      label: formatDate(h.trained_at),
      rps: h.rps,
      n_current_season_matches: h.n_current_season_matches,
      n_train: h.n_train,
    })) ?? [];

  return (
    <div>
      <h2 className="mb-2 text-lg font-semibold text-pl-text">Model freshness</h2>
      <p className="mb-3 max-w-2xl text-xs text-pl-text-faint">
        Every retrain folds whatever's been played of the current season into training (never into the fixed
        held-out validation season above, so this stays a fair comparison over time){" "}
        <InfoTooltip
          text="The underlying model doesn't retrain itself automatically — click 'Retrain models' periodically (e.g. weekly, or after a batch of fixtures) to pull in newly-played results."
          align="left"
        />
        .
      </p>

      <div className="mb-3 clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
        <span className="text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
          Current-season matches in this training run
        </span>
        <p className="mt-1 text-2xl font-bold text-pl-text">
          {manifest.n_current_season_matches}
          <span className="ml-2 text-xs font-normal text-pl-text-faint">
            {manifest.n_current_season_matches === 0
              ? "— season hasn't produced results yet, or the model hasn't been retrained since it did"
              : "of this season's played matches, on top of 8 completed seasons"}
          </span>
        </p>
      </div>

      {error && <p className="text-xs text-loss">{error}</p>}

      {history && history.length < 2 && (
        <p className="text-xs text-pl-text-faint">
          Only one retrain logged so far — this chart fills in as you retrain again through the season, showing
          whether accuracy trends up as more of this season's own results get folded in.
        </p>
      )}

      {history && history.length >= 2 && (
        <div className="clip-corner-lg h-56 rounded-xl border border-pl-border bg-pl-850/70 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--color-pl-border)" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }} />
              <YAxis tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "var(--color-pl-900)", border: "1px solid var(--color-pl-border)", borderRadius: 8 }}
                labelStyle={{ color: "var(--color-pl-text-faint)" }}
                formatter={(value, name) =>
                  name === "rps" ? [typeof value === "number" ? value.toFixed(4) : value, "RPS"] : [value, name]
                }
              />
              <Line type="monotone" dataKey="rps" stroke="var(--color-pl-pink)" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
