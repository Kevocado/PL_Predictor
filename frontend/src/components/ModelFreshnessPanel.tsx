import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { ManifestHistoryEntry, ManifestResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";

interface Props {
  manifest: ManifestResponse;
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
    history?.map((h, index) => ({
      update: `Update ${index + 1}`,
      rps: h.rps,
      brier: h.brier,
      trained_at: h.trained_at,
      n_current_season_matches: h.n_current_season_matches,
      n_train: h.n_train,
    })) ?? [];

  return (
    <div>
      <h2 className="mb-2 text-lg font-semibold text-pl-text">Model freshness</h2>
      <p className="mb-3 max-w-2xl text-xs text-pl-text-faint">
        The model’s training coverage and the live results feed are shown separately. The chart only records a
        held-out validation-score change, so repeated retrains do not create fake visual movement.{" "}
        <InfoTooltip
          text="A live final score can update the standings immediately. It enters the scoreline model only when the stat-complete training source is available, so missing match statistics are never silently treated as zero."
          align="left"
        />
        .
      </p>

      <div className="mb-3 clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
        <span className="text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
          Current season: live results / this model version
        </span>
        <p className="mt-1 text-2xl font-bold text-pl-text">
          {manifest.live_current_season_matches ?? manifest.n_current_season_matches}
          <span className="mx-2 text-pl-text-faint">/</span>
          {manifest.n_current_season_matches}
          <span className="ml-2 text-xs font-normal text-pl-text-faint">
            {manifest.live_current_season_matches !== undefined && manifest.live_current_season_matches !== manifest.n_current_season_matches
              ? "live results detected / stat-complete results included in this model"
              : "live results and model-training coverage currently match"}
          </span>
        </p>
      </div>

      {error && <p className="text-xs text-loss">{error}</p>}

      {history && history.length < 2 && (
        <p className="text-xs text-pl-text-faint">
          Only one held-out score change logged so far — this chart appears once another model update changes Brier or RPS.
        </p>
      )}

      {history && history.length >= 2 && (
        <div className="clip-corner-lg h-56 rounded-xl border border-pl-border bg-pl-850/70 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--color-pl-border)" strokeDasharray="3 3" />
              <XAxis dataKey="update" tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }} />
              <YAxis tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "var(--color-pl-900)", border: "1px solid var(--color-pl-border)", borderRadius: 8 }}
                labelStyle={{ color: "var(--color-pl-text-faint)" }}
                labelFormatter={(label, payload) => {
                  const point = payload[0]?.payload as { trained_at?: string; n_current_season_matches?: number } | undefined;
                  return point?.trained_at
                    ? `${label} · ${point.n_current_season_matches ?? 0} current-season matches in model · ${new Date(point.trained_at).toLocaleString()}`
                    : label;
                }}
                formatter={(value, name) =>
                  typeof value === "number" ? [value.toFixed(4), name === "brier" ? "Brier" : "RPS"] : [value, name]
                }
              />
              <Line type="linear" dataKey="brier" name="brier" stroke="var(--color-pl-pink)" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
