import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { FeatureImportance } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { formatFeatureName } from "../lib/featureNames";

const METRIC_INFO: Record<"permutation" | "gain", string> = {
  permutation:
    "How much held-out accuracy (R²) drops when this one feature's values are randomly shuffled — the more it drops, the more the model actually relies on it. More trustworthy than gain, but slower to compute.",
  gain: "XGBoost's own internal score for how useful a feature was for splitting decisions while training. Can overstate a feature the model happens to split on often without it truly driving accuracy.",
};

interface Props {
  title: string;
  importance: FeatureImportance;
  topN?: number;
}

export function FeatureImportanceChart({ title, importance, topN = 10 }: Props) {
  const [metric, setMetric] = useState<"permutation" | "gain">("permutation");

  const entries = Object.entries(importance[metric])
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .reverse()
    .map(([name, value]) => ({ name: formatFeatureName(name), value }));

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
          {title} — top {topN} features
          <InfoTooltip text={METRIC_INFO[metric]} align="left" />
        </h4>
        <div className="flex gap-1 rounded-md border border-pl-border bg-pl-850/60 p-0.5">
          {(["permutation", "gain"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`rounded px-2 py-0.5 text-[11px] font-medium capitalize transition ${
                metric === m ? "bg-pl-pink text-white" : "text-pl-text-faint hover:text-pl-text"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="h-72 rounded-xl border border-pl-border bg-pl-850/70 p-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={entries} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid stroke="var(--color-pl-border)" strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fill: "var(--color-pl-text-faint)", fontSize: 10 }} />
            <YAxis
              type="category"
              dataKey="name"
              width={200}
              tick={{ fill: "var(--color-pl-text-dim)", fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{ background: "var(--color-pl-900)", border: "1px solid var(--color-pl-border)", borderRadius: 8 }}
              labelStyle={{ color: "var(--color-pl-text)" }}
              formatter={(value) => (typeof value === "number" ? value.toFixed(4) : value)}
            />
            <Bar dataKey="value" fill="var(--color-pl-pink)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
