import { useState } from "react";
import { Bar, BarChart, Cell, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { FeatureImportance } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { explainFeatureName, formatFeatureName } from "../lib/featureNames";

const METRIC_INFO: Record<"shap" | "permutation" | "gain", string> = {
  shap: "Each prediction decomposed into exactly how much every feature pushed it up or down from the average (SHAP values), then averaged — keeping the sign — across held-out matches. Pink bars push the prediction up, blue bars push it down; bar length is the size of that average push. Grounded in real per-prediction effects, which usually makes it the most intuitive of the three at a glance.",
  permutation:
    "How much held-out accuracy (R²) drops when this one feature's values are randomly shuffled — the more it drops, the more the model actually relies on it. More trustworthy than gain, but slower to compute.",
  gain: "XGBoost's own internal score for how useful a feature was for splitting decisions while training. Can overstate a feature the model happens to split on often without it truly driving accuracy.",
};

interface Entry {
  raw: string;
  name: string;
  value: number;
}

interface TooltipPayloadItem {
  payload: Entry;
}

function ChartTooltip({ active, payload, metric }: { active?: boolean; payload?: TooltipPayloadItem[]; metric: "shap" | "permutation" | "gain" }) {
  if (!active || !payload || payload.length === 0) return null;
  const entry = payload[0].payload;
  const valueLabel = metric === "shap" ? "Avg. contribution" : metric === "gain" ? "Gain" : "Accuracy drop when shuffled";
  return (
    <div className="max-w-xs rounded-lg border border-pl-border bg-pl-900 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-semibold text-pl-text">{entry.name}</div>
      <div className="mb-1.5 text-pl-text-dim">
        {valueLabel}: <span className="font-semibold text-pl-text">{entry.value >= 0 && metric === "shap" ? "+" : ""}{entry.value.toFixed(4)}</span>
      </div>
      <p className="leading-snug text-pl-text-faint">{explainFeatureName(entry.raw)}</p>
    </div>
  );
}

interface Props {
  title: string;
  importance: FeatureImportance;
  topN?: number;
}

export function FeatureImportanceChart({ title, importance, topN = 10 }: Props) {
  const [metric, setMetric] = useState<"shap" | "permutation" | "gain">("shap");

  const entries: Entry[] = Object.entries(importance[metric])
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, topN)
    .reverse()
    .map(([raw, value]) => ({ raw, name: formatFeatureName(raw), value }));

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
          {title} — top {topN} features
          <InfoTooltip text={METRIC_INFO[metric]} align="left" />
        </h4>
        <div className="flex gap-1 rounded-md border border-pl-border bg-pl-850/60 p-0.5">
          {(["shap", "permutation", "gain"] as const).map((m) => (
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
              content={(props) => <ChartTooltip active={props.active} payload={props.payload as unknown as TooltipPayloadItem[] | undefined} metric={metric} />}
              cursor={{ fill: "var(--color-pl-border)", opacity: 0.15 }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {entries.map((entry, i) => (
                <Cell key={i} fill={metric === "shap" && entry.value < 0 ? "var(--color-pl-cyan)" : "var(--color-pl-pink)"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {metric === "shap" && (
        <div className="mt-2 flex items-center gap-4 text-[10px] text-pl-text-faint">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={{ background: "var(--color-pl-pink)" }} /> pushes prediction up
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={{ background: "var(--color-pl-cyan)" }} /> pushes prediction down
          </span>
        </div>
      )}
    </div>
  );
}
