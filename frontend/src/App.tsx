import { useEffect, useState } from "react";
import { FixturesPage } from "./pages/FixturesPage";
import { CalibrationPage } from "./pages/CalibrationPage";
import { ModelSummaryPage } from "./pages/ModelSummaryPage";
import { DataHubPage } from "./pages/DataHubPage";
import { FPLPage } from "./pages/FPLPage";
import { PUBLIC_MODE } from "./lib/publicMode";
import { api } from "./api/client";

type Tab = "fixtures" | "calibration" | "hub" | "fpl";

function App() {
  const [tab, setTab] = useState<Tab>("fixtures");
  const [mountedTabs, setMountedTabs] = useState<Set<Tab>>(() => new Set(["fixtures"]));

  const selectTab = (next: Tab) => {
    setTab(next);
    setMountedTabs((current) => new Set(current).add(next));
  };

  useEffect(() => {
    // Let the first Fixtures paint, then make the dashboard data available
    // in the background. The API client reuses these promises when the user
    // opens Data Hub or Calibration, instead of issuing a second cold call.
    const id = window.setTimeout(() => { void api.preloadDashboards(); }, 750);
    return () => window.clearTimeout(id);
  }, []);

  return (
    <div className="mx-auto min-h-screen max-w-7xl px-6 py-8">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="clip-corner flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-pl-pink to-pl-600 font-black text-white">
            PL
          </div>
          <div>
            <h1 className="text-xl font-extrabold tracking-tight text-pl-text">PL Predictor</h1>
            <p className="text-xs text-pl-text-faint">Match outcomes, scorelines &amp; betting markets</p>
          </div>
        </div>
        <nav className="flex flex-wrap gap-1 rounded-lg border border-pl-border bg-pl-850/60 p-1">
          {(
            [
              ["fixtures", "Fixtures"],
              ["hub", "Data Hub"],
              ["fpl", "FPL"],
              ["calibration", PUBLIC_MODE ? "Model" : "Calibration & Backtest"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => selectTab(key)}
              className={`rounded-md px-3.5 py-1.5 text-sm font-medium transition ${
                tab === key ? "bg-pl-pink text-white" : "text-pl-text-dim hover:text-pl-text"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {/* Tabs mount on first visit, then stay mounted. Dashboard data itself
          is warmed after the Fixtures page has painted. */}
      <main>
        {mountedTabs.has("fixtures") && <div style={{ display: tab === "fixtures" ? "block" : "none" }}>
          <FixturesPage />
        </div>}
        {mountedTabs.has("hub") && <div style={{ display: tab === "hub" ? "block" : "none" }}>
          <DataHubPage />
        </div>}
        {mountedTabs.has("fpl") && <div style={{ display: tab === "fpl" ? "block" : "none" }}>
          <FPLPage />
        </div>}
        {mountedTabs.has("calibration") && <div style={{ display: tab === "calibration" ? "block" : "none" }}>
          {PUBLIC_MODE ? <ModelSummaryPage /> : <CalibrationPage />}
        </div>}
      </main>
    </div>
  );
}

export default App;
