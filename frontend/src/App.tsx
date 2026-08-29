import { useState } from "react";
import { FixturesPage } from "./pages/FixturesPage";
import { CalibrationPage } from "./pages/CalibrationPage";
import { ModelSummaryPage } from "./pages/ModelSummaryPage";
import { DataHubPage } from "./pages/DataHubPage";
import { FPLPage } from "./pages/FPLPage";
import { PUBLIC_MODE } from "./lib/publicMode";

type Tab = "fixtures" | "calibration" | "hub" | "fpl";

function App() {
  const [tab, setTab] = useState<Tab>("fixtures");

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
              onClick={() => setTab(key)}
              className={`rounded-md px-3.5 py-1.5 text-sm font-medium transition ${
                tab === key ? "bg-pl-pink text-white" : "text-pl-text-dim hover:text-pl-text"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {/* All three pages mount immediately on page load (not on first
          click) and stay mounted for the rest of the session — hidden with
          CSS rather than unmounted, so switching tabs never re-fetches data
          that's already loaded. */}
      <main>
        <div style={{ display: tab === "fixtures" ? "block" : "none" }}>
          <FixturesPage />
        </div>
        <div style={{ display: tab === "hub" ? "block" : "none" }}>
          <DataHubPage />
        </div>
        <div style={{ display: tab === "fpl" ? "block" : "none" }}>
          <FPLPage />
        </div>
        <div style={{ display: tab === "calibration" ? "block" : "none" }}>
          {PUBLIC_MODE ? <ModelSummaryPage /> : <CalibrationPage />}
        </div>
      </main>
    </div>
  );
}

export default App;
