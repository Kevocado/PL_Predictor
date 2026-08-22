import { useState } from "react";
import { FixturesPage } from "./pages/FixturesPage";
import { CalibrationPage } from "./pages/CalibrationPage";
import { DataHubPage } from "./pages/DataHubPage";

type Tab = "fixtures" | "calibration" | "hub";

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
              ["calibration", "Calibration & Backtest"],
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

      <main>
        {tab === "fixtures" && <FixturesPage />}
        {tab === "hub" && <DataHubPage />}
        {tab === "calibration" && <CalibrationPage />}
      </main>
    </div>
  );
}

export default App;
