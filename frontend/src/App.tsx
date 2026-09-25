import { useEffect, useState } from "react";
import { FixturesPage } from "./pages/FixturesPage";
import { CalibrationPage } from "./pages/CalibrationPage";
import { ModelSummaryPage } from "./pages/ModelSummaryPage";
import { DataHubPage } from "./pages/DataHubPage";
import { FPLPage } from "./pages/FPLPage";
import { PUBLIC_MODE } from "./lib/publicMode";
import { api } from "./api/client";
import { AppFrame } from "./predictor-ui";
import { SITES } from "./lib/sites";

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

  const tabs = [
    { id: "fixtures", label: "Fixtures" },
    { id: "hub", label: "Data Hub" },
    { id: "fpl", label: "FPL" },
    { id: "calibration", label: PUBLIC_MODE ? "Model" : "Calibration & Backtest" },
  ];

  // Tabs mount on first visit, then stay mounted (hidden). Dashboard data
  // itself is warmed after the Fixtures page has painted.
  return (
    <AppFrame sport="pl" sportName="PL" sites={SITES} tabs={tabs} activeTab={tab} onTab={(id) => selectTab(id as Tab)}>
      {mountedTabs.has("fixtures") && (
        <div data-testid="page-fixtures" hidden={tab !== "fixtures"}>
          <FixturesPage />
        </div>
      )}
      {mountedTabs.has("hub") && (
        <div data-testid="page-hub" hidden={tab !== "hub"}>
          <DataHubPage />
        </div>
      )}
      {mountedTabs.has("fpl") && (
        <div data-testid="page-fpl" hidden={tab !== "fpl"}>
          <FPLPage />
        </div>
      )}
      {mountedTabs.has("calibration") && (
        <div data-testid="page-calibration" hidden={tab !== "calibration"}>
          {PUBLIC_MODE ? <ModelSummaryPage /> : <CalibrationPage />}
        </div>
      )}
    </AppFrame>
  );
}

export default App;
