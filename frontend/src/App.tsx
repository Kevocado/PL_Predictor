import { useEffect, useState, type ReactNode } from "react";
import { FixturesPage } from "./pages/FixturesPage";
import { CalibrationPage } from "./pages/CalibrationPage";
import { ModelSummaryPage } from "./pages/ModelSummaryPage";
import { DataHubPage } from "./pages/DataHubPage";
import { FPLPage } from "./pages/FPLPage";
import { PUBLIC_MODE } from "./lib/publicMode";
import { api } from "./api/client";
import { AppFrame } from "./predictor-ui";
import { SITES } from "./lib/sites";

type Tab =
  | "fixtures"
  | "teams"
  | "players"
  | "power-rankings"
  | "projected-table"
  | "track-record"
  | "fpl"
  | "calibration";

const TABS: { id: Tab; label: string }[] = [
  { id: "fixtures", label: "Fixtures" },
  { id: "teams", label: "Teams" },
  { id: "players", label: "Players" },
  { id: "power-rankings", label: "Power rankings" },
  { id: "projected-table", label: "Projected table" },
  { id: "track-record", label: "Track record" },
  { id: "fpl", label: "FPL" },
  { id: "calibration", label: PUBLIC_MODE ? "Model" : "Calibration & Backtest" },
];

// Data Hub panels each have their own tab now; the aggregate "Data Hub" tab
// is dropped. Old #hub deep links redirect to Teams, the first promoted tab.
const PANEL_FOR_TAB: Record<Tab, () => ReactNode> = {
  fixtures: () => <FixturesPage />,
  teams: () => <DataHubPage initialTab="Team Hub" />,
  players: () => <DataHubPage initialTab="Player Hub" />,
  "power-rankings": () => <DataHubPage initialTab="Power Rankings" />,
  "projected-table": () => <DataHubPage initialTab="Projected Table" />,
  "track-record": () => <DataHubPage initialTab="Track Record" />,
  fpl: () => <FPLPage />,
  calibration: () => (PUBLIC_MODE ? <ModelSummaryPage /> : <CalibrationPage />),
};

function App() {
  // Hash tabs: read the initial tab from the URL, keep back/forward working.
  // The initial tab is mounted straight away so its panel renders on load.
  const initialHash = window.location.hash.slice(1);
  const initialTab = TABS.some((t) => t.id === initialHash) ? (initialHash as Tab) : "fixtures";
  const [tab, setTab] = useState<Tab>(initialTab);
  const [mountedTabs, setMountedTabs] = useState<Set<Tab>>(() => new Set([initialTab]));

  useEffect(() => {
    const onHashChange = () => {
      const next = window.location.hash.slice(1);
      if (TABS.some((t) => t.id === next)) setTab(next as Tab);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const selectTab = (next: string) => {
    const tab = next as Tab;
    setTab(tab);
    setMountedTabs((current) => new Set(current).add(tab));
    window.location.hash = `#${tab}`;
  };

  // Let the first Fixtures paint, then make the dashboard data available
  // in the background. The API client reuses these promises when the user
  // opens Data Hub or Calibration, instead of issuing a second cold call.
  useEffect(() => {
    const id = window.setTimeout(() => { void api.preloadDashboards(); }, 750);
    return () => window.clearTimeout(id);
  }, []);

  // Tabs mount on first visit, then stay mounted (hidden), so switching tabs
  // never re-fetches data that's already loaded.
  return (
    <AppFrame
      sport="pl"
      sportName="PL"
      sites={SITES}
      tabs={TABS.map(({ id, label }) => ({ id, label }))}
      activeTab={tab}
      paperTradingHref="https://40-160-91-131.sslip.io/paper.html"
      onTab={selectTab}
    >
      {mountedTabs.has("fixtures") && <div data-testid="page-fixtures" hidden={tab !== "fixtures"}>{PANEL_FOR_TAB.fixtures()}</div>}
      {mountedTabs.has("teams") && (
        <div data-testid="page-teams" hidden={tab !== "teams"}>{PANEL_FOR_TAB.teams()}</div>
      )}
      {mountedTabs.has("players") && (
        <div data-testid="page-players" hidden={tab !== "players"}>{PANEL_FOR_TAB.players()}</div>
      )}
      {mountedTabs.has("power-rankings") && (
        <div data-testid="page-power-rankings" hidden={tab !== "power-rankings"}>{PANEL_FOR_TAB["power-rankings"]()}</div>
      )}
      {mountedTabs.has("projected-table") && (
        <div data-testid="page-projected-table" hidden={tab !== "projected-table"}>{PANEL_FOR_TAB["projected-table"]()}</div>
      )}
      {mountedTabs.has("track-record") && (
        <div data-testid="page-track-record" hidden={tab !== "track-record"}>{PANEL_FOR_TAB["track-record"]()}</div>
      )}
      {mountedTabs.has("fpl") && <div data-testid="page-fpl" hidden={tab !== "fpl"}>{PANEL_FOR_TAB.fpl()}</div>}
      {mountedTabs.has("calibration") && (
        <div data-testid="page-calibration" hidden={tab !== "calibration"}>{PANEL_FOR_TAB.calibration()}</div>
      )}
    </AppFrame>
  );
}

export default App;
