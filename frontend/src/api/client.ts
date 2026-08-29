import type {
  BacktestResponse,
  CalibrationResponse,
  CurrentGameweekResponse,
  FixtureDetail,
  FixturePlayers,
  FixturePlayerReview,
  FixtureSummary,
  ManifestHistoryResponse,
  ManifestResponse,
  ProjectedTableResponse,
  RankingsResponse,
  PlayerHubResponse,
  ScorerAccuracyResponse,
  SquadContinuityResponse,
  TeamHubResponse,
  TrackRecordResponse,
  ValueBetTrackRecordResponse,
  WalkForwardBettingResponse,
} from "../types";

// Derived from wherever this page was loaded from, not hardcoded to
// "localhost" — that would resolve to the *viewing device* (a phone on
// Tailscale, say), not the machine actually running the backend.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? `${window.location.protocol}//${window.location.hostname}:8000/api`;

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post<T>(path: string, params?: Record<string, string>): Promise<T> {
  const query = params ? `?${new URLSearchParams(params)}` : "";
  const res = await fetch(`${BASE_URL}${path}${query}`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  fixtures: () => get<FixtureSummary[]>("/fixtures"),
  currentGameweek: (gameweek?: number) =>
    get<CurrentGameweekResponse>(gameweek ? `/fixtures/gameweek?gameweek=${gameweek}` : "/fixtures/gameweek"),
  fixtureDetail: (eventId: string) => get<FixtureDetail>(`/fixtures/${eventId}`),
  fixturePlayers: (eventId: string) => get<FixturePlayers>(`/fixtures/${eventId}/players`),
  fixturePlayerReview: (eventId: string) => get<FixturePlayerReview | null>(`/fixtures/${eventId}/player-review`),
  manifest: () => get<ManifestResponse>("/manifest"),
  manifestHistory: () => get<ManifestHistoryResponse>("/manifest/history"),
  calibration: () => get<CalibrationResponse>("/calibration"),
  backtest: (staking: "kelly" | "flat" = "kelly") => post<BacktestResponse>("/backtest", { staking }),
  valueBetTrackRecord: (staking: "kelly" | "flat" = "kelly") =>
    get<ValueBetTrackRecordResponse>(`/value-bets/track-record?staking=${staking}`),
  walkForwardValueBetValidation: () => get<WalkForwardBettingResponse>("/value-bets/walk-forward"),
  retrain: () => post<ManifestResponse>("/retrain"),
  refreshOdds: () => post<{ status: string }>("/refresh-odds"),
  refreshFixtures: () => post<{ status: string }>("/refresh-fixtures"),
  rankings: () => get<RankingsResponse>("/hub/rankings"),
  projectedTable: () => get<ProjectedTableResponse>("/hub/table"),
  trackRecord: () => get<TrackRecordResponse>("/hub/track-record"),
  teamHub: () => get<TeamHubResponse>("/hub/teams"),
  playerHub: () => get<PlayerHubResponse>("/hub/players"),
  scorerTrackRecord: () => get<ScorerAccuracyResponse>("/scorer-track-record"),
  squadContinuity: () => get<SquadContinuityResponse>("/squad-continuity"),
};

export class ApiError extends Error {}
