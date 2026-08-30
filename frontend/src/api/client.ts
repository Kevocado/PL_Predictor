import type {
  BacktestResponse,
  CalibrationResponse,
  CurrentGameweekResponse,
  FixtureDetail,
  FixturePlayers,
  FixturePlayerReview,
  FixtureSummary,
  FPLProjectionsResponse,
  FPLRecommendation,
  FPLSquadResponse,
  FPLTransfersResponse,
  ManifestHistoryResponse,
  ManifestResponse,
  ProjectedTableResponse,
  RankingsResponse,
  PlayerHubResponse,
  ScorerAccuracyResponse,
  TeamHubResponse,
  TrackRecordResponse,
  ValueBetTrackRecordResponse,
  WalkForwardBettingResponse,
} from "../types";

// Same-origin is robust for both the production FastAPI static site and the
// Vite development proxy.  Set VITE_API_BASE_URL only for a deliberately
// separate API deployment.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";
const READ_TIMEOUT_MS = 20_000;
const READ_CACHE_TTL_MS = 45_000;
const readCache = new Map<string, { expiresAt: number; promise: Promise<unknown> }>();

async function fetchRead<T>(path: string): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), READ_TIMEOUT_MS);
    try {
      const res = await fetch(`${BASE_URL}${path}`, { signal: controller.signal });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new ApiError(body.detail ?? `${res.status} ${res.statusText}`);
      }
      return res.json();
    } catch (error) {
      lastError = error;
      if (attempt === 0 && !(error instanceof ApiError)) continue;
    } finally {
      window.clearTimeout(timeout);
    }
  }
  if (lastError instanceof DOMException && lastError.name === "AbortError") {
    throw new ApiError("The API is still preparing data. Please retry in a moment.");
  }
  throw new ApiError("Could not reach the API. Check that the backend is running, then retry.");
}

async function get<T>(path: string, force = false): Promise<T> {
  const cached = readCache.get(path);
  if (!force && cached && cached.expiresAt > Date.now()) {
    return cached.promise as Promise<T>;
  }

  const promise = fetchRead<T>(path);
  readCache.set(path, { expiresAt: Date.now() + READ_CACHE_TTL_MS, promise });
  void promise.catch(() => {
    if (readCache.get(path)?.promise === promise) readCache.delete(path);
  });
  return promise;
}

function clearReadCache(): void {
  readCache.clear();
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

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail ?? `${res.status} ${res.statusText}`);
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
  manifest: (force = false) => get<ManifestResponse>("/manifest", force),
  manifestHistory: (force = false) => get<ManifestHistoryResponse>("/manifest/history", force),
  calibration: (force = false) => get<CalibrationResponse>("/calibration", force),
  backtest: (staking: "kelly" | "flat" = "kelly") => post<BacktestResponse>("/backtest", { staking }),
  valueBetTrackRecord: (staking: "kelly" | "flat" = "kelly") =>
    get<ValueBetTrackRecordResponse>(`/value-bets/track-record?staking=${staking}`),
  walkForwardValueBetValidation: () => get<WalkForwardBettingResponse>("/value-bets/walk-forward"),
  retrain: () => post<ManifestResponse>("/retrain"),
  refreshOdds: async () => {
    const result = await post<{ status: string }>("/refresh-odds");
    clearReadCache();
    return result;
  },
  refreshFixtures: async () => {
    const result = await post<{ status: string }>("/refresh-fixtures");
    clearReadCache();
    return result;
  },
  rankings: (force = false) => get<RankingsResponse>("/hub/rankings", force),
  projectedTable: (force = false) => get<ProjectedTableResponse>("/hub/table", force),
  trackRecord: (force = false) => get<TrackRecordResponse>("/hub/track-record", force),
  teamHub: (force = false) => get<TeamHubResponse>("/hub/teams", force),
  playerHub: (force = false) => get<PlayerHubResponse>("/hub/players", force),
  scorerTrackRecord: (force = false) => get<ScorerAccuracyResponse>("/scorer-track-record", force),
  preloadDashboards: () => Promise.allSettled([
    get<RankingsResponse>("/hub/rankings"),
    get<ProjectedTableResponse>("/hub/table"),
    get<TrackRecordResponse>("/hub/track-record"),
    get<TeamHubResponse>("/hub/teams"),
    get<PlayerHubResponse>("/hub/players"),
    get<CalibrationResponse>("/calibration"),
    get<ManifestResponse>("/manifest"),
    get<ScorerAccuracyResponse>("/scorer-track-record"),
  ]),
  fplProjections: () => get<FPLProjectionsResponse>("/fpl/projections"),
  fplOptimalXi: (formation?: string) => get<FPLRecommendation & { gameweek: number; model_source: string }>(`/fpl/optimal-xi${formation ? `?formation=${formation}` : ""}`),
  fplSquad: () => get<FPLSquadResponse>("/fpl/squad"),
  fplManualTransfers: (playerIds: number[], bank: number, freeTransfers: number) =>
    postJson<FPLTransfersResponse>("/fpl/transfers/manual", { player_ids: playerIds, bank, free_transfers: freeTransfers }),
  fplEntryTransfers: (entryId: number, freeTransfers: number) =>
    get<FPLTransfersResponse>(`/fpl/entry/${entryId}/transfers?free_transfers=${freeTransfers}`),
};

export class ApiError extends Error {}
