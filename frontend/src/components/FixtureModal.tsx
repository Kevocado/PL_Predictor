import { useEffect, useState, type ReactNode } from "react";
import type { FixtureDetail, FixturePlayers, MarketEdge } from "../types";
import { api } from "../api/client";
import { TeamBadge } from "./TeamBadge";
import { ScorelineHeatmap } from "./ScorelineHeatmap";
import { FormStrip } from "./FormStrip";
import { InfoTooltip } from "./InfoTooltip";
import { PlayerScorerList } from "./PlayerScorerList";
import { GLOSSARY } from "../lib/glossary";

interface Props {
  eventId: string;
  onClose: () => void;
}

const MARKET_LABELS: Record<string, string> = {
  home_win: "Home win",
  draw: "Draw",
  away_win: "Away win",
  over_2_5: "Over 2.5",
  under_2_5: "Under 2.5",
};

function MarketRow({ label, edge, flagged }: { label: ReactNode; edge: MarketEdge; flagged: boolean }) {
  return (
    <div
      className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${
        flagged ? "bg-pl-pink/10 ring-1 ring-pl-pink/40" : "bg-pl-850/60"
      }`}
    >
      <span className="text-pl-text-dim">{label}</span>
      <div className="flex items-center gap-3">
        <span className="font-semibold text-pl-text">{(edge.prob * 100).toFixed(1)}%</span>
        {edge.implied !== null && (
          <>
            <span className="text-xs text-pl-text-faint">mkt {(edge.implied * 100).toFixed(1)}%</span>
            <span className={`text-xs font-semibold ${(edge.edge ?? 0) > 0 ? "text-pl-cyan" : "text-pl-text-faint"}`}>
              {(edge.edge ?? 0) > 0 ? "+" : ""}
              {((edge.edge ?? 0) * 100).toFixed(1)}%
            </span>
          </>
        )}
      </div>
    </div>
  );
}

function OverUnderRow({ label, lam, line, over }: { label: string; lam: number; line: number; over: number }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-2 text-sm">
      <span className="text-pl-text-dim">{label}</span>
      <div className="flex items-center gap-3">
        <span className="text-xs text-pl-text-faint">exp. {lam.toFixed(1)}</span>
        <span className="font-semibold text-pl-text">
          O{line} {(over * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

export function FixtureModal({ eventId, onClose }: Props) {
  const [detail, setDetail] = useState<FixtureDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [players, setPlayers] = useState<FixturePlayers | null>(null);
  const [playersError, setPlayersError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setError(null);
    api
      .fixtureDetail(eventId)
      .then(setDetail)
      .catch((e) => setError(e.message));

    setPlayers(null);
    setPlayersError(null);
    api
      .fixturePlayers(eventId)
      .then(setPlayers)
      .catch((e) => setPlayersError(e.message));
  }, [eventId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      <div className="animate-modal-in relative flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-pl-border bg-pl-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-pl-border px-6 py-4">
          <span className="text-sm font-semibold text-pl-text-dim">Fixture detail</span>
          <button
            onClick={onClose}
            className="rounded-full p-1.5 text-pl-text-dim transition hover:bg-pl-800 hover:text-pl-text"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="overflow-y-auto px-6 py-6">
          {error && <div className="text-sm text-loss">{error}</div>}
          {!detail && !error && <div className="flex h-64 items-center justify-center text-pl-text-faint">Loading…</div>}

          {detail && (
            <div className="flex flex-col gap-6">
              <div className="flex items-center justify-center gap-10">
                <div className="flex flex-col items-center gap-2">
                  <TeamBadge team={detail.team_home} size="lg" />
                  <span className="text-base font-semibold text-pl-text">{detail.team_home}</span>
                  <div className="flex items-center gap-1.5">
                    <FormStrip results={detail.home_recent_form} />
                    <InfoTooltip text={GLOSSARY.recentForm} />
                  </div>
                </div>
                <div className="flex flex-col items-center gap-1 text-center">
                  <span className="text-xs font-medium uppercase text-pl-text-faint">
                    {new Date(detail.commence_time).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })}
                  </span>
                  <span className="text-2xl font-black text-pl-text-faint">vs</span>
                  <span className="text-xs text-pl-text-faint">
                    {new Date(detail.commence_time).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <TeamBadge team={detail.team_away} size="lg" />
                  <span className="text-base font-semibold text-pl-text">{detail.team_away}</span>
                  <FormStrip results={detail.away_recent_form} />
                </div>
              </div>

              {detail.data_confidence === "new" && (
                <div className="flex items-center gap-2 rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-dim">
                  <span className="rounded bg-pl-700/60 px-1.5 py-0.5 font-semibold text-pl-text-dim">new team</span>
                  {GLOSSARY.dataConfidenceNew}
                </div>
              )}
              {detail.data_confidence === "limited" && (
                <div className="flex items-center gap-2 rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-dim">
                  <span className="rounded bg-pl-700/60 px-1.5 py-0.5 font-semibold text-pl-text-dim">limited data</span>
                  {GLOSSARY.dataConfidenceLimited}
                </div>
              )}

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div className="flex flex-col gap-6">
                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                      Scoreline probability
                      <InfoTooltip text={GLOSSARY.scorelineGrid} align="left" />
                    </h3>
                    <ScorelineHeatmap
                      grid={detail.score_grid}
                      homeTeam={detail.team_home}
                      awayTeam={detail.team_away}
                      topScorelines={detail.top_scorelines}
                    />
                  </section>

                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                      Head-to-head
                    </h3>
                    {detail.head_to_head.length === 0 ? (
                      <p className="text-xs text-pl-text-faint">No meetings in the loaded seasons.</p>
                    ) : (
                      <div className="flex flex-col gap-1">
                        {detail.head_to_head.map((m, i) => (
                          <div key={i} className="flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-1.5 text-xs">
                            <span className="text-pl-text-faint">{m.date}</span>
                            <span className="font-medium text-pl-text">
                              {m.team_home} {m.goals_home}-{m.goals_away} {m.team_away}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </section>
                </div>

                <div className="flex flex-col gap-6">
                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                      Match result &amp; goals
                      <InfoTooltip text={GLOSSARY.edge} align="right" />
                    </h3>
                    <div className="flex flex-col gap-1.5">
                      {(["home_win", "draw", "away_win", "over_2_5", "under_2_5"] as const).map((k) => (
                        <MarketRow
                          key={k}
                          label={MARKET_LABELS[k]}
                          edge={detail[k]}
                          flagged={detail.value_bet_flags.includes(k)}
                        />
                      ))}
                      <MarketRow
                        label={
                          <span className="inline-flex items-center gap-1.5">
                            BTTS: Yes <InfoTooltip text={GLOSSARY.btts} align="right" />
                          </span>
                        }
                        edge={{ prob: detail.btts_yes_prob, implied: null, edge: null }}
                        flagged={false}
                      />
                    </div>
                    {!detail.has_live_odds && <p className="mt-2 text-xs text-pl-text-faint">{GLOSSARY.noLiveMarket}</p>}
                  </section>

                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                      Corners &amp; cards
                      <InfoTooltip text={GLOSSARY.noLiveMarket} align="right" />
                    </h3>
                    <div className="flex flex-col gap-1.5">
                      <OverUnderRow label="Total corners" lam={detail.corners.lambda_} line={detail.corners.line} over={detail.corners.over} />
                      <OverUnderRow label="Total cards" lam={detail.cards.lambda_} line={detail.cards.line} over={detail.cards.over} />
                    </div>
                  </section>
                </div>
              </div>

              <section>
                <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                  Likely scorers &amp; assists
                  <InfoTooltip text={GLOSSARY.anytimeScorer} align="left" />
                  <InfoTooltip text={GLOSSARY.playerAvailability} align="left" />
                </h3>
                {playersError && <p className="text-xs text-loss">{playersError}</p>}
                {!players && !playersError && <p className="text-xs text-pl-text-faint">Loading player predictions…</p>}
                {players && (
                  <PlayerScorerList
                    homeTeam={detail.team_home}
                    awayTeam={detail.team_away}
                    homePlayers={players.home_players}
                    awayPlayers={players.away_players}
                  />
                )}
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
