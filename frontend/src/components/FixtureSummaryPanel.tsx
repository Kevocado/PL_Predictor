import { useCallback, useEffect, useMemo, useState } from "react";
import { ExplainerPanel, type Explanation } from "../predictor-ui";
import { panelFacts } from "../lib/panelFacts";
import type { FixtureSummary } from "../types";

/**
 * The "In plain English" panel on its own, so every surface that opens a
 * fixture gets the same behaviour without each one re-implementing the fetch:
 * request the summary for this id, never block what is already on screen, and
 * offer a retry when it fails.
 *
 * `fetcher` defaults to the site's own client. A caller with no explainer
 * deployed (or a fixture the service has nothing for) passes nothing and the
 * panel renders nothing at all, rather than an empty box.
 */
export function FixtureSummaryPanel({
  eventId,
  fetcher,
  fixture,
  sport = "pl",
  className = "",
}: {
  eventId: string;
  fetcher?: (sport: string, id: string) => Promise<Explanation>;
  /** The fixture itself, when the caller already has it. The panel renders no
   *  figure of its own, so without this it would show the explanation's words
   *  and nothing else — and for PL that means no split bar at all, which is
   *  the one thing a three-way market needs. Optional: a caller that has not
   *  loaded the fixture yet gets the words alone, which is better than a box of
   *  dashes. */
  fixture?: FixtureSummary | null;
  sport?: string;
  className?: string;
}) {
  const [data, setData] = useState<Explanation | null>(null);
  // True from the first paint when a fetcher exists, so there is no frame of
  // empty wrapper before the loading Skeleton appears.
  const [loading, setLoading] = useState(Boolean(fetcher));
  const [error, setError] = useState(false);

  const load = useCallback(() => {
    if (!fetcher) return;
    let cancelled = false;
    setLoading(true);
    setError(false);
    fetcher(sport, eventId)
      .then((r) => { if (!cancelled) setData(r); })
      .catch(() => { if (!cancelled) { setData(null); setError(true); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [fetcher, sport, eventId]);

  useEffect(() => load(), [load]);

  // Derived, not fetched — and memoised, because `panelFacts` allocates fresh
  // arrays and the panel takes them as props. Without this the tiles re-render
  // on every keystroke anywhere above.
  const panel = useMemo(() => panelFacts(fixture), [fixture]);

  if (!fetcher) return null;
  return (
    <div className={className}>
      <ExplainerPanel
        data={data}
        loading={loading}
        error={error}
        onRetry={() => load()}
        // The figures come from the fixture, not from the explanation. The panel
        // renders what it is handed and must not be what decides it, or the two
        // could disagree on the same screen.
        tiles={panel.tiles}
        segments={panel.segments}
        legend={panel.legend}
      />
    </div>
  );
}
