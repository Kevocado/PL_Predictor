import { useCallback, useEffect, useState } from "react";
import { ExplainerPanel, type Explanation } from "../predictor-ui";

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
  sport = "pl",
  className = "",
}: {
  eventId: string;
  fetcher?: (sport: string, id: string) => Promise<Explanation>;
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

  if (!fetcher) return null;
  return (
    <div className={className}>
      <ExplainerPanel data={data} loading={loading} error={error} onRetry={() => load()} />
    </div>
  );
}
