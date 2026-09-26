// Synced from predictor-ui@5cfbec69fb05. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useState } from "react";
import { ErrorState, Skeleton } from "./States";
import { StatusBadge, type Moment } from "./StatusBadge";

export type Explanation = {
  headline: string;
  sections: { market: string; title: string; text: string }[];
  /** "llm" means a model wrote these words; "template" means the site's own
   *  copy did, from the same numbers. The panel never blurs the two. */
  source: "llm" | "template";
  model: string;
  generated_at: string;
  /** The sport, so the panel words the rebuilt status for that moment. */
  sport: string;
  /** When the pick itself was made. "rebuilt" means the model was asked again
   *  after the event had begun, so the pick is shown but never counted. This
   *  travels with the answer rather than being read out of the prose: neither
   *  the model nor the template is a reliable source for a status label. */
  pick_timing: "pre_kickoff" | "rebuilt" | "none";
};

/** The moment an F1 pick has to beat is the session, not a kick-off. */
const MOMENT_OF: Record<string, Moment> = { f1: "the session", nba: "tip-off" };

/** What "after the moment" means in a sentence. StatusBadge owns the label;
 *  this owns why the pick is not being counted. */
const STARTED: Record<Moment, string> = {
  kickoff: "after the game started",
  "tip-off": "after the game started",
  "the session": "after the session started",
};

/** How long ago a summary was written, in the reader's own units. A summary
 *  from 20 seconds ago is "just now", not "0 min ago" — nobody writes that. */
function ago(iso: string, now: number): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} ${hours === 1 ? "hr" : "hrs"} ago`;
  const days = Math.round(hours / 24);
  return `${days} ${days === 1 ? "day" : "days"} ago`;
}

/**
 * The panel's footer is the honesty contract, so it is written as one sentence
 * a reader can check: a model-written summary names the model and its age, and
 * a templated one says so in as many words and never says "AI".
 */
function footer(data: Explanation, now: number): string {
  if (data.source === "template") return "Summary written from the model's numbers";
  const age = ago(data.generated_at, now);
  return `AI summary of the model's numbers · ${data.model}${age ? ` · ${age}` : ""}`;
}

const toggleClass =
  "font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-accent underline-offset-4 transition-colors hover:text-pr-text focus-visible:text-pr-text";

export function ExplainerPanel({
  data,
  loading,
  error,
  onRetry,
  collapsed = false,
  moment,
}: {
  data: Explanation | null;
  loading: boolean;
  error: boolean;
  onRetry: () => void;
  collapsed?: boolean;
  /** Overrides the moment the sport implies. Sites pass their own so the
   *  status reads in the words that site already uses everywhere else. */
  moment?: Moment;
}) {
  const [opened, setOpened] = useState(false);
  const showBody = !collapsed || opened;

  if (loading) return <Skeleton label="Writing the summary…" />;
  if (error) return <ErrorState message="The summary didn't come through. The numbers below are still good." onRetry={onRetry} />;
  if (!data) return null;

  const now = Date.now();
  const when = moment ?? MOMENT_OF[data.sport] ?? "kickoff";
  const rebuilt = data.pick_timing === "rebuilt";

  return (
    <section aria-labelledby="explainer-heading" className="flex flex-col gap-3">
      <h3
        id="explainer-heading"
        className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim"
      >
        In plain English
      </h3>

      <p className="max-w-[70ch] text-lg font-medium leading-snug text-pr-text">{data.headline}</p>

      {rebuilt && (
        <p className="flex max-w-[70ch] flex-wrap items-center gap-2 text-sm text-pr-text-dim">
          <StatusBadge status="rebuilt" moment={when} />
          <span>
            This pick was made {STARTED[when]}, so it is shown for reference and not counted.
          </span>
        </p>
      )}

      {collapsed && (
        <button
          type="button"
          onClick={() => setOpened((was) => !was)}
          aria-expanded={showBody}
          className={`mt-1 self-start underline ${toggleClass}`}
        >
          {showBody ? "Hide the race story" : "Read the race story"}
        </button>
      )}

      {showBody && data.sections.length > 0 && (
        <div className="flex max-w-[70ch] flex-col gap-4 border-t border-pr-rule pt-4">
          {data.sections.map((section) => (
            <div key={`${section.market}-${section.title}`} className="flex flex-col gap-1">
              <h4 className="font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text-dim">
                {section.title}
              </h4>
              <p className="text-sm leading-relaxed text-pr-text-dim">{section.text}</p>
            </div>
          ))}
        </div>
      )}

      <p className="mt-1 text-xs text-pr-text-faint">{footer(data, now)}</p>
    </section>
  );
}
