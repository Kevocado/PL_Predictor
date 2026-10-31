import { useEffect, useState } from "react";

export interface SectionLink {
  id: string;
  label: string;
}

/** The bar under a fixture header: jumps to each section and marks the one in
 *  view. `root` is the scrolling container the sections live in. */
export function SectionBar({ links, root }: { links: SectionLink[]; root: React.RefObject<HTMLElement | null> }) {
  const [current, setCurrent] = useState(links[0]?.id ?? "");
  const key = links.map((l) => l.id).join("|");

  useEffect(() => {
    const container = root.current;
    if (!container || typeof IntersectionObserver === "undefined") return;
    const seen = new Map<string, boolean>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) seen.set(e.target.id, e.isIntersecting);
        const first = links.find((l) => seen.get(l.id));
        if (first) setCurrent(first.id);
      },
      { root: container, rootMargin: "-96px 0px -55% 0px", threshold: 0 },
    );
    links.forEach((l) => {
      const el = container.querySelector(`#${CSS.escape(l.id)}`);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, root]);

  const go = (id: string) => {
    const el = root.current?.querySelector(`#${CSS.escape(id)}`);
    if (el && "scrollIntoView" in el) (el as HTMLElement).scrollIntoView({ behavior: "smooth", block: "start" });
    setCurrent(id);
  };

  return (
    <nav aria-label="Sections" className="flex gap-1 overflow-x-auto border-t border-pr-rule px-3 sm:px-6">
      {links.map((l) => (
        <button
          key={l.id}
          type="button"
          onClick={() => go(l.id)}
          aria-current={current === l.id ? "true" : undefined}
          className={`shrink-0 border-b-2 px-3 py-2.5 text-sm font-semibold transition ${
            current === l.id ? "border-pr-accent text-pr-text" : "border-transparent text-pr-text-faint hover:text-pr-text"
          }`}
        >
          {l.label}
        </button>
      ))}
    </nav>
  );
}
