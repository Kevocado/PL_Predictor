import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { FixtureSummaryPanel } from "./FixtureSummaryPanel";

const explanation = {
  headline: "Tottenham are the narrow favourites, but this is close to a coin flip.",
  sections: [{ market: "result", title: "Why Tottenham", text: "The model has them at 44% at home." }],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "pl",
  pick_timing: "pre_kickoff" as const,
};

describe("FixtureSummaryPanel", () => {
  it("asks for this fixture's summary and shows its headline", async () => {
    const fetcher = vi.fn().mockResolvedValue(explanation);
    render(<FixtureSummaryPanel eventId="e1" fetcher={fetcher} />);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledWith("pl", "e1");
  });

  it("says it is writing, which is a state and not a blank space", () => {
    const fetcher = vi.fn().mockReturnValue(new Promise<typeof explanation>(() => {}));
    render(<FixtureSummaryPanel eventId="e1" fetcher={fetcher} />);
    expect(screen.getByRole("status")).toHaveTextContent("Writing the summary…");
  });

  it("offers a retry that asks again, and recovers", async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error("explainer down"));
    render(<FixtureSummaryPanel eventId="e1" fetcher={fetcher} />);
    const retry = await screen.findByRole("button", { name: "Try again" });
    fetcher.mockResolvedValue(explanation);
    fireEvent.click(retry);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("renders nothing at all when there is no explainer", () => {
    const { container } = render(<FixtureSummaryPanel eventId="e1" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("labels a rebuilt pick in the site's own words", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ...explanation, pick_timing: "rebuilt" });
    render(<FixtureSummaryPanel eventId="e1" fetcher={fetcher} />);
    expect(await screen.findByText("Rebuilt after kickoff")).toBeInTheDocument();
    expect(screen.getByText(/not counted/)).toBeInTheDocument();
  });
});
