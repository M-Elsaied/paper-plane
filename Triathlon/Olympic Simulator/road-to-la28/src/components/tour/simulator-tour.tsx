"use client";
import { HelpCircle } from "lucide-react";
import { ProductTour, startTour, type TourStep } from "./product-tour";

const STEPS: TourStep[] = [
  {
    target: "sim-position",
    title: "Set the finish position",
    body: "Drag the slider to choose where the athlete finishes — from a win down to 40th. The points they'd earn update instantly.",
  },
  {
    target: "sim-tier",
    title: "Pick the race",
    body: "Different races are worth different points — a WTCS win pays far more than a Continental Cup. Choose the event you're imagining.",
  },
  {
    target: "sim-board",
    title: "Watch the ranking move — live",
    body: "The whole Olympic ranking re-sorts in real time and the athlete's row slides across the glowing gold qualification line. Real math, zero delay.",
  },
  {
    target: "sim-outcome",
    title: "See the result",
    body: "Their projected new rank and the points swing. If they cross into the qualifying zone, you'll see it light up.",
  },
];

/** Mounts the simulator tour (auto-starts once) + a small replay button. */
export function SimulatorTour() {
  return (
    <>
      <ProductTour id="simulator" steps={STEPS} />
      <button
        onClick={() => startTour("simulator")}
        className="inline-flex items-center gap-1 text-[11px] font-semibold text-ink-faint transition hover:text-ink-dim"
      >
        <HelpCircle size={13} /> Tour
      </button>
    </>
  );
}
