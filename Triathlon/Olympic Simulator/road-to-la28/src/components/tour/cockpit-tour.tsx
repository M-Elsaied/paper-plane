"use client";
import { HelpCircle } from "lucide-react";
import { ProductTour, startTour, type TourStep } from "./product-tour";

const STEPS: TourStep[] = [
  {
    target: "status",
    title: "Are they going to the Olympics?",
    body: "At a glance: QUALIFYING (inside the line), CHASING (still short), or BLOCKED (their nation's spots are full). Tap Share to post the card.",
  },
  {
    target: "hero",
    title: "Rank, points & the line",
    body: "Their live Olympic ranking and points. The glowing gold line is the qualification cut — the marker shows how far inside or outside it they sit.",
  },
  {
    target: "simulate",
    title: "Play out a race",
    body: "The fun part. Drag a finish position and watch the whole ranking re-sort live and the athlete cross the line — real qualification math, instantly.",
  },
  {
    target: "explain",
    title: "Why they're here — in plain English",
    body: "No black box. This spells out exactly what the ranking math means for them and what they need to do next.",
  },
  {
    target: "window",
    title: "How scoring works",
    body: "Only an athlete's best 12 results count, max 7 from each of the two qualifying periods. This shows what's locked in and where today sits.",
  },
  {
    target: "nav",
    title: "Explore everything",
    body: "Full rankings with the line drawn in, this week's races, the biggest movers, and the Mixed Relay pathway.",
  },
];

/** Mounts the cockpit tour (auto-starts once) + a small replay button. */
export function CockpitTour() {
  return (
    <>
      <ProductTour id="cockpit" steps={STEPS} />
      <button
        onClick={() => startTour("cockpit")}
        className="inline-flex items-center gap-1 text-[11px] font-semibold text-ink-faint transition hover:text-ink-dim"
      >
        <HelpCircle size={13} /> Take the tour
      </button>
    </>
  );
}
