/**
 * Qualification Status Spine — one canonical way to describe where an athlete
 * stands, reused across the cockpit, rankings, simulator, and pulse so the whole
 * app speaks the same language. Pure function over the engine's QualLine.
 */
import type { QualLine } from "./types";

export type StatusCode = "in" | "chasing" | "blocked" | "ineligible";
export type StatusTone = "good" | "electric" | "warn" | "muted";

export interface QualStatus {
  code: StatusCode;
  label: string;
  tone: StatusTone;
  detail: string;
}

export function athleteStatus(line: QualLine, athleteId: number): QualStatus {
  if (line.qualified.some((q) => q.athleteId === athleteId)) {
    return { code: "in", label: "QUALIFYING", tone: "good", detail: "Inside the individual line" };
  }
  const skip = line.skipped.find((s) => s.athleteId === athleteId);
  if (skip?.reason === "noc_cap" || skip?.reason === "pathway_consumed") {
    return { code: "blocked", label: "BLOCKED", tone: "warn", detail: skip.detail };
  }
  if (skip?.reason === "ineligible") {
    return { code: "ineligible", label: "INELIGIBLE", tone: "muted", detail: skip.detail };
  }
  return { code: "chasing", label: "CHASING", tone: "electric", detail: "Outside the individual line" };
}

/** Tailwind tone classes shared by every status surface. */
export const STATUS_TONE: Record<StatusTone, { text: string; bg: string; dot: string }> = {
  good: { text: "text-good", bg: "bg-good/15", dot: "bg-good" },
  electric: { text: "text-electric-bright", bg: "bg-electric/15", dot: "bg-electric" },
  warn: { text: "text-la-gold", bg: "bg-la-gold/15", dot: "bg-la-gold" },
  muted: { text: "text-ink-faint", bg: "bg-surface-2", dot: "bg-ink-faint" },
};
