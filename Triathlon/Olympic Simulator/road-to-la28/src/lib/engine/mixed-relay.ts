/**
 * Mixed Relay pathway tracker — read-only classification (relay is tracked, not
 * simulated in v1). Given the Mixed Relay Olympic Qualification Ranking and the
 * known/assumed World Champions, describe a nation's relay standing so the
 * athlete cockpit can show the relay context that affects their odds.
 */
import type { PathwayAssumptions } from "@/config/pathways";

/** The 16 MR Olympic ranking places. */
export const MR_OQR_PLACES = 16;
/** Continental-guarantee window inside the ranking. */
export const MR_CONTINENTAL_TOP = 15;

export interface MrNationEntry {
  noc: string;
  rank: number;
  total: number;
}

export interface MrNationStatus {
  noc: string;
  rank: number | null;
  total: number | null;
  insideTop16: boolean;
  gapToTop16: number | null; // points behind the 16th nation (0 if inside)
  worldChampsSlot: "2026" | "2027" | null;
}

export function nationMrStatus(
  entries: MrNationEntry[],
  noc: string,
  assumptions: Pick<PathwayAssumptions, "mrWorldChamps2026" | "mrWorldChamps2027">,
): MrNationStatus {
  const sorted = [...entries].sort((a, b) => a.rank - b.rank);
  const entry = sorted.find((e) => e.noc === noc) ?? null;
  const sixteenth = sorted.find((e) => e.rank === MR_OQR_PLACES) ?? sorted[MR_OQR_PLACES - 1] ?? null;

  const worldChampsSlot =
    assumptions.mrWorldChamps2026 === noc
      ? "2026"
      : assumptions.mrWorldChamps2027 === noc
        ? "2027"
        : null;

  if (!entry) {
    return {
      noc,
      rank: null,
      total: null,
      insideTop16: false,
      gapToTop16: sixteenth ? sixteenth.total : null,
      worldChampsSlot,
    };
  }

  const insideTop16 = entry.rank <= MR_OQR_PLACES;
  const gapToTop16 = insideTop16 ? 0 : sixteenth ? Math.max(0, sixteenth.total - entry.total) : null;

  return {
    noc,
    rank: entry.rank,
    total: entry.total,
    insideTop16,
    gapToTop16,
    worldChampsSlot,
  };
}
