/**
 * Mixed Relay pathway tracker — read-only classification (relay is tracked, not
 * simulated in v1). Given the Mixed Relay Olympic Qualification Ranking and the
 * known/assumed World Champions, describe a nation's relay standing so the
 * athlete cockpit can show the relay context that affects their odds.
 *
 * The relay ranking cut is the top MR_RANKING_TEAMS (8) nations. The host and
 * the 2026 / 2027 Mixed Relay World Champions qualify outside the ranking,
 * which is how the 11 relay teams (22 places per gender) in `QUOTA` add up.
 */
import { MR_RANKING_TEAMS, type PathwayAssumptions } from "@/config/pathways";

/** Ranking places that qualify a relay team (mirrors QUOTA.relay.rankingTeams). */
export const MR_OQR_PLACES = MR_RANKING_TEAMS;

export interface MrNationEntry {
  noc: string;
  rank: number;
  total: number;
}

export interface MrNationStatus {
  noc: string;
  rank: number | null;
  total: number | null;
  /** Inside the relay ranking cut (top MR_OQR_PLACES). */
  insideRelayCut: boolean;
  /** Points behind the last qualifying nation (0 if inside). */
  gapToRelayCut: number | null;
  worldChampsSlot: "2026" | "2027" | null;
}

export function nationMrStatus(
  entries: MrNationEntry[],
  noc: string,
  assumptions: Pick<PathwayAssumptions, "mrWorldChamps2026" | "mrWorldChamps2027">,
): MrNationStatus {
  const sorted = [...entries].sort((a, b) => a.rank - b.rank);
  const entry = sorted.find((e) => e.noc === noc) ?? null;
  const cutNation = sorted.find((e) => e.rank === MR_OQR_PLACES) ?? sorted[MR_OQR_PLACES - 1] ?? null;

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
      insideRelayCut: false,
      gapToRelayCut: cutNation ? cutNation.total : null,
      worldChampsSlot,
    };
  }

  const insideRelayCut = entry.rank <= MR_OQR_PLACES;
  const gapToRelayCut = insideRelayCut ? 0 : cutNation ? Math.max(0, cutNation.total - entry.total) : null;

  return {
    noc,
    rank: entry.rank,
    total: entry.total,
    insideRelayCut,
    gapToRelayCut,
    worldChampsSlot,
  };
}
