/**
 * Server-side data access.
 *
 * v1 reads the committed seed JSON (produced by `npm run seed`) so the app runs
 * with real World Triathlon data and zero infra. This is the single seam to
 * swap for live Neon/Drizzle reads later — every page goes through these
 * functions, so nothing else changes when the DB path lands.
 */
import "server-only";
import type { Gender } from "@/config/pathways";
import type { QualState, AthleteScores } from "@/lib/engine/types";
import type { MrNationEntry } from "@/lib/engine/mixed-relay";
import type { UpcomingEvent } from "@/lib/wt-api/events";
import men from "@/data/qual-state-men.json";
import women from "@/data/qual-state-women.json";
import mrNations from "@/data/mr-nations.json";
import events from "@/data/events.json";
import seedMeta from "@/data/seed-meta.json";

const STATES: Record<Gender, QualState> = {
  male: men as unknown as QualState,
  female: women as unknown as QualState,
};

export function getQualState(gender: Gender): QualState {
  return STATES[gender];
}

export function getBothStates(): QualState[] {
  return [STATES.male, STATES.female];
}

export function getMrNations(): MrNationEntry[] {
  return mrNations as unknown as MrNationEntry[];
}

export function getUpcomingEvents(): UpcomingEvent[] {
  return events as unknown as UpcomingEvent[];
}

export function getSeedMeta() {
  return seedMeta as { today?: string; menPublished: string; womenPublished: string };
}

/** Biggest official rank movers since the previous ranking, for the Pulse view. */
export function getMovers(gender: Gender, limit = 8) {
  const state = STATES[gender];
  return state.athletes
    .filter((a) => typeof a.change === "number" && a.change !== 0)
    .map((a) => ({
      athleteId: a.athleteId,
      fullName: a.fullName,
      noc: a.noc,
      change: a.change as number,
      rank: a.publishedRank ?? 0,
      flag: a.flag,
      profileImage: a.profileImage,
    }))
    .sort((x, y) => Math.abs(y.change) - Math.abs(x.change))
    .slice(0, limit);
}

/** Find an athlete + their gender across both rankings. */
export function findAthlete(
  athleteId: number,
): { athlete: AthleteScores; state: QualState } | null {
  for (const state of getBothStates()) {
    const athlete = state.athletes.find((a) => a.athleteId === athleteId);
    if (athlete) return { athlete, state };
  }
  return null;
}

/** Lightweight directory for the picker (no score arrays). */
export function getAthleteDirectory() {
  return getBothStates().flatMap((s) =>
    s.athletes.map((a, i) => ({
      athleteId: a.athleteId,
      fullName: a.fullName,
      noc: a.noc,
      gender: a.gender,
      rank: i + 1,
      flag: a.flag,
      profileImage: a.profileImage,
    })),
  );
}
