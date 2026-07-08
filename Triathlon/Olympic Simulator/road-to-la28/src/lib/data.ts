/**
 * Server-side data access — the single seam between the UI and the data source.
 *
 * Reads from Neon when a snapshot exists; otherwise falls back to the committed
 * seed JSON so the app always runs (local dev, fresh deploy, DB hiccup). Every
 * page goes through these functions, so switching sources changed nothing else.
 * Reads are async now; results are per-request memoized with React cache().
 */
import "server-only";
import { cache } from "react";
import type { Gender } from "@/config/pathways";
import type { QualState, AthleteScores } from "@/lib/engine/types";
import type { MrNationEntry } from "@/lib/engine/mixed-relay";
import type { UpcomingEvent } from "@/lib/wt-api/events";
import type { WorldRankEntry } from "@/lib/wt-api/rankings";
import { readQualState, readMrNations, readWorldRanking, readRankTrajectory } from "@/lib/db-read";
import type { TrajectoryPoint } from "@/lib/trajectory";
import men from "@/data/qual-state-men.json";
import women from "@/data/qual-state-women.json";
import worldMen from "@/data/world-ranking-men.json";
import worldWomen from "@/data/world-ranking-women.json";
import establishedNocs from "@/data/established-nocs.json";
import mrNations from "@/data/mr-nations.json";
import events from "@/data/events.json";
import seedMeta from "@/data/seed-meta.json";

const SEED: Record<Gender, QualState> = {
  male: men as unknown as QualState,
  female: women as unknown as QualState,
};

export const getQualState = cache(async (gender: Gender): Promise<QualState> => {
  try {
    const fromDb = await readQualState(gender);
    if (fromDb && fromDb.athletes.length) return fromDb;
  } catch {
    // fall through to seed JSON
  }
  return SEED[gender];
});

export async function getBothStates(): Promise<QualState[]> {
  return Promise.all([getQualState("male"), getQualState("female")]);
}

const SEED_WORLD: Record<Gender, WorldRankEntry[]> = {
  male: worldMen as unknown as WorldRankEntry[],
  female: worldWomen as unknown as WorldRankEntry[],
};

export const getWorldRanking = cache(async (gender: Gender): Promise<WorldRankEntry[]> => {
  try {
    const fromDb = await readWorldRanking(gender);
    if (fromDb && fromDb.length) return fromDb;
  } catch {
    // fall through to seed JSON
  }
  return SEED_WORLD[gender];
});

export const getMrNations = cache(async (): Promise<MrNationEntry[]> => {
  try {
    const fromDb = await readMrNations();
    if (fromDb && fromDb.length) return fromDb;
  } catch {
    // fall through
  }
  return mrNations as unknown as MrNationEntry[];
});

/** An athlete's rank history from stored snapshots; [] when none (seed/first run). */
export const getRankTrajectory = cache(async (gender: Gender, athleteId: number): Promise<TrajectoryPoint[]> => {
  try {
    const fromDb = await readRankTrajectory(gender, athleteId);
    if (fromDb && fromDb.length >= 2) return fromDb;
  } catch {
    // fall through — the cockpit synthesizes a last→now trend from the athlete record
  }
  return [];
});

export function getUpcomingEvents(): UpcomingEvent[] {
  return events as unknown as UpcomingEvent[];
}

/** NOCs that already have Olympic triathlon history → NOT New Flag eligible.
 *  Slow-moving reference data, refreshed by `npm run seed`. */
export function getEstablishedNocs(): Set<string> {
  return new Set(establishedNocs as string[]);
}

export function getSeedMeta() {
  return seedMeta as { today?: string; menPublished: string; womenPublished: string };
}

/** Find an athlete + their gender across both rankings. */
export async function findAthlete(
  athleteId: number,
): Promise<{ athlete: AthleteScores; state: QualState } | null> {
  for (const state of await getBothStates()) {
    const athlete = state.athletes.find((a) => a.athleteId === athleteId);
    if (athlete) return { athlete, state };
  }
  return null;
}

/** Lightweight directory for the picker (no score arrays). */
export async function getAthleteDirectory() {
  const both = await getBothStates();
  return both.flatMap((s) =>
    s.athletes.map((a, i) => ({
      athleteId: a.athleteId,
      fullName: a.fullName,
      noc: a.noc,
      gender: a.gender,
      rank: a.publishedRank ?? i + 1,
      flag: a.flag,
      profileImage: a.profileImage,
    })),
  );
}

/** Biggest official rank movers since the previous ranking, for the Pulse view. */
export async function getMovers(gender: Gender, limit = 8) {
  const state = await getQualState(gender);
  return state.athletes
    .filter((a) => typeof a.change === "number" && a.change !== 0)
    .map((a, i) => ({
      athleteId: a.athleteId,
      fullName: a.fullName,
      noc: a.noc,
      change: a.change as number,
      rank: a.publishedRank ?? i + 1,
      flag: a.flag,
      profileImage: a.profileImage,
    }))
    .sort((x, y) => Math.abs(y.change) - Math.abs(x.change))
    .slice(0, limit);
}
