/**
 * Fetch + normalize World Triathlon rankings into engine QualState.
 *
 * The ranking payload gives each athlete their COUNTING scores as arrays of
 * point values (padded with nulls), split into current/previous period. We map
 * current -> period 1, previous -> period 2 and drop the null padding. For v1
 * this counting-score pool is enough for the qualification line and what-if
 * (a new hypothetical score correctly displaces the lowest counting one).
 */
import type { Gender } from "@/config/pathways";
import type { PeriodId } from "@/config/qualification";
import type { AthleteScores, QualState, Score } from "@/lib/engine/types";
import type { MrNationEntry } from "@/lib/engine/mixed-relay";
import { wtGet } from "./client";

export interface RawRankingAthlete {
  athlete_id: number;
  athlete_full_name: string;
  athlete_noc: string;
  athlete_gender: Gender;
  athlete_yob?: number;
  athlete_profile_image?: string | null;
  athlete_flag_circle?: string | null;
  athlete_country_name?: string;
  /** Previous Games this athlete qualified for, e.g. ["olympics_2024"]. */
  olympics_qualifications?: string[] | null;
  rank: number;
  last_rank?: number;
  change?: number | string; // API sends "NEW" for new entrants
  total: number;
  scores_current_period?: (number | null)[];
  scores_previous_period?: (number | null)[];
}

export interface RawRanking {
  ranking_id: number;
  ranking_name: string;
  ranking_cat_name: string;
  published: string;
  total: number;
  rankings: RawRankingAthlete[];
}

function toScores(values: (number | null)[] | undefined, period: PeriodId): Score[] {
  return (values ?? [])
    .filter((v): v is number => typeof v === "number" && v > 0)
    .map((points) => ({ points, period }));
}

export function normalizeRanking(raw: RawRanking, gender: Gender): QualState {
  const athletes: AthleteScores[] = raw.rankings.map((a) => ({
    athleteId: a.athlete_id,
    fullName: a.athlete_full_name,
    noc: a.athlete_noc,
    gender: a.athlete_gender ?? gender,
    yearOfBirth: a.athlete_yob,
    profileImage: a.athlete_profile_image ?? undefined,
    flag: a.athlete_flag_circle ?? undefined,
    publishedRank: a.rank,
    lastRank: a.last_rank,
    // `change` may be "NEW" (string) for new entrants — keep only real numbers.
    change: typeof a.change === "number" && Number.isFinite(a.change) ? a.change : undefined,
    scores: [
      ...toScores(a.scores_current_period, 1),
      ...toScores(a.scores_previous_period, 2),
    ],
  }));

  return {
    gender,
    publishedAt: raw.published,
    rankingId: raw.ranking_id,
    athletes,
  };
}

export async function fetchRankingState(
  rankingId: number,
  gender: Gender,
  limit = 1000,
): Promise<QualState> {
  const res = await wtGet<RawRanking>(`/rankings/${rankingId}`, { limit });
  return normalizeRanking(res.data, gender);
}

interface RawMrTeam {
  team_noc?: string;
  team_country_name?: string;
  team_title?: string;
  team_flag_circle?: string;
  rank: number;
  total: number;
}

/** A compact World Ranking entry — enough for New Flag continental analysis. */
export interface WorldRankEntry {
  athleteId: number;
  fullName: string;
  noc: string;
  gender: Gender;
  rank: number;
  total: number;
  flag?: string;
  profileImage?: string;
  /** True if this athlete has raced a previous Olympics (marks their NOC as
   *  an established — NOT New Flag — nation). */
  olympicHistory?: boolean;
}

export function normalizeWorldRanking(raw: RawRanking, gender: Gender): WorldRankEntry[] {
  return raw.rankings.map((a) => ({
    athleteId: a.athlete_id,
    fullName: a.athlete_full_name,
    noc: a.athlete_noc,
    gender: a.athlete_gender ?? gender,
    rank: a.rank,
    total: a.total,
    flag: a.athlete_flag_circle ?? undefined,
    profileImage: a.athlete_profile_image ?? undefined,
    olympicHistory: (a.olympics_qualifications?.length ?? 0) > 0,
  }));
}

/** NOCs with any athlete who has Olympic history → established (not New Flag). */
export function establishedNocsFrom(...lists: WorldRankEntry[][]): string[] {
  const s = new Set<string>();
  for (const list of lists) for (const e of list) if (e.olympicHistory) s.add(e.noc);
  return [...s].sort();
}

export async function fetchWorldRanking(
  rankingId: number,
  gender: Gender,
  limit = 500,
): Promise<WorldRankEntry[]> {
  const res = await wtGet<RawRanking>(`/rankings/${rankingId}`, { limit });
  return normalizeWorldRanking(res.data, gender);
}

/** Mixed Relay Olympic ranking -> per-nation entries (nations, not athletes). */
export async function fetchMrNations(rankingId: number, limit = 100): Promise<MrNationEntry[]> {
  const res = await wtGet<{ rankings: RawMrTeam[] }>(`/rankings/${rankingId}`, { limit });
  return res.data.rankings.map((r) => ({
    noc: r.team_noc || r.team_country_name || r.team_title || "—",
    rank: r.rank,
    total: r.total,
  }));
}
