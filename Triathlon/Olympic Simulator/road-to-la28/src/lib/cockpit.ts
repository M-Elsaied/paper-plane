/**
 * Server view-model builders — turn raw QualState + engine output into the
 * exact shapes the cockpit and rankings screens render.
 */
import "server-only";
import type { Gender } from "@/config/pathways";
import { QUAL } from "@/config/qualification";
import { rankAthletes, computeQualificationLine } from "@/lib/engine/qualification";
import { nationMrStatus } from "@/lib/engine/mixed-relay";
import { DEFAULT_ASSUMPTIONS } from "@/config/pathways";
import { getQualState, getMrNations, findAthlete } from "@/lib/data";
import type { RankedAthlete } from "@/lib/engine/types";

export interface RankingRow {
  athleteId: number;
  rank: number;
  fullName: string;
  noc: string;
  total: number;
  qualified: boolean;
  flag?: string;
  profileImage?: string;
}

export function buildRanking(gender: Gender) {
  const state = getQualState(gender);
  const ranked = rankAthletes(state.athletes);
  const line = computeQualificationLine(state.athletes);
  const qualifiedIds = new Set(line.qualified.map((q) => q.athleteId));

  const rows: RankingRow[] = ranked.map((a, i) => ({
    athleteId: a.athleteId,
    rank: i + 1,
    fullName: a.fullName,
    noc: a.noc,
    total: a.total,
    qualified: qualifiedIds.has(a.athleteId),
    flag: a.flag,
    profileImage: a.profileImage,
  }));

  return { state, ranked, line, rows };
}

export interface CockpitModel {
  athleteId: number;
  fullName: string;
  noc: string;
  gender: Gender;
  flag?: string;
  profileImage?: string;
  rank: number;
  total: number;
  qualified: boolean;
  /** Points to the individual cut line (negative = inside, positive = needed). */
  gapToLine: number;
  cutRank: number | null;
  cutPoints: number | null;
  /** Best-12 counted scores + the marginal "points to defend". */
  counted: { points: number; period: 1 | 2 }[];
  marginalPoints: number | null;
  periodCount: Record<1 | 2, number>;
  periodFull: Record<1 | 2, boolean>;
  daysToDeadline: number;
  mr: ReturnType<typeof nationMrStatus>;
  /** A few chasers just behind, for context. */
  chasers: RankingRow[];
  publishedAt: string;
}

export function buildCockpit(athleteId: number): CockpitModel | null {
  const found = findAthlete(athleteId);
  if (!found) return null;
  const gender = found.athlete.gender;
  const { ranked, line, rows } = buildRanking(gender);

  const idx = ranked.findIndex((a) => a.athleteId === athleteId);
  const me = ranked[idx] as RankedAthlete;
  const rank = idx + 1;
  const qualified = line.qualified.some((q) => q.athleteId === athleteId);
  const gapToLine =
    line.cutPoints != null ? Math.round((line.cutPoints - me.total) * 100) / 100 : 0;

  const mr = nationMrStatus(getMrNations(), me.noc, DEFAULT_ASSUMPTIONS);

  const chasers = rows.filter((r) => Math.abs(r.rank - rank) <= 2 && r.athleteId !== athleteId);

  return {
    athleteId,
    fullName: me.fullName,
    noc: me.noc,
    gender,
    flag: me.flag,
    profileImage: me.profileImage,
    rank,
    total: me.total,
    qualified,
    gapToLine,
    cutRank: line.cutRank,
    cutPoints: line.cutPoints,
    counted: me.counted.counted.map((s) => ({ points: s.points, period: s.period })),
    marginalPoints: me.counted.marginal?.points ?? null,
    periodCount: me.counted.perPeriodCount,
    periodFull: me.counted.periodFull,
    daysToDeadline: Math.max(
      0,
      Math.ceil((new Date(QUAL.deadline).getTime() - Date.now()) / 86_400_000),
    ),
    mr,
    chasers,
    publishedAt: found.state.publishedAt,
  };
}
