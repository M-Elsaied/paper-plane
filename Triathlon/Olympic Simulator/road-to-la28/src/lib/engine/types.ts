/**
 * Core engine types. Everything here is plain data — the engine is pure and
 * isomorphic (identical behaviour on the server and in the browser).
 */
import type { PeriodId } from "@/config/qualification";
import type { PointsTier } from "@/config/points-tables";

export type Gender = "male" | "female";

/** A single scoring result an athlete has earned (or a hypothetical one). */
export interface Score {
  /** Points earned. For real results this is the WT-published value. */
  points: number;
  period: PeriodId;
  /** Optional provenance — present for real results, absent for bare values. */
  eventId?: number;
  programId?: number;
  eventTitle?: string;
  date?: string; // ISO
  position?: number;
  tier?: PointsTier;
  /** Marks a hypothetical score injected by the what-if simulator. */
  hypothetical?: boolean;
}

/** An athlete as the engine sees them: identity + their full score pool. */
export interface AthleteScores {
  athleteId: number;
  fullName: string;
  noc: string;
  gender: Gender;
  /** All scores across both periods (not yet filtered to the best 12). */
  scores: Score[];
  /** Age/flag eligibility already-known false disqualifies regardless of rank. */
  eligible?: boolean;
  yearOfBirth?: number;
  profileImage?: string;
  flag?: string;
  /** Official published rank + movement since the previous official ranking. */
  publishedRank?: number;
  lastRank?: number;
  change?: number;
}

/** Output of best-score selection for one athlete. */
export interface CountedScores {
  counted: Score[];
  total: number;
  perPeriodCount: Record<PeriodId, number>;
  /** Lowest counting score — the bar a new result must beat to improve the total. */
  marginal: Score | null;
  /** True when a period is already at its max cap (7). */
  periodFull: Record<PeriodId, boolean>;
}

/** An athlete ranked by counted total, as fed to the qualification walk. */
export interface RankedAthlete extends AthleteScores {
  counted: CountedScores;
  total: number;
}

/** A slot in the qualification line result. */
export interface QualSlot {
  athleteId: number;
  fullName: string;
  noc: string;
  rank: number; // rank in the individual ranking
  total: number;
}

/** Why an athlete high in the ranking did NOT take an individual slot. */
export interface SkipReason {
  athleteId: number;
  fullName: string;
  noc: string;
  rank: number;
  reason: "noc_cap" | "ineligible" | "pathway_consumed";
  detail: string;
}

/** Full qualification-line computation. */
export interface QualLine {
  gender: Gender;
  qualified: QualSlot[];
  /** The cut: last rank/points that still make the individual line. */
  cutRank: number | null;
  cutPoints: number | null;
  /** Just-outside athletes (the bubble) for "who's chasing" views. */
  bubble: QualSlot[];
  perNocUsage: Record<string, { cap: number; used: number }>;
  skipped: SkipReason[];
}

/** The compact per-gender state serialized to the client for instant what-if. */
export interface QualState {
  gender: Gender;
  /** Snapshot provenance for "as of" labelling. */
  publishedAt: string;
  rankingId: number;
  athletes: AthleteScores[];
}
