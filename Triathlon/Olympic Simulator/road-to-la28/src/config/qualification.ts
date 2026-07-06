/**
 * LA28 Olympic triathlon qualification rules — the single source of truth.
 *
 * Every number the qualification engine uses lives here so the rules can be
 * tuned without touching engine logic. Sourced from the World Triathlon
 * "Los Angeles 2028 Olympic Games Qualification Criteria" (IOC-approved).
 */

export type PeriodId = 1 | 2;

export const QUAL = {
  /** The two scoring periods. Best-score caps apply within each. */
  periods: {
    1: { id: 1 as PeriodId, label: "First", from: "2026-05-18", to: "2027-05-18" },
    2: { id: 2 as PeriodId, label: "Second", from: "2027-05-19", to: "2028-05-18" },
  },

  /** An athlete's best `bestN` results count, no more than `maxPerPeriod` from one period. */
  bestN: 12,
  maxPerPeriod: 7,

  /** Quota + NOC caps. */
  quotaPerGender: 55,
  nocCapDefault: 2,
  nocCapIfDepth: 3,
  /** A NOC earns the 3-athlete cap only if it has `nocCapDepthCount` athletes inside the top `nocCapDepthWindow`. */
  nocCapDepthCount: 3,
  nocCapDepthWindow: 30,

  /** Places filled directly off the Individual Olympic Qualification Ranking. */
  individualRankingSlots: 21,

  /** Eligibility gate: must sit inside the top `eligibilityTopRank` of the World Ranking by the deadline. */
  eligibilityTopRank: 160,

  /** Final ranking snapshot that decides individual places. */
  deadline: "2028-05-18",
  /** Minimum birth date to be age-eligible (born on/before this day). */
  minBirthDate: "2010-12-31",
} as const;

/** Returns which scoring period a date falls in, or null if outside the qualification window. */
export function periodForDate(iso: string): PeriodId | null {
  if (iso >= QUAL.periods[1].from && iso <= QUAL.periods[1].to) return 1;
  if (iso >= QUAL.periods[2].from && iso <= QUAL.periods[2].to) return 2;
  return null;
}
