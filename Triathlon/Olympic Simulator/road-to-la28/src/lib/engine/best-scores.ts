/**
 * Best-score selection: an athlete's best `bestN` results count, with no more
 * than `maxPerPeriod` from either scoring period. This is the single rule every
 * athlete lives by every week, and it also defines the "points to defend"
 * (the marginal counting score a new result must beat).
 */
import { QUAL, type PeriodId } from "@/config/qualification";
import type { AthleteScores, CountedScores, Score } from "./types";

export function selectCountingScores(
  scores: Score[],
  cfg = QUAL,
): CountedScores {
  // Highest points first; greedily take while respecting the per-period cap.
  const sorted = [...scores].sort((a, b) => b.points - a.points);
  const counted: Score[] = [];
  const perPeriodCount: Record<PeriodId, number> = { 1: 0, 2: 0 };

  for (const s of sorted) {
    if (counted.length >= cfg.bestN) break;
    if (perPeriodCount[s.period] >= cfg.maxPerPeriod) continue;
    counted.push(s);
    perPeriodCount[s.period] += 1;
  }

  const total = round2(counted.reduce((sum, s) => sum + s.points, 0));
  const marginal = counted.length ? counted[counted.length - 1] : null;

  return {
    counted,
    total,
    perPeriodCount,
    marginal,
    periodFull: {
      1: perPeriodCount[1] >= cfg.maxPerPeriod,
      2: perPeriodCount[2] >= cfg.maxPerPeriod,
    },
  };
}

/** Convenience: counted total for an athlete's full score pool. */
export function countedTotal(athlete: AthleteScores, cfg = QUAL): number {
  return selectCountingScores(athlete.scores, cfg).total;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
