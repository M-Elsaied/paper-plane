/**
 * Race projection engine — the Live Race Companion's brain.
 *
 * Deterministic and explainable (no ML): seed each racing athlete's expected
 * finish from their current ranking strength within the field, blended with a
 * recency-of-form proxy (rank movement). Those projected finishes flow through
 * the SAME qualification engine to produce a "projected Monday ranking" — the
 * Olympic ranking as it would stand after the race, with line crossings.
 *
 * Pure functions → run on the server (initial render) and in the browser (live
 * as the user drags a finish order).
 */
import { PROJECTION } from "@/config/projection";
import { pointsForPosition, type PointsTier } from "@/config/points-tables";
import type { PeriodId } from "@/config/qualification";
import { selectCountingScores } from "./best-scores";
import { rankAthletes, computeQualificationLine } from "./qualification";
import type { AthleteScores, QualLine, Score } from "./types";

export interface FieldAthlete extends AthleteScores {
  /** Current best-12 counting total (0 if unranked). */
  currentTotal: number;
  /** Current Olympic rank (null if outside the ranking). */
  currentRank: number | null;
  /** Rank movement on the last official update (+ = rising). Form proxy. */
  change?: number;
}

export interface SeededAthlete {
  athleteId: number;
  seed: number;
  rankingComponent: number;
  formComponent: number;
}

/** Model's default projected finish order (best → worst). */
export function seedFinishOrder(field: FieldAthlete[], cfg = PROJECTION): SeededAthlete[] {
  const totals = field.map((a) => a.currentTotal);
  const maxTotal = Math.max(1, ...totals);
  // `change` can arrive as a non-number (e.g. "NEW" for new entrants) — coerce.
  const chg = (a: FieldAthlete) => (typeof a.change === "number" && Number.isFinite(a.change) ? a.change : 0);
  const maxAbsChange = Math.max(1, ...field.map((a) => Math.abs(chg(a))));

  const seeded = field.map((a) => {
    // Ranking strength within the field (0..1).
    const rankingComponent = a.currentTotal / maxTotal;
    // Form proxy: rising athletes get a small boost (0..1, 0.5 = neutral).
    const formComponent = 0.5 + chg(a) / (2 * maxAbsChange);
    const seed =
      cfg.weights.ranking * rankingComponent + cfg.weights.form * clamp01(formComponent);
    return { athleteId: a.athleteId, seed, rankingComponent, formComponent: clamp01(formComponent) };
  });

  seeded.sort((a, b) => b.seed - a.seed);
  return seeded;
}

/** Expected points for a given finish order at a given race tier. */
export function pointsForOrder(
  orderedAthleteIds: number[],
  tier: PointsTier,
): Map<number, { position: number; expectedPoints: number }> {
  const map = new Map<number, { position: number; expectedPoints: number }>();
  orderedAthleteIds.forEach((id, i) => {
    map.set(id, { position: i + 1, expectedPoints: pointsForPosition(tier, i + 1) });
  });
  return map;
}

export interface ProjectedRow {
  athleteId: number;
  fullName: string;
  noc: string;
  projectedPosition: number;
  expectedPoints: number;
  currentRank: number | null;
  currentTotal: number;
  projectedRank: number;
  projectedTotal: number;
  rankDelta: number; // + = moved up
  pointsDelta: number;
  crossesLine: boolean;
  dropsOut: boolean;
  qualified: boolean;
}

export interface BoardRow {
  athleteId: number;
  fullName: string;
  noc: string;
  projectedRank: number;
  currentRank: number | null;
  rankDelta: number; // + = moved up
  projectedTotal: number;
  qualified: boolean;
  isRacer: boolean;
  crossesLine: boolean;
  dropsOut: boolean;
}

export interface ProjectedRanking {
  rows: ProjectedRow[];
  /** Top-N of the full projected ranking (everyone, not just racers). */
  board: BoardRow[];
  projectedLine: QualLine;
  baselineLine: QualLine;
  cutRank: number | null;
  cutPoints: number | null;
}

/**
 * Run a projected finish order through the qualification engine and diff against
 * the current ranking. `orderedAthleteIds` is the projected finish order for the
 * racing field; only athletes present in `allAthletes` (the ranking) affect the
 * line, but their projected finish positions come from the race order.
 */
export function projectRanking(
  allAthletes: AthleteScores[],
  field: FieldAthlete[],
  orderedAthleteIds: number[],
  tier: PointsTier,
  period: PeriodId,
  boardSize = 25,
): ProjectedRanking {
  const pointsMap = pointsForOrder(orderedAthleteIds, tier);

  // Baseline (today).
  const baseRanked = rankAthletes(allAthletes);
  const baselineLine = computeQualificationLine(allAthletes);
  const baseRankById = new Map(baseRanked.map((a, i) => [a.athleteId, i + 1]));

  // Apply each field athlete's projected points as a new hypothetical score.
  const projectedAthletes: AthleteScores[] = allAthletes.map((a) => {
    const proj = pointsMap.get(a.athleteId);
    if (!proj) return a;
    const hypo: Score = { points: proj.expectedPoints, period, tier, position: proj.position, hypothetical: true };
    return { ...a, scores: [...a.scores, hypo] };
  });

  const projRanked = rankAthletes(projectedAthletes);
  const projectedLine = computeQualificationLine(projectedAthletes);
  const projRankById = new Map(projRanked.map((a, i) => [a.athleteId, i + 1]));
  const baseQual = new Set(baselineLine.qualified.map((q) => q.athleteId));
  const projQual = new Set(projectedLine.qualified.map((q) => q.athleteId));

  const fieldById = new Map(field.map((f) => [f.athleteId, f]));

  const rows: ProjectedRow[] = field
    .filter((f) => pointsMap.has(f.athleteId))
    .map((f) => {
      const proj = pointsMap.get(f.athleteId)!;
      const currentRank = baseRankById.get(f.athleteId) ?? null;
      const projectedRank = projRankById.get(f.athleteId) ?? projRanked.length;
      const currentTotal = selectCountingScores(fieldById.get(f.athleteId)!.scores).total;
      const projectedTotal = selectCountingScores(
        projectedAthletes.find((a) => a.athleteId === f.athleteId)!.scores,
      ).total;
      return {
        athleteId: f.athleteId,
        fullName: f.fullName,
        noc: f.noc,
        projectedPosition: proj.position,
        expectedPoints: proj.expectedPoints,
        currentRank,
        currentTotal,
        projectedRank,
        projectedTotal,
        rankDelta: (currentRank ?? projectedRank) - projectedRank,
        pointsDelta: Math.round((projectedTotal - currentTotal) * 100) / 100,
        crossesLine: !baseQual.has(f.athleteId) && projQual.has(f.athleteId),
        dropsOut: baseQual.has(f.athleteId) && !projQual.has(f.athleteId),
        qualified: projQual.has(f.athleteId),
      };
    })
    .sort((a, b) => a.projectedPosition - b.projectedPosition);

  const racerIds = new Set(pointsMap.keys());
  const board: BoardRow[] = projRanked.slice(0, boardSize).map((a, i) => {
    const projectedRank = i + 1;
    const currentRank = baseRankById.get(a.athleteId) ?? null;
    return {
      athleteId: a.athleteId,
      fullName: a.fullName,
      noc: a.noc,
      projectedRank,
      currentRank,
      rankDelta: (currentRank ?? projectedRank) - projectedRank,
      projectedTotal: a.total,
      qualified: projQual.has(a.athleteId),
      isRacer: racerIds.has(a.athleteId),
      crossesLine: !baseQual.has(a.athleteId) && projQual.has(a.athleteId),
      dropsOut: baseQual.has(a.athleteId) && !projQual.has(a.athleteId),
    };
  });

  return {
    rows,
    board,
    projectedLine,
    baselineLine,
    cutRank: projectedLine.cutRank,
    cutPoints: projectedLine.cutPoints,
  };
}

function clamp01(n: number) {
  return Math.min(1, Math.max(0, n));
}
