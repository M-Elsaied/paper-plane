/**
 * What-if simulator — the interactive heart of the cockpit.
 *
 * Given the current per-gender state and a hypothetical result ("athlete X
 * finishes Pth at a WTCS race"), recompute the whole ranking + qualification
 * line and diff it against the baseline. Pure and fast (~160 athletes x ~15
 * scores re-ranks in well under a millisecond) so it runs client-side on every
 * slider tick for zero-latency interactivity.
 */
import { DEFAULT_ASSUMPTIONS, type PathwayAssumptions } from "@/config/pathways";
import { pointsForPosition, type PointsTier } from "@/config/points-tables";
import type { PeriodId } from "@/config/qualification";
import { rankAthletes, computeQualificationLine } from "./qualification";
import type { AthleteScores, QualLine, QualState, Score } from "./types";

export interface WhatIfScenario {
  athleteId: number;
  tier: PointsTier;
  position: number;
  period: PeriodId;
  /** If set, overrides the tier/position points (used for direct entry). */
  points?: number;
}

export interface WhatIfResult {
  athleteId: number;
  hypotheticalPoints: number;
  before: { rank: number; total: number; qualified: boolean };
  after: { rank: number; total: number; qualified: boolean };
  rankDelta: number; // positive = moved up
  pointsDelta: number;
  crossesLine: boolean; // moved from outside -> inside the line
  dropsOut: boolean; // moved from inside -> outside the line
  /** Athletes pushed out of / down within the qualified set by this scenario. */
  displaced: { athleteId: number; fullName: string; noc: string; fromRank: number; toRank: number }[];
  line: QualLine; // the post-scenario line (for rendering)
  baselineLine: QualLine;
}

function rankOf(line: QualLine, ranked: ReturnType<typeof rankAthletes>, athleteId: number): number {
  const idx = ranked.findIndex((a) => a.athleteId === athleteId);
  return idx >= 0 ? idx + 1 : ranked.length + 1;
}

function isQualified(line: QualLine, athleteId: number): boolean {
  return line.qualified.some((q) => q.athleteId === athleteId);
}

export function applyWhatIf(
  state: QualState,
  scenario: WhatIfScenario,
  assumptions: PathwayAssumptions = DEFAULT_ASSUMPTIONS,
): WhatIfResult {
  const points =
    scenario.points ?? pointsForPosition(scenario.tier, scenario.position);

  const hypo: Score = {
    points,
    period: scenario.period,
    tier: scenario.tier,
    position: scenario.position,
    hypothetical: true,
  };

  // Baseline.
  const baseRanked = rankAthletes(state.athletes);
  const baselineLine = computeQualificationLine(state.athletes, assumptions);

  // Scenario: clone the target athlete's score pool with the hypothetical added.
  const modified: AthleteScores[] = state.athletes.map((a) =>
    a.athleteId === scenario.athleteId
      ? { ...a, scores: [...a.scores, hypo] }
      : a,
  );
  const afterRanked = rankAthletes(modified);
  const afterLine = computeQualificationLine(modified, assumptions);

  const beforeRank = rankOf(baselineLine, baseRanked, scenario.athleteId);
  const afterRank = rankOf(afterLine, afterRanked, scenario.athleteId);
  const beforeTotal = baseRanked.find((a) => a.athleteId === scenario.athleteId)?.total ?? 0;
  const afterTotal = afterRanked.find((a) => a.athleteId === scenario.athleteId)?.total ?? 0;
  const beforeQual = isQualified(baselineLine, scenario.athleteId);
  const afterQual = isQualified(afterLine, scenario.athleteId);

  // Who slipped a rank inside the qualified region because of this?
  const beforeRankById = new Map(baseRanked.map((a, i) => [a.athleteId, i + 1]));
  const displaced = afterRanked
    .map((a, i) => ({ athleteId: a.athleteId, fullName: a.fullName, noc: a.noc, toRank: i + 1, fromRank: beforeRankById.get(a.athleteId) ?? i + 1 }))
    .filter((d) => d.athleteId !== scenario.athleteId && d.toRank > d.fromRank && d.fromRank <= (baselineLine.cutRank ?? 0))
    .slice(0, 6);

  return {
    athleteId: scenario.athleteId,
    hypotheticalPoints: points,
    before: { rank: beforeRank, total: beforeTotal, qualified: beforeQual },
    after: { rank: afterRank, total: afterTotal, qualified: afterQual },
    rankDelta: beforeRank - afterRank,
    pointsDelta: Math.round((afterTotal - beforeTotal) * 100) / 100,
    crossesLine: !beforeQual && afterQual,
    dropsOut: beforeQual && !afterQual,
    displaced,
    line: afterLine,
    baselineLine,
  };
}
