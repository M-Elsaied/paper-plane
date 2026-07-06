/**
 * World Triathlon points tables — versioned.
 *
 * Two responsibilities:
 *  1. Map a World Triathlon event category id -> our internal points tier.
 *  2. Provide position -> points for HYPOTHETICAL results (what-if scenarios,
 *     race-week projections). For REAL past results the engine always prefers
 *     the API's own per-score point value; these tables are the fallback.
 *
 * The position curve below is the standard WT allocation shape (winner = base
 * category points, decaying by a published percentage per place). Values are
 * validated in CI against real published OQR totals (see tests/engine/golden).
 * Bump POINTS_TABLES_VERSION whenever any number here changes.
 */

export const POINTS_TABLES_VERSION = 1;

/** Internal tiers. WT event `cat_id` -> tier is resolved via CATEGORY_TIER. */
export type PointsTier =
  | "grand_final"
  | "wtcs"
  | "wtcs_final"
  | "world_cup"
  | "continental_champs"
  | "games"
  | "continental_cup"
  | "other";

/** Base points awarded to the winner of an event in each tier. */
export const TIER_BASE_POINTS: Record<PointsTier, number> = {
  grand_final: 1250,
  wtcs_final: 1250,
  wtcs: 1000,
  games: 1000,
  continental_champs: 800,
  world_cup: 500,
  continental_cup: 350,
  other: 250,
};

/**
 * Percentage of the winner's points earned by each finishing position (1-based).
 * Index 0 -> 1st place. Standard World Triathlon decay curve.
 * Positions beyond the array earn TAIL_PERCENT.
 */
export const POSITION_PERCENT: number[] = [
  100.0, 92.5, 86.0, 80.5, 75.5, 71.0, 67.0, 63.5, 60.5, 57.5, // 1-10
  55.0, 52.5, 50.0, 47.5, 45.0, 42.5, 40.0, 38.0, 36.0, 34.0, // 11-20
  32.0, 30.0, 28.0, 26.0, 24.0, 22.5, 21.0, 19.5, 18.0, 16.5, // 21-30
  15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, // 31-40
];
const TAIL_PERCENT = 5.0;

/** WT event `cat_id` -> internal tier. Extend as new category ids are observed. */
export const CATEGORY_TIER: Record<number, PointsTier> = {
  351: "wtcs", // World Championship Series
  // Discovered/extended by scripts/discover-ranking-ids + ingestion. Common ids:
  // World Cup, Continental Championships, Games, etc. are added as encountered.
};

/** Human labels for tiers (UI). */
export const TIER_LABEL: Record<PointsTier, string> = {
  grand_final: "Grand Final",
  wtcs_final: "WTCS Final",
  wtcs: "WTCS",
  games: "Games",
  continental_champs: "Continental Champs",
  world_cup: "World Cup",
  continental_cup: "Continental Cup",
  other: "Other",
};

/** Points a given finishing position would earn in a given tier (hypothetical). */
export function pointsForPosition(tier: PointsTier, position: number): number {
  const base = TIER_BASE_POINTS[tier];
  const pct =
    position >= 1 && position <= POSITION_PERCENT.length
      ? POSITION_PERCENT[position - 1]
      : TAIL_PERCENT;
  return Math.round(base * (pct / 100) * 100) / 100;
}

/** Resolve a WT category id to our tier (defaults to "other"). */
export function tierForCategory(catId: number | null | undefined): PointsTier {
  if (catId == null) return "other";
  return CATEGORY_TIER[catId] ?? "other";
}
