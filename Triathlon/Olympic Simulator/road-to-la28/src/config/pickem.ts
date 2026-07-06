/**
 * Race Pick-'Em scoring — tunable. Predict the podium (1st/2nd/3rd) before a
 * race; auto-scored against the official result.
 */
export const PICKEM = {
  /** Right athlete in the exact predicted slot. */
  exact: { 1: 10, 2: 6, 3: 4 } as Record<number, number>,
  /** Right athlete made the podium but in a different slot. */
  onPodiumWrongSpot: 2,
  /** Bonus for a perfect podium (all three exact). */
  perfectBonus: 5,
  /** Max achievable (for progress bars): 10+6+4+5. */
  maxScore: 25,
} as const;
