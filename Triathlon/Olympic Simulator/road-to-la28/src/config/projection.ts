/**
 * Race-week projection model — deterministic and explainable (no ML).
 * All weights tunable here; the engine records a per-projection breakdown so
 * the UI can show exactly why an athlete is projected where they are.
 */

export const PROJECTION = {
  /** Seed = weights.ranking * fieldRankComponent + weights.form * formComponent. */
  weights: { ranking: 0.6, form: 0.4 },
  form: {
    /** Consider the athlete's last N scored results. */
    lastN: 5,
    /** Recency half-life in days (older results decay). */
    halfLifeDays: 90,
    /** Below this many usable results, fall back to ranking component alone. */
    minResults: 2,
  },
} as const;
