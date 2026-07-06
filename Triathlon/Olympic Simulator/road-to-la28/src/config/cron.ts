/**
 * Cron cadences — kept in sync with vercel.json. Documented here so schedules
 * are visible next to the rest of the tunable config.
 *
 * Note: Vercel Hobby allows only daily crons (max 2). For hourly race-weekend
 * result polling either upgrade to Pro or drive the protected routes from a
 * GitHub Actions schedule (see README).
 */
export const CRON = {
  syncRankings: "0 6 * * *", // daily 06:00 UTC — hash-noop unless WT published
  syncEvents: "0 5 * * *", // daily 05:00 UTC — events window -7d..+90d
  syncResultsWeekend: "0 * * * 5,6,0", // hourly Fri/Sat/Sun
  syncResultsMonday: "0 7 * * 1", // Monday sweep
  syncAthletes: "0 4 * * 3", // weekly Wed 04:00 UTC
  /** Guard: ignore a new trigger if a run of the same job is <N minutes old. */
  inFlightGuardMinutes: 10,
  /** Event raw-payload retention (days). Ranking payloads are kept forever. */
  eventPayloadRetentionDays: 90,
} as const;
