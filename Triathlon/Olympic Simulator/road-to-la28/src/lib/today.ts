import seedMeta from "@/data/seed-meta.json";

/**
 * The app's notion of "today" (ISO yyyy-mm-dd).
 * - RTLA28_TODAY pins it (E2E, screenshot captures, reproducible demos).
 * - Seed mode (no DATABASE_URL) uses the seed's pinned date so the committed
 *   calendar and rankings stay coherent with each other.
 * - Otherwise the real clock: production reads live data (Neon + WT), so the
 *   calendar, period maths and "as of" labels must move with the calendar.
 */
export function todayIso(): string {
  const pinned = process.env.RTLA28_TODAY?.trim();
  if (pinned) return pinned;
  if (!process.env.DATABASE_URL && seedMeta.today) return seedMeta.today;
  return new Date().toISOString().slice(0, 10);
}

export function plusDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}
