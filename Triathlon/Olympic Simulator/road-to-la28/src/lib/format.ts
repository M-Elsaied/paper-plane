import { QUAL } from "@/config/qualification";

/** Days remaining until the individual-ranking deadline (18 May 2028). */
export function daysToDeadline(nowIso?: string): number {
  const now = nowIso ? new Date(nowIso) : new Date();
  const end = new Date(QUAL.deadline);
  return Math.max(0, Math.ceil((end.getTime() - now.getTime()) / 86_400_000));
}

export function fmtPoints(n: number | null | undefined): string {
  if (n == null) return "—";
  return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

export function fmtDelta(n: number): string {
  const r = Math.round(n);
  return r > 0 ? `+${r}` : `${r}`;
}

export function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

/** Flag emoji fallback from a 2-letter ISO code (unused if flag image present). */
export function nocLabel(noc: string): string {
  return noc;
}
