/**
 * Fetch upcoming World Triathlon events for the race-week view. We pull every
 * elite category that awards Olympic qualification points in a forward window
 * and keep a minimal normalized shape. `fetchEvent` resolves a single event by
 * id (any date) so the race companion works for races outside the window.
 *
 * An event's tier comes from ITS OWN category tags (an event can carry several,
 * e.g. the Asian Games are "Major Games" + "Continental Championships"), never
 * from which query it happened to be returned by.
 */
import { wtGet } from "./client";
import {
  tierForCategory,
  TIER_BASE_POINTS,
  TIER_LABEL,
  CATEGORY_TIER,
  type PointsTier,
} from "@/config/points-tables";

/** WT category ids whose elite races award Olympic qualification points.
 *  Ids per GET /events/categories (343 is "Major Games", NOT World Cup = 349). */
export const ELITE_EVENT_CATEGORIES: { id: number; label: string; tier: PointsTier }[] = [
  { id: 624, label: "WTCS Final", tier: "wtcs_final" },
  { id: 351, label: "WTCS", tier: "wtcs" },
  { id: 349, label: "World Cup", tier: "world_cup" },
  { id: 343, label: "Major Games", tier: "games" },
  { id: 340, label: "Continental Champs", tier: "continental_champs" },
];

/** Titles that share a points-scoring category but award no elite OQR points
 *  (youth/junior/age-group fields, para, non-triathlon multisport). Tunable. */
export const NON_ELITE_TITLE =
  /\b(youth|junior|u23|age[- ]group|para|duathlon|aquathlon|long[- ]distance|cross|winter|indoor|arena)\b/i;

export interface RawEvent {
  event_id: number;
  event_title: string;
  event_date: string;
  event_finish_date?: string;
  event_venue?: string;
  event_country_name?: string;
  event_flag?: string;
  event_categories?: { cat_id: number; cat_name: string }[];
  /** Discipline/format tags, e.g. "Triathlon", "Sprint", "Mixed Relay". */
  event_specifications?: { cat_id: number; cat_name: string }[];
}

export interface UpcomingEvent {
  eventId: number;
  title: string;
  date: string;
  endDate?: string;
  venue?: string;
  country?: string;
  flag?: string;
  categoryLabel: string;
  tier: PointsTier;
  tierLabel: string;
}

/** Elite triathlon only: the discipline must be Triathlon (when WT tags it) and
 *  the title must not mark a non-elite or non-triathlon field. */
export function isEliteTriathlon(e: RawEvent): boolean {
  const specs = (e.event_specifications ?? []).map((s) => s.cat_name.toLowerCase());
  if (specs.length && !specs.includes("triathlon")) return false;
  return !NON_ELITE_TITLE.test(e.event_title);
}

/** Highest-scoring tier among the event's own category tags; `fallback` when
 *  none of them is a known points category. */
export function resolveTier(e: RawEvent, fallback: PointsTier = "other"): PointsTier {
  const tiers = (e.event_categories ?? [])
    .map((c) => tierForCategory(c.cat_id))
    .filter((t) => t !== "other");
  if (!tiers.length) return fallback;
  return tiers.reduce((best, t) => (TIER_BASE_POINTS[t] > TIER_BASE_POINTS[best] ? t : best));
}

function normalizeEvent(e: RawEvent, tier: PointsTier): UpcomingEvent {
  const known = ELITE_EVENT_CATEGORIES.find((c) => c.tier === tier);
  return {
    eventId: e.event_id,
    title: e.event_title,
    date: e.event_date,
    endDate: e.event_finish_date,
    venue: e.event_venue,
    country: e.event_country_name,
    flag: e.event_flag,
    categoryLabel: known?.label ?? e.event_categories?.[0]?.cat_name ?? TIER_LABEL[tier],
    tier,
    tierLabel: TIER_LABEL[tier],
  };
}

export async function fetchUpcomingEvents(
  fromIso: string,
  toIso: string,
): Promise<UpcomingEvent[]> {
  const byId = new Map<number, UpcomingEvent>();
  for (const cat of ELITE_EVENT_CATEGORIES) {
    const res = await wtGet<RawEvent[]>("/events", {
      category_id: cat.id,
      start_date: fromIso,
      end_date: toIso,
      per_page: 50,
      order: "asc",
    });
    for (const e of res.data ?? []) {
      if (byId.has(e.event_id) || !isEliteTriathlon(e)) continue;
      byId.set(e.event_id, normalizeEvent(e, resolveTier(e, cat.tier)));
    }
  }
  return [...byId.values()].sort((a, b) => a.date.localeCompare(b.date));
}

/** A single event by id (past or future). Not filtered — the caller asked for it. */
export async function fetchEvent(eventId: number): Promise<UpcomingEvent | null> {
  const res = await wtGet<RawEvent>(`/events/${eventId}`);
  const e = res.data;
  if (!e || !e.event_id) return null;
  return normalizeEvent(e, resolveTier(e));
}

/** Sanity: every queried category must map to the tier we label it with. */
for (const c of ELITE_EVENT_CATEGORIES) {
  if (CATEGORY_TIER[c.id] !== c.tier) {
    throw new Error(`points-tables CATEGORY_TIER[${c.id}] must be "${c.tier}" (got "${CATEGORY_TIER[c.id]}")`);
  }
}
