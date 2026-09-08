/**
 * Fetch upcoming World Triathlon events for the race-week view. We pull the
 * points-relevant elite categories in a forward window and keep a minimal
 * normalized shape. `fetchEvent` resolves a single event by id (any date) so
 * the race companion works for races that have dropped out of the window.
 */
import { wtGet } from "./client";
import { tierForCategory, TIER_LABEL, type PointsTier } from "@/config/points-tables";

/** WT category ids whose elite races award Olympic qualification points. */
export const ELITE_EVENT_CATEGORIES: { id: number; label: string; tier: PointsTier }[] = [
  { id: 351, label: "WTCS", tier: "wtcs" },
  { id: 343, label: "World Cup", tier: "world_cup" },
];

export interface RawEvent {
  event_id: number;
  event_title: string;
  event_date: string;
  event_finish_date?: string;
  event_venue?: string;
  event_country_name?: string;
  event_flag?: string;
  event_categories?: { cat_id: number; cat_name: string }[];
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

function normalizeEvent(e: RawEvent, cat: { label: string; tier: PointsTier }): UpcomingEvent {
  return {
    eventId: e.event_id,
    title: e.event_title,
    date: e.event_date,
    endDate: e.event_finish_date,
    venue: e.event_venue,
    country: e.event_country_name,
    flag: e.event_flag,
    categoryLabel: cat.label,
    tier: cat.tier,
    tierLabel: TIER_LABEL[cat.tier],
  };
}

export async function fetchUpcomingEvents(
  fromIso: string,
  toIso: string,
): Promise<UpcomingEvent[]> {
  const all: UpcomingEvent[] = [];
  for (const cat of ELITE_EVENT_CATEGORIES) {
    const res = await wtGet<RawEvent[]>("/events", {
      category_id: cat.id,
      start_date: fromIso,
      end_date: toIso,
      per_page: 50,
      order: "asc",
    });
    for (const e of res.data ?? []) {
      const tier = tierForCategory(cat.id) === "other" ? cat.tier : tierForCategory(cat.id);
      all.push(normalizeEvent(e, { label: cat.label, tier }));
    }
  }
  return all.sort((a, b) => a.date.localeCompare(b.date));
}

/** A single event by id (past or future). Tier comes from its own categories. */
export async function fetchEvent(eventId: number): Promise<UpcomingEvent | null> {
  const res = await wtGet<RawEvent>(`/events/${eventId}`);
  const e = res.data;
  if (!e || !e.event_id) return null;
  const known = ELITE_EVENT_CATEGORIES.find((c) => e.event_categories?.some((ec) => ec.cat_id === c.id));
  const firstCat = e.event_categories?.[0];
  const cat = known ?? { label: firstCat?.cat_name ?? "Other", tier: tierForCategory(firstCat?.cat_id) };
  return normalizeEvent(e, cat);
}
