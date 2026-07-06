/**
 * Fetch upcoming World Triathlon events for the race-week view. We pull the
 * points-relevant elite categories in a forward window and keep a minimal
 * normalized shape.
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
      all.push({
        eventId: e.event_id,
        title: e.event_title,
        date: e.event_date,
        endDate: e.event_finish_date,
        venue: e.event_venue,
        country: e.event_country_name,
        flag: e.event_flag,
        categoryLabel: cat.label,
        tier,
        tierLabel: TIER_LABEL[tier],
      });
    }
  }
  return all.sort((a, b) => a.date.localeCompare(b.date));
}
