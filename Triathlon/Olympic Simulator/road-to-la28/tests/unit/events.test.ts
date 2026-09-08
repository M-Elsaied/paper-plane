import { describe, it, expect, afterEach, vi } from "vitest";
import { fetchUpcomingEvents, fetchEvent, isEliteTriathlon, resolveTier, type RawEvent } from "@/lib/wt-api/events";
import { installWtFetch, wtOk } from "../helpers/mock-wt";

afterEach(() => vi.unstubAllGlobals());

const TRI = [{ cat_id: 357, cat_name: "Triathlon" }];
const asianGames: RawEvent = {
  event_id: 195355,
  event_title: "2026 Aichi-Nagoya Asian Games",
  event_date: "2026-09-20",
  event_categories: [
    { cat_id: 343, cat_name: "Major Games" },
    { cat_id: 340, cat_name: "Continental Championships" },
  ],
  event_specifications: TRI,
};
const yog: RawEvent = {
  event_id: 194999,
  event_title: "Dakar 2026 Youth Olympic Games",
  event_date: "2026-11-05",
  event_categories: [{ cat_id: 343, cat_name: "Major Games" }],
  event_specifications: TRI,
};
const rome: RawEvent = {
  event_id: 1,
  event_title: "2026 World Triathlon Cup Rome",
  event_date: "2026-10-03",
  event_categories: [{ cat_id: 349, cat_name: "World Cup" }],
  event_specifications: TRI,
};
const karlovy: RawEvent = {
  event_id: 195152,
  event_title: "2026 World Triathlon Championship Series Karlovy Vary",
  event_date: "2026-09-13",
  event_categories: [{ cat_id: 351, cat_name: "World Championship Series" }],
  event_specifications: TRI,
};
const duathlon: RawEvent = {
  event_id: 2,
  event_title: "2026 Americas Triathlon Duathlon Championships Merida",
  event_date: "2026-09-19",
  event_categories: [{ cat_id: 340, cat_name: "Continental Championships" }],
  event_specifications: [{ cat_id: 358, cat_name: "Duathlon" }],
};
const youthChamps: RawEvent = {
  event_id: 3,
  event_title: "2026 Europe Triathlon Youth Championships Melilla",
  event_date: "2026-10-16",
  event_categories: [{ cat_id: 340, cat_name: "Continental Championships" }],
  event_specifications: TRI,
};
const hurghada: RawEvent = {
  event_id: 4,
  event_title: "2026 Africa Triathlon Championships Hurghada",
  event_date: "2026-10-09",
  event_categories: [{ cat_id: 340, cat_name: "Continental Championships" }],
  event_specifications: TRI,
};

/** Route the calendar queries by their category_id param, like the real API. */
function calendarRoutes() {
  const byCat: Record<string, RawEvent[]> = {
    "624": [],
    "351": [karlovy],
    "349": [rome],
    "343": [asianGames, yog],
    "340": [duathlon, asianGames, hurghada, youthChamps],
  };
  return [
    {
      match: "/events?",
      response: (url: string) => ({ body: wtOk(byCat[new URL(url).searchParams.get("category_id") ?? ""] ?? []) }),
    },
  ];
}

describe("resolveTier / isEliteTriathlon", () => {
  it("picks the highest-scoring of an event's own category tags", () => {
    expect(resolveTier(asianGames)).toBe("games");
    expect(resolveTier(hurghada)).toBe("continental_champs");
    expect(resolveTier(rome)).toBe("world_cup");
    expect(resolveTier({ ...rome, event_categories: [] }, "wtcs")).toBe("wtcs");
  });

  it("keeps elite triathlon and drops youth / non-triathlon fields", () => {
    expect(isEliteTriathlon(asianGames)).toBe(true);
    expect(isEliteTriathlon(yog)).toBe(false);
    expect(isEliteTriathlon(duathlon)).toBe(false);
    expect(isEliteTriathlon(youthChamps)).toBe(false);
  });
});

describe("fetchUpcomingEvents", () => {
  it("tags by the event's own categories, dedupes across queries, excludes non-elite", async () => {
    installWtFetch(calendarRoutes());
    const events = await fetchUpcomingEvents("2026-09-08", "2027-01-06");
    const ids = events.map((e) => e.eventId);
    expect(ids).toEqual([195152, 195355, 1, 4]); // date order, one Asian Games, no YOG/duathlon/youth
    const byId = Object.fromEntries(events.map((e) => [e.eventId, e]));
    expect(byId[195355].tier).toBe("games");
    expect(byId[195355].tierLabel).toBe("Games");
    expect(byId[195355].categoryLabel).toBe("Major Games");
    expect(byId[1].tier).toBe("world_cup");
    expect(byId[195152].tier).toBe("wtcs");
    expect(byId[4].tier).toBe("continental_champs");
  });
});

describe("fetchEvent", () => {
  it("resolves a single event's tier from its tags", async () => {
    installWtFetch([{ match: "/events/195355", response: wtOk(asianGames) }]);
    const e = await fetchEvent(195355);
    expect(e?.tier).toBe("games");
    expect(e?.title).toContain("Asian Games");
  });
});
