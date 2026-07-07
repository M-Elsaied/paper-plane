import { describe, it, expect, afterEach, vi } from "vitest";
import { fetchRaceStartLists } from "@/lib/wt-api/start-list";
import { installWtFetch } from "../helpers/mock-wt";

afterEach(() => vi.unstubAllGlobals());

const programs = { data: [{ prog_id: 10, prog_name: "Elite Men" }, { prog_id: 11, prog_name: "Elite Women" }] };

describe("fetchRaceStartLists", () => {
  it("excludes waitlisted athletes and sorts by start number", async () => {
    installWtFetch([
      { match: "/programs/10/entries", response: { data: { entries: [
        { athlete_id: 3, start_num: 3 },
        { athlete_id: 1, start_num: 1 },
        { athlete_id: 99, start_num: 5, wait_pos: 2 }, // waitlisted → excluded
        { athlete_id: 2, start_num: 2 },
      ] } } },
      { match: "/programs/11/entries", response: { data: { entries: [] } } },
      { match: "/programs", response: programs },
    ]);
    const lists = await fetchRaceStartLists(1);
    expect(lists.men).toEqual([1, 2, 3]);
    expect(lists.women).toEqual([]);
  });

  it("sorts null start numbers last", async () => {
    installWtFetch([
      { match: "/programs/10/entries", response: { data: { entries: [
        { athlete_id: 5, start_num: null }, { athlete_id: 1, start_num: 1 },
      ] } } },
      { match: "/programs/11/entries", response: { data: { entries: [] } } },
      { match: "/programs", response: programs },
    ]);
    const lists = await fetchRaceStartLists(1);
    expect(lists.men).toEqual([1, 5]);
  });
});
