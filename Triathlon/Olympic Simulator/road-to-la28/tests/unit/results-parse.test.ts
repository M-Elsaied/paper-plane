import { describe, it, expect, afterEach, vi } from "vitest";
import { fetchRacePodiums } from "@/lib/wt-api/results";
import { installWtFetch } from "../helpers/mock-wt";

afterEach(() => vi.unstubAllGlobals());

const programs = { code: 200, status: "ok", data: [{ prog_id: 10, prog_name: "Elite Men" }, { prog_id: 11, prog_name: "Elite Women" }] };

describe("fetchRacePodiums", () => {
  it("extracts the top-3 athlete ids in order", async () => {
    installWtFetch([
      { match: "/programs/10/results", response: { data: { results: [
        { athlete_id: 30, position: 3 }, { athlete_id: 10, position: 1 }, { athlete_id: 20, position: 2 }, { athlete_id: 40, position: 4 },
      ] } } },
      { match: "/programs/11/results", response: { data: { results: [] } } },
      { match: "/programs", response: programs },
    ]);
    const p = await fetchRacePodiums(1);
    expect(p.male).toEqual([10, 20, 30]);
    expect(p.female).toEqual([]);
  });

  it("handles string/DNF positions and ignores non-podium", async () => {
    installWtFetch([
      { match: "/programs/10/results", response: { data: { results: [
        { athlete_id: 1, position: "1" }, { athlete_id: 2, position: "DNF" }, { athlete_id: 3, position: 2 }, { athlete_id: 4, position: 3 },
      ] } } },
      { match: "/programs/11/results", response: { data: { results: [] } } },
      { match: "/programs", response: programs },
    ]);
    const p = await fetchRacePodiums(1);
    expect(p.male).toEqual([1, 3, 4]);
  });

  it("accepts results as a bare array (asList) and returns empty when none", async () => {
    installWtFetch([
      { match: "/programs/10/results", response: { data: [{ athlete_id: 7, position: 1 }] } },
      { match: "/programs/11/results", response: { data: { results: [] } } },
      { match: "/programs", response: programs },
    ]);
    const p = await fetchRacePodiums(1);
    expect(p.male).toEqual([7]); // fewer than 3 → what's available
  });
});
