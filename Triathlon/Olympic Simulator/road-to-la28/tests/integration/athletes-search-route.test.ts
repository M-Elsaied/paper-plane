import { describe, it, expect, afterEach, vi } from "vitest";
import { GET } from "@/app/api/athletes/search/route";
import { installWtFetch } from "../helpers/mock-wt";

afterEach(() => vi.unstubAllGlobals());

const get = (q: string) => GET(new Request(`http://test/api/athletes/search?q=${encodeURIComponent(q)}`));

describe("athlete search route", () => {
  it("annotates OQR-ranked hits and sorts them first", async () => {
    // 86042 (Vilaca) is #1 in the committed seed OQR; 70338 is unranked.
    installWtFetch([
      { match: "/search/athletes", response: { data: [
        { athlete_id: 70338, athlete_title: "Mohamed Elsaied", athlete_noc: "EGY", athlete_gender: "male" },
        { athlete_id: 86042, athlete_title: "Vasco Vilaca", athlete_noc: "POR", athlete_gender: "male" },
      ] } },
    ]);
    const body = await (await get("el")).json();
    expect(body.results[0].athleteId).toBe(86042); // ranked → first
    expect(body.results[0].rank).toBe(1);
    expect(body.results[1].athleteId).toBe(70338);
    expect(body.results[1].rank).toBeNull(); // unranked
  });

  it("returns empty for a <2-char query without calling WT", async () => {
    const spy = installWtFetch([]);
    const body = await (await get("x")).json();
    expect(body.results).toEqual([]);
    expect(spy).not.toHaveBeenCalled();
  });

  it("degrades gracefully when WT search errors", async () => {
    // 500s trigger the client's retry+backoff, so allow extra time.
    installWtFetch([{ match: "/search/athletes", response: () => ({ body: "boom", status: 500 }) }]);
    const body = await (await get("Elsaied")).json();
    expect(body.results).toEqual([]);
  }, 20000);
});
