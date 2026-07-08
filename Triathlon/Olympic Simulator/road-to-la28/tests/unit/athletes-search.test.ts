import { describe, it, expect, afterEach, vi } from "vitest";
import { searchAthletes, fetchAthleteProfile, fetchAthleteResults } from "@/lib/wt-api/athletes";
import { installWtFetch } from "../helpers/mock-wt";

afterEach(() => vi.unstubAllGlobals());

describe("searchAthletes", () => {
  it("normalizes hits and requires a 2-char query", async () => {
    installWtFetch([
      { match: "/search/athletes", response: { data: [
        { athlete_id: 70338, athlete_title: "Mohamed Elsaied", athlete_noc: "EGY", athlete_gender: "male", athlete_yob: "1993", athlete_profile_image: "img", athlete_flag_circle: "flag" },
      ] } },
    ]);
    expect(await searchAthletes("a")).toEqual([]); // too short → no call
    const hits = await searchAthletes("Elsaied");
    expect(hits[0]).toMatchObject({ athleteId: 70338, fullName: "Mohamed Elsaied", noc: "EGY", gender: "male", yearOfBirth: 1993, profileImage: "img", flag: "flag" });
  });
});

describe("fetchAthleteProfile", () => {
  it("reads a profile object and coerces yob", async () => {
    installWtFetch([
      { match: "/athletes/70338", response: { data: { athlete_id: 70338, athlete_title: "Mohamed Elsaied", athlete_noc: "EGY", athlete_country_name: "Egypt", athlete_gender: "male", athlete_yob: 1993 } } },
    ]);
    const p = (await fetchAthleteProfile(70338))!;
    expect(p.fullName).toBe("Mohamed Elsaied");
    expect(p.countryName).toBe("Egypt");
    expect(p.yearOfBirth).toBe(1993);
  });

  it("returns null for an unknown athlete", async () => {
    installWtFetch([{ match: "/athletes/999", response: { data: [] } }]);
    expect(await fetchAthleteProfile(999)).toBeNull();
  });
});

describe("fetchAthleteResults", () => {
  it("maps results and preserves DNF/string positions", async () => {
    installWtFetch([
      { match: "/athletes/70338/results", response: { data: [
        { event_id: 1, event_title: "2023 Africa Champs", event_date: "2023-10-13", prog_name: "Elite Men", position: "DNF" },
        { event_id: 2, event_title: "2022 AG Champs", event_date: "2022-11-24", prog_name: "AG", position: 14 },
      ] } },
    ]);
    const r = await fetchAthleteResults(70338);
    expect(r).toHaveLength(2);
    expect(r[0].position).toBe("DNF");
    expect(r[1].position).toBe(14);
  });
});
