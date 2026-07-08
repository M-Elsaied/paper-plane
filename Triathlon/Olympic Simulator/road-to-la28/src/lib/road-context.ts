/**
 * Server assembler for "Your Road". Finds the athlete (ranked in the OQR, or
 * unranked via a live World Triathlon profile), builds the analysis context for
 * their gender, and returns the road plus display info.
 */
import "server-only";
import { rankAthletes, computeQualificationLine } from "@/lib/engine/qualification";
import { analyzeRoad, type Road } from "@/lib/engine/road";
import { continentOf } from "@/config/continents";
import { DEFAULT_ASSUMPTIONS } from "@/config/pathways";
import { getQualState, getMrNations, getWorldRanking, findAthlete } from "@/lib/data";
import { fetchAthleteProfile, fetchAthleteResults, type AthleteResult } from "@/lib/wt-api/athletes";
import type { Gender } from "@/config/pathways";

export interface RoadView {
  road: Road;
  display: {
    profileImage?: string;
    flag?: string;
    countryName?: string;
    continentLabel: string | null;
  };
  /** Recent results, shown for unranked athletes. */
  results?: AthleteResult[];
}

export async function buildRoad(athleteId: number): Promise<RoadView | null> {
  const found = await findAthlete(athleteId);

  if (found) {
    // Ranked athlete — full context from their gender's OQR.
    const gender = found.athlete.gender as Gender;
    const state = await getQualState(gender);
    const ranked = rankAthletes(state.athletes);
    const line = computeQualificationLine(state.athletes);
    const road = analyzeRoad({
      ranked,
      line,
      mrNations: await getMrNations(),
      worldRanking: await getWorldRanking(gender),
      assumptions: DEFAULT_ASSUMPTIONS,
      subjectId: athleteId,
    });
    return {
      road,
      display: {
        profileImage: found.athlete.profileImage,
        flag: found.athlete.flag,
        continentLabel: continentOf(found.athlete.noc),
      },
    };
  }

  // Unranked — live profile drives the subject; still analyze against the field.
  const profile = await fetchAthleteProfile(athleteId).catch(() => null);
  if (!profile) return null;
  const gender = profile.gender;
  const state = await getQualState(gender);
  const ranked = rankAthletes(state.athletes);
  const line = computeQualificationLine(state.athletes);
  const road = analyzeRoad({
    ranked,
    line,
    mrNations: await getMrNations(),
    worldRanking: await getWorldRanking(gender),
    assumptions: DEFAULT_ASSUMPTIONS,
    subjectId: athleteId,
    subjectFallback: { name: profile.fullName, noc: profile.noc, gender, worldRank: null },
  });
  const results = await fetchAthleteResults(athleteId, 5).catch(() => []);
  return {
    road,
    display: {
      profileImage: profile.profileImage,
      flag: profile.flag,
      countryName: profile.countryName,
      continentLabel: continentOf(profile.noc),
    },
    results,
  };
}
