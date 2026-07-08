/**
 * Profile view for an athlete who is NOT in the Olympic Qualification Ranking.
 * We can't show a rank/qualification cockpit (they have no OQR points), so we
 * show an honest profile: who they are, their nation, recent results, and what
 * it takes to enter the ranking.
 */
import "server-only";
import { fetchAthleteProfile, fetchAthleteResults, type AthleteResult } from "@/lib/wt-api/athletes";
import { QUAL } from "@/config/qualification";
import type { Gender } from "@/config/pathways";

export interface UnrankedProfile {
  athleteId: number;
  fullName: string;
  noc: string;
  countryName?: string;
  gender: Gender;
  age?: number;
  profileImage?: string;
  flag?: string;
  results: AthleteResult[];
  eligibilityTopRank: number;
}

export async function buildUnrankedProfile(
  athleteId: number,
  todayIso?: string,
): Promise<UnrankedProfile | null> {
  const profile = await fetchAthleteProfile(athleteId);
  if (!profile) return null;
  let results: AthleteResult[] = [];
  try {
    results = await fetchAthleteResults(athleteId, 8);
  } catch {
    results = [];
  }
  const now = todayIso ? new Date(todayIso) : new Date();
  const age = profile.yearOfBirth ? now.getUTCFullYear() - profile.yearOfBirth : undefined;

  return {
    athleteId: profile.athleteId,
    fullName: profile.fullName,
    noc: profile.noc,
    countryName: profile.countryName,
    gender: profile.gender,
    age,
    profileImage: profile.profileImage,
    flag: profile.flag,
    results,
    eligibilityTopRank: QUAL.eligibilityTopRank,
  };
}
