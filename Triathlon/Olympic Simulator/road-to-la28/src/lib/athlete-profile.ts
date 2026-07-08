/**
 * Profile view for an athlete who is NOT in the Olympic Qualification Ranking.
 * We can't show a rank/qualification cockpit (they have no OQR points), so we
 * show a rich, honest profile: who they are, their nation, career stats, recent
 * results, and what it takes to enter the ranking.
 */
import "server-only";
import { fetchAthleteProfile, fetchAthleteResults, type AthleteResult, type CareerStats } from "@/lib/wt-api/athletes";
import { QUAL } from "@/config/qualification";
import type { Gender } from "@/config/pathways";

export interface UnrankedProfile {
  athleteId: number;
  fullName: string;
  noc: string;
  countryName?: string;
  countryIso?: string;
  gender: Gender;
  yearOfBirth?: number;
  age?: number;
  profileImage?: string;
  flag?: string;
  stats?: CareerStats;
  ageGroupRank?: number;
  results: AthleteResult[];
  eligibilityTopRank: number;
}

export async function buildUnrankedProfile(
  athleteId: number,
  todayIso?: string,
): Promise<UnrankedProfile | null> {
  const profile = await fetchAthleteProfile(athleteId);
  if (!profile) return null;

  // The profile payload already carries recent results; only fall back to a
  // dedicated call if it came back empty.
  let results = profile.latestResults;
  if (!results.length) {
    try {
      results = await fetchAthleteResults(athleteId, 8);
    } catch {
      results = [];
    }
  }
  const now = todayIso ? new Date(todayIso) : new Date();
  const age = profile.age ?? (profile.yearOfBirth ? now.getUTCFullYear() - profile.yearOfBirth : undefined);

  return {
    athleteId: profile.athleteId,
    fullName: profile.fullName,
    noc: profile.noc,
    countryName: profile.countryName,
    countryIso: profile.countryIso,
    gender: profile.gender,
    yearOfBirth: profile.yearOfBirth,
    age,
    profileImage: profile.profileImage,
    flag: profile.flag,
    stats: profile.stats,
    ageGroupRank: profile.ageGroupRank,
    results,
    eligibilityTopRank: QUAL.eligibilityTopRank,
  };
}
