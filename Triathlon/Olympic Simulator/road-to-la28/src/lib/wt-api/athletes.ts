/**
 * World Triathlon athlete search + profile + results. Lets the app find and show
 * ANY athlete, not just the ~160 in the Olympic Qualification Ranking.
 */
import { wtGet } from "./client";
import type { Gender } from "@/config/pathways";

export interface AthleteHit {
  athleteId: number;
  fullName: string;
  noc: string;
  gender: Gender;
  yearOfBirth?: number;
  profileImage?: string;
  flag?: string;
}

interface RawSearchAthlete {
  athlete_id: number;
  athlete_title: string;
  athlete_noc: string;
  athlete_gender: Gender;
  athlete_yob?: number | string;
  athlete_profile_image?: string | null;
  athlete_flag?: string | null;
  athlete_flag_circle?: string | null;
}

function normHit(a: RawSearchAthlete): AthleteHit {
  return {
    athleteId: a.athlete_id,
    fullName: a.athlete_title,
    noc: a.athlete_noc,
    gender: a.athlete_gender,
    yearOfBirth: a.athlete_yob ? Number(a.athlete_yob) : undefined,
    profileImage: a.athlete_profile_image ?? undefined,
    flag: a.athlete_flag_circle ?? a.athlete_flag ?? undefined,
  };
}

/** Free-text search across the whole World Triathlon athlete database. */
export async function searchAthletes(query: string, limit = 20): Promise<AthleteHit[]> {
  const q = query.trim();
  if (q.length < 2) return [];
  const res = await wtGet<RawSearchAthlete[]>("/search/athletes", { query: q, per_page: limit });
  return (res.data ?? []).map(normHit);
}

export interface CareerStats {
  starts: number;
  finishes: number;
  finishPct: number;
  wins: number;
  podiums: number;
  podiumPct: number;
}

export interface AthleteProfile {
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
  flagSquare?: string;
  stats?: CareerStats;
  ageGroupRank?: number;
  /** Recent results inline from the profile payload (no extra call). */
  latestResults: AthleteResult[];
}

interface RawStats {
  race_starts?: number;
  race_finishes?: number;
  finish_percentage?: number;
  race_wins?: number;
  race_podiums?: number;
  race_podium_percentage?: number;
}
interface RawProfile {
  athlete_id: number;
  athlete_title: string;
  athlete_noc: string;
  athlete_country_name?: string;
  athlete_country_isoa2?: string;
  athlete_gender: Gender;
  athlete_yob?: number | string;
  athlete_age?: number;
  athlete_profile_image?: string | null;
  athlete_flag_circle?: string | null;
  athlete_flag?: string | null;
  stats?: RawStats | null;
  current_rankings?: { age_group?: { rank?: number } | null } | null;
  latest_results?: RawResult[] | null;
}

function mapStats(s?: RawStats | null): CareerStats | undefined {
  if (!s || s.race_starts == null) return undefined;
  return {
    starts: s.race_starts ?? 0,
    finishes: s.race_finishes ?? 0,
    finishPct: s.finish_percentage ?? 0,
    wins: s.race_wins ?? 0,
    podiums: s.race_podiums ?? 0,
    podiumPct: s.race_podium_percentage ?? 0,
  };
}

export async function fetchAthleteProfile(id: number): Promise<AthleteProfile | null> {
  const res = await wtGet<RawProfile | RawProfile[]>(`/athletes/${id}`);
  const d = Array.isArray(res.data) ? res.data[0] : res.data;
  if (!d?.athlete_id) return null;
  const yob = d.athlete_yob ? Number(d.athlete_yob) : undefined;
  return {
    athleteId: d.athlete_id,
    fullName: d.athlete_title,
    noc: d.athlete_noc,
    countryName: d.athlete_country_name,
    countryIso: d.athlete_country_isoa2,
    gender: d.athlete_gender,
    yearOfBirth: yob,
    age: d.athlete_age,
    profileImage: d.athlete_profile_image ?? undefined,
    flag: d.athlete_flag_circle ?? undefined,
    flagSquare: d.athlete_flag ?? undefined,
    stats: mapStats(d.stats),
    ageGroupRank: d.current_rankings?.age_group?.rank ?? undefined,
    latestResults: mapResults(d.latest_results ?? []),
  };
}

export interface AthleteResult {
  eventId: number;
  eventTitle: string;
  date: string;
  program?: string;
  position: number | string | null;
  totalTime?: string;
  venue?: string;
  eventFlag?: string;
  eventIso?: string;
}

interface RawResult {
  event_id: number;
  event_title: string;
  event_date: string;
  event_venue?: string;
  event_flag?: string | null;
  event_flag_circle?: string | null;
  event_country_isoa2?: string;
  prog_name?: string;
  position?: number | string | null;
  total_time?: string | null;
}

function mapResults(list: RawResult[]): AthleteResult[] {
  return list.map((r) => ({
    eventId: r.event_id,
    eventTitle: r.event_title,
    date: r.event_date,
    program: r.prog_name,
    position: r.position ?? null,
    totalTime: r.total_time ?? undefined,
    venue: r.event_venue,
    eventFlag: r.event_flag_circle ?? r.event_flag ?? undefined,
    eventIso: r.event_country_isoa2,
  }));
}

export async function fetchAthleteResults(id: number, limit = 8): Promise<AthleteResult[]> {
  const res = await wtGet<RawResult[] | { results: RawResult[] }>(`/athletes/${id}/results`, { per_page: limit });
  const list = Array.isArray(res.data) ? res.data : (res.data?.results ?? []);
  return mapResults(list.slice(0, limit));
}
