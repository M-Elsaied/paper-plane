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

export interface AthleteProfile {
  athleteId: number;
  fullName: string;
  noc: string;
  countryName?: string;
  gender: Gender;
  yearOfBirth?: number;
  profileImage?: string;
  flag?: string;
}

interface RawProfile {
  athlete_id: number;
  athlete_title: string;
  athlete_noc: string;
  athlete_country_name?: string;
  athlete_gender: Gender;
  athlete_yob?: number | string;
  athlete_profile_image?: string | null;
  athlete_flag_circle?: string | null;
}

export async function fetchAthleteProfile(id: number): Promise<AthleteProfile | null> {
  const res = await wtGet<RawProfile | RawProfile[]>(`/athletes/${id}`);
  const d = Array.isArray(res.data) ? res.data[0] : res.data;
  if (!d?.athlete_id) return null;
  return {
    athleteId: d.athlete_id,
    fullName: d.athlete_title,
    noc: d.athlete_noc,
    countryName: d.athlete_country_name,
    gender: d.athlete_gender,
    yearOfBirth: d.athlete_yob ? Number(d.athlete_yob) : undefined,
    profileImage: d.athlete_profile_image ?? undefined,
    flag: d.athlete_flag_circle ?? undefined,
  };
}

export interface AthleteResult {
  eventId: number;
  eventTitle: string;
  date: string;
  program?: string;
  position: number | string | null;
}

interface RawResult {
  event_id: number;
  event_title: string;
  event_date: string;
  prog_name?: string;
  position?: number | string | null;
}

export async function fetchAthleteResults(id: number, limit = 8): Promise<AthleteResult[]> {
  const res = await wtGet<RawResult[] | { results: RawResult[] }>(`/athletes/${id}/results`, { per_page: limit });
  const list = Array.isArray(res.data) ? res.data : (res.data?.results ?? []);
  return list.slice(0, limit).map((r) => ({
    eventId: r.event_id,
    eventTitle: r.event_title,
    date: r.event_date,
    program: r.prog_name,
    position: r.position ?? null,
  }));
}
