/**
 * Fetch the elite start lists (entries) for a World Triathlon event, so the Live
 * Race Companion projects only the athletes actually racing.
 */
import { wtGet } from "./client";

interface RawProgram {
  prog_id: number;
  prog_name: string;
}
interface RawEntry {
  athlete_id: number;
  start_num?: number | null;
  wait_pos?: number | null;
}

export interface RaceStartLists {
  eventId: number;
  men: number[]; // athlete ids on the Elite Men start list, in start order
  women: number[];
}

function asList<T>(data: unknown, key: string): T[] {
  if (Array.isArray(data)) return data as T[];
  if (data && typeof data === "object") {
    const v = (data as Record<string, unknown>)[key];
    if (Array.isArray(v)) return v as T[];
  }
  return [];
}

async function fetchEntries(eventId: number, progId: number): Promise<number[]> {
  const res = await wtGet<unknown>(`/events/${eventId}/programs/${progId}/entries`);
  const entries = asList<RawEntry>(res.data, "entries");
  return entries
    .filter((e) => e.athlete_id && (e.wait_pos == null || e.wait_pos === 0))
    .sort((a, b) => (a.start_num ?? 9999) - (b.start_num ?? 9999))
    .map((e) => e.athlete_id);
}

export async function fetchRaceStartLists(eventId: number): Promise<RaceStartLists> {
  const res = await wtGet<unknown>(`/events/${eventId}/programs`);
  const programs = asList<RawProgram>(res.data, "programs");
  const menProg = programs.find((p) => p.prog_name === "Elite Men");
  const womenProg = programs.find((p) => p.prog_name === "Elite Women");

  const [men, women] = await Promise.all([
    menProg ? fetchEntries(eventId, menProg.prog_id) : Promise.resolve<number[]>([]),
    womenProg ? fetchEntries(eventId, womenProg.prog_id) : Promise.resolve<number[]>([]),
  ]);

  return { eventId, men, women };
}
