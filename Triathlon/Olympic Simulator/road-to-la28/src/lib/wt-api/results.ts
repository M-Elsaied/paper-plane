/**
 * Fetch official race results (finishing podiums) for Pick-'Em scoring.
 */
import { wtGet } from "./client";

interface RawProgram {
  prog_id: number;
  prog_name: string;
}
interface RawResult {
  athlete_id: number;
  position?: number | string | null;
}

export interface RacePodiums {
  eventId: number;
  male: number[]; // [1st, 2nd, 3rd] athlete ids (empty if no results yet)
  female: number[];
}

function asList<T>(data: unknown, key: string): T[] {
  if (Array.isArray(data)) return data as T[];
  if (data && typeof data === "object") {
    const v = (data as Record<string, unknown>)[key];
    if (Array.isArray(v)) return v as T[];
  }
  return [];
}

async function podium(eventId: number, progId: number): Promise<number[]> {
  const res = await wtGet<unknown>(`/events/${eventId}/programs/${progId}/results`);
  const results = asList<RawResult>(res.data, "results");
  const top: { pos: number; id: number }[] = [];
  for (const r of results) {
    const pos = Number(r.position);
    if (Number.isFinite(pos) && pos >= 1 && pos <= 3 && r.athlete_id) {
      top.push({ pos, id: r.athlete_id });
    }
  }
  top.sort((a, b) => a.pos - b.pos);
  return top.map((t) => t.id);
}

export async function fetchRacePodiums(eventId: number): Promise<RacePodiums> {
  const res = await wtGet<unknown>(`/events/${eventId}/programs`);
  const programs = asList<RawProgram>(res.data, "programs");
  const menProg = programs.find((p) => p.prog_name === "Elite Men");
  const womenProg = programs.find((p) => p.prog_name === "Elite Women");

  const [male, female] = await Promise.all([
    menProg ? podium(eventId, menProg.prog_id) : Promise.resolve<number[]>([]),
    womenProg ? podium(eventId, womenProg.prog_id) : Promise.resolve<number[]>([]),
  ]);

  return { eventId, male, female };
}
