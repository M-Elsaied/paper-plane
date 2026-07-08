/**
 * Head-to-head builder. Assembles a qualification-centric comparison of two
 * athletes plus their real race-by-race record (common events, who finished
 * ahead) — the raw material for the Versus page and persistent rivalries.
 */
import "server-only";
import { buildCockpit } from "@/lib/cockpit";
import { fetchAthleteProfile, fetchAthleteResults, type CareerStats, type AthleteResult } from "@/lib/wt-api/athletes";
import { getWorldRanking } from "@/lib/data";
import { continentOf, type Continent } from "@/config/continents";
import type { QualStatus } from "@/lib/engine/status";
import type { Gender } from "@/config/pathways";

export interface VersusAthlete {
  athleteId: number;
  name: string;
  noc: string;
  gender: Gender;
  continent: Continent | null;
  flag?: string;
  image?: string;
  age?: number;
  oqrRank: number | null;
  points: number | null;
  qualified: boolean;
  gapToLine: number | null; // negative = inside the line
  worldRank: number | null;
  status: QualStatus | null;
  stats: CareerStats | null;
}

export interface Meeting {
  eventTitle: string;
  date: string;
  program?: string;
  aPos: number;
  bPos: number;
  winner: "a" | "b";
}

export interface HeadToHead {
  meetings: Meeting[]; // most recent first, capped
  total: number;
  aWins: number;
  bWins: number;
}

export interface VersusModel {
  a: VersusAthlete;
  b: VersusAthlete;
  sameNoc: boolean;
  sameGender: boolean;
  /** a.points - b.points when both are ranked. */
  pointsGap: number | null;
  /** "a" | "b" | null — who is ahead on the Olympic Qualification Ranking. */
  oqrLeader: "a" | "b" | null;
  h2h: HeadToHead;
}

async function versusAthlete(id: number): Promise<VersusAthlete | null> {
  const profile = await fetchAthleteProfile(id).catch(() => null);
  if (!profile) return null;
  const [cockpit, world] = await Promise.all([
    buildCockpit(id).catch(() => null),
    getWorldRanking(profile.gender).catch(() => []),
  ]);
  const worldRank = world.find((w) => w.athleteId === id)?.rank ?? null;
  return {
    athleteId: id,
    name: profile.fullName,
    noc: profile.noc,
    gender: profile.gender,
    continent: continentOf(profile.noc),
    flag: profile.flag ?? cockpit?.flag,
    image: profile.profileImage ?? cockpit?.profileImage,
    age: profile.age,
    oqrRank: cockpit?.rank ?? null,
    points: cockpit?.total ?? null,
    qualified: cockpit?.qualified ?? false,
    gapToLine: cockpit?.gapToLine ?? null,
    worldRank,
    status: cockpit?.status ?? null,
    stats: profile.stats ?? null,
  };
}

function keyOf(r: AthleteResult) {
  return `${r.eventId}|${r.program ?? ""}`;
}

async function headToHead(aId: number, bId: number): Promise<HeadToHead> {
  const [ra, rb] = await Promise.all([
    fetchAthleteResults(aId, 40).catch(() => [] as AthleteResult[]),
    fetchAthleteResults(bId, 40).catch(() => [] as AthleteResult[]),
  ]);
  const bByKey = new Map(rb.map((r) => [keyOf(r), r]));
  const meetings: Meeting[] = [];
  for (const r of ra) {
    const other = bByKey.get(keyOf(r));
    if (!other) continue;
    const pa = Number(r.position);
    const pb = Number(other.position);
    if (!Number.isFinite(pa) || !Number.isFinite(pb) || pa === pb) continue;
    meetings.push({
      eventTitle: r.eventTitle,
      date: r.date,
      program: r.program,
      aPos: pa,
      bPos: pb,
      winner: pa < pb ? "a" : "b",
    });
  }
  meetings.sort((x, y) => y.date.localeCompare(x.date));
  return {
    meetings: meetings.slice(0, 6),
    total: meetings.length,
    aWins: meetings.filter((m) => m.winner === "a").length,
    bWins: meetings.filter((m) => m.winner === "b").length,
  };
}

export async function buildVersus(aId: number, bId: number): Promise<VersusModel | null> {
  if (aId === bId) return null;
  const [a, b] = await Promise.all([versusAthlete(aId), versusAthlete(bId)]);
  if (!a || !b) return null;

  const sameGender = a.gender === b.gender;
  const pointsGap = a.points != null && b.points != null ? Math.round((a.points - b.points) * 100) / 100 : null;
  const oqrLeader =
    a.oqrRank != null && b.oqrRank != null ? (a.oqrRank < b.oqrRank ? "a" : "b") : a.oqrRank != null ? "a" : b.oqrRank != null ? "b" : null;

  // Head-to-head record only makes sense within the same gender's races.
  const h2h = sameGender ? await headToHead(aId, bId) : { meetings: [], total: 0, aWins: 0, bWins: 0 };

  return { a, b, sameNoc: a.noc === b.noc, sameGender, pointsGap, oqrLeader, h2h };
}
