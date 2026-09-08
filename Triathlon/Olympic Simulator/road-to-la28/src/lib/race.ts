/**
 * Server builder for the Live Race Companion. Cross-references a race's official
 * start list with the current Olympic ranking to build the racing field, then
 * seeds an initial projected finish order. Everything needed for the client to
 * recompute live (drag a finish → projected Monday ranking) is serialized out.
 */
import "server-only";
import type { Gender } from "@/config/pathways";
import { periodForDate, type PeriodId } from "@/config/qualification";
import type { PointsTier } from "@/config/points-tables";
import type { AthleteScores } from "@/lib/engine/types";
import { seedFinishOrder, type FieldAthlete } from "@/lib/engine/projection";
import { rankAthletes } from "@/lib/engine/qualification";
import { getQualState, getUpcomingEvents } from "@/lib/data";
import { fetchEvent, type UpcomingEvent } from "@/lib/wt-api/events";
import { todayIso } from "@/lib/today";
import { fetchRaceStartLists } from "@/lib/wt-api/start-list";

export interface RaceCompanionModel {
  eventId: number;
  title: string;
  tier: PointsTier;
  tierLabel: string;
  date: string;
  venue?: string;
  gender: Gender;
  period: PeriodId;
  /** Racing athletes that are in the Olympic ranking (contenders). */
  field: FieldAthlete[];
  /** Full ranking (for the client to recompute the projected line). */
  allAthletes: AthleteScores[];
  /** Model's initial projected finish order (athlete ids). */
  initialOrder: number[];
  /** Entrants on the start list who aren't (yet) in the ranking. */
  unrankedEntrants: number;
  /** True when we used the official start list; false = ranking-based field. */
  officialStartList: boolean;
}

/** The race calendar first; any other WT event id resolves live (past races too). */
export async function getEvent(eventId: number): Promise<UpcomingEvent | null> {
  if (!Number.isInteger(eventId) || eventId <= 0) return null;
  const upcoming = (await getUpcomingEvents()).find((e) => e.eventId === eventId);
  if (upcoming) return upcoming;
  try {
    return await fetchEvent(eventId);
  } catch {
    return null;
  }
}

export async function buildRaceCompanion(
  eventId: number,
  gender: Gender,
): Promise<RaceCompanionModel | null> {
  const event = await getEvent(eventId);
  if (!event) return null;

  const state = await getQualState(gender);
  const ranked = rankAthletes(state.athletes);
  const rankById = new Map(ranked.map((a, i) => [a.athleteId, i + 1]));

  // Try the official start list; fall back to the ranking's top contenders.
  let startIds: number[] = [];
  let official = false;
  let unranked = 0;
  try {
    const lists = await fetchRaceStartLists(eventId);
    startIds = gender === "male" ? lists.men : lists.women;
    official = startIds.length > 0;
  } catch {
    official = false;
  }

  let field: FieldAthlete[];
  if (official) {
    const rankedIds = new Set(ranked.map((a) => a.athleteId));
    unranked = startIds.filter((id) => !rankedIds.has(id)).length;
    field = startIds
      .filter((id) => rankedIds.has(id))
      .map((id) => toFieldAthlete(ranked.find((a) => a.athleteId === id)!, rankById))
      .filter(Boolean);
  } else {
    // Expected field: the top ~30 contenders in the ranking.
    field = ranked.slice(0, 30).map((a) => toFieldAthlete(a, rankById));
  }

  const period = periodForDate(event.date) ?? currentPeriod();
  const initialOrder = seedFinishOrder(field).map((s) => s.athleteId);

  return {
    eventId,
    title: event.title.replace(/^\d{4}\s+/, ""),
    tier: event.tier,
    tierLabel: event.tierLabel,
    date: event.date,
    venue: event.venue ?? event.country,
    gender,
    period,
    field,
    allAthletes: state.athletes,
    initialOrder,
    unrankedEntrants: unranked,
    officialStartList: official,
  };
}

function toFieldAthlete(
  a: ReturnType<typeof rankAthletes>[number],
  rankById: Map<number, number>,
): FieldAthlete {
  return {
    athleteId: a.athleteId,
    fullName: a.fullName,
    noc: a.noc,
    gender: a.gender,
    scores: a.scores,
    flag: a.flag,
    profileImage: a.profileImage,
    change: a.change,
    currentTotal: a.total,
    currentRank: rankById.get(a.athleteId) ?? null,
  };
}

function currentPeriod(): PeriodId {
  return periodForDate(todayIso()) ?? 1;
}
