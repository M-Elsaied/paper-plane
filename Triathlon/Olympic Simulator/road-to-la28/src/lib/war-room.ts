/**
 * NOC Slot War Room — the nation-level view of LA28 qualification.
 *
 * Every nation may send at most 3 athletes per gender (2 by default; 3 only with
 * enough depth). The engine already computes, per gender, how many capped places
 * each NOC has effectively secured (`QualLine.perNocUsage`, which folds in the
 * host / Mixed-Relay / tripartite pathway pre-consumption) and which high-ranked
 * athletes were pushed out by that cap (`QualLine.skipped`). This module groups
 * both genders by NOC into a single "war room" model: who is secured, who is
 * fighting for the open slots, and who is strong enough to qualify but is locked
 * out by their own compatriots — the internal battle that makes the cap bite.
 */
import { MR_OQR_PLACES } from "@/lib/engine/mixed-relay";
import "server-only";
import { buildRanking, type RankingRow } from "@/lib/cockpit";
import { getMrNations } from "@/lib/data";
import { DEFAULT_ASSUMPTIONS } from "@/config/pathways";
import { nationMrStatus, type MrNationEntry, type MrNationStatus } from "@/lib/engine/mixed-relay";
import { continentOf, type Continent } from "@/config/continents";
import { nationName } from "@/config/noc-names";
import type { Gender, QualLine } from "@/lib/engine/types";

/** One of a nation's athletes in the fight for its capped slots. */
export interface NocContender {
  athleteId: number;
  fullName: string;
  rank: number;
  total: number;
  /** Holds an individual Olympic place. */
  qualified: boolean;
  /** Would rank inside the line but is locked out by the nation cap. */
  blockedByCap: boolean;
  flag?: string;
}

/** A nation's slot picture for one gender. */
export interface NocGenderSlots {
  gender: Gender;
  cap: number;
  /** Capped places already effectively secured (individual + pathway). */
  secured: number;
  /** Places still open to win individually (cap − secured, floored at 0). */
  open: number;
  /** Places secured through the host / relay / tripartite pathway, not an individual line spot. */
  pathwaySecured: number;
  /** This nation's athletes, in ranking order. */
  contenders: NocContender[];
  qualifiedCount: number;
  /** Athletes good enough to qualify on merit but capped out — the heartbreak count. */
  blockedCount: number;
}

/** A nation's full War Room card. */
export interface NocWarRoom {
  noc: string;
  name: string;
  continent: Continent | null;
  flag?: string;
  men: NocGenderSlots;
  women: NocGenderSlots;
  mr: MrNationStatus;
  securedTotal: number;
  capTotal: number;
  openTotal: number;
  contenderTotal: number;
  blockedTotal: number;
  /** Sort/heat key: secured places weighted up, internal battles surfaced. */
  heat: number;
}

export interface WarRoomModel {
  nocs: NocWarRoom[];
  publishedAt: string;
  totals: {
    nations: number;
    securedMen: number;
    securedWomen: number;
    mrNationsInside: number;
  };
}

type RankingLike = { rows: RankingRow[]; line: QualLine };

function genderSlots(gender: Gender, noc: string, r: RankingLike): NocGenderSlots {
  const usage = r.line.perNocUsage[noc] ?? { cap: 2, used: 0 };
  const cappedOut = new Set(
    r.line.skipped
      .filter((s) => s.reason === "noc_cap" || s.reason === "pathway_consumed")
      .map((s) => s.athleteId),
  );

  // rows are already rank-ordered; keep only this nation's athletes.
  const contenders: NocContender[] = r.rows
    .filter((row) => row.noc === noc)
    .map((row) => ({
      athleteId: row.athleteId,
      fullName: row.fullName,
      rank: row.rank,
      total: row.total,
      qualified: row.qualified,
      blockedByCap: cappedOut.has(row.athleteId),
      flag: row.flag,
    }));

  const qualifiedCount = contenders.filter((c) => c.qualified).length;
  const blockedCount = contenders.filter((c) => c.blockedByCap).length;
  const secured = usage.used;
  // Places secured beyond this nation's own individual-line qualifiers came from a pathway.
  const pathwaySecured = Math.max(0, secured - qualifiedCount);

  return {
    gender,
    cap: usage.cap,
    secured,
    open: Math.max(0, usage.cap - secured),
    pathwaySecured,
    contenders,
    qualifiedCount,
    blockedCount,
  };
}

/**
 * Pure assembly — given both genders' ranking output + the MR ranking, produce
 * the War Room model. Kept free of I/O so it is cheap to unit-test.
 */
export function assembleWarRoom(inputs: {
  men: RankingLike;
  women: RankingLike;
  mrNations: MrNationEntry[];
  publishedAt: string;
}): WarRoomModel {
  const { men, women, mrNations, publishedAt } = inputs;

  // The universe of nations: anyone with a ranked athlete, plus MR-ranked nations.
  const nocs = new Set<string>();
  men.rows.forEach((r) => nocs.add(r.noc));
  women.rows.forEach((r) => nocs.add(r.noc));
  mrNations.forEach((m) => nocs.add(m.noc));

  const cards: NocWarRoom[] = [];
  for (const noc of nocs) {
    const m = genderSlots("male", noc, men);
    const w = genderSlots("female", noc, women);
    const mr = nationMrStatus(mrNations, noc, DEFAULT_ASSUMPTIONS);

    const contenderTotal = m.contenders.length + w.contenders.length;
    const securedTotal = m.secured + w.secured;
    // Drop nations with nothing at stake here: no ranked athlete, no secured
    // place, and no Mixed Relay standing.
    if (contenderTotal === 0 && securedTotal === 0 && mr.rank == null) continue;

    const blockedTotal = m.blockedCount + w.blockedCount;
    const flag = m.contenders[0]?.flag ?? w.contenders[0]?.flag;

    cards.push({
      noc,
      name: nationName(noc),
      continent: continentOf(noc),
      flag,
      men: m,
      women: w,
      mr,
      securedTotal,
      capTotal: m.cap + w.cap,
      openTotal: m.open + w.open,
      contenderTotal,
      blockedTotal,
      heat: securedTotal * 10 + blockedTotal * 4 + contenderTotal + (mr.insideRelayCut ? 3 : 0),
    });
  }

  // Powerhouses first (most places secured), then the fiercest internal battles.
  cards.sort(
    (a, b) =>
      b.securedTotal - a.securedTotal ||
      b.blockedTotal - a.blockedTotal ||
      b.contenderTotal - a.contenderTotal ||
      a.noc.localeCompare(b.noc),
  );

  return {
    nocs: cards,
    publishedAt,
    totals: {
      nations: cards.length,
      securedMen: cards.reduce((s, c) => s + c.men.secured, 0),
      securedWomen: cards.reduce((s, c) => s + c.women.secured, 0),
      mrNationsInside: mrNations.filter((m) => m.rank <= MR_OQR_PLACES).length,
    },
  };
}

/** Server view-model: run the engine for both genders and group by NOC. */
export async function buildWarRoom(): Promise<WarRoomModel> {
  const [men, women, mrNations] = await Promise.all([
    buildRanking("male"),
    buildRanking("female"),
    getMrNations(),
  ]);
  return assembleWarRoom({
    men,
    women,
    mrNations,
    publishedAt: men.state.publishedAt,
  });
}

/** Server view-model for a single nation (its own detail page). */
export async function buildNocWarRoom(noc: string): Promise<NocWarRoom | null> {
  const upper = noc.toUpperCase();
  const model = await buildWarRoom();
  return model.nocs.find((c) => c.noc === upper) ?? null;
}
