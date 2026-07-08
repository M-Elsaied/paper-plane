/**
 * "Your Road to LA28" — the athlete-centric, game-theoretic pathway analyzer.
 *
 * For one athlete it evaluates EVERY route to the Games, personalised to their
 * nation, continent, ranking (or lack of one), and — crucially — the specific
 * OTHER athletes competing for the same slots. Each route is a game with
 * different players: teammates fighting the NOC cap, continental rivals fighting
 * for one New Flag place, the bubble fighting for the last individual spots.
 *
 * Pure + deterministic. The server assembles the context (rankings, relay
 * standings); this module reasons about it.
 */
import { QUAL } from "@/config/qualification";
import { QUOTA, MR_RANKING_TEAMS, type PathwayAssumptions } from "@/config/pathways";
import { continentOf, type Continent } from "@/config/continents";
import type { MrNationEntry } from "./mixed-relay";
import type { Gender, QualLine, RankedAthlete } from "./types";
import { selectCountingScores } from "./best-scores";

export type RouteKey =
  | "individual"
  | "relay"
  | "host"
  | "mr_champ"
  | "newflag_continental"
  | "newflag_ranking"
  | "universality";

export type RouteStatus = "on_track" | "in_contention" | "stretch" | "locked_out";

export interface Competitor {
  athleteId?: number;
  name: string;
  noc: string;
  /** Why this player matters to the subject on this route. */
  note: string;
  /** Are they ahead of the subject on this route? */
  ahead?: boolean;
}

export interface RouteAssessment {
  key: RouteKey;
  label: string;
  /** One-line plain-English mechanic. */
  mechanic: string;
  status: RouteStatus;
  /** The headline for this route as it applies to the subject. */
  headline: string;
  /** What it concretely takes. */
  detail: string;
  /** 0–100 how realistic this route is for the subject (drives ordering). */
  realism: number;
  /** The other players in this specific game. */
  competitors: Competitor[];
}

export interface RoadSubject {
  athleteId: number;
  name: string;
  noc: string;
  gender: Gender;
  continent: Continent | null;
  oqrRank: number | null;
  total: number;
  worldRank?: number | null;
}

export interface Road {
  subject: RoadSubject;
  qualifiedNocs: string[];
  routes: RouteAssessment[];
  primary: RouteAssessment | null;
  verdict: string;
}

export interface RoadContext {
  /** Subject's gender OQR pool, already ranked (index 0 = rank 1). */
  ranked: RankedAthlete[];
  line: QualLine;
  mrNations: MrNationEntry[];
  assumptions: PathwayAssumptions;
  /** Subject identity — required when the subject is unranked (not in `ranked`). */
  subjectId: number;
  subjectFallback?: { name: string; noc: string; gender: Gender; worldRank?: number | null };
}

const REALISM: Record<RouteStatus, number> = { on_track: 1, in_contention: 0.66, stretch: 0.35, locked_out: 0.1 };
const BASE: Record<RouteKey, number> = {
  individual: 100,
  host: 95,
  relay: 88,
  newflag_continental: 82,
  newflag_ranking: 76,
  mr_champ: 34,
  universality: 18,
};
function realism(key: RouteKey, status: RouteStatus): number {
  return Math.round(BASE[key] * REALISM[status]);
}

/** Nations that already hold (or are on track to hold) a place → New-Flag-ineligible. */
export function computeQualifiedNocs(ctx: RoadContext): Set<string> {
  const q = new Set<string>();
  q.add(ctx.assumptions.host.noc);
  if (ctx.assumptions.mrWorldChamps2026) q.add(ctx.assumptions.mrWorldChamps2026);
  if (ctx.assumptions.mrWorldChamps2027) q.add(ctx.assumptions.mrWorldChamps2027);
  // Top MR ranking teams qualify their nation.
  [...ctx.mrNations].sort((a, b) => a.rank - b.rank).slice(0, MR_RANKING_TEAMS).forEach((n) => q.add(n.noc));
  // Any nation with an athlete inside the individual line.
  ctx.line.qualified.forEach((s) => q.add(s.noc));
  return q;
}

function subjectOf(ctx: RoadContext): RoadSubject {
  const idx = ctx.ranked.findIndex((a) => a.athleteId === ctx.subjectId);
  if (idx >= 0) {
    const a = ctx.ranked[idx];
    return {
      athleteId: a.athleteId,
      name: a.fullName,
      noc: a.noc,
      gender: a.gender,
      continent: continentOf(a.noc),
      oqrRank: idx + 1,
      total: a.total,
      worldRank: ctx.subjectFallback?.worldRank ?? null,
    };
  }
  const f = ctx.subjectFallback;
  return {
    athleteId: ctx.subjectId,
    name: f?.name ?? "Athlete",
    noc: f?.noc ?? "",
    gender: f?.gender ?? "male",
    continent: continentOf(f?.noc),
    oqrRank: null,
    total: 0,
    worldRank: f?.worldRank ?? null,
  };
}

export function analyzeRoad(ctx: RoadContext): Road {
  const s = subjectOf(ctx);
  const qualified = computeQualifiedNocs(ctx);
  const routes: RouteAssessment[] = [];

  const push = (r: RouteAssessment | null) => r && routes.push(r);
  push(assessIndividual(ctx, s));
  push(assessRelay(ctx, s));
  push(assessHost(ctx, s));
  push(assessMrChamp(ctx, s));
  if (!qualified.has(s.noc)) {
    push(assessNewFlagContinental(ctx, s, qualified));
    push(assessNewFlagRanking(ctx, s, qualified));
    push(assessUniversality(ctx, s, qualified));
  }

  routes.sort((a, b) => b.realism - a.realism);
  const primary = routes[0] ?? null;

  return { subject: s, qualifiedNocs: [...qualified], routes, primary, verdict: verdictOf(s, primary) };
}

// ---------- individual ----------
function assessIndividual(ctx: RoadContext, s: RoadSubject): RouteAssessment | null {
  const mechanic = `The ${QUOTA.individual} highest-ranked eligible athletes on the Individual Olympic Qualification Ranking on 18 May 2028 qualify.`;
  if (s.oqrRank == null) {
    return {
      key: "individual",
      label: "Individual Olympic Ranking",
      mechanic,
      status: "locked_out",
      headline: "Not on the Olympic Qualification Ranking yet",
      detail: `${first(s)} needs counting results at World Triathlon events to enter the ranking, then climb toward the top ${QUOTA.individual}. Right now this is out of reach.`,
      realism: realism("individual", "locked_out"),
      competitors: [],
    };
  }
  const inLine = ctx.line.qualified.some((q) => q.athleteId === s.athleteId);
  const cut = ctx.line.cutPoints ?? 0;
  const gap = Math.round((cut - s.total) * 100) / 100;
  const nocAhead = ctx.ranked
    .filter((a, i) => a.noc === s.noc && i + 1 < (s.oqrRank ?? 0))
    .map((a, i) => ({ athleteId: a.athleteId, name: a.fullName, noc: a.noc, note: "teammate ahead — the NOC cap means you must out-point them", ahead: true }));
  const capReason = ctx.line.skipped.find((k) => k.athleteId === s.athleteId && (k.reason === "noc_cap" || k.reason === "pathway_consumed"));

  // nearest bubble (other-NOC) rivals around the cut
  const cutRank = ctx.line.cutRank ?? QUOTA.individual;
  const bubble = ctx.ranked
    .map((a, i) => ({ a, rank: i + 1 }))
    .filter(({ a, rank }) => a.athleteId !== s.athleteId && Math.abs(rank - cutRank) <= 3)
    .slice(0, 4)
    .map(({ a, rank }) => ({ athleteId: a.athleteId, name: a.fullName, noc: a.noc, note: `#${rank} · on the cut line`, ahead: rank < (s.oqrRank ?? 0) }));

  let status: RouteStatus;
  let headline: string;
  let detail: string;
  if (inLine) {
    status = "on_track";
    headline = `Inside the ${QUOTA.individual} qualifying places (#${s.oqrRank})`;
    detail = `${first(s)} is projected to qualify directly. Defend the position — the counting-score buffer to the cut is ${fmt(-gap)} pts.`;
  } else if (capReason) {
    status = "locked_out";
    headline = `Blocked by the ${s.noc} team cap`;
    detail = `${first(s)} would make the line on points, but ${s.noc} has filled its places. The real fight is out-pointing ${nocAhead.length} teammate${nocAhead.length === 1 ? "" : "s"}, not the field.`;
  } else if (gap <= 250) {
    status = "in_contention";
    headline = `On the bubble — ${fmt(gap)} pts from the cut`;
    detail = `${first(s)} sits just outside #${cutRank}. One strong counting result (worth more than the current weakest score) swings it.`;
  } else if ((s.oqrRank ?? 999) <= 60) {
    status = "stretch";
    headline = `In the chase — #${s.oqrRank}, ${fmt(gap)} pts back`;
    detail = `Making the individual line means a run of top results to close a ${fmt(gap)}-pt gap before 18 May 2028.`;
  } else {
    status = "stretch";
    headline = `A long climb — #${s.oqrRank}`;
    detail = `The individual route needs a major rise up the ranking; realistically another route is a better bet.`;
  }

  return {
    key: "individual",
    label: "Individual Olympic Ranking",
    mechanic,
    status,
    headline,
    detail,
    realism: realism("individual", status),
    competitors: dedupeCompetitors([...nocAhead, ...bubble]).slice(0, 6),
  };
}

/** Drop duplicate athletes (a teammate ahead can also be a bubble athlete). */
function dedupeCompetitors(list: Competitor[]): Competitor[] {
  const seen = new Set<number>();
  const out: Competitor[] = [];
  for (const c of list) {
    if (c.athleteId != null && seen.has(c.athleteId)) continue;
    if (c.athleteId != null) seen.add(c.athleteId);
    out.push(c);
  }
  return out;
}

// ---------- relay ----------
function relayNation(ctx: RoadContext, noc: string) {
  const sorted = [...ctx.mrNations].sort((a, b) => a.rank - b.rank);
  const entry = sorted.find((n) => n.noc === noc) ?? null;
  return { entry, sorted };
}
function nationCompatriots(ctx: RoadContext, s: RoadSubject) {
  return ctx.ranked
    .map((a, i) => ({ a, rank: i + 1 }))
    .filter(({ a }) => a.noc === s.noc && a.athleteId !== s.athleteId);
}

function assessRelay(ctx: RoadContext, s: RoadSubject): RouteAssessment | null {
  if (s.noc === ctx.assumptions.host.noc) return null; // host route handled separately
  const { entry } = relayNation(ctx, s.noc);
  if (!entry) return null;
  const mechanic = `The top ${MR_RANKING_TEAMS} nations on the Mixed Relay Olympic Qualification Ranking each qualify a relay team of 2 men + 2 women.`;
  const qualifiesTeam = entry.rank <= MR_RANKING_TEAMS;
  const mates = nationCompatriots(ctx, s);
  const matesAhead = mates.filter((m) => (s.oqrRank ?? 999) > m.rank);
  const topTwoInNation = matesAhead.length < QUOTA.relay.perTeamPerGender; // subject among nation's top 2

  const competitors: Competitor[] = mates.slice(0, 4).map((m) => ({
    athleteId: m.a.athleteId,
    name: m.a.fullName,
    noc: m.a.noc,
    note: `${s.noc} teammate — competes for the ${QUOTA.relay.perTeamPerGender} relay places`,
    ahead: (s.oqrRank ?? 999) > m.rank,
  }));

  let status: RouteStatus;
  let headline: string;
  let detail: string;
  if (qualifiesTeam && topTwoInNation) {
    status = "on_track";
    headline = `${s.noc} is inside the relay top ${MR_RANKING_TEAMS} (#${entry.rank})`;
    detail = `${s.noc} qualifies a relay team, and ${first(s)} is among the nation's top ${QUOTA.relay.perTeamPerGender} — the likely relay picks. This is a strong route.`;
  } else if (qualifiesTeam) {
    status = "in_contention";
    headline = `${s.noc} qualifies a relay team, but the spots are contested`;
    detail = `${s.noc} is inside the relay top ${MR_RANKING_TEAMS}, but ${matesAhead.length} teammate${matesAhead.length === 1 ? "" : "s"} rank ahead for the ${QUOTA.relay.perTeamPerGender} relay places. ${first(s)} must be a top-${QUOTA.relay.perTeamPerGender} national pick.`;
  } else if (entry.rank <= MR_RANKING_TEAMS + 4) {
    status = "stretch";
    headline = `${s.noc} is just outside the relay top ${MR_RANKING_TEAMS} (#${entry.rank})`;
    detail = `${s.noc} needs to climb into the relay top ${MR_RANKING_TEAMS} at mixed relay events — then ${first(s)} must be a national relay pick.`;
  } else {
    status = "stretch";
    headline = `${s.noc}'s relay is a long way from qualifying (#${entry.rank})`;
    detail = `The relay route needs a big rise up the Mixed Relay ranking.`;
  }

  return {
    key: "relay",
    label: "Mixed Relay",
    mechanic,
    status,
    headline,
    detail,
    realism: realism("relay", status),
    competitors,
  };
}

function assessHost(ctx: RoadContext, s: RoadSubject): RouteAssessment | null {
  if (s.noc !== ctx.assumptions.host.noc) return null;
  const mates = nationCompatriots(ctx, s);
  const matesAhead = mates.filter((m) => (s.oqrRank ?? 999) > m.rank);
  const topTwo = matesAhead.length < ctx.assumptions.host.perGender;
  const status: RouteStatus = topTwo ? "on_track" : "in_contention";
  return {
    key: "host",
    label: "Host nation (USA)",
    mechanic: `As hosts, ${ctx.assumptions.host.noc} field a relay team and are guaranteed ${ctx.assumptions.host.perGender} places per gender.`,
    status,
    headline: topTwo
      ? `Among the top ${ctx.assumptions.host.perGender} ${s.noc} athletes — host places are in reach`
      : `${s.noc} host places contested by teammates ranked higher`,
    detail: topTwo
      ? `The host places are ${first(s)}'s strongest route — stay among the top ${ctx.assumptions.host.perGender} in ${s.noc}.`
      : `${matesAhead.length} ${s.noc} athletes rank ahead; ${first(s)} needs to be a top-${ctx.assumptions.host.perGender} national pick.`,
    realism: realism("host", status),
    competitors: mates.slice(0, 4).map((m) => ({ athleteId: m.a.athleteId, name: m.a.fullName, noc: m.a.noc, note: `${s.noc} teammate — competes for host places`, ahead: (s.oqrRank ?? 999) > m.rank })),
  };
}

function assessMrChamp(ctx: RoadContext, s: RoadSubject): RouteAssessment | null {
  if (s.noc === ctx.assumptions.host.noc) return null;
  const { entry } = relayNation(ctx, s.noc);
  if (!entry || entry.rank > 4) return null; // only credible for genuine relay powers
  return {
    key: "mr_champ",
    label: "Mixed Relay World Champion",
    mechanic: `Winning the 2026 or 2027 Mixed Relay World Championships qualifies that nation a relay team (2 per gender).`,
    status: "in_contention",
    headline: `${s.noc} is a Mixed Relay medal threat (#${entry.rank})`,
    detail: `A wildcard: if ${s.noc} wins Mixed Relay Worlds in 2026 or 2027 and ${first(s)} is a top relay pick, that books the team directly.`,
    realism: realism("mr_champ", "in_contention"),
    competitors: nationCompatriots(ctx, s).slice(0, 3).map((m) => ({ athleteId: m.a.athleteId, name: m.a.fullName, noc: m.a.noc, note: `${s.noc} relay teammate`, ahead: (s.oqrRank ?? 999) > m.rank })),
  };
}

// ---------- new flag ----------
/** Not-yet-qualified athletes on the subject's continent, best-ranked first. */
function continentalRivals(ctx: RoadContext, s: RoadSubject, qualified: Set<string>) {
  if (!s.continent) return [];
  return ctx.ranked
    .map((a, i) => ({ a, rank: i + 1 }))
    .filter(({ a }) => a.noc !== s.noc && !qualified.has(a.noc) && continentOf(a.noc) === s.continent)
    // one representative (best) per nation — New Flag is one place per NOC-block
    .filter((x, _i, arr) => arr.findIndex((y) => y.a.noc === x.a.noc) === arr.indexOf(x));
}

function newFlagStatus(s: RoadSubject, rivalsAhead: number): RouteStatus {
  if (s.oqrRank == null && (s.worldRank == null || s.worldRank > 160)) return "stretch";
  if (rivalsAhead === 0) return "on_track";
  if (rivalsAhead <= 2) return "in_contention";
  return "stretch";
}

function assessNewFlagContinental(ctx: RoadContext, s: RoadSubject, qualified: Set<string>): RouteAssessment | null {
  if (!s.continent) return null;
  const rivals = continentalRivals(ctx, s, qualified);
  const ahead = rivals.filter((r) => s.oqrRank == null || r.rank < s.oqrRank).length;
  const status = newFlagStatus(s, ahead);
  const label = "New Flag — Continental Games";
  return {
    key: "newflag_continental",
    label,
    mechanic: `One place per continent goes to the best not-yet-qualified nation at the 2026–27 Continental Games (${s.continent}).`,
    status,
    headline:
      status === "on_track"
        ? `${first(s)} could be ${s.noc}'s New Flag hope at the ${s.continent} Games`
        : `Contested New Flag route via the ${s.continent} Continental Games`,
    detail:
      s.oqrRank == null
        ? `With ${s.noc} not yet qualified, a strong ${s.continent} Continental Games could earn this place — but ${first(s)} must be that Games' best athlete from a not-yet-qualified nation.`
        : `${ahead === 0 ? `${first(s)} is the top-ranked not-yet-qualified ${s.continent} athlete here.` : `${ahead} continental rival${ahead === 1 ? "" : "s"} rank ahead.`} Peak for the ${s.continent} Games.`,
    realism: realism("newflag_continental", status),
    competitors: rivals.slice(0, 5).map((r) => ({ athleteId: r.a.athleteId, name: r.a.fullName, noc: r.a.noc, note: `${r.a.noc} · ${s.continent} New Flag rival`, ahead: s.oqrRank == null || r.rank < s.oqrRank })),
  };
}

function assessNewFlagRanking(ctx: RoadContext, s: RoadSubject, qualified: Set<string>): RouteAssessment | null {
  if (!s.continent) return null;
  const rivals = continentalRivals(ctx, s, qualified);
  const ahead = rivals.filter((r) => s.oqrRank == null || r.rank < s.oqrRank).length;
  const status = newFlagStatus(s, ahead);
  return {
    key: "newflag_ranking",
    label: "New Flag — World Ranking",
    mechanic: `One place per continent goes to the top-ranked athlete from a not-yet-qualified nation on the World Ranking on 18 May 2028 (${s.continent}).`,
    status,
    headline:
      status === "on_track"
        ? `Top not-yet-qualified ${s.continent} athlete — a live New Flag route`
        : `New Flag via the World Ranking (${s.continent})`,
    detail:
      s.worldRank == null && s.oqrRank == null
        ? `${first(s)} first needs a World Triathlon Ranking. This place goes to the best-ranked ${s.continent} athlete whose nation hasn't qualified — build ranking points to enter that race.`
        : `Be the highest-ranked ${s.continent} athlete from a not-yet-qualified nation by 18 May 2028. ${ahead === 0 ? "Currently that's you." : `${ahead} rival${ahead === 1 ? "" : "s"} ahead.`}`,
    realism: realism("newflag_ranking", status),
    competitors: rivals.slice(0, 5).map((r) => ({ athleteId: r.a.athleteId, name: r.a.fullName, noc: r.a.noc, note: `${r.a.noc} · ${s.continent} New Flag rival`, ahead: s.oqrRank == null || r.rank < s.oqrRank })),
  };
}

function assessUniversality(ctx: RoadContext, s: RoadSubject, qualified: Set<string>): RouteAssessment | null {
  const nationRanked = ctx.ranked.filter((a) => a.noc === s.noc).length;
  if (nationRanked > 1) return null; // only a realistic fallback for tiny programs
  return {
    key: "universality",
    label: "Tripartite / Universality",
    mechanic: `Up to ${QUOTA.universality} invitation places per gender go to nations with small Olympic delegations, via the Tripartite Commission.`,
    status: "stretch",
    headline: `A longshot invitation route for ${s.noc}`,
    detail: `If ${s.noc} qualifies no athlete on merit, a Tripartite invitation is a last-resort possibility — decided by committee, not results.`,
    realism: realism("universality", "stretch"),
    competitors: [],
  };
}

// ---------- helpers ----------
function first(s: RoadSubject) {
  return s.name.split(" ")[0];
}
function fmt(n: number) {
  return `${Math.round(n)}`;
}
function verdictOf(s: RoadSubject, primary: RouteAssessment | null): string {
  if (!primary) return `${first(s)}'s road to LA28 isn't clear from the current data.`;
  const map: Record<RouteStatus, string> = {
    on_track: "is the clearest route",
    in_contention: "is the most realistic route, and it's live",
    stretch: "is the most realistic route, though it's a stretch",
    locked_out: "is currently closed",
  };
  return `For ${first(s)}, the ${primary.label} route ${map[primary.status]}.`;
}

/** Convenience: recompute a subject's counting total (for context assembly). */
export function subjectTotal(scores: { points: number; period: 1 | 2 }[]): number {
  return selectCountingScores(scores, QUAL).total;
}
