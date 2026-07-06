/**
 * Qualification-line computation for the Individual Olympic Qualification
 * Ranking. Encodes the LA28 rules: rank by best-12 counted total, apply per-NOC
 * caps (3 if a nation has depth in the top 30, else 2), account for places a
 * nation has already consumed through other pathways (host, relay), and walk the
 * ranking to allocate the 21 individual places.
 *
 * Pure and deterministic. Every exclusion carries a human-readable reason so the
 * UI can explain exactly why a highly-ranked athlete misses the line.
 */
import { QUAL } from "@/config/qualification";
import {
  DEFAULT_ASSUMPTIONS,
  type PathwayAssumptions,
} from "@/config/pathways";
import { selectCountingScores } from "./best-scores";
import type {
  AthleteScores,
  QualLine,
  QualSlot,
  RankedAthlete,
  SkipReason,
} from "./types";

/** Rank a pool of athletes by best-12 counted total (desc). Stable tie-break. */
export function rankAthletes(
  athletes: AthleteScores[],
  cfg = QUAL,
): RankedAthlete[] {
  const withTotals = athletes.map((a) => {
    const counted = selectCountingScores(a.scores, cfg);
    return { ...a, counted, total: counted.total };
  });
  withTotals.sort(
    (a, b) => b.total - a.total || a.fullName.localeCompare(b.fullName),
  );
  return withTotals;
}

/** Per-NOC cap: 3 if the nation has `nocCapDepthCount` athletes inside the
 *  top `nocCapDepthWindow`, otherwise 2. */
export function computeNocCaps(
  ranked: RankedAthlete[],
  cfg = QUAL,
): Record<string, number> {
  const census: Record<string, number> = {};
  ranked.slice(0, cfg.nocCapDepthWindow).forEach((a) => {
    census[a.noc] = (census[a.noc] ?? 0) + 1;
  });
  const caps: Record<string, number> = {};
  ranked.forEach((a) => {
    if (caps[a.noc] == null) {
      caps[a.noc] =
        (census[a.noc] ?? 0) >= cfg.nocCapDepthCount
          ? cfg.nocCapIfDepth
          : cfg.nocCapDefault;
    }
  });
  return caps;
}

/** Places a nation has already consumed through non-individual pathways —
 *  these count against the nation's cap and so limit its individual places. */
function preConsumedByNoc(
  assumptions: PathwayAssumptions,
  perGender: number,
): Record<string, number> {
  const used: Record<string, number> = {};
  const add = (noc: string | null, n: number) => {
    if (!noc) return;
    used[noc] = (used[noc] ?? 0) + n;
  };
  add(assumptions.host.noc, assumptions.host.perGender);
  add(assumptions.mrWorldChamps2026, 2 * perGender > 0 ? 1 : 1); // 1 per gender
  add(assumptions.mrWorldChamps2027, 1);
  assumptions.mrOqrNations.forEach((noc) => add(noc, 1)); // ~1 slot/gender/team
  assumptions.tripartite.forEach((noc) => add(noc, 1));
  return used;
}

export function computeQualificationLine(
  athletes: AthleteScores[],
  assumptions: PathwayAssumptions = DEFAULT_ASSUMPTIONS,
  cfg = QUAL,
): QualLine {
  const ranked = rankAthletes(athletes, cfg);
  const gender = ranked[0]?.gender ?? "male";
  const caps = computeNocCaps(ranked, cfg);
  const preUsed = preConsumedByNoc(assumptions, 1);

  // usage starts from pathway-consumed places (clamped to the cap).
  const usage: Record<string, number> = {};
  Object.keys(caps).forEach((noc) => {
    usage[noc] = Math.min(preUsed[noc] ?? 0, caps[noc]);
  });

  const qualified: QualSlot[] = [];
  const skipped: SkipReason[] = [];
  const bubble: QualSlot[] = [];

  const slots = assumptions.individualSlots ?? cfg.individualRankingSlots;

  ranked.forEach((a, i) => {
    const rank = i + 1;
    const slot: QualSlot = {
      athleteId: a.athleteId,
      fullName: a.fullName,
      noc: a.noc,
      rank,
      total: a.total,
    };

    if (qualified.length >= slots) {
      if (bubble.length < 8) bubble.push(slot);
      return;
    }

    const eligible = a.eligible !== false && rank <= cfg.eligibilityTopRank;
    if (!eligible) {
      skipped.push({
        ...slotBase(a, rank),
        reason: "ineligible",
        detail:
          rank > cfg.eligibilityTopRank
            ? `outside top ${cfg.eligibilityTopRank}`
            : "age/nationality ineligible",
      });
      return;
    }

    const cap = caps[a.noc] ?? cfg.nocCapDefault;
    if ((usage[a.noc] ?? 0) >= cap) {
      skipped.push({
        ...slotBase(a, rank),
        reason: preUsed[a.noc] ? "pathway_consumed" : "noc_cap",
        detail: `${a.noc} cap reached (${cap})`,
      });
      return;
    }

    usage[a.noc] = (usage[a.noc] ?? 0) + 1;
    qualified.push(slot);
  });

  const cut = qualified[qualified.length - 1] ?? null;
  const perNocUsage: Record<string, { cap: number; used: number }> = {};
  Object.keys(caps).forEach((noc) => {
    perNocUsage[noc] = { cap: caps[noc], used: usage[noc] ?? 0 };
  });

  return {
    gender,
    qualified,
    cutRank: cut?.rank ?? null,
    cutPoints: cut?.total ?? null,
    bubble,
    perNocUsage,
    skipped,
  };
}

function slotBase(a: RankedAthlete, rank: number) {
  return { athleteId: a.athleteId, fullName: a.fullName, noc: a.noc, rank };
}
