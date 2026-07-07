/**
 * Live-reality check (nightly). Fetches TODAY's rankings from World Triathlon
 * and asserts the engine still reproduces published totals + structural
 * invariants, and that the ranking ids / names haven't drifted. Read-only, no
 * DB, a few polite requests. Exits non-zero on any failure.  npm run verify:live
 */
import { RANKING_IDS } from "../src/config/wt-api";
import { fetchRankingState } from "../src/lib/wt-api/rankings";
import { wtGet } from "../src/lib/wt-api/client";
import { selectCountingScores } from "../src/lib/engine/best-scores";
import { computeQualificationLine } from "../src/lib/engine/qualification";
import type { RawRanking } from "../src/lib/wt-api/rankings";
import type { Gender } from "../src/config/pathways";

const fails: string[] = [];
const warn: string[] = [];
function check(cond: boolean, msg: string) {
  if (!cond) fails.push(msg);
}

async function verifyGender(id: number, expectName: string, gender: Gender) {
  const res = await wtGet<RawRanking>(`/rankings/${id}`, { limit: 1000 });
  const raw = res.data;
  check(raw.ranking_name?.includes(expectName), `ranking ${id} name drift: "${raw.ranking_name}" !~ "${expectName}"`);

  const state = await fetchRankingState(id, gender);
  check(state.athletes.length >= 120, `${gender}: only ${state.athletes.length} athletes (<120)`);

  // every recomputed total within half a cent of published
  const publishedById = new Map(raw.rankings.map((r) => [r.athlete_id, r.total]));
  let offenders = 0;
  for (const a of state.athletes) {
    const computed = selectCountingScores(a.scores).total;
    const published = publishedById.get(a.athleteId)!;
    if (Math.abs(computed - published) > 0.005) offenders++;
    const c = selectCountingScores(a.scores);
    check(c.counted.length <= 12, `${gender}: ${a.fullName} has ${c.counted.length} counted (>12)`);
    check(c.perPeriodCount[1] <= 7 && c.perPeriodCount[2] <= 7, `${gender}: ${a.fullName} exceeds 7/period`);
  }
  check(offenders === 0, `${gender}: ${offenders} athletes' totals disagree with published (rule drift)`);

  // qualification line sanity
  const line = computeQualificationLine(state.athletes);
  check(line.cutRank != null && line.cutRank <= 55, `${gender}: cutRank ${line.cutRank} out of range`);
  for (const [noc, u] of Object.entries(line.perNocUsage)) {
    check(u.used <= u.cap && u.cap <= 3, `${gender}: ${noc} usage ${u.used}/${u.cap} invalid`);
  }
  console.log(`  ${gender}: ${state.athletes.length} athletes, cut #${line.cutRank} @ ${line.cutPoints}pts, ${offenders} total-mismatches`);
}

async function main() {
  console.log("verify-live: checking today's World Triathlon rankings…");
  await verifyGender(RANKING_IDS.oqr_men.id, RANKING_IDS.oqr_men.expectName, "male");
  await verifyGender(RANKING_IDS.oqr_women.id, RANKING_IDS.oqr_women.expectName, "female");

  // MR ranking name canary
  const mr = await wtGet<{ ranking_cat_name: string }>(`/rankings/${RANKING_IDS.mr_olympic.id}`, { limit: 5 });
  check(mr.data.ranking_cat_name?.includes("Mixed Relay"), "MR ranking 64 name drift");

  if (warn.length) console.warn("WARN:\n  " + warn.join("\n  "));
  if (fails.length) {
    console.error(`\n✗ ${fails.length} check(s) failed:\n  ` + fails.join("\n  "));
    process.exit(1);
  }
  console.log("\n✓ live reality check passed");
}

main().catch((e) => {
  console.error("verify-live crashed:", e);
  process.exit(1);
});
