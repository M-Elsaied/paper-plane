/**
 * Post-deploy smoke test — codifies the manual curl checks. No deps, <20s.
 *   npm run smoke -- https://road-to-la28.vercel.app [--secret <CRON_SECRET>]
 * Exits non-zero on any failure.
 */
const base = (process.argv[2] || "https://road-to-la28.vercel.app").replace(/\/$/, "");
const secretIdx = process.argv.indexOf("--secret");
const secret = secretIdx > -1 ? process.argv[secretIdx + 1] : null;

const fails = [];
const ok = (cond, msg) => { if (!cond) fails.push(msg); else console.log("  ✓", msg); };

async function get(path, init) {
  const res = await fetch(base + path, init);
  return res;
}

async function main() {
  console.log("smoke:", base);

  // 1. public pages 200
  for (const p of ["/", "/rankings", "/race-week", "/pulse", "/relay", "/account"]) {
    ok((await get(p)).status === 200, `GET ${p} → 200`);
  }

  // 2. cockpit renders a known athlete + OG meta
  const cockpit = await (await get("/athlete/86042")).text();
  ok(cockpit.includes("Vilaca"), "cockpit contains Vilaca");
  ok(cockpit.includes('og:image') || cockpit.includes("/athlete/86042/card"), "cockpit has OG image meta");

  // 3. share card is a real PNG of the right size
  const card = await get("/athlete/86042/card");
  const buf = Buffer.from(await card.arrayBuffer());
  ok(card.headers.get("content-type")?.includes("image/png"), "card content-type PNG");
  ok(buf.length > 30000, `card body ${buf.length}B > 30kB (not blank)`);
  ok(buf.readUInt32BE(16) === 1080 && buf.readUInt32BE(20) === 1350, "card is 1080x1350");

  // 4. cron guards
  ok((await get("/api/cron/sync-rankings")).status === 401, "sync-rankings 401 without bearer");
  ok((await get("/api/cron/score-picks")).status === 401, "score-picks 401 without bearer");
  if (secret) {
    const s = await get("/api/cron/sync-rankings", { headers: { authorization: `Bearer ${secret}` } });
    ok([200].includes(s.status), "sync-rankings 200 with bearer");
  }

  // 5. API validation
  ok((await get("/api/picks?raceId=")).status === 400, "picks GET without race → 400");
  const acc = await (await get("/api/account")).json();
  ok(acc.claimed === false, "account unclaimed for anonymous");

  // 6. PWA assets
  const manifest = await get("/manifest.webmanifest");
  ok(manifest.status === 200 && (await manifest.json()).name.includes("Road to LA28"), "manifest served");
  ok((await get("/sw.js")).headers.get("content-type")?.includes("javascript"), "sw.js served");

  // 7. forged claim redirect
  const claim = await get("/claim?t=garbage", { redirect: "manual" });
  ok([302, 307].includes(claim.status) && claim.headers.get("location")?.includes("error=invalid"), "forged claim → error redirect");

  if (fails.length) {
    console.error(`\n✗ ${fails.length} smoke check(s) failed:\n  ` + fails.join("\n  "));
    process.exit(1);
  }
  console.log("\n✓ prod smoke passed");
}

main().catch((e) => { console.error("smoke crashed:", e); process.exit(1); });
