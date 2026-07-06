/**
 * Print the World Triathlon rankings catalog so the Olympic / WTCS / Mixed Relay
 * ranking ids can be (re)pinned in src/config/wt-api.ts. Run:  npm run discover-rankings
 */
import { wtGet } from "../src/lib/wt-api/client";

interface CatalogRow {
  ranking_id: number;
  ranking_cat_name: string;
  ranking_name: string;
  published?: string;
}

async function main() {
  const res = await wtGet<CatalogRow[]>("/rankings", { limit: 300 });
  const rows = res.data.filter((r) =>
    /olympic|series|relay/i.test(`${r.ranking_cat_name} ${r.ranking_name}`),
  );
  console.log("id  | category | name | published");
  for (const r of rows) {
    console.log(`${r.ranking_id}\t| ${r.ranking_cat_name} | ${r.ranking_name} | ${r.published ?? ""}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
