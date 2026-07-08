/**
 * World Triathlon API configuration.
 *
 * Auth: send the key as the `apikey` header on every request. Register a free
 * key at https://apps.api.triathlon.org and set WT_API_KEY in the environment.
 * Until then the public key published in WT's own OpenAPI spec is used as a
 * fallback so the app runs out of the box — keep request volume polite.
 */

/** Base URL. `WT_API_BASE_OVERRIDE` lets tests point at a local fixture server. */
export const WT_API_BASE =
  process.env.WT_API_BASE_OVERRIDE || "https://api.triathlon.org/v1";

/** Public fallback key from WT's OpenAPI spec. Replace via WT_API_KEY env. */
export const WT_PUBLIC_FALLBACK_KEY = "2649776ef9ece4c391003b521cbfce7a";

export function wtApiKey(): string {
  return process.env.WT_API_KEY?.trim() || WT_PUBLIC_FALLBACK_KEY;
}

/**
 * Ranking ids, discovered from GET /rankings and pinned here.
 * Ingestion asserts the fetched ranking's name still matches (WT can renumber).
 * Re-discover any time with: npm run discover-rankings
 */
export const RANKING_IDS = {
  oqr_men: { id: 11, expectName: "Elite Men", category: "Olympic" },
  oqr_women: { id: 12, expectName: "Elite Women", category: "Olympic" },
  world_men: { id: 13, expectName: "Elite Men", category: "World Rankings" },
  world_women: { id: 14, expectName: "Elite Women", category: "World Rankings" },
  wtcs_men: { id: 15, expectName: "Elite Men", category: "World Triathlon Series" },
  wtcs_women: { id: 16, expectName: "Elite Women", category: "World Triathlon Series" },
  mr_olympic: { id: 64, expectName: "Mixed Relay", category: "Mixed Relay Olympic" },
} as const;

export type RankingKey = keyof typeof RANKING_IDS;

/** Politeness / resilience knobs for the client. */
export const WT_CLIENT = {
  concurrency: 2,
  minDelayMs: 250,
  retries: 3,
  timeoutMs: 15000,
  rankingLimit: 1000,
} as const;
