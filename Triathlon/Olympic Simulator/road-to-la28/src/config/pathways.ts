/**
 * The eight LA28 qualification pathways and their known/assumed slot holders.
 *
 * The qualification engine consumes a `PathwayAssumptions` object so unknown
 * future outcomes (e.g. the 2027 Mixed Relay World Champions) are explicit,
 * nullable inputs rather than magic constants. The UI can later expose an
 * "assumptions" drawer that edits this without any engine change.
 */

export type Gender = "male" | "female";

export interface PathwayAssumptions {
  /** Host nation automatic places (per gender). */
  host: { noc: string; perGender: number };
  /** 2026 Mixed Relay World Championship winning nation (2 places). Null until decided. */
  mrWorldChamps2026: string | null;
  /** 2027 Mixed Relay World Championship winning nation (2 places). Null until decided. */
  mrWorldChamps2027: string | null;
  /**
   * Nations expected to take the 16 Mixed Relay Olympic Qualification Ranking
   * places (top of the MR Olympic ranking). Drives per-NOC quota consumption.
   */
  mrOqrNations: string[];
  /** Tripartite / universality nations (up to 2). */
  tripartite: string[];
  /** Total individual-ranking places (mirrors QUAL.individualRankingSlots). */
  individualSlots: number;
}

/**
 * Working defaults as of the current qualification window. All future-dated
 * outcomes are null/empty and get filled by config edits or an admin drawer.
 * These are TUNABLE placeholders, not locked facts.
 */
export const DEFAULT_ASSUMPTIONS: PathwayAssumptions = {
  host: { noc: "USA", perGender: 2 },
  mrWorldChamps2026: null,
  mrWorldChamps2027: null,
  mrOqrNations: [],
  tripartite: [],
  individualSlots: 21,
};

/**
 * The full LA28 quota allocation per gender (sums to 55). Sourced from the
 * IOC-approved qualification system. Each Mixed Relay pathway qualifies a NATION
 * a relay team = 2 athletes per gender; those relay athletes count toward the
 * nation's per-gender Olympic team cap (2, or 3 with depth in the top 30).
 */
export const QUOTA = {
  perGender: 55,
  /** Mixed Relay = 22 per gender: 11 qualified teams x 2 athletes/gender. */
  relay: {
    total: 22,
    teams: 11,
    host: 1, // USA relay team
    worldChamps2026: 1,
    worldChamps2027: 1,
    rankingTeams: 8, // top 8 nations on the Mixed Relay Olympic Qualification Ranking
    perTeamPerGender: 2,
  },
  /** Individual Olympic Qualification Ranking. */
  individual: 21,
  /** New Flag = 10 per gender: one per continent x two sub-routes. */
  newFlag: {
    total: 10,
    continentalGames: 5, // one per continent, via the 2026-27 Continental Games
    worldRanking: 5, // one per continent, via the World Ranking on 18 May 2028
  },
  /** Tripartite Commission universality invitations. */
  universality: 2,
} as const;

/** Mixed Relay nation slots (host + champs + ranking teams) — teams whose 2
 *  athletes/gender qualify through relay. */
export const MR_RANKING_TEAMS = QUOTA.relay.rankingTeams;

/** The pathways, for UI explanation (per gender, sums to 55). */
export const PATHWAYS = [
  { key: "relay", label: "Mixed Relay teams (host + 2 champions + top 8 ranking)", places: 22 },
  { key: "individual", label: "Individual Olympic Qualification Ranking", places: 21 },
  { key: "newflag_continental", label: "New Flag — Continental Games (1 per continent)", places: 5 },
  { key: "newflag_ranking", label: "New Flag — World Ranking (1 per continent)", places: 5 },
  { key: "universality", label: "Tripartite / Universality", places: 2 },
] as const;
