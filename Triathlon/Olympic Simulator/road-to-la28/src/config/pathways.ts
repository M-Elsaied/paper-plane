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

/** The eight pathways, for UI explanation. */
export const PATHWAYS = [
  { key: "host", label: "Host nation (USA)", places: 2 },
  { key: "mr_wc_2026", label: "2026 Mixed Relay World Champions", places: 2 },
  { key: "mr_wc_2027", label: "2027 Mixed Relay World Champions", places: 2 },
  { key: "mr_oqr", label: "Mixed Relay Olympic Qualification Ranking", places: 16 },
  { key: "individual", label: "Individual Olympic Qualification Ranking", places: 21 },
  { key: "tripartite", label: "Tripartite / Universality", places: 2 },
] as const;
