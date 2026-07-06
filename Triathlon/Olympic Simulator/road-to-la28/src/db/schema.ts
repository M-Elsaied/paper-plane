/**
 * Neon / Drizzle schema — the production ingestion + recompute store.
 *
 * The app runs on committed seed JSON out of the box (see src/lib/data.ts). This
 * schema is the production path: Vercel Cron writes raw payloads + versioned
 * snapshots here, the engine recomputes qualification states, and the data layer
 * swaps its reads from JSON to these tables with no change to any page.
 *
 * Design notes:
 *  - WT natural ids are primary keys where they exist.
 *  - rankingSnapshots is append-only, keyed by contentHash (the change detector).
 *  - Every normalized row can be recomputed from rawPayloads.
 */
import {
  pgTable,
  integer,
  text,
  boolean,
  timestamp,
  jsonb,
  doublePrecision,
  serial,
  uniqueIndex,
  index,
} from "drizzle-orm/pg-core";

export const athletes = pgTable("athletes", {
  athleteId: integer("athlete_id").primaryKey(),
  fullName: text("full_name").notNull(),
  givenName: text("given_name"),
  familyName: text("family_name"),
  noc: text("noc"),
  gender: text("gender"),
  yearOfBirth: integer("year_of_birth"),
  headshotUrl: text("headshot_url"),
  flagUrl: text("flag_url"),
  raw: jsonb("raw"),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
});

export const events = pgTable("events", {
  eventId: integer("event_id").primaryKey(),
  title: text("title").notNull(),
  venue: text("venue"),
  countryNoc: text("country_noc"),
  startDate: text("start_date"),
  endDate: text("end_date"),
  categoryIds: jsonb("category_ids"),
  pointsCategory: text("points_category"),
  status: text("status"),
  raw: jsonb("raw"),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
});

export const programs = pgTable("programs", {
  programId: integer("program_id").primaryKey(),
  eventId: integer("event_id").references(() => events.eventId),
  name: text("name"),
  gender: text("gender"),
  programDate: text("program_date"),
  isRelay: boolean("is_relay").default(false),
  resultsAvailable: boolean("results_available").default(false),
  raw: jsonb("raw"),
});

export const startListEntries = pgTable(
  "start_list_entries",
  {
    id: serial("id").primaryKey(),
    programId: integer("program_id").references(() => programs.programId),
    athleteId: integer("athlete_id").references(() => athletes.athleteId),
    bib: text("bib"),
    status: text("status"),
    firstSeenAt: timestamp("first_seen_at", { withTimezone: true }).defaultNow(),
    removedAt: timestamp("removed_at", { withTimezone: true }),
  },
  (t) => [uniqueIndex("sle_prog_ath").on(t.programId, t.athleteId)],
);

export const results = pgTable(
  "results",
  {
    id: serial("id").primaryKey(),
    programId: integer("program_id").references(() => programs.programId),
    athleteId: integer("athlete_id").references(() => athletes.athleteId),
    position: integer("position"),
    status: text("status"),
    totalTime: text("total_time"),
    computedPoints: doublePrecision("computed_points"),
    pointsConfigVersion: integer("points_config_version"),
  },
  (t) => [uniqueIndex("res_prog_ath").on(t.programId, t.athleteId)],
);

export const rankingSnapshots = pgTable(
  "ranking_snapshots",
  {
    id: serial("id").primaryKey(),
    rankingId: integer("ranking_id").notNull(),
    rankingType: text("ranking_type").notNull(), // oqr_men | oqr_women | wtcs_* | mr_olympic
    contentHash: text("content_hash").notNull(),
    publishedAt: text("published_at"),
    fetchedAt: timestamp("fetched_at", { withTimezone: true }).defaultNow(),
    isOfficial: boolean("is_official").default(true),
    rawPayloadId: integer("raw_payload_id").references(() => rawPayloads.id),
  },
  (t) => [uniqueIndex("snap_type_hash").on(t.rankingType, t.contentHash)],
);

export const rankingEntries = pgTable(
  "ranking_entries",
  {
    id: serial("id").primaryKey(),
    snapshotId: integer("snapshot_id").references(() => rankingSnapshots.id),
    athleteId: integer("athlete_id"),
    nationNoc: text("nation_noc"),
    rank: integer("rank").notNull(),
    lastRank: integer("last_rank"),
    change: text("change"),
    totalPoints: doublePrecision("total_points"),
    scores: jsonb("scores"), // per-score breakdown {points, period, counted}
  },
  (t) => [index("re_snapshot").on(t.snapshotId)],
);

export const pointsConfig = pgTable("points_config", {
  version: integer("version").primaryKey(),
  effectiveFrom: text("effective_from"),
  tables: jsonb("tables"),
  notes: text("notes"),
});

export const projections = pgTable(
  "projections",
  {
    id: serial("id").primaryKey(),
    programId: integer("program_id"),
    athleteId: integer("athlete_id"),
    projectedPosition: integer("projected_position"),
    expectedPoints: doublePrecision("expected_points"),
    seedScore: doublePrecision("seed_score"),
    breakdown: jsonb("breakdown"),
    engineVersion: integer("engine_version"),
    configVersion: integer("config_version"),
    computedAt: timestamp("computed_at", { withTimezone: true }).defaultNow(),
  },
  (t) => [uniqueIndex("proj_prog_ath_eng").on(t.programId, t.athleteId, t.engineVersion)],
);

export const qualificationStates = pgTable("qualification_states", {
  id: serial("id").primaryKey(),
  snapshotId: integer("snapshot_id").references(() => rankingSnapshots.id),
  gender: text("gender").notNull(),
  line: jsonb("line"), // qualified list, cut rank/points, per-NOC usage, assumptions
  engineVersion: integer("engine_version"),
  computedAt: timestamp("computed_at", { withTimezone: true }).defaultNow(),
});

export const syncRuns = pgTable("sync_runs", {
  id: serial("id").primaryKey(),
  job: text("job").notNull(),
  status: text("status").notNull(), // running | ok | error | noop
  startedAt: timestamp("started_at", { withTimezone: true }).defaultNow(),
  finishedAt: timestamp("finished_at", { withTimezone: true }),
  stats: jsonb("stats"),
  error: text("error"),
});

/** Account bridge: a synced "board" (my athlete + follows) with no password —
 *  identity is a signed session cookie; recovery is a signed link. */
export const accounts = pgTable("accounts", {
  id: text("id").primaryKey(), // uuid
  board: jsonb("board"), // { myAthlete, follows }
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
});

/** Web Push subscriptions. Tied to an account when claimed, else standalone.
 *  `follows` is the cached athlete-id list used to target relevant events. */
export const pushSubscriptions = pgTable(
  "push_subscriptions",
  {
    id: serial("id").primaryKey(),
    accountId: text("account_id"),
    endpoint: text("endpoint").notNull(),
    p256dh: text("p256dh").notNull(),
    auth: text("auth").notNull(),
    follows: jsonb("follows"), // number[] athlete ids
    lastSnapshotMen: integer("last_snapshot_men"),
    lastSnapshotWomen: integer("last_snapshot_women"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  },
  (t) => [uniqueIndex("push_endpoint").on(t.endpoint)],
);

/** Pick-'Em: a user's podium prediction for a race, keyed by owner (account id
 *  when claimed, else a device id). Auto-scored once results are official. */
export const picks = pgTable(
  "picks",
  {
    id: serial("id").primaryKey(),
    raceId: integer("race_id").notNull(),
    gender: text("gender").notNull(),
    ownerKey: text("owner_key").notNull(),
    podium: jsonb("podium").notNull(), // [firstId, secondId, thirdId]
    score: integer("score"),
    perfect: boolean("perfect"),
    scoredAt: timestamp("scored_at", { withTimezone: true }),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  },
  (t) => [uniqueIndex("pick_owner").on(t.raceId, t.gender, t.ownerKey)],
);

export const rawPayloads = pgTable("raw_payloads", {
  id: serial("id").primaryKey(),
  syncRunId: integer("sync_run_id"),
  endpoint: text("endpoint").notNull(),
  params: jsonb("params"),
  payload: jsonb("payload"),
  payloadHash: text("payload_hash"),
  dedupeOfId: integer("dedupe_of_id"),
  fetchedAt: timestamp("fetched_at", { withTimezone: true }).defaultNow(),
});
