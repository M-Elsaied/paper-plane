CREATE TABLE "accounts" (
	"id" text PRIMARY KEY NOT NULL,
	"board" jsonb,
	"created_at" timestamp with time zone DEFAULT now(),
	"updated_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "athletes" (
	"athlete_id" integer PRIMARY KEY NOT NULL,
	"full_name" text NOT NULL,
	"given_name" text,
	"family_name" text,
	"noc" text,
	"gender" text,
	"year_of_birth" integer,
	"headshot_url" text,
	"flag_url" text,
	"raw" jsonb,
	"updated_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "events" (
	"event_id" integer PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"venue" text,
	"country_noc" text,
	"start_date" text,
	"end_date" text,
	"category_ids" jsonb,
	"points_category" text,
	"status" text,
	"raw" jsonb,
	"updated_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "picks" (
	"id" serial PRIMARY KEY NOT NULL,
	"race_id" integer NOT NULL,
	"gender" text NOT NULL,
	"owner_key" text NOT NULL,
	"podium" jsonb NOT NULL,
	"score" integer,
	"perfect" boolean,
	"scored_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "points_config" (
	"version" integer PRIMARY KEY NOT NULL,
	"effective_from" text,
	"tables" jsonb,
	"notes" text
);
--> statement-breakpoint
CREATE TABLE "programs" (
	"program_id" integer PRIMARY KEY NOT NULL,
	"event_id" integer,
	"name" text,
	"gender" text,
	"program_date" text,
	"is_relay" boolean DEFAULT false,
	"results_available" boolean DEFAULT false,
	"raw" jsonb
);
--> statement-breakpoint
CREATE TABLE "projections" (
	"id" serial PRIMARY KEY NOT NULL,
	"program_id" integer,
	"athlete_id" integer,
	"projected_position" integer,
	"expected_points" double precision,
	"seed_score" double precision,
	"breakdown" jsonb,
	"engine_version" integer,
	"config_version" integer,
	"computed_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "push_subscriptions" (
	"id" serial PRIMARY KEY NOT NULL,
	"account_id" text,
	"endpoint" text NOT NULL,
	"p256dh" text NOT NULL,
	"auth" text NOT NULL,
	"follows" jsonb,
	"last_snapshot_men" integer,
	"last_snapshot_women" integer,
	"created_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "qualification_states" (
	"id" serial PRIMARY KEY NOT NULL,
	"snapshot_id" integer,
	"gender" text NOT NULL,
	"line" jsonb,
	"engine_version" integer,
	"computed_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ranking_entries" (
	"id" serial PRIMARY KEY NOT NULL,
	"snapshot_id" integer,
	"athlete_id" integer,
	"nation_noc" text,
	"rank" integer NOT NULL,
	"last_rank" integer,
	"change" text,
	"total_points" double precision,
	"scores" jsonb
);
--> statement-breakpoint
CREATE TABLE "ranking_snapshots" (
	"id" serial PRIMARY KEY NOT NULL,
	"ranking_id" integer NOT NULL,
	"ranking_type" text NOT NULL,
	"content_hash" text NOT NULL,
	"published_at" text,
	"fetched_at" timestamp with time zone DEFAULT now(),
	"is_official" boolean DEFAULT true,
	"raw_payload_id" integer
);
--> statement-breakpoint
CREATE TABLE "raw_payloads" (
	"id" serial PRIMARY KEY NOT NULL,
	"sync_run_id" integer,
	"endpoint" text NOT NULL,
	"params" jsonb,
	"payload" jsonb,
	"payload_hash" text,
	"dedupe_of_id" integer,
	"fetched_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "results" (
	"id" serial PRIMARY KEY NOT NULL,
	"program_id" integer,
	"athlete_id" integer,
	"position" integer,
	"status" text,
	"total_time" text,
	"computed_points" double precision,
	"points_config_version" integer
);
--> statement-breakpoint
CREATE TABLE "start_list_entries" (
	"id" serial PRIMARY KEY NOT NULL,
	"program_id" integer,
	"athlete_id" integer,
	"bib" text,
	"status" text,
	"first_seen_at" timestamp with time zone DEFAULT now(),
	"removed_at" timestamp with time zone
);
--> statement-breakpoint
CREATE TABLE "sync_runs" (
	"id" serial PRIMARY KEY NOT NULL,
	"job" text NOT NULL,
	"status" text NOT NULL,
	"started_at" timestamp with time zone DEFAULT now(),
	"finished_at" timestamp with time zone,
	"stats" jsonb,
	"error" text
);
--> statement-breakpoint
ALTER TABLE "programs" ADD CONSTRAINT "programs_event_id_events_event_id_fk" FOREIGN KEY ("event_id") REFERENCES "public"."events"("event_id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "qualification_states" ADD CONSTRAINT "qualification_states_snapshot_id_ranking_snapshots_id_fk" FOREIGN KEY ("snapshot_id") REFERENCES "public"."ranking_snapshots"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ranking_entries" ADD CONSTRAINT "ranking_entries_snapshot_id_ranking_snapshots_id_fk" FOREIGN KEY ("snapshot_id") REFERENCES "public"."ranking_snapshots"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ranking_snapshots" ADD CONSTRAINT "ranking_snapshots_raw_payload_id_raw_payloads_id_fk" FOREIGN KEY ("raw_payload_id") REFERENCES "public"."raw_payloads"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "results" ADD CONSTRAINT "results_program_id_programs_program_id_fk" FOREIGN KEY ("program_id") REFERENCES "public"."programs"("program_id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "results" ADD CONSTRAINT "results_athlete_id_athletes_athlete_id_fk" FOREIGN KEY ("athlete_id") REFERENCES "public"."athletes"("athlete_id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "start_list_entries" ADD CONSTRAINT "start_list_entries_program_id_programs_program_id_fk" FOREIGN KEY ("program_id") REFERENCES "public"."programs"("program_id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "start_list_entries" ADD CONSTRAINT "start_list_entries_athlete_id_athletes_athlete_id_fk" FOREIGN KEY ("athlete_id") REFERENCES "public"."athletes"("athlete_id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE UNIQUE INDEX "pick_owner" ON "picks" USING btree ("race_id","gender","owner_key");--> statement-breakpoint
CREATE UNIQUE INDEX "proj_prog_ath_eng" ON "projections" USING btree ("program_id","athlete_id","engine_version");--> statement-breakpoint
CREATE UNIQUE INDEX "push_endpoint" ON "push_subscriptions" USING btree ("endpoint");--> statement-breakpoint
CREATE INDEX "re_snapshot" ON "ranking_entries" USING btree ("snapshot_id");--> statement-breakpoint
CREATE UNIQUE INDEX "snap_type_hash" ON "ranking_snapshots" USING btree ("ranking_type","content_hash");--> statement-breakpoint
CREATE UNIQUE INDEX "res_prog_ath" ON "results" USING btree ("program_id","athlete_id");--> statement-breakpoint
CREATE UNIQUE INDEX "sle_prog_ath" ON "start_list_entries" USING btree ("program_id","athlete_id");