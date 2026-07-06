# Road to LA28 🥇

An Olympic qualification cockpit for professional triathletes. Pick any athlete
from the World Triathlon rankings and see exactly where they stand on the road to
the **Los Angeles 2028 Olympics** — live ranking, the glowing qualification line,
points to defend, and an instant client-side **what-if simulator**.

Built by a former pro, for the field. Installable as a phone app (PWA).

## What it does

- **Athlete cockpit** — current Olympic rank, points gap to the cut, best-12
  counting scores (max 7/period), countdown to 18 May 2028, and Mixed Relay
  pathway status for the athlete's nation.
- **What-if simulator (the money shot)** — drag a finish position and watch the
  Olympic ranking re-sort live and the athlete cross the qualification line. Every
  number is the real LA28 qualification math, computed on-device for zero latency.
- **Rankings / Race Week / Pulse / Relay** — full OQR with the line drawn in,
  upcoming races with points on offer, official movers, and the relay top-16.

## Stack

Next.js 16 (App Router) · Neon Postgres + Drizzle · Tailwind v4 + Framer Motion ·
Vercel Cron · installable PWA. Data from the official
[World Triathlon API](https://developers.triathlon.org).

## Architecture

```
World Triathlon API ──► ingestion (seed script / Vercel Cron) ──► store ──► engine ──► UI
                                                                   (JSON or Neon)
```

- **Engine** (`src/lib/engine/`) — pure, isomorphic TypeScript. The LA28 rules
  (best-12/max-7, NOC caps, pathway slots, qualification line) and the what-if +
  projection logic. Runs on the server (precompute) **and** in the browser (the
  live slider). Unit-tested with Vitest, including tests over real ranking data.
- **Config** (`src/config/`) — every tunable: rule numbers, points tables,
  pathway assumptions, projection weights, cron cadences, pinned ranking ids.
- **Data layer** (`src/lib/data.ts`) — the single seam. Reads committed seed JSON
  today; swap to Neon/Drizzle reads with no page changes.
- **Ingestion** (`src/app/api/cron/`, `src/lib/ingest/`, `src/db/`) — the
  production path: Vercel Cron writes versioned, hash-deduped snapshots to Neon
  and the engine recomputes qualification states.

## Getting started

```bash
npm install
npm run seed     # pull real World Triathlon OQR + relay + events into src/data/
npm run dev      # http://localhost:3000
npm test         # engine test suite
```

The app runs fully on the seeded JSON — **no database required**. To enable the
production ingestion path:

```bash
cp .env.example .env        # set WT_API_KEY, DATABASE_URL, CRON_SECRET
npm run db:push             # create the Neon schema
```

## Deploying to Vercel

1. Provision Neon via the Vercel Marketplace (auto-sets `DATABASE_URL`).
2. Set `WT_API_KEY` and `CRON_SECRET` in project env vars.
3. Deploy. `vercel.json` registers the daily rankings sync.
   > Vercel Hobby allows only daily crons (max 2). For hourly race-weekend result
   > polling, upgrade to Pro or drive the protected routes from a GitHub Actions
   > schedule.

## Data & rules notes

- Ranking ids are pinned in `src/config/wt-api.ts` (OQR men 11, women 12, WTCS
  15/16, Mixed Relay Olympic 64) and re-discoverable via `npm run discover-rankings`.
- Point tables in `src/config/points-tables.ts` follow the standard WT curve and
  are validated against published totals; real past results use the API's own
  per-score values. Transcribe exact tables from the official criteria PDF and the
  golden tests will flag any drift.
- Mixed Relay is **tracked, not simulated** in v1 (per-nation pathway status).
