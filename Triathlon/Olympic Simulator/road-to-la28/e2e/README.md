# E2E tests (Playwright)

Two projects (see `playwright.config.ts`):

## `seed` — default, offline, deterministic
`npm run e2e`

Runs the app with `DATABASE_URL` **unset** → pure committed-seed-JSON mode
(165 men / 138 women, Vilaça #1). World Triathlon calls are routed to a local
fixture server (`e2e/helpers/wt-fixture-server.mjs`, started by globalSetup)
via `WT_API_BASE_OVERRIDE`. No network, no database. 12 journeys covering the
picker→cockpit flow, simulator line-crossing + celebration, rankings + nav,
explain-the-line, tours, race companion, share card, and PWA/404.

## `db` — opt-in write flows
`TEST_DATABASE_URL=<neon-test-branch> npm run e2e:db`

Covers the two flows that need persistence: Pick-'Em (lock a podium → crowd
grows across two browser contexts) and account claim → recovery-link → sync onto
a fresh device. **Requires a dedicated Neon test branch** — never point it at the
production database. These same flows are also covered hermetically by the
PGlite integration tests (`tests/integration/{picks-api,score-picks-cron,account-api}.test.ts`),
so `e2e:db` is a belt-and-braces check to run before releases touching picks,
accounts, or push.
