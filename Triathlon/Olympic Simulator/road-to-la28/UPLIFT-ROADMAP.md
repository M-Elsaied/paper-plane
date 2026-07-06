# Road to LA28 — Feature Uplift Roadmap

_Consolidated from a 5-expert × 2-round review (UI, UX, gamification, triathlon domain, healthy-habit design) → master orchestrator. 41 proposals in round 1, cross-reviewed and scored in round 2._

## 1. Unicorn thesis

**Road to LA28 is the live broadcast layer for Olympic qualification — the only place a fan, coach, or athlete can watch a real triathlete cross the qualification line in real time and know exactly what must happen next.** The emotional core is *the line*: a single glowing threshold between an Olympic dream and heartbreak, and the recurring, earned thrill of watching someone move across it. Everything else — sharing, predicting, rivalries, streaks — is scaffolding around that one broadcast-grade moment, made trustworthy by a deterministic rules engine no competitor has bothered to build.

## 2. The money-shot upgrade

**Wire the dormant projections engine into a Live Race Companion → Projected Monday Ranking.** Every single lens independently reached for it, and it's the biggest latent asset already sitting dark in the codebase. A projected (or live, or user-dragged) finish flows through the *same* exact rules engine to a projected OQR row that visibly crosses — or slips below — the gold line before results are official. It turns a static reference tool ("where do they stand?") into a live broadcast companion ("watch them qualify right now") — the appointment-viewing hook that gives race Sundays a reason to open the app. Low incremental effort (the config exists), category-defining payoff.

## 3. Prioritized master feature table

| # | Feature | Lens(es) | Impact | Effort | Unicorn leverage |
|---|---------|----------|--------|--------|------------------|
| 1 | **Live Race Companion → Projected Monday Ranking** (wire projections engine) | All 5 | H | L | Appointment-viewing; activates dead code; broadcast moat |
| 2 | **Shareable Qualification/Versus/Milestone Card + deep link** (one OG-image engine, 4 triggers) | All 5 | H | M | The only organic growth loop; k-factor engine |
| 3 | **60-Second First Scenario onboarding** (land inside the animating re-sort) | UI, UX, Tri, Behav | H | M | Activation; every retention loop needs a user this creates |
| 4 | **The Return Loop / Flightpath digest** (one honest "since you were away" home) | All 5 | H | M | Daily retention canvas; fixes the missing "home" IA |
| 5 | **Real-event notifications** (event-triggered, never engagement-triggered) | UX, Tri, Behav | H | M | The ethical trigger that reopens the app |
| 6 | **Race Pick-'Em / "Call It"** (deterministic auto-scored predictions) | UI, Gam, Tri, Behav | H | M | Best daily-play loop; seeds proprietary forecast data |
| 7 | **Head-to-head Versus / persistent Rivalry** (seeded intra-nation quota duel) | All 5 | H | M | Built-in drama; feeds share card + return triggers |
| 8 | **Line-cross celebration moment** (900ms cinematic → auto-offer share) | UI, Gam, Tri, Behav | H | S | The emotional payoff; delight-per-effort leader |
| 9 | **Explain the line** (deterministic engine as plain-language sentences + 8 pathways) | UX, Tri, Behav, UI | H | S | Trust moat made visible; reused everywhere |
| 10 | **NOC Slot War Room** (intra-nation depth chart, "out-point teammate by X") | UI, Tri | H | M | Novel, screenshot-worthy, exposes hardest real rule |
| 11 | **Investment layer / My Board** (roster + calls + pinned rivals + notes + discovery lanes) | UX, Tri, Behav | H | M | Compounding switch-cost; personalizes every loop |
| 12 | **Points Expiry Cliff** (decay timeline, loss-framed honest trigger) | Tri, Behav | M | S | Cheapest credibility win; quiet-day content teeth |
| 13 | **Qualification Status Spine** (one IN/CHASING/BLOCKED component everywhere) | UX | M | S | Connective tissue; makes 8 features feel like one app |
| 14 | **Athlete Duotone Portrait pipeline** (auto face-crop, duotone, nation accent, fallback) | UI | M | M | Invisible enabler that makes every share card look premium at scale |
| 15 | **Claim your Board** (passkey/magic-link local→sync bridge) | UX | M | M | Silent dependency that makes push + sharing physically possible |
| 16 | **Private Leagues & Mini-Leagues** (join-code container around Pick-'Em) | Gam | H | M | The k>1 viral + social-obligation engine (FPL model) |
| 17 | **Crowd Forecast + explainable Monte-Carlo odds** (fans-vs-model probability strip) | Gam, UI, Tri, Behav | M | M/L | Proprietary data moat; network-effect flywheel |
| 18 | **Start-List & Schedule Intelligence** (confirmed entries + field-strength) | Tri | M | M/L | Data plumbing that makes projections/Points Path *real* |
| 19 | **Points Path — Race Targeting Optimizer** | Tri | H | L | The pro moat; coach/agent evangelism; natural paid tier |
| 20 | **Pro Desk — multi-athlete roster cockpit** | Tri | H | L | The venture-scale B2B wedge; the path to real revenue |
| 21 | **Rank Trajectory sparkline** ("the climb" + expiry overlay) | UI, Tri, Behav | M | S | First historical trend; reads as data, feeds Wrapped |
| 22 | **Reduced-motion & colorblind-safe accessibility pass** | All 5 | M | S | Credibility/inclusion floor; keeps press + federations clean |
| 23 | **Race DNA fingerprint** (swim/bike/run profile) | UI, Tri | M | L | Scouting differentiator; honest input for odds |
| 24 | **Season Wrapped — "Your Road"** (longitudinal recap) | Behav | M | M | Identity artifact; proven viral re-engagement spike |

_Folded into the above rather than tracked separately: Broadcast Qualification Gauge, unified dataviz language, odometer numerals, Custom Watches (rides on notifications), Open-Loops tray (rides on Pick-'Em), Today on the Road (a card inside Flightpath), Calls & Convictions badges (a Pick-'Em skin), Trust Layer/backtest (folds into Explain-the-line)._

## 4. NOW / NEXT / LATER roadmap

### NOW — the spine (quick wins + the flagship's foundation)
- **Explain the line** (S) — trust backbone, reused in every later surface
- **Line-cross celebration** (S) — the emotional payoff, wired to auto-offer a share
- **Points Expiry Cliff** (S) — cheapest credibility, gives quiet days content
- **Rank Trajectory sparkline** (S) — first trend surface
- **Qualification Status Spine** (S/M) — build the one reused status component *before* proliferating screens
- **Accessibility pass** (S) — do it while surfaces are few
- **Athlete Duotone Portrait pipeline** (M) — enabler; makes the share card look premium
- **60-Second onboarding** (M) — activation; reuse the celebration choreography as the payoff
- **Flightpath return digest** (M) — the real home; honest quiet state
- **Shareable Card v1** (M) — the growth loop, seeded by the celebration + milestone triggers

### NEXT — the flagship + the loops
- **Live Race Companion → Projected Monday** (L) — *the money-shot upgrade*
- **Claim your Board** account bridge (M) — unblocks push + sync + share-back
- **Real-event notifications** (M) — trigger half of the loop (depends on account bridge)
- **Race Pick-'Em / Call It** (M) — daily-play + the open loop
- **Head-to-head Versus / Rivalry** (M) — seeded intra-nation quota duel
- **My Board investment layer + discovery lanes** (M) — the compounding moat
- **NOC Slot War Room** (M) — the signature differentiator viz
- **Start-List Intelligence** (M/L) — makes projections honest

### LATER — the moat + the money
- **Private Leagues** (M) — the viral container around Pick-'Em (needs accounts + predictions)
- **Crowd Forecast + Monte-Carlo odds** (M/L) — data moat (needs Pick-'Em volume + Race DNA)
- **Points Path optimizer** (L) — pro moat / paid tier
- **Pro Desk multi-athlete cockpit** (L) — B2B revenue wedge
- **Race DNA** (L) — data-heavy scouting layer
- **Season Wrapped** (M) — periodic viral spike
- **Selector Mode** (L) — deepest retention ceiling, *only* after the core loop is proven

## 5. Retention loop (healthy by design)

Runs on the sport's **real heartbeat** — the weekly Monday ranking update and the live race calendar — never on manufactured urgency.

- **Trigger (external):** a *real* event fires a deep-linked push — "Monday drop: Beaugrand crossed the line," "Rival banked a counting score," "Race starting in 15 min." Quiet hours respected; weekly-digest fallback; visible promise: *"we only ping when something actually happened."*
- **Action:** open → land on the **Flightpath digest** resolving the open loop you left with. On a race day, open the **Live Race Companion** and watch the row move. Lock a **Pick-'Em** call. Check a **rivalry** gap.
- **Variable reward:** the **line-cross celebration** fires only on genuine threshold crossings. Pick-'Em auto-resolves against real results. The reward is *being right about a sport you know* and *watching your athlete climb* — intrinsic, not currency.
- **Investment:** every follow, personal call, pinned rival, note, and locked prediction tailors the next digest, notification, and forecast — switching cost compounds with use. The **honest streak** counts only on real ranking-update events, auto-freezes during off-weeks, never breaks for a single miss.

Anti-dark-pattern proof point: the "**nothing changed, quiet**" state is a design *requirement*. Restraint is the moat — pros abandon apps that cry wolf.

## 6. Moat & monetization

**Defensible because:**
1. **The rules engine** — a pure, exact, explainable encoding of the real LA28 criteria. Competitors ship black-box guesses; this ships *sentences a coach can reconcile against the World Triathlon PDF*. A public backtest ("our engine matched the Paris cycle's final quota allocation") turns correctness into marketing.
2. **Proprietary prediction data** — aggregated Pick-'Em + Selector picks become a *fan qualification forecast* no one can copy without the user base. A genuine data flywheel.
3. **The projections + start-list layer** — "points realistically on offer given who's actually racing" is data casual tools can't fake.
4. **Viral distribution** — deep-linked share cards (k-factor) + private mini-leagues (social obligation) compound acquisition at near-zero CAC.

**Money (freemium + B2B):**
- **Free:** follow, cockpit, simulator, digest, celebration, Pick-'Em, basic sharing — the whole habit loop. Never gate the emotional core.
- **Pro / Superfan ($):** Points Path optimizer, unlimited rivalries + custom watches, Monte-Carlo odds, Race DNA, ad-free Wrapped, advanced share exports.
- **Pro Desk B2B ($$$ — the venture line):** multi-athlete roster cockpit for agents, coaches, and federations. Runs War Room math, Points Path, and Expiry Cliff across a whole squad. Federations and agencies *pay* for qualification planning — the realistic route from "nice fan app" to unicorn revenue. Per-seat / per-federation, with a public-methodology trust page as the opener.

## 7. Deliberately cut

- **Odometer / slot-roll hero numerals** — killed by three lenses. Pure motion polish, zero effect on activation/clarity/retention. Folds into the celebration + gauge.
- **Standalone Weekly Missions strip** — dark-pattern/busywork risk with a serious audience. Its useful parts fold into onboarding and Flightpath.
- **Global Predictor Leaderboards as a headline** — vanity, invites compulsion. Keep accuracy ranking only *inside* private leagues.
- **Federation Criteria Tracker** — high-maintenance manual data entry; the rules engine already computes the truth.
- **Selector Mode / Build Your Federation — deferred, not killed.** Deepest retention ceiling and most defensible "fantasy," but L-effort and forks the IA into a parallel game *before* the core loop is proven. Revisit after onboarding + return-loop + Pick-'Em + compare are validated.

**The discipline across all cuts: nothing ships that doesn't serve the line — the broadcast moment, the deterministic trust behind it, or the honest loop that brings people back to watch it.**
