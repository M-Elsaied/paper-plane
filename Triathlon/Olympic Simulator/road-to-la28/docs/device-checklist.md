# Manual device checklist

The automated pyramid (unit → golden → integration → E2E → smoke) covers
everything driveable headlessly. These are the bits that genuinely need a real
phone. Run once per meaningful release on **one Android** and **one iPhone
(iOS ≥ 16.4** — required for web push).

## Install (PWA)
- [ ] Visiting the site offers "Add to Home Screen" (Android) / Share → Add to Home Screen (iOS)
- [ ] Installed app launches **standalone** (no browser chrome) with the LA28 icon + navy splash
- [ ] Opens straight to the cockpit / last screen; bottom nav clears the home indicator (safe-area)

## Push notifications
- [ ] `/account` → "Turn on alerts" → OS permission prompt appears; granting succeeds
- [ ] "Send a test notification" → notification arrives within seconds, with the app icon + correct copy
- [ ] Tapping the notification **deep-links** to the right screen (not just the home page)
- [ ] After a real Monday ranking update, a followed athlete's move produces a push with correct copy (crossed-line / climbed N) and intact emoji
- [ ] Turning alerts off stops future pushes

## Sharing
- [ ] Cockpit "Share" opens the **native share sheet** (not just a copied link) on a device that supports it
- [ ] The shared link unfurls with the broadcast card image in WhatsApp / iMessage preview

## Interaction (touch-only paths)
- [ ] Simulator slider drags smoothly; crossing the line fires the celebration at **full motion** (device not in reduced-motion)
- [ ] Race Companion "Projected finish" — drag an athlete to reorder via the touch handle (Framer Motion's touch path differs from mouse)
- [ ] Pick-'Em — tap three athletes into the podium, lock in; card shows locked state

## Offline
- [ ] After visiting a few pages, enable airplane mode → previously visited pages still render from the service-worker cache

## Notes
- Real push delivery cannot be asserted in CI — this checklist is the source of
  truth for it. If a push regression is suspected, verify VAPID env vars are set
  in Vercel prod and re-run "Send a test notification".
