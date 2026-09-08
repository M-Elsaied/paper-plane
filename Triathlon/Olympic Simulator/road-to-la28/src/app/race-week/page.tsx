import Link from "next/link";
import { CalendarDays, MapPin, Trophy, ArrowRight } from "lucide-react";
import { getUpcomingEvents } from "@/lib/data";
import { todayIso } from "@/lib/today";
import type { UpcomingEvent } from "@/lib/wt-api/events";
import { TIER_BASE_POINTS } from "@/config/points-tables";
import { fmtPoints } from "@/lib/format";

export const revalidate = 300;

export default async function RaceWeekPage() {
  const events = await getUpcomingEvents();
  const today = todayIso();

  const grouped = groupByWeek(events);

  return (
    <main className="mx-auto px-4 pt-6 lg:max-w-2xl">
      <header className="mb-4">
        <h1 className="text-2xl font-extrabold">Race Week</h1>
        <p className="text-sm text-ink-faint">
          Upcoming elite races with Olympic points on the line. Projected start lists and finishes
          populate as World Triathlon publishes them.
        </p>
      </header>

      {grouped.length === 0 && (
        <div className="card p-6 text-center text-sm text-ink-faint">No upcoming races in range.</div>
      )}

      <div className="space-y-5">
        {grouped.map((g) => (
          <section key={g.week}>
            <div className="mb-2 flex items-center gap-2 text-[11px] font-bold uppercase tracking-wide text-ink-faint">
              <CalendarDays size={13} /> {g.week}
            </div>
            <div className="space-y-2">
              {g.events.map((e) => (
                <Link key={e.eventId} href={`/race/${e.eventId}`} className="card block p-4 transition hover:border-electric/40">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="mb-1 inline-flex items-center gap-1 rounded-full bg-surface-2 px-2 py-0.5 text-[10px] font-bold text-electric-bright">
                        {e.tierLabel}
                      </div>
                      <h2 className="truncate text-sm font-bold leading-tight">
                        {e.title.replace(/^\d{4}\s+/, "")}
                      </h2>
                      {(e.venue || e.country) && (
                        <div className="mt-0.5 flex items-center gap-1 text-[11px] text-ink-faint">
                          <MapPin size={11} /> {e.venue || e.country}
                        </div>
                      )}
                    </div>
                    <div className="shrink-0 text-right">
                      <div className="flex items-center gap-1 text-la-gold">
                        <Trophy size={13} />
                        <span className="tnum text-lg font-extrabold">{fmtPoints(TIER_BASE_POINTS[e.tier])}</span>
                      </div>
                      <div className="text-[10px] text-ink-faint">win points</div>
                    </div>
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <span className="text-[11px] text-ink-faint">{formatDate(e.date)}</span>
                    <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-electric-bright">
                      Project this race <ArrowRight size={12} />
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>

      <p className="pt-4 text-center text-[11px] text-ink-faint">Schedule as of {today}</p>
    </main>
  );
}

function isoWeekLabel(iso: string): string {
  const d = new Date(iso);
  const day = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  const dayNum = (day.getUTCDay() + 6) % 7;
  day.setUTCDate(day.getUTCDate() - dayNum + 3);
  const firstThursday = new Date(Date.UTC(day.getUTCFullYear(), 0, 4));
  const week =
    1 +
    Math.round(
      ((day.getTime() - firstThursday.getTime()) / 86_400_000 - 3 + ((firstThursday.getUTCDay() + 6) % 7)) / 7,
    );
  return `${day.getUTCFullYear()} · Week ${week}`;
}

function groupByWeek(events: UpcomingEvent[]) {
  const map = new Map<string, typeof events>();
  for (const e of events) {
    const w = isoWeekLabel(e.date);
    if (!map.has(w)) map.set(w, []);
    map.get(w)!.push(e);
  }
  return [...map.entries()].map(([week, evs]) => ({ week, events: evs }));
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
}
