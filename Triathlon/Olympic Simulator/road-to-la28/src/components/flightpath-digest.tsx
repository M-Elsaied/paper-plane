import Link from "next/link";
import { Activity, CalendarDays, ArrowRight } from "lucide-react";
import { getMovers, getUpcomingEvents } from "@/lib/data";
import { MovementArrow } from "./movement-arrow";
import { TIER_BASE_POINTS } from "@/config/points-tables";
import { fmtPoints } from "@/lib/format";

/**
 * Flightpath — the "since you were away" digest. A calm daily home brief: who
 * moved on the latest ranking and what's racing next. Honest quiet state when
 * nothing has changed.
 */
export async function FlightpathDigest() {
  const [movers, women] = await Promise.all([getMovers("male", 1), getMovers("female", 1)]);
  const topMovers = [movers[0], women[0]].filter(Boolean);
  const nextRace = getUpcomingEvents()[0];

  if (!topMovers.length && !nextRace) return null;

  return (
    <section className="card mb-5 overflow-hidden">
      <div className="flex items-center gap-2 border-b border-hairline px-4 py-2.5">
        <span className="h-1.5 w-1.5 rounded-full bg-good" />
        <span className="text-[11px] font-bold uppercase tracking-wide text-ink-dim">
          On the road today
        </span>
      </div>

      <div className="divide-y divide-white/6">
        {topMovers.map((m) => (
          <Link
            key={m.athleteId}
            href={`/athlete/${m.athleteId}`}
            className="flex items-center gap-3 px-4 py-2.5 transition hover:bg-surface-2"
          >
            <Activity size={16} className="text-electric-bright" />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-semibold">{m.fullName}</div>
              <div className="text-[11px] text-ink-faint">
                {m.noc} · now #{m.rank}
              </div>
            </div>
            <MovementArrow delta={m.change} />
          </Link>
        ))}

        {nextRace && (
          <div className="flex items-center gap-3 px-4 py-2.5">
            <CalendarDays size={16} className="text-la-gold" />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-semibold">
                {nextRace.title.replace(/^\d{4}\s+/, "")}
              </div>
              <div className="text-[11px] text-ink-faint">
                {nextRace.tierLabel} · {fmtPoints(TIER_BASE_POINTS[nextRace.tier])} win pts ·{" "}
                {new Date(nextRace.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
              </div>
            </div>
            <Link href="/race-week" className="text-ink-faint">
              <ArrowRight size={16} />
            </Link>
          </div>
        )}
      </div>
    </section>
  );
}
