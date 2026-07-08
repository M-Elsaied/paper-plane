import Link from "next/link";
import { Info, CalendarDays, Compass } from "lucide-react";
import type { UnrankedProfile } from "@/lib/athlete-profile";
import { AthleteAvatar } from "./athlete-avatar";
import { ShareButton } from "./share-button";
import { cn } from "@/lib/utils";

/** Cockpit variant for an athlete not (yet) on the Olympic Qualification Ranking. */
export function UnrankedProfileView({ p }: { p: UnrankedProfile }) {
  return (
    <main className="px-4 pt-6">
      <header className="mb-5 flex items-center gap-3">
        <AthleteAvatar name={p.fullName} src={p.profileImage} size={56} />
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-xl font-extrabold leading-tight">{p.fullName}</h1>
          <div className="flex items-center gap-2 text-sm text-ink-dim">
            <span className="font-semibold">{p.noc}</span>
            {p.countryName && <span className="truncate text-ink-faint">· {p.countryName}</span>}
            <span className="text-ink-faint">·</span>
            <span className="capitalize">{p.gender === "male" ? "Men" : "Women"}</span>
            {p.age != null && <span className="text-ink-faint">· {p.age}y</span>}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1.5">
          <span className="rounded-full bg-white/10 px-2.5 py-1 text-[11px] font-bold text-ink-dim">UNRANKED</span>
          <ShareButton athleteId={p.athleteId} name={p.fullName} />
        </div>
      </header>

      {/* Honest status */}
      <section className="card mb-4 p-4">
        <div className="mb-2 flex items-center gap-2">
          <Info size={15} className="text-electric-bright" />
          <h2 className="text-sm font-bold">Not yet on the Olympic Qualification Ranking</h2>
        </div>
        <p className="text-[13px] leading-snug text-ink-dim">
          <span className="font-semibold text-ink">{p.fullName.split(" ")[0]}</span>{" "}
          isn&apos;t currently among the ranked athletes chasing an LA 2028 individual place. To appear
          on the road to qualification, an athlete needs counting results at World Triathlon events and
          must climb inside the top {p.eligibilityTopRank} of the World Ranking by 18 May 2028.
        </p>
        <Link
          href={`/athlete/${p.athleteId}/road`}
          className="mt-3 flex items-center justify-between rounded-xl border border-electric/30 bg-electric/10 px-3 py-2.5 transition active:scale-[0.99]"
        >
          <span className="flex items-center gap-2 text-sm font-bold text-electric-bright">
            <Compass size={16} /> See {p.fullName.split(" ")[0]}&apos;s realistic routes
          </span>
          <span className="text-xs font-semibold text-ink-dim">→</span>
        </Link>
      </section>

      {/* Recent results */}
      <section className="mb-4">
        <h2 className="mb-2 text-sm font-bold">Recent results</h2>
        {p.results.length === 0 ? (
          <div className="card p-6 text-center text-sm text-ink-faint">No recent results on record.</div>
        ) : (
          <ul className="space-y-1">
            {p.results.map((r, i) => (
              <li key={`${r.eventId}-${i}`} className="flex items-center gap-2 rounded-lg bg-white/[0.03] px-3 py-2 text-sm">
                <CalendarDays size={13} className="shrink-0 text-ink-faint" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{r.eventTitle.replace(/^\d{4}\s+/, "")}</span>
                  <span className="text-[11px] text-ink-faint">
                    {r.date}
                    {r.program ? ` · ${r.program}` : ""}
                  </span>
                </span>
                <span
                  className={cn(
                    "tnum shrink-0 rounded-md px-2 py-0.5 text-xs font-bold",
                    typeof r.position === "number" && r.position <= 3
                      ? "bg-la-gold/20 text-la-gold"
                      : "bg-white/8 text-ink-dim",
                  )}
                >
                  {r.position == null ? "—" : typeof r.position === "number" ? `${r.position}` : r.position}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="text-center">
        <Link href="/rankings" className="text-[11px] font-semibold text-electric-bright">
          See who&apos;s in the qualifying places →
        </Link>
      </div>
    </main>
  );
}
