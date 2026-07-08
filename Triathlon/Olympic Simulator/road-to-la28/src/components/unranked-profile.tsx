import Link from "next/link";
import { Info, Compass, Trophy, Medal, CalendarDays, MapPin, Hash } from "lucide-react";
import type { UnrankedProfile } from "@/lib/athlete-profile";
import { AthleteAvatar } from "./athlete-avatar";
import { Flag } from "./flag";
import { ShareButton } from "./share-button";
import { fmtPoints } from "@/lib/format";
import { cn } from "@/lib/utils";

/** Rich profile for an athlete not (yet) on the Olympic Qualification Ranking. */
export function UnrankedProfileView({ p }: { p: UnrankedProfile }) {
  const first = p.fullName.split(" ")[0];
  return (
    <main className="mx-auto px-4 pt-6 lg:max-w-2xl">
      {/* Hero */}
      <header className="card mb-4 overflow-hidden">
        <div className="flex items-start gap-3 p-4">
          <AthleteAvatar name={p.fullName} src={p.profileImage} size={64} />
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-xl font-extrabold leading-tight">{p.fullName}</h1>
            <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-sm text-ink-dim">
              <span className="inline-flex items-center gap-1.5 font-semibold">
                <Flag src={p.flag} iso={p.countryIso} noc={p.noc} size={18} />
                {p.noc}
              </span>
              {p.countryName && <span className="text-ink-faint">{p.countryName}</span>}
              <span className="text-ink-faint">·</span>
              <span>{p.gender === "male" ? "Men" : "Women"}</span>
              {p.yearOfBirth && (
                <>
                  <span className="text-ink-faint">·</span>
                  <span className="text-ink-faint">
                    b.{p.yearOfBirth}
                    {p.age != null ? ` (${p.age})` : ""}
                  </span>
                </>
              )}
            </div>
            <div className="mt-1.5 flex items-center gap-1 text-[11px] text-ink-faint">
              <Hash size={11} /> WT Athlete ID {p.athleteId}
            </div>
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <span className="rounded-full bg-surface-2 px-2.5 py-1 text-[11px] font-bold text-ink-dim">UNRANKED</span>
            <ShareButton athleteId={p.athleteId} name={p.fullName} />
          </div>
        </div>

        {/* Career stats */}
        {p.stats && (
          <div className="grid grid-cols-4 divide-x divide-white/8 border-t border-hairline">
            <Stat icon={<CalendarDays size={13} />} value={p.stats.starts} label="Starts" />
            <Stat icon={<Medal size={13} />} value={p.stats.podiums} label="Podiums" accent={p.stats.podiums > 0} />
            <Stat icon={<Trophy size={13} />} value={p.stats.wins} label="Wins" accent={p.stats.wins > 0} />
            <Stat value={`${p.stats.finishPct}%`} label="Finish" />
          </div>
        )}
        {p.ageGroupRank != null && (
          <div className="flex items-center justify-between border-t border-hairline px-4 py-2 text-[12px]">
            <span className="text-ink-faint">Age-Group World Ranking</span>
            <span className="tnum font-bold text-ink">#{fmtPoints(p.ageGroupRank)}</span>
          </div>
        )}
      </header>

      {/* Honest status + CTA */}
      <section className="card mb-4 p-4">
        <div className="mb-2 flex items-center gap-2">
          <Info size={15} className="text-electric-bright" />
          <h2 className="text-sm font-bold">Not yet on the Olympic Qualification Ranking</h2>
        </div>
        <p className="text-[13px] leading-snug text-ink-dim">
          <span className="font-semibold text-ink">{first}</span>
          {" isn't currently among the ranked athletes chasing an LA 2028 individual place. "} To appear on the road to qualification, an
          athlete needs counting results at World Triathlon events and must climb inside the top{" "}
          {p.eligibilityTopRank} of the World Ranking by 18 May 2028.
        </p>
        <Link
          href={`/athlete/${p.athleteId}/road`}
          className="mt-3 flex items-center justify-between rounded-xl border border-electric/30 bg-electric/10 px-3 py-2.5 transition active:scale-[0.99]"
        >
          <span className="flex items-center gap-2 text-sm font-bold text-electric-bright">
            <Compass size={16} /> See {first}&apos;s realistic routes
          </span>
          <span className="text-xs font-semibold text-ink-dim">→</span>
        </Link>
      </section>

      {/* Results */}
      <section className="mb-2">
        <h2 className="mb-2 text-sm font-bold">Recent results</h2>
        {p.results.length === 0 ? (
          <div className="card p-6 text-center text-sm text-ink-faint">No recent results on record.</div>
        ) : (
          <ul className="space-y-1.5">
            {p.results.map((r, i) => (
              <li key={`${r.eventId}-${i}`} className="card flex items-center gap-3 p-3">
                <PositionBadge position={r.position} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <Flag src={r.eventFlag} iso={r.eventIso} size={14} />
                    <span className="truncate text-[13px] font-semibold">
                      {r.eventTitle.replace(/^\d{4}\s+/, "")}
                    </span>
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[11px] text-ink-faint">
                    <span>{r.date}</span>
                    {r.venue && (
                      <span className="inline-flex items-center gap-0.5">
                        <MapPin size={10} /> {r.venue}
                      </span>
                    )}
                    {r.program && <span>· {r.program}</span>}
                  </div>
                </div>
                {r.totalTime && <span className="tnum shrink-0 text-[12px] font-semibold text-ink-dim">{r.totalTime}</span>}
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="pt-3 text-center">
        <Link href="/rankings" className="text-[11px] font-semibold text-electric-bright">
          See who&apos;s in the qualifying places →
        </Link>
      </div>
    </main>
  );
}

function Stat({
  icon,
  value,
  label,
  accent,
}: {
  icon?: React.ReactNode;
  value: React.ReactNode;
  label: string;
  accent?: boolean;
}) {
  return (
    <div className="flex flex-col items-center py-3">
      <div className={cn("tnum flex items-center gap-1 text-lg font-extrabold", accent ? "text-la-gold" : "text-ink")}>
        {icon && <span className={accent ? "text-la-gold" : "text-ink-faint"}>{icon}</span>}
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-wide text-ink-faint">{label}</div>
    </div>
  );
}

function PositionBadge({ position }: { position: number | string | null }) {
  const isPodium = typeof position === "number" && position <= 3;
  const medal = position === 1 ? "text-la-gold" : position === 2 ? "text-ink" : position === 3 ? "text-[#cd7f32]" : "";
  return (
    <span
      className={cn(
        "tnum flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-sm font-black",
        isPodium ? "bg-la-gold/15" : "bg-surface-2",
        isPodium ? medal : "text-ink-dim",
      )}
    >
      {position == null ? "—" : position}
    </span>
  );
}
