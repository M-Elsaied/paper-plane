import Link from "next/link";
import { notFound } from "next/navigation";
import { SlidersHorizontal, ShieldCheck, TriangleAlert, Users } from "lucide-react";
import { buildCockpit } from "@/lib/cockpit";
import { explainPosition } from "@/lib/explain";
import { AthleteAvatar } from "@/components/athlete-avatar";
import { CountUp } from "@/components/count-up";
import { QualMeter } from "@/components/qual-meter";
import { StatusBadge } from "@/components/status-badge";
import { ExplainLineCard } from "@/components/explain-line";
import { PeriodTimeline } from "@/components/period-timeline";
import { ShareButton } from "@/components/share-button";
import { CockpitTour } from "@/components/tour/cockpit-tour";
import { UnrankedProfileView } from "@/components/unranked-profile";
import { buildUnrankedProfile } from "@/lib/athlete-profile";
import { fmtPoints } from "@/lib/format";
import { getSeedMeta, findAthlete } from "@/lib/data";
import { fetchAthleteProfile } from "@/lib/wt-api/athletes";
import { cn } from "@/lib/utils";

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const found = await findAthlete(Number(id));
  // Ranked athletes get the broadcast card; unranked get a plain title.
  if (!found) {
    const profile = await fetchAthleteProfile(Number(id)).catch(() => null);
    const nm = profile?.fullName ?? "Athlete";
    return { title: `${nm} · Road to LA28`, description: `${nm}'s World Triathlon profile.` };
  }
  const name = found.athlete.fullName;
  return {
    title: `${name} · Road to LA28`,
    description: `${name}'s road to the LA 2028 Olympics — live qualification ranking and what-if simulator.`,
    openGraph: {
      title: `${name} · Road to LA28`,
      images: [{ url: `/athlete/${id}/card`, width: 1080, height: 1350 }],
    },
    twitter: { card: "summary_large_image", images: [`/athlete/${id}/card`] },
  };
}

export default async function AthleteCockpit({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const m = await buildCockpit(Number(id));
  if (!m) {
    // Not in the OQR — show an honest profile (or 404 if the id is unknown to WT).
    const profile = await buildUnrankedProfile(Number(id), getSeedMeta().today).catch(() => null);
    if (!profile) notFound();
    return <UnrankedProfileView p={profile} />;
  }

  const insideBy = -m.gapToLine; // positive when inside the line
  const explain = explainPosition(m);
  const today = getSeedMeta().today;

  return (
    <main className="px-4 pt-6">
      {/* Identity header */}
      <header className="mb-5 flex items-center gap-3">
        <AthleteAvatar name={m.fullName} src={m.profileImage} size={56} ring={m.qualified} />
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-xl font-extrabold leading-tight">{m.fullName}</h1>
          <div className="flex items-center gap-2 text-sm text-ink-dim">
            <span className="font-semibold">{m.noc}</span>
            <span className="text-ink-faint">·</span>
            <span className="capitalize">{m.gender === "male" ? "Elite Men" : "Elite Women"}</span>
          </div>
        </div>
        <div data-tour="status" className="flex flex-col items-end gap-1.5">
          <StatusBadge status={m.status} />
          <ShareButton athleteId={m.athleteId} name={m.fullName} />
        </div>
      </header>

      {/* Hero: rank + points */}
      <section data-tour="hero" className="card mb-4 p-4">
        <div className="flex items-end justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-ink-faint">Olympic Rank</div>
            <div className="tnum text-6xl font-black leading-none">
              <span className="text-ink-faint text-2xl align-top">#</span>
              <CountUp value={m.rank} />
            </div>
          </div>
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-wide text-ink-faint">Points</div>
            <div className="tnum text-3xl font-extrabold la-gradient-text">
              <CountUp value={m.total} />
            </div>
          </div>
        </div>

        <div className="mt-4">
          <QualMeter total={m.total} cutPoints={m.cutPoints} qualified={m.qualified} />
        </div>

        <div className="mt-1 flex items-center justify-center gap-1.5 text-sm">
          {m.qualified ? (
            <span className="font-semibold text-good">
              Inside the line by {fmtPoints(insideBy)} pts
            </span>
          ) : (
            <span className="font-semibold text-electric-bright">
              {fmtPoints(m.gapToLine)} pts from the cut (#{m.cutRank})
            </span>
          )}
        </div>
      </section>

      {/* CTA — the money shot */}
      <Link
        data-tour="simulate"
        href={`/athlete/${m.athleteId}/simulate`}
        className="mb-4 flex items-center justify-between rounded-2xl border border-white/10 la-gradient px-4 py-3.5 text-navy-950 shadow-lg transition active:scale-[0.99]"
      >
        <span className="flex items-center gap-2 font-extrabold">
          <SlidersHorizontal size={18} /> Simulate a race result
        </span>
        <span className="text-sm font-bold">What if →</span>
      </Link>

      {/* Explain the line — the engine in plain language */}
      <div data-tour="explain" className="mb-4">
        <ExplainLineCard lines={explain} />
      </div>

      {/* Countdown + best-12 */}
      <section className="mb-4 grid grid-cols-2 gap-3">
        <div className="card p-3">
          <div className="text-[10px] uppercase tracking-wide text-ink-faint">To deadline</div>
          <div className="tnum mt-1 text-2xl font-extrabold">
            <CountUp value={m.daysToDeadline} /> <span className="text-sm text-ink-faint">days</span>
          </div>
          <div className="text-[11px] text-ink-faint">18 May 2028</div>
        </div>
        <div className="card p-3">
          <div className="text-[10px] uppercase tracking-wide text-ink-faint">Points to defend</div>
          <div className="tnum mt-1 text-2xl font-extrabold text-la-gold">
            {fmtPoints(m.marginalPoints)}
          </div>
          <div className="text-[11px] text-ink-faint">weakest counting score</div>
        </div>
      </section>

      {/* Best-12 breakdown */}
      <section className="card mb-4 p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-bold">Best 12 counting scores</h2>
          <span className="text-[11px] text-ink-faint">
            P1 {m.periodCount[1]}/7 · P2 {m.periodCount[2]}/7
          </span>
        </div>
        <PeriodBars counted={m.counted} periodFull={m.periodFull} />
        {(m.periodFull[1] || m.periodFull[2]) && (
          <div className="mt-3 flex items-start gap-1.5 rounded-lg bg-la-gold/10 px-2.5 py-2 text-[11px] text-la-gold">
            <TriangleAlert size={13} className="mt-0.5 shrink-0" />
            <span>
              Period {m.periodFull[1] ? "1" : "2"} is full (7/7) — only a score above{" "}
              {fmtPoints(m.marginalPoints)} in that period will improve the total.
            </span>
          </div>
        )}
      </section>

      {/* Qualification window / points expiry */}
      <div data-tour="window" className="mb-4">
        <PeriodTimeline periodCount={m.periodCount} periodFull={m.periodFull} todayIso={today} />
      </div>

      {/* Mixed Relay strip */}
      <section className="card mb-4 flex items-center gap-3 p-4">
        <Users size={20} className={m.mr.insideTop16 ? "text-good" : "text-ink-faint"} />
        <div className="flex-1">
          <div className="text-sm font-bold">{m.noc} Mixed Relay pathway</div>
          <div className="text-[11px] text-ink-dim">
            {m.mr.rank
              ? `MR Olympic rank #${m.mr.rank} · ${
                  m.mr.insideTop16 ? "inside the top 16 ✓" : `${fmtPoints(m.mr.gapToTop16)} pts outside`
                }`
              : "not currently in the Mixed Relay Olympic ranking"}
          </div>
        </div>
        {m.mr.insideTop16 && <ShieldCheck size={18} className="text-good" />}
      </section>

      {/* Chasers */}
      <section className="mb-2">
        <h2 className="mb-2 text-sm font-bold">Around the athlete</h2>
        <ul className="space-y-1">
          {m.chasers.map((c) => (
            <li
              key={c.athleteId}
              className="flex items-center gap-2 rounded-lg bg-white/[0.03] px-3 py-2 text-sm"
            >
              <span className="tnum w-6 text-ink-faint">#{c.rank}</span>
              <span className="flex-1 truncate">{c.fullName}</span>
              <span className="text-ink-faint">{c.noc}</span>
              <span className="tnum font-semibold">{fmtPoints(c.total)}</span>
            </li>
          ))}
        </ul>
      </section>

      <div className="flex items-center justify-center gap-3 pt-3 text-center">
        <span className="text-[11px] text-ink-faint">
          Ranking published {m.publishedAt.slice(0, 10)}
        </span>
        <span className="text-ink-faint">·</span>
        <CockpitTour />
      </div>
    </main>
  );
}

function PeriodBars({
  counted,
  periodFull,
}: {
  counted: { points: number; period: 1 | 2 }[];
  periodFull: Record<1 | 2, boolean>;
}) {
  const max = Math.max(1, ...counted.map((c) => c.points));
  return (
    <div className="space-y-1.5">
      {counted.map((c, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-6 text-[10px] font-semibold text-ink-faint">P{c.period}</span>
          <div className="h-3 flex-1 overflow-hidden rounded-full bg-white/6">
            <div
              className={cn("h-full rounded-full", c.period === 1 ? "bg-electric" : "bg-la-violet")}
              style={{ width: `${(c.points / max) * 100}%` }}
            />
          </div>
          <span className="tnum w-12 text-right text-xs font-semibold">{fmtPoints(c.points)}</span>
        </div>
      ))}
    </div>
  );
}
