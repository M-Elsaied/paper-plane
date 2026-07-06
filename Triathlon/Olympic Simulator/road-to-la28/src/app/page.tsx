import { getAthleteDirectory } from "@/lib/data";
import { buildRanking } from "@/lib/cockpit";
import { AthletePicker } from "@/components/athlete-picker";
import { MyAthleteResume } from "@/components/my-athlete-resume";
import { fmtPoints, daysToDeadline } from "@/lib/format";

export default function Home() {
  const directory = getAthleteDirectory();
  const men = buildRanking("male");
  const women = buildRanking("female");
  const days = daysToDeadline();

  return (
    <main className="px-4 pt-8">
      {/* Hero */}
      <header className="mb-6">
        <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] font-semibold text-ink-dim">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-good" />
          LIVE · Olympic Qualification Ranking
        </div>
        <h1 className="font-[family-name:var(--font-display)] text-5xl font-black leading-[0.95] tracking-tight">
          ROAD TO
          <br />
          <span className="la-gradient-text">LOS ANGELES 28</span>
        </h1>
        <p className="mt-3 max-w-xs text-sm text-ink-dim">
          Every athlete&apos;s road to the Olympic start line — live rankings, the
          qualification line, and instant what-if simulation.
        </p>
      </header>

      {/* Headline stats */}
      <section className="mb-6 grid grid-cols-3 gap-2">
        <Stat label="Days to deadline" value={fmtPoints(days)} accent />
        <Stat label="Men's cut" value={`${fmtPoints(men.line.cutPoints)} pt`} sub={`#${men.line.cutRank}`} />
        <Stat label="Women's cut" value={`${fmtPoints(women.line.cutPoints)} pt`} sub={`#${women.line.cutRank}`} />
      </section>

      <MyAthleteResume />

      <div className="my-5 flex items-center gap-3">
        <div className="h-px flex-1 bg-white/10" />
        <span className="text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
          Pick your athlete
        </span>
        <div className="h-px flex-1 bg-white/10" />
      </div>

      <AthletePicker directory={directory} />
    </main>
  );
}

function Stat({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div className="card p-3">
      <div className="text-[10px] uppercase tracking-wide text-ink-faint">{label}</div>
      <div className={`tnum mt-1 text-lg font-extrabold ${accent ? "la-gradient-text" : "text-ink"}`}>
        {value}
      </div>
      {sub && <div className="text-[11px] text-ink-faint">{sub}</div>}
    </div>
  );
}
