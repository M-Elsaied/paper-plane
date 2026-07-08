import { QUAL } from "@/config/qualification";
import { cn } from "@/lib/utils";

/**
 * Points Expiry Cliff — the two-period window as a timeline. Shows how many
 * counting scores are locked in each period, where "today" sits, and the cap
 * pressure (7/7 = a period is full and can only be improved, not added to).
 */
export function PeriodTimeline({
  periodCount,
  periodFull,
  todayIso,
}: {
  periodCount: Record<1 | 2, number>;
  periodFull: Record<1 | 2, boolean>;
  todayIso?: string;
}) {
  const today = todayIso ?? new Date().toISOString().slice(0, 10);
  const p1 = QUAL.periods[1];
  const p2 = QUAL.periods[2];
  const span = (a: string, b: string) => new Date(b).getTime() - new Date(a).getTime();
  const total = span(p1.from, p2.to);
  const nowPct = clamp(((new Date(today).getTime() - new Date(p1.from).getTime()) / total) * 100);
  const p1EndPct = (span(p1.from, p1.to) / total) * 100;

  return (
    <section className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold">Qualification window</h2>
        <span className="text-[11px] text-ink-faint">closes {QUAL.deadline}</span>
      </div>

      {/* Track */}
      <div className="relative mt-1 h-8">
        <div className="absolute inset-x-0 top-3 flex h-2 overflow-hidden rounded-full">
          <div className="h-full bg-electric/40" style={{ width: `${p1EndPct}%` }} />
          <div className="h-full bg-la-violet/40" style={{ width: `${100 - p1EndPct}%` }} />
        </div>
        {/* today marker */}
        <div className="absolute top-0 bottom-0" style={{ left: `${nowPct}%` }}>
          <div className="h-full w-0.5 bg-la-gold" />
          <span className="absolute -top-0.5 left-1 whitespace-nowrap text-[9px] font-bold text-la-gold">
            today
          </span>
        </div>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-3">
        <PeriodChip n={1} count={periodCount[1]} full={periodFull[1]} from={p1.from} to={p1.to} accent="electric" />
        <PeriodChip n={2} count={periodCount[2]} full={periodFull[2]} from={p2.from} to={p2.to} accent="violet" />
      </div>
    </section>
  );
}

function PeriodChip({
  n,
  count,
  full,
  from,
  to,
  accent,
}: {
  n: number;
  count: number;
  full: boolean;
  from: string;
  to: string;
  accent: "electric" | "violet";
}) {
  return (
    <div className="rounded-xl bg-surface p-2.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-bold">Period {n}</span>
        <span
          className={cn(
            "tnum rounded-md px-1.5 py-0.5 text-[10px] font-bold",
            full ? "bg-la-gold/20 text-la-gold" : accent === "electric" ? "bg-electric/15 text-electric-bright" : "bg-la-violet/20 text-la-violet",
          )}
        >
          {count}/{QUAL.maxPerPeriod}
        </span>
      </div>
      <div className="mt-1 text-[10px] text-ink-faint">
        {shortDate(from)} – {shortDate(to)}
      </div>
      {full && <div className="mt-1 text-[10px] font-semibold text-la-gold">Full — improve only</div>}
    </div>
  );
}

function shortDate(iso: string) {
  const d = new Date(iso);
  return `${d.toLocaleDateString("en-US", { month: "short" })} '${String(d.getFullYear()).slice(2)}`;
}
function clamp(n: number) {
  return Math.min(100, Math.max(0, n));
}
