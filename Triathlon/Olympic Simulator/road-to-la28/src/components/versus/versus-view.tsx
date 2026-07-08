import Link from "next/link";
import { Swords, ShieldAlert, CalendarDays } from "lucide-react";
import type { VersusModel, VersusAthlete } from "@/lib/versus";
import { AthleteAvatar } from "../athlete-avatar";
import { Flag } from "../flag";
import { RivalButton } from "./rival-button";
import { fmtPoints } from "@/lib/format";
import { cn } from "@/lib/utils";

type Side = "a" | "b";

export function VersusView({ v }: { v: VersusModel }) {
  const { a, b, h2h } = v;
  return (
    <div className="space-y-4">
      {/* Dual header */}
      <div className="card overflow-hidden">
        <div className="grid grid-cols-[1fr_auto_1fr] items-stretch">
          <AthleteHead athlete={a} align="left" />
          <div className="flex flex-col items-center justify-center gap-1 border-x border-hairline bg-surface px-3">
            <Swords size={20} className="text-la-gold" />
            <span className="text-[10px] font-black uppercase tracking-wider text-ink-faint">vs</span>
          </div>
          <AthleteHead athlete={b} align="right" />
        </div>
      </div>

      {/* Same-nation cap duel */}
      {v.sameNoc && (
        <div className="card flex items-start gap-2.5 p-3">
          <ShieldAlert size={18} className="mt-0.5 shrink-0 text-la-gold" />
          <p className="text-[13px] leading-snug text-ink-dim">
            <span className="font-semibold text-ink">Same nation.</span> {a.name.split(" ")[0]} and{" "}
            {b.name.split(" ")[0]} both race for {a.noc}, so they compete for the same limited quota
            places — a nation may send at most 3 (2 without depth in the top 30). Beating the field
            isn&apos;t enough; one has to out-point the other.
          </p>
        </div>
      )}

      {/* Comparison rows */}
      <div className="card divide-y divide-hairline">
        <CompareRow label="Olympic rank" av={fmtRank(a.oqrRank)} bv={fmtRank(b.oqrRank)} winner={betterLow(a.oqrRank, b.oqrRank)} />
        <CompareRow label="Points" av={fmtPoints(a.points)} bv={fmtPoints(b.points)} winner={betterHigh(a.points, b.points)} />
        <CompareRow label="Gap to the line" av={fmtGap(a.gapToLine)} bv={fmtGap(b.gapToLine)} winner={betterLow(a.gapToLine, b.gapToLine)} />
        <CompareRow label="World ranking" av={fmtRank(a.worldRank)} bv={fmtRank(b.worldRank)} winner={betterLow(a.worldRank, b.worldRank)} />
        <CompareRow label="Career starts" av={a.stats?.starts} bv={b.stats?.starts} winner={betterHigh(a.stats?.starts, b.stats?.starts)} />
        <CompareRow label="Podiums" av={a.stats?.podiums} bv={b.stats?.podiums} winner={betterHigh(a.stats?.podiums, b.stats?.podiums)} />
        <CompareRow label="Wins" av={a.stats?.wins} bv={b.stats?.wins} winner={betterHigh(a.stats?.wins, b.stats?.wins)} />
      </div>

      {/* Head-to-head record */}
      {v.sameGender && (
        <section className="card p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold">Head-to-head record</h2>
            <span className="text-[11px] text-ink-faint">{h2h.total} career meetings</span>
          </div>
          {h2h.total === 0 ? (
            <p className="text-[13px] text-ink-faint">They haven&apos;t raced the same event yet.</p>
          ) : (
            <>
              <div className="mb-3 flex items-center gap-3">
                <span className={cn("tnum text-2xl font-black", h2h.aWins >= h2h.bWins ? "text-good" : "text-ink-dim")}>
                  {h2h.aWins}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                  <div className="h-full la-gradient" style={{ width: `${pct(h2h.aWins, h2h.total)}%` }} />
                </div>
                <span className={cn("tnum text-2xl font-black", h2h.bWins > h2h.aWins ? "text-good" : "text-ink-dim")}>
                  {h2h.bWins}
                </span>
              </div>
              <ul className="space-y-1">
                {h2h.meetings.map((m, i) => (
                  <li key={i} className="flex items-center gap-2 rounded-lg bg-surface px-2.5 py-2 text-sm">
                    <CalendarDays size={12} className="shrink-0 text-ink-faint" />
                    <span className="min-w-0 flex-1 truncate text-[13px]">{m.eventTitle.replace(/^\d{4}\s+/, "")}</span>
                    <span className="text-[11px] text-ink-faint">{m.date}</span>
                    <span className="tnum shrink-0 text-xs font-bold">
                      <span className={m.winner === "a" ? "text-good" : "text-ink-faint"}>{m.aPos}</span>
                      <span className="text-ink-faint"> · </span>
                      <span className={m.winner === "b" ? "text-good" : "text-ink-faint"}>{m.bPos}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}

      <div className="flex items-center justify-center gap-4 pt-1">
        <RivalButton a={{ athleteId: a.athleteId, name: a.name }} b={{ athleteId: b.athleteId, name: b.name }} />
        <Link href={`/athlete/${a.athleteId}`} className="text-[11px] font-semibold text-electric-bright">
          {a.name.split(" ")[0]}&apos;s cockpit →
        </Link>
      </div>
    </div>
  );
}

function AthleteHead({ athlete, align }: { athlete: VersusAthlete; align: "left" | "right" }) {
  const right = align === "right";
  return (
    <Link
      href={`/athlete/${athlete.athleteId}`}
      className={cn("flex flex-col gap-2 p-4", right ? "items-end text-right" : "items-start")}
    >
      <AthleteAvatar name={athlete.name} src={athlete.image} size={52} ring={athlete.qualified} />
      <div className={cn("min-w-0", right && "text-right")}>
        <div className="truncate text-sm font-extrabold leading-tight">{athlete.name}</div>
        <div className={cn("mt-0.5 flex items-center gap-1 text-[11px] text-ink-faint", right && "justify-end")}>
          <Flag src={athlete.flag} noc={athlete.noc} size={12} /> {athlete.noc}
        </div>
      </div>
      <div className={cn("tnum", right && "text-right")}>
        <span className="text-lg font-black">{athlete.oqrRank ? `#${athlete.oqrRank}` : "—"}</span>
        <span className="ml-1 text-[11px] text-ink-faint">{athlete.points != null ? `${fmtPoints(athlete.points)} pt` : "unranked"}</span>
      </div>
    </Link>
  );
}

function CompareRow({
  label,
  av,
  bv,
  winner,
}: {
  label: string;
  av: React.ReactNode;
  bv: React.ReactNode;
  winner: Side | null;
}) {
  return (
    <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 px-3 py-2.5 text-sm">
      <span className={cn("tnum text-left font-bold", winner === "a" ? "text-good" : "text-ink-dim")}>{av ?? "—"}</span>
      <span className="text-center text-[10px] uppercase tracking-wide text-ink-faint">{label}</span>
      <span className={cn("tnum text-right font-bold", winner === "b" ? "text-good" : "text-ink-dim")}>{bv ?? "—"}</span>
    </div>
  );
}

// ---- formatting + comparison ----
function fmtRank(n: number | null) {
  return n != null ? `#${n}` : "—";
}
function fmtGap(n: number | null) {
  if (n == null) return "—";
  return n <= 0 ? `+${fmtPoints(-n)}` : `-${fmtPoints(n)}`;
}
function betterLow(a?: number | null, b?: number | null): Side | null {
  if (a == null && b == null) return null;
  if (a == null) return "b";
  if (b == null) return "a";
  return a < b ? "a" : b < a ? "b" : null;
}
function betterHigh(a?: number | null, b?: number | null): Side | null {
  if (a == null && b == null) return null;
  if (a == null) return "b";
  if (b == null) return "a";
  return a > b ? "a" : b > a ? "b" : null;
}
function pct(n: number, total: number) {
  return total ? Math.round((n / total) * 100) : 50;
}
