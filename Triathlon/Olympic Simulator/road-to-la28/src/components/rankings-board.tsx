"use client";
import { Fragment, useState } from "react";
import Link from "next/link";
import { AthleteAvatar } from "./athlete-avatar";
import { Flag } from "./flag";
import type { RankingRow } from "@/lib/cockpit";
import { useMyAthlete } from "@/lib/local/use-my-athlete";
import { fmtPoints } from "@/lib/format";
import { cn } from "@/lib/utils";

/** The row list for one gender (shared by the mobile toggle + desktop columns). */
function RankingList({
  rows,
  cutRank,
  mineId,
}: {
  rows: RankingRow[];
  cutRank: number | null;
  mineId?: number;
}) {
  return (
    <ul className="space-y-1">
      {rows.map((r) => {
        const isMine = mineId === r.athleteId;
        return (
          <Fragment key={r.athleteId}>
            <li>
              <Link
                href={`/athlete/${r.athleteId}`}
                className={cn(
                  "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition",
                  isMine ? "bg-electric/15 ring-1 ring-electric/40" : "bg-surface hover:bg-surface-2",
                )}
              >
                <span className={cn("tnum w-7 text-center font-bold", r.qualified ? "text-good" : "text-ink-faint")}>
                  {r.rank}
                </span>
                <AthleteAvatar name={r.fullName} src={r.profileImage} size={30} ring={isMine} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-semibold">{r.fullName}</span>
                  <span className="flex items-center gap-1 text-[11px] text-ink-faint">
                    <Flag src={r.flag} noc={r.noc} size={12} /> {r.noc}
                  </span>
                </span>
                <span className="tnum font-semibold">{fmtPoints(r.total)}</span>
              </Link>
            </li>
            {cutRank != null && r.rank === cutRank && (
              <li aria-hidden className="relative py-2">
                <div className="qual-line" />
                <span className="absolute -top-1 left-1/2 -translate-x-1/2 rounded-full bg-elevated px-2 text-[9px] font-bold uppercase tracking-wider text-la-gold">
                  Qualification line · 21 places
                </span>
              </li>
            )}
          </Fragment>
        );
      })}
    </ul>
  );
}

export function RankingsBoard({
  men,
  women,
  cut,
}: {
  men: RankingRow[];
  women: RankingRow[];
  cut: { male: number | null; female: number | null };
}) {
  const [gender, setGender] = useState<"male" | "female">("male");
  const { athlete: mine } = useMyAthlete();

  return (
    <>
      {/* Mobile / tablet: a gender toggle + single list. */}
      <div data-testid="rankings-mobile" className="lg:hidden">
        <div className="mb-3 flex rounded-xl border border-hairline bg-surface p-0.5 text-sm font-semibold">
          {(["male", "female"] as const).map((g) => (
            <button
              key={g}
              onClick={() => setGender(g)}
              className={cn(
                "flex-1 rounded-lg py-2 transition",
                gender === g ? "la-gradient text-navy-950" : "text-ink-dim",
              )}
            >
              {g === "male" ? "Elite Men" : "Elite Women"}
            </button>
          ))}
        </div>
        <RankingList rows={gender === "male" ? men : women} cutRank={gender === "male" ? cut.male : cut.female} mineId={mine?.athleteId} />
      </div>

      {/* Desktop: both genders side by side. */}
      <div className="hidden gap-8 lg:grid lg:grid-cols-2">
        <div>
          <h2 className="mb-2 text-sm font-bold text-ink-dim">Elite Men</h2>
          <RankingList rows={men} cutRank={cut.male} mineId={mine?.athleteId} />
        </div>
        <div>
          <h2 className="mb-2 text-sm font-bold text-ink-dim">Elite Women</h2>
          <RankingList rows={women} cutRank={cut.female} mineId={mine?.athleteId} />
        </div>
      </div>
    </>
  );
}
