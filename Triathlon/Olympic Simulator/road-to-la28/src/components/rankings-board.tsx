"use client";
import { Fragment, useState } from "react";
import Link from "next/link";
import { AthleteAvatar } from "./athlete-avatar";
import { Flag } from "./flag";
import type { RankingRow } from "@/lib/cockpit";
import { useMyAthlete } from "@/lib/local/use-my-athlete";
import { fmtPoints } from "@/lib/format";
import { cn } from "@/lib/utils";

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
  const rows = gender === "male" ? men : women;
  const cutRank = gender === "male" ? cut.male : cut.female;

  return (
    <div>
      <div className="mb-3 flex rounded-xl border border-white/10 bg-white/5 p-0.5 text-sm font-semibold">
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

      <ul className="space-y-1">
        {rows.map((r) => {
          const isMine = mine?.athleteId === r.athleteId;
          return (
            <Fragment key={r.athleteId}>
              <li>
                <Link
                  href={`/athlete/${r.athleteId}`}
                  className={cn(
                    "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition",
                    isMine ? "bg-electric/15 ring-1 ring-electric/40" : "bg-white/[0.03] hover:bg-white/[0.06]",
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
                  <span className="absolute -top-1 left-1/2 -translate-x-1/2 rounded-full bg-navy-950 px-2 text-[9px] font-bold uppercase tracking-wider text-la-gold">
                    Qualification line · 21 individual places
                  </span>
                </li>
              )}
            </Fragment>
          );
        })}
      </ul>
    </div>
  );
}
