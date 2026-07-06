"use client";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Search, Star } from "lucide-react";
import { AthleteAvatar } from "./athlete-avatar";
import { useMyAthlete } from "@/lib/local/use-my-athlete";
import { cn } from "@/lib/utils";

export interface DirEntry {
  athleteId: number;
  fullName: string;
  noc: string;
  gender: "male" | "female";
  rank: number;
  flag?: string;
  profileImage?: string;
}

export function AthletePicker({ directory }: { directory: DirEntry[] }) {
  const router = useRouter();
  const { athlete: mine, choose } = useMyAthlete();
  const [q, setQ] = useState("");
  const [gender, setGender] = useState<"male" | "female">("male");

  const results = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return directory
      .filter((d) => d.gender === gender)
      .filter(
        (d) =>
          !needle ||
          d.fullName.toLowerCase().includes(needle) ||
          d.noc.toLowerCase().includes(needle),
      )
      .sort((a, b) => a.rank - b.rank)
      .slice(0, 40);
  }, [q, gender, directory]);

  async function pick(d: DirEntry) {
    await choose({ athleteId: d.athleteId, fullName: d.fullName, noc: d.noc, gender: d.gender });
    router.push(`/athlete/${d.athleteId}`);
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2.5">
          <Search size={16} className="text-ink-faint" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search athletes or nations…"
            className="w-full bg-transparent text-sm outline-none placeholder:text-ink-faint"
          />
        </div>
        <div className="flex rounded-xl border border-white/10 bg-white/5 p-0.5 text-xs font-semibold">
          {(["male", "female"] as const).map((g) => (
            <button
              key={g}
              onClick={() => setGender(g)}
              className={cn(
                "rounded-lg px-2.5 py-2 transition",
                gender === g ? "la-gradient text-navy-950" : "text-ink-dim",
              )}
            >
              {g === "male" ? "Men" : "Women"}
            </button>
          ))}
        </div>
      </div>

      <ul className="space-y-1.5">
        {results.map((d) => {
          const isMine = mine?.athleteId === d.athleteId;
          return (
            <li key={d.athleteId}>
              <button
                onClick={() => pick(d)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-xl border px-3 py-2 text-left transition",
                  isMine
                    ? "border-electric/50 bg-electric/10"
                    : "border-white/5 bg-white/[0.03] hover:bg-white/[0.06]",
                )}
              >
                <span className="tnum w-7 text-center text-sm text-ink-faint">{d.rank}</span>
                <AthleteAvatar name={d.fullName} src={d.profileImage} size={36} ring={isMine} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-semibold">{d.fullName}</span>
                  <span className="text-xs text-ink-faint">{d.noc}</span>
                </span>
                {isMine && <Star size={16} className="text-la-gold" fill="currentColor" />}
              </button>
            </li>
          );
        })}
        {results.length === 0 && (
          <li className="py-8 text-center text-sm text-ink-faint">No athletes found.</li>
        )}
      </ul>
    </div>
  );
}
