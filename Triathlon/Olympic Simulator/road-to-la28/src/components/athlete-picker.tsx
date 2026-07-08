"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Search, Star, Loader2, Globe } from "lucide-react";
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

interface SearchHit {
  athleteId: number;
  fullName: string;
  noc: string;
  gender: "male" | "female";
  rank: number | null;
  profileImage?: string;
}

export function AthletePicker({ directory }: { directory: DirEntry[] }) {
  const router = useRouter();
  const { athlete: mine, choose } = useMyAthlete();
  const [q, setQ] = useState("");
  const [gender, setGender] = useState<"male" | "female">("male");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const seq = useRef(0);

  const searching = q.trim().length >= 2;

  // Browse mode (short/empty query): the local OQR directory, gender-filtered.
  const browse = useMemo(() => {
    return directory
      .filter((d) => d.gender === gender)
      .sort((a, b) => a.rank - b.rank)
      .slice(0, 50);
  }, [gender, directory]);

  // Search mode (≥2 chars): debounced live search across the whole WT database.
  useEffect(() => {
    if (!searching) {
      setHits(null);
      setLoading(false);
      return;
    }
    const id = ++seq.current;
    setLoading(true);
    const t = setTimeout(async () => {
      try {
        const res = await fetch(`/api/athletes/search?q=${encodeURIComponent(q.trim())}`, { cache: "no-store" });
        const data = await res.json();
        if (id === seq.current) setHits(data.results ?? []);
      } catch {
        if (id === seq.current) setHits([]);
      } finally {
        if (id === seq.current) setLoading(false);
      }
    }, 280);
    return () => clearTimeout(t);
  }, [q, searching]);

  async function pick(a: { athleteId: number; fullName: string; noc: string; gender: "male" | "female" }) {
    await choose({ athleteId: a.athleteId, fullName: a.fullName, noc: a.noc, gender: a.gender });
    router.push(`/athlete/${a.athleteId}`);
  }

  const list: SearchHit[] = searching
    ? hits ?? []
    : browse.map((d) => ({ ...d, rank: d.rank }));

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2.5">
          {loading ? (
            <Loader2 size={16} className="animate-spin text-electric-bright" />
          ) : (
            <Search size={16} className="text-ink-faint" />
          )}
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search any athlete or nation…"
            className="w-full bg-transparent text-sm outline-none placeholder:text-ink-faint"
          />
        </div>
        {!searching && (
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
        )}
      </div>

      {searching && (
        <div className="flex items-center gap-1.5 px-1 text-[11px] text-ink-faint">
          <Globe size={12} /> Searching the full World Triathlon database
        </div>
      )}

      <ul className="space-y-1.5">
        {list.map((d) => {
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
                <span className="tnum w-7 shrink-0 text-center text-sm text-ink-faint">
                  {d.rank ? `#${d.rank}` : "–"}
                </span>
                <AthleteAvatar name={d.fullName} src={d.profileImage} size={36} ring={isMine} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-semibold">{d.fullName}</span>
                  <span className="text-xs text-ink-faint">
                    {d.noc}
                    {searching && (
                      <span className="ml-1.5 text-ink-faint">· {d.gender === "male" ? "M" : "W"}</span>
                    )}
                    {searching && !d.rank && <span className="ml-1.5 text-ink-faint">· unranked</span>}
                  </span>
                </span>
                {isMine && <Star size={16} className="text-la-gold" fill="currentColor" />}
              </button>
            </li>
          );
        })}
        {searching && !loading && list.length === 0 && (
          <li className="py-8 text-center text-sm text-ink-faint">
            No athletes match &ldquo;{q.trim()}&rdquo;.
          </li>
        )}
      </ul>
    </div>
  );
}
