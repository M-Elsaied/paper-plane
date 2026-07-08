"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { Swords, ChevronRight } from "lucide-react";
import { getRivalries, type Rivalry } from "@/lib/local/athlete-store";

/** Shows the pinned rivalries involving this athlete (on their cockpit). */
export function RivalriesStrip({ athleteId }: { athleteId: number }) {
  const [rivalries, setRivalries] = useState<Rivalry[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    getRivalries().then((all) => {
      setRivalries(all.filter((r) => r.a.athleteId === athleteId || r.b.athleteId === athleteId));
      setReady(true);
    });
  }, [athleteId]);

  if (!ready || rivalries.length === 0) return null;

  return (
    <section className="card mb-4 p-4">
      <div className="mb-2 flex items-center gap-1.5">
        <Swords size={15} className="text-la-gold" />
        <h2 className="text-sm font-bold">Your rivalries</h2>
      </div>
      <ul className="space-y-1">
        {rivalries.map((r) => {
          const opp = r.a.athleteId === athleteId ? r.b : r.a;
          return (
            <li key={`${r.a.athleteId}-${r.b.athleteId}`}>
              <Link
                href={`/versus/${athleteId}/${opp.athleteId}`}
                className="flex items-center gap-2 rounded-lg bg-surface px-3 py-2 text-sm transition hover:bg-surface-2"
              >
                <span className="text-[11px] font-bold uppercase tracking-wide text-ink-faint">vs</span>
                <span className="flex-1 truncate font-semibold">{opp.name}</span>
                <ChevronRight size={15} className="text-ink-faint" />
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
