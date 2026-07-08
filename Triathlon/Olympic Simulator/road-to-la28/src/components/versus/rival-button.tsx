"use client";
import { useEffect, useState } from "react";
import { Swords, Check } from "lucide-react";
import { toggleRivalry, isRivalry, type RivalRef } from "@/lib/local/athlete-store";
import { cn } from "@/lib/utils";

/** Pin/unpin this head-to-head as a persistent rivalry (stored on-device). */
export function RivalButton({ a, b }: { a: RivalRef; b: RivalRef }) {
  const [pinned, setPinned] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    isRivalry(a.athleteId, b.athleteId).then((v) => {
      setPinned(v);
      setReady(true);
    });
  }, [a.athleteId, b.athleteId]);

  async function toggle() {
    const next = await toggleRivalry({ a, b });
    setPinned(next.some((r) => [r.a.athleteId, r.b.athleteId].includes(a.athleteId) && [r.a.athleteId, r.b.athleteId].includes(b.athleteId)));
  }

  if (!ready) return <span className="inline-block h-7 w-28" aria-hidden />;

  return (
    <button
      onClick={toggle}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-bold transition",
        pinned ? "bg-la-gold/15 text-la-gold" : "la-gradient text-navy-950",
      )}
    >
      {pinned ? <Check size={13} /> : <Swords size={13} />}
      {pinned ? "Rivalry pinned" : "Pin this rivalry"}
    </button>
  );
}
