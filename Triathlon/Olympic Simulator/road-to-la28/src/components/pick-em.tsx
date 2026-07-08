"use client";
import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Trophy, Lock, Check, Users, Sparkles } from "lucide-react";
import { getDeviceId } from "@/lib/local/device";
import { PICKEM } from "@/config/pickem";
import { cn } from "@/lib/utils";

export interface PickAthlete {
  athleteId: number;
  fullName: string;
  noc: string;
}

interface CrowdForecast {
  total: number;
  athletes: { athleteId: number; win: number; podium: number }[];
}
interface MyPick {
  podium: number[];
  score: number | null;
  perfect: boolean | null;
}

const MEDAL = ["🥇", "🥈", "🥉"];

export function PickEm({
  raceId,
  gender,
  field,
}: {
  raceId: number;
  gender: "male" | "female";
  field: PickAthlete[];
}) {
  const nameById = useMemo(() => new Map(field.map((f) => [f.athleteId, f])), [field]);
  const [loaded, setLoaded] = useState(false);
  const [myPick, setMyPick] = useState<MyPick | null>(null);
  const [crowd, setCrowd] = useState<CrowdForecast>({ total: 0, athletes: [] });
  const [draft, setDraft] = useState<number[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const dev = getDeviceId();
    fetch(`/api/picks?raceId=${raceId}&gender=${gender}&deviceId=${dev}`, { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        setMyPick(d.myPick ?? null);
        setCrowd(d.crowd ?? { total: 0, athletes: [] });
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
  }, [raceId, gender]);

  function tap(id: number) {
    setDraft((d) => (d.includes(id) ? d.filter((x) => x !== id) : d.length < 3 ? [...d, id] : d));
  }

  async function lockIn() {
    if (draft.length !== 3) return;
    setSaving(true);
    const dev = getDeviceId();
    const res = await fetch("/api/picks", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ raceId, gender, podium: draft, deviceId: dev }),
    });
    if (res.ok) {
      setMyPick({ podium: draft, score: null, perfect: null });
      const r = await fetch(`/api/picks?raceId=${raceId}&gender=${gender}&deviceId=${dev}`, { cache: "no-store" });
      const d = await r.json();
      setCrowd(d.crowd ?? crowd);
    }
    setSaving(false);
  }

  if (!loaded) return <div className="card h-40 animate-pulse" />;

  const locked = !!myPick;
  const scored = locked && myPick!.score != null;

  return (
    <section className="card overflow-hidden p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="flex items-center gap-1.5 text-sm font-bold">
          <Trophy size={15} className="text-la-gold" /> Call the podium
        </h2>
        {crowd.total > 0 && (
          <span className="flex items-center gap-1 text-[11px] text-ink-faint">
            <Users size={12} /> {crowd.total} {crowd.total === 1 ? "call" : "calls"}
          </span>
        )}
      </div>

      {/* Podium slots */}
      <div className="mb-3 grid grid-cols-3 gap-2">
        {[0, 1, 2].map((slot) => {
          const id = locked ? myPick!.podium[slot] : draft[slot];
          const a = id != null ? nameById.get(id) : undefined;
          return (
            <div
              key={slot}
              className={cn(
                "flex min-h-[64px] flex-col items-center justify-center rounded-xl border px-1 py-2 text-center",
                a ? "border-la-gold/40 bg-la-gold/5" : "border-dashed border-hairline bg-surface",
              )}
            >
              <span className="text-lg leading-none">{MEDAL[slot]}</span>
              {a ? (
                <span className="mt-1 line-clamp-2 text-[11px] font-semibold leading-tight">{a.fullName}</span>
              ) : (
                <span className="mt-1 text-[10px] text-ink-faint">pick</span>
              )}
            </div>
          );
        })}
      </div>

      {scored && (
        <div className="mb-3 flex items-center justify-between rounded-xl bg-good/15 px-3 py-2">
          <span className="text-sm font-bold text-good">You scored</span>
          <span className="tnum text-lg font-black text-good">
            {myPick!.score}
            <span className="text-xs text-ink-dim">/{PICKEM.maxScore}</span>
            {myPick!.perfect && <span className="ml-1">🎯</span>}
          </span>
        </div>
      )}

      {locked && !scored && (
        <div className="mb-3 flex items-center gap-1.5 rounded-lg bg-surface px-3 py-2 text-[12px] text-ink-dim">
          <Lock size={13} className="text-la-gold" /> Locked in — scored automatically after the race.
        </div>
      )}

      {/* Picker (only when not locked) */}
      {!locked && (
        <>
          <button
            onClick={lockIn}
            disabled={draft.length !== 3 || saving}
            className={cn(
              "mb-3 flex w-full items-center justify-center gap-2 rounded-xl py-2.5 text-sm font-bold transition",
              draft.length === 3 ? "la-gradient text-navy-950" : "bg-surface text-ink-faint",
            )}
          >
            <Lock size={15} />
            {saving ? "Locking…" : draft.length === 3 ? "Lock in your call" : `Pick ${3 - draft.length} more`}
          </button>
          <div className="max-h-[38vh] space-y-1 overflow-y-auto hide-scrollbar">
            {field.map((a) => {
              const idx = draft.indexOf(a.athleteId);
              const picked = idx >= 0;
              return (
                <button
                  key={a.athleteId}
                  onClick={() => tap(a.athleteId)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-lg border px-2.5 py-2 text-left text-sm transition",
                    picked ? "border-la-gold/50 bg-la-gold/10" : "border-hairline bg-surface",
                  )}
                >
                  <span className="w-6 text-center">{picked ? MEDAL[idx] : <span className="text-ink-faint">+</span>}</span>
                  <span className="min-w-0 flex-1 truncate font-semibold">{a.fullName}</span>
                  <span className="text-[11px] text-ink-faint">{a.noc}</span>
                </button>
              );
            })}
          </div>
        </>
      )}

      {/* Crowd forecast */}
      <AnimatePresence>
        {crowd.total > 0 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-3 border-t border-hairline pt-3">
            <div className="mb-2 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wide text-ink-faint">
              <Sparkles size={12} /> Who the crowd is calling to win
            </div>
            <div className="space-y-1.5">
              {crowd.athletes
                .filter((c) => c.win > 0)
                .slice(0, 4)
                .map((c) => {
                  const a = nameById.get(c.athleteId);
                  const pct = Math.round((c.win / crowd.total) * 100);
                  return (
                    <div key={c.athleteId} className="flex items-center gap-2">
                      <span className="w-24 shrink-0 truncate text-[12px] font-semibold">
                        {a?.fullName ?? c.athleteId}
                      </span>
                      <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-surface">
                        <div className="h-full rounded-full la-gradient" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="tnum w-9 text-right text-[11px] font-semibold text-ink-dim">{pct}%</span>
                    </div>
                  );
                })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
