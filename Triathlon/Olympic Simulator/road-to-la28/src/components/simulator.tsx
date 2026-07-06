"use client";
import { Fragment, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowUp, ArrowDown, Sparkles } from "lucide-react";
import { applyWhatIf } from "@/lib/engine/what-if";
import { rankAthletes, computeQualificationLine } from "@/lib/engine/qualification";
import type { QualState } from "@/lib/engine/types";
import { TIER_BASE_POINTS, TIER_LABEL, type PointsTier } from "@/config/points-tables";
import { fmtPoints, fmtDelta, ordinal } from "@/lib/format";
import { cn } from "@/lib/utils";

const TIERS: PointsTier[] = ["wtcs_final", "wtcs", "world_cup", "continental_champs", "continental_cup"];

export function Simulator({
  state,
  athleteId,
  athleteName,
}: {
  state: QualState;
  athleteId: number;
  athleteName: string;
}) {
  const [position, setPosition] = useState(5);
  const [tier, setTier] = useState<PointsTier>("wtcs");
  const [period, setPeriod] = useState<1 | 2>(1);

  const result = useMemo(
    () => applyWhatIf(state, { athleteId, tier, position, period }),
    [state, athleteId, tier, position, period],
  );

  // Build the post-scenario ranking window around the athlete for the live board.
  const board = useMemo(() => {
    const ranked = rankAthletes(
      state.athletes.map((a) =>
        a.athleteId === athleteId
          ? { ...a, scores: [...a.scores, { points: result.hypotheticalPoints, period }] }
          : a,
      ),
    );
    const line = computeQualificationLine(
      state.athletes.map((a) =>
        a.athleteId === athleteId
          ? { ...a, scores: [...a.scores, { points: result.hypotheticalPoints, period }] }
          : a,
      ),
    );
    const cut = line.cutRank ?? 21;
    const meIdx = ranked.findIndex((a) => a.athleteId === athleteId);
    const start = Math.max(0, Math.min(meIdx - 4, cut - 3));
    return ranked.slice(start, start + 9).map((a, i) => ({
      athleteId: a.athleteId,
      rank: start + i + 1,
      fullName: a.fullName,
      noc: a.noc,
      total: a.total,
      isMe: a.athleteId === athleteId,
      qualified: start + i + 1 <= cut,
      cut,
    }));
  }, [state, athleteId, result.hypotheticalPoints, period]);

  const cutRank = board[0]?.cut ?? 21;

  return (
    <div className="space-y-4">
      {/* Outcome banner */}
      <motion.div
        layout
        className={cn(
          "card overflow-hidden p-4",
          result.crossesLine && "border-good/50",
          result.dropsOut && "border-bad/50",
        )}
      >
        <div className="flex items-center justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-ink-faint">
              Projected rank
            </div>
            <div className="tnum flex items-baseline gap-2">
              <span className="text-ink-faint text-2xl">#{result.before.rank}</span>
              <span className="text-ink-faint">→</span>
              <motion.span
                key={result.after.rank}
                initial={{ y: 8, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="text-5xl font-black"
              >
                #{result.after.rank}
              </motion.span>
            </div>
          </div>
          <DeltaChip rankDelta={result.rankDelta} pointsDelta={result.pointsDelta} />
        </div>

        <AnimatePresence mode="wait">
          {result.crossesLine && (
            <Flash key="in" tone="good" text="Crosses into the qualifying zone" icon={<ArrowUp size={15} />} />
          )}
          {result.dropsOut && (
            <Flash key="out" tone="bad" text="Drops out of the qualifying zone" icon={<ArrowDown size={15} />} />
          )}
        </AnimatePresence>
      </motion.div>

      {/* Controls */}
      <div className="card space-y-4 p-4">
        <div>
          <div className="mb-2 flex items-center justify-between">
            <label className="text-sm font-bold">Finish position</label>
            <span className="tnum rounded-lg bg-white/10 px-2.5 py-1 text-lg font-extrabold la-gradient-text">
              {ordinal(position)}
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={40}
            value={position}
            onChange={(e) => setPosition(Number(e.target.value))}
            className="w-full accent-[var(--color-la-violet)]"
          />
          <div className="mt-1 flex justify-between text-[10px] text-ink-faint">
            <span>Win</span>
            <span>+{fmtPoints(result.hypotheticalPoints)} pts</span>
            <span>40th</span>
          </div>
        </div>

        <div>
          <label className="mb-2 block text-sm font-bold">Race tier</label>
          <div className="flex flex-wrap gap-1.5">
            {TIERS.map((t) => (
              <button
                key={t}
                onClick={() => setTier(t)}
                className={cn(
                  "rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition",
                  tier === t
                    ? "border-transparent la-gradient text-navy-950"
                    : "border-white/10 bg-white/5 text-ink-dim",
                )}
              >
                {TIER_LABEL[t]}
                <span className="ml-1 text-[10px] opacity-70">{TIER_BASE_POINTS[t]}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="mb-2 block text-sm font-bold">Scoring period</label>
          <div className="flex gap-1.5">
            {([1, 2] as const).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={cn(
                  "flex-1 rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition",
                  period === p
                    ? "border-electric/50 bg-electric/15 text-electric-bright"
                    : "border-white/10 bg-white/5 text-ink-dim",
                )}
              >
                Period {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Live re-sorting board */}
      <div className="card p-2">
        <div className="px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
          Live ranking · qualification line at #{cutRank}
        </div>
        <motion.ul layout className="relative space-y-1">
          {board.map((row) => (
            <Fragment key={row.athleteId}>
              <motion.li
                layout
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm",
                  row.isMe ? "bg-electric/15 ring-1 ring-electric/50" : "bg-white/[0.03]",
                )}
              >
                <span className={cn("tnum w-7 text-center font-bold", row.qualified ? "text-good" : "text-ink-faint")}>
                  {row.rank}
                </span>
                <span className={cn("flex-1 truncate", row.isMe && "font-bold")}>
                  {row.isMe ? athleteName : row.fullName}
                </span>
                <span className="text-[11px] text-ink-faint">{row.noc}</span>
                <span className="tnum w-14 text-right font-semibold">{fmtPoints(row.total)}</span>
              </motion.li>
              {row.rank === cutRank && (
                <li aria-hidden className="relative py-1">
                  <div className="qual-line" />
                  <span className="absolute -top-0.5 right-2 text-[9px] font-bold uppercase tracking-wider text-la-gold">
                    Qualification line
                  </span>
                </li>
              )}
            </Fragment>
          ))}
        </motion.ul>
      </div>
    </div>
  );
}

function DeltaChip({ rankDelta, pointsDelta }: { rankDelta: number; pointsDelta: number }) {
  const up = rankDelta > 0;
  return (
    <div
      className={cn(
        "rounded-xl px-3 py-2 text-right",
        rankDelta === 0 ? "bg-white/5" : up ? "bg-good/15" : "bg-bad/15",
      )}
    >
      <div className={cn("tnum text-xl font-black", rankDelta === 0 ? "text-ink-dim" : up ? "text-good" : "text-bad")}>
        {rankDelta === 0 ? "—" : `${up ? "↑" : "↓"}${Math.abs(rankDelta)}`}
      </div>
      <div className="tnum text-[11px] font-semibold text-ink-dim">{fmtDelta(pointsDelta)} pts</div>
    </div>
  );
}

function Flash({ tone, text, icon }: { tone: "good" | "bad"; text: string; icon: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      className={cn(
        "mt-3 flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-bold",
        tone === "good" ? "bg-good/15 text-good" : "bg-bad/15 text-bad",
      )}
    >
      {icon}
      {text}
      <Sparkles size={14} className="ml-auto opacity-70" />
    </motion.div>
  );
}
