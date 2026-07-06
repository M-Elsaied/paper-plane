"use client";
import { Fragment, useMemo, useState } from "react";
import Link from "next/link";
import { Reorder, motion } from "motion/react";
import { GripVertical, RotateCcw, Trophy, Radio, Info } from "lucide-react";
import { projectRanking, pointsForOrder, type FieldAthlete } from "@/lib/engine/projection";
import type { AthleteScores } from "@/lib/engine/types";
import type { PointsTier } from "@/config/points-tables";
import type { PeriodId } from "@/config/qualification";
import type { Gender } from "@/config/pathways";
import { MovementArrow } from "./movement-arrow";
import { fmtPoints, fmtDelta, ordinal } from "@/lib/format";
import { cn } from "@/lib/utils";

export interface RaceModel {
  eventId: number;
  title: string;
  tier: PointsTier;
  tierLabel: string;
  date: string;
  venue?: string;
  gender: Gender;
  period: PeriodId;
  field: FieldAthlete[];
  allAthletes: AthleteScores[];
  initialOrder: number[];
  unrankedEntrants: number;
  officialStartList: boolean;
}

export function LiveRaceCompanion({ model }: { model: RaceModel }) {
  const [order, setOrder] = useState<number[]>(model.initialOrder);
  const fieldById = useMemo(
    () => new Map(model.field.map((f) => [f.athleteId, f])),
    [model.field],
  );

  const proj = useMemo(
    () => projectRanking(model.allAthletes, model.field, order, model.tier, model.period),
    [model.allAthletes, model.field, order, model.tier, model.period],
  );
  const pts = useMemo(() => pointsForOrder(order, model.tier), [order, model.tier]);

  const changed = order.join() !== model.initialOrder.join();
  const cutRank = proj.cutRank ?? 21;

  return (
    <div className="space-y-4">
      {/* Race header */}
      <div className="card p-4">
        <div className="mb-1 flex items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full bg-la-gold/15 px-2 py-0.5 text-[10px] font-bold text-la-gold">
            <Radio size={11} /> {model.tierLabel}
          </span>
          <span className="text-[11px] text-ink-faint">
            {new Date(model.date).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" })}
          </span>
        </div>
        <h1 className="text-lg font-extrabold leading-tight">{model.title}</h1>
        <p className="mt-0.5 text-[11px] text-ink-faint">
          {model.officialStartList
            ? `Official start list · ${model.field.length} ranked contenders${model.unrankedEntrants ? ` (+${model.unrankedEntrants} unranked)` : ""}`
            : `Expected field · top ${model.field.length} in the ranking`}
        </p>
      </div>

      {/* Projected finish — drag to change */}
      <div className="card p-3">
        <div className="mb-2 flex items-center justify-between px-1">
          <h2 className="text-sm font-bold">Projected finish</h2>
          <div className="flex items-center gap-2">
            {changed && (
              <button
                onClick={() => setOrder(model.initialOrder)}
                className="inline-flex items-center gap-1 text-[11px] font-semibold text-ink-faint transition hover:text-ink-dim"
              >
                <RotateCcw size={12} /> Reset
              </button>
            )}
          </div>
        </div>
        <div className="mb-2 flex items-center gap-1 px-1 text-[11px] text-ink-faint">
          <GripVertical size={12} /> Drag athletes to set who finishes where
        </div>

        <Reorder.Group
          axis="y"
          values={order}
          onReorder={setOrder}
          className="max-h-[42vh] space-y-1 overflow-y-auto hide-scrollbar"
        >
          {order.map((id) => {
            const a = fieldById.get(id);
            if (!a) return null;
            const p = pts.get(id);
            return (
              <Reorder.Item
                key={id}
                value={id}
                className="flex touch-none items-center gap-2 rounded-lg border border-white/8 bg-white/[0.04] px-2.5 py-2 text-sm"
              >
                <span
                  className={cn(
                    "tnum w-6 text-center font-bold",
                    p && p.position <= 3 ? "text-la-gold" : "text-ink-faint",
                  )}
                >
                  {p?.position}
                </span>
                <span className="min-w-0 flex-1 truncate font-semibold">{a.fullName}</span>
                <span className="text-[11px] text-ink-faint">{a.noc}</span>
                <span className="tnum w-12 text-right text-xs font-semibold text-la-gold">
                  +{fmtPoints(p?.expectedPoints ?? 0)}
                </span>
                <GripVertical size={14} className="text-ink-faint" />
              </Reorder.Item>
            );
          })}
        </Reorder.Group>
      </div>

      {/* Projected Monday ranking */}
      <div className="card p-2">
        <div className="flex items-center justify-between px-2 py-1.5">
          <h2 className="text-[11px] font-bold uppercase tracking-wide text-ink-dim">
            Projected ranking · Monday
          </h2>
          <span className="text-[11px] text-ink-faint">line at #{cutRank}</span>
        </div>
        <motion.ul layout className="space-y-1">
          {proj.board.map((r) => (
            <Fragment key={r.athleteId}>
              <motion.li
                layout
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm",
                  r.crossesLine
                    ? "bg-good/15 ring-1 ring-good/50"
                    : r.dropsOut
                      ? "bg-bad/10 ring-1 ring-bad/40"
                      : r.isRacer
                        ? "bg-white/[0.05]"
                        : "bg-white/[0.02]",
                )}
              >
                <span className={cn("tnum w-7 text-center font-bold", r.qualified ? "text-good" : "text-ink-faint")}>
                  {r.projectedRank}
                </span>
                <span className="w-8">
                  <MovementArrow delta={r.rankDelta} />
                </span>
                <span className={cn("min-w-0 flex-1 truncate", r.isRacer && "font-semibold")}>
                  {r.fullName}
                  {r.isRacer && <Trophy size={11} className="ml-1 inline text-la-gold" />}
                </span>
                <span className="text-[11px] text-ink-faint">{r.noc}</span>
                <span className="tnum w-14 text-right font-semibold">{fmtPoints(r.projectedTotal)}</span>
              </motion.li>
              {r.projectedRank === cutRank && (
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

      <p className="flex items-center justify-center gap-1.5 px-4 text-center text-[11px] text-ink-faint">
        <Info size={12} />
        Projection from current ranking strength + form. Not a prediction — drag to explore.
      </p>

      <div className="text-center">
        <Link href="/race-week" className="text-[11px] font-semibold text-electric-bright">
          ← All races
        </Link>
      </div>
    </div>
  );
}
