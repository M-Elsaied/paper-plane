"use client";
import { useState } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import { AthleteAvatar } from "./athlete-avatar";
import { MovementArrow } from "./movement-arrow";
import { cn } from "@/lib/utils";

export interface Mover {
  athleteId: number;
  fullName: string;
  noc: string;
  change: number;
  rank: number;
  flag?: string;
  profileImage?: string;
}

export function PulseBoard({ men, women }: { men: Mover[]; women: Mover[] }) {
  const [gender, setGender] = useState<"male" | "female">("male");
  const movers = gender === "male" ? men : women;

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

      <ul className="space-y-1.5">
        {movers.map((m, i) => (
          <motion.li
            key={m.athleteId}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
          >
            <Link
              href={`/athlete/${m.athleteId}`}
              className="flex items-center gap-3 rounded-xl bg-white/[0.03] px-3 py-2.5 transition hover:bg-white/[0.06]"
            >
              <AthleteAvatar name={m.fullName} src={m.profileImage} size={36} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-semibold">{m.fullName}</div>
                <div className="text-[11px] text-ink-faint">
                  {m.noc} · now #{m.rank}
                </div>
              </div>
              <div
                className={cn(
                  "flex items-center rounded-lg px-2 py-1 text-sm",
                  m.change > 0 ? "bg-good/15" : "bg-bad/15",
                )}
              >
                <MovementArrow delta={m.change} />
              </div>
            </Link>
          </motion.li>
        ))}
        {movers.length === 0 && (
          <li className="card p-6 text-center text-sm text-ink-faint">
            No movement since the last official ranking. New movers appear after each update.
          </li>
        )}
      </ul>
    </div>
  );
}
