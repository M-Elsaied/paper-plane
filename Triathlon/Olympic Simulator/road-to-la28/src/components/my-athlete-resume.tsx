"use client";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { useMyAthlete } from "@/lib/local/use-my-athlete";
import { AthleteAvatar } from "./athlete-avatar";

/** Shown at the top of the landing page when a "my athlete" is already stored. */
export function MyAthleteResume() {
  const { athlete, hydrated } = useMyAthlete();
  if (!hydrated || !athlete) return null;

  return (
    <Link
      href={`/athlete/${athlete.athleteId}`}
      className="card flex items-center gap-3 p-3 transition hover:border-electric/40"
    >
      <AthleteAvatar name={athlete.fullName} size={44} ring />
      <div className="min-w-0 flex-1">
        <div className="text-[11px] uppercase tracking-wide text-ink-faint">Continue tracking</div>
        <div className="truncate font-semibold">{athlete.fullName}</div>
      </div>
      <span className="flex items-center gap-1 text-sm font-semibold electric-text">
        Open <ArrowRight size={16} className="text-electric-bright" />
      </span>
    </Link>
  );
}
