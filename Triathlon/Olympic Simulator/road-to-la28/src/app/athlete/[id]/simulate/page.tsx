import Link from "next/link";
import { notFound } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { findAthlete, getQualState } from "@/lib/data";
import { Simulator } from "@/components/simulator";
import { SimulatorTour } from "@/components/tour/simulator-tour";

export default async function SimulatePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const athleteId = Number(id);
  const found = await findAthlete(athleteId);
  if (!found) notFound();

  const state = await getQualState(found.athlete.gender);

  return (
    <main className="px-4 pt-6">
      <div className="mb-4 flex items-center gap-2">
        <Link
          href={`/athlete/${athleteId}`}
          className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/5 text-ink-dim"
        >
          <ChevronLeft size={18} />
        </Link>
        <div className="flex-1">
          <h1 className="text-lg font-extrabold leading-tight">What-if simulator</h1>
          <p className="text-xs text-ink-faint">{found.athlete.fullName} · {found.athlete.noc}</p>
        </div>
        <SimulatorTour />
      </div>

      <p className="mb-4 text-sm text-ink-dim">
        Drag the finish position and watch the Olympic ranking re-sort live — every
        number is the real qualification math, computed on your device.
      </p>

      <Simulator state={state} athleteId={athleteId} athleteName={found.athlete.fullName} />
    </main>
  );
}
