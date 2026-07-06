import Link from "next/link";
import { notFound } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { buildRaceCompanion } from "@/lib/race";
import { LiveRaceCompanion } from "@/components/live-race-companion";
import type { Gender } from "@/config/pathways";
import { cn } from "@/lib/utils";

export const revalidate = 300;

export default async function RacePage({
  params,
  searchParams,
}: {
  params: Promise<{ eventId: string }>;
  searchParams: Promise<{ g?: string }>;
}) {
  const { eventId } = await params;
  const { g } = await searchParams;
  const gender: Gender = g === "women" ? "female" : "male";

  const model = await buildRaceCompanion(Number(eventId), gender);
  if (!model) notFound();

  return (
    <main className="px-4 pt-6">
      <div className="mb-4 flex items-center gap-2">
        <Link
          href="/race-week"
          className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/5 text-ink-dim"
        >
          <ChevronLeft size={18} />
        </Link>
        <h1 className="text-lg font-extrabold leading-tight">Live Race Companion</h1>
      </div>

      {/* Gender toggle */}
      <div className="mb-4 flex rounded-xl border border-white/10 bg-white/5 p-0.5 text-sm font-semibold">
        {(
          [
            { key: "men", label: "Elite Men" },
            { key: "women", label: "Elite Women" },
          ] as const
        ).map((t) => {
          const active = (t.key === "women") === (gender === "female");
          return (
            <Link
              key={t.key}
              href={`/race/${eventId}?g=${t.key}`}
              scroll={false}
              className={cn(
                "flex-1 rounded-lg py-2 text-center transition",
                active ? "la-gradient text-navy-950" : "text-ink-dim",
              )}
            >
              {t.label}
            </Link>
          );
        })}
      </div>

      <LiveRaceCompanion model={model} />
    </main>
  );
}
