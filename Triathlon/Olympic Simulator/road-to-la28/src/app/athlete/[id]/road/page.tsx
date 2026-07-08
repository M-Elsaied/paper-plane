import Link from "next/link";
import { notFound } from "next/navigation";
import { ChevronLeft, Compass, Flag } from "lucide-react";
import { buildRoad } from "@/lib/road-context";
import { AthleteAvatar } from "@/components/athlete-avatar";
import { Flag as CountryFlag } from "@/components/flag";
import { RouteCard } from "@/components/road/route-card";
import { GameBoard } from "@/components/road/game-board";
import { CONTINENT_LABEL } from "@/config/continents";

export const revalidate = 300;

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const view = await buildRoad(Number(id)).catch(() => null);
  const name = view?.road.subject.name ?? "Athlete";
  return { title: `${name}'s Road to LA28`, description: `Every route ${name} could take to qualify for the LA 2028 Olympics.` };
}

export default async function RoadPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const view = await buildRoad(Number(id)).catch(() => null);
  if (!view) notFound();
  const { road, display } = view;
  const s = road.subject;

  return (
    <main className="mx-auto px-4 pt-6 lg:max-w-2xl">
      <div className="mb-4 flex items-center gap-2">
        <Link
          href={`/athlete/${s.athleteId}`}
          className="flex h-9 w-9 items-center justify-center rounded-full border border-hairline bg-surface text-ink-dim"
        >
          <ChevronLeft size={18} />
        </Link>
        <div className="flex items-center gap-1.5">
          <Compass size={18} className="text-electric-bright" />
          <h1 className="text-lg font-extrabold leading-tight">Your Road to LA28</h1>
        </div>
      </div>

      {/* Subject + verdict */}
      <section className="card mb-4 p-4">
        <div className="flex items-center gap-3">
          <AthleteAvatar name={s.name} src={display.profileImage} size={52} />
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-lg font-extrabold leading-tight">{s.name}</h2>
            <div className="flex items-center gap-2 text-sm text-ink-dim">
              <span className="inline-flex items-center gap-1.5 font-semibold">
                <CountryFlag src={display.flag} noc={s.noc} size={18} /> {s.noc}
              </span>
              {s.continent && (
                <>
                  <span className="text-ink-faint">·</span>
                  <span className="flex items-center gap-1 text-ink-faint">
                    <Flag size={11} /> {CONTINENT_LABEL[s.continent]}
                  </span>
                </>
              )}
              {s.oqrRank && (
                <>
                  <span className="text-ink-faint">·</span>
                  <span className="tnum">OQR #{s.oqrRank}</span>
                </>
              )}
            </div>
          </div>
        </div>
        <p className="mt-3 text-[13px] font-semibold la-gradient-text">{road.verdict}</p>
      </section>

      {/* Game board for the primary route */}
      {road.primary && road.primary.competitors.length > 0 && (
        <div className="mb-4">
          <GameBoard road={road} />
        </div>
      )}

      {/* The route portfolio */}
      <div className="mb-2 flex items-center justify-between px-1">
        <h2 className="text-sm font-bold">Every route, ranked by realism</h2>
        <span className="text-[11px] text-ink-faint">{road.routes.length} routes</span>
      </div>
      <div className="space-y-2.5">
        {road.routes.map((r, i) => (
          <RouteCard key={r.key} route={r} primary={i === 0} />
        ))}
      </div>

      {/* Unranked: recent results for context */}
      {view.results && view.results.length > 0 && (
        <section className="mt-4">
          <h2 className="mb-2 text-sm font-bold">Recent results</h2>
          <ul className="space-y-1">
            {view.results.map((r, i) => (
              <li key={`${r.eventId}-${i}`} className="flex items-center gap-2 rounded-lg bg-surface px-3 py-2 text-sm">
                <span className="min-w-0 flex-1 truncate">{r.eventTitle.replace(/^\d{4}\s+/, "")}</span>
                <span className="text-[11px] text-ink-faint">{r.date}</span>
                <span className="tnum shrink-0 rounded-md bg-surface-2 px-2 py-0.5 text-xs font-bold text-ink-dim">
                  {r.position ?? "—"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <p className="pt-4 text-center text-[11px] leading-snug text-ink-faint">
        Routes are assessed from the live rankings and the LA28 criteria — a strategic guide, not a
        guarantee. New Flag and relay outcomes depend on events still to come.
      </p>
    </main>
  );
}
