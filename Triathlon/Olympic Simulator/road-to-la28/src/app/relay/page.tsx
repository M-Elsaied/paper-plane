import { Fragment } from "react";
import { getMrNations } from "@/lib/data";
import { MR_OQR_PLACES } from "@/lib/engine/mixed-relay";
import { fmtPoints } from "@/lib/format";
import { cn } from "@/lib/utils";

export default function RelayPage() {
  const nations = [...getMrNations()].sort((a, b) => a.rank - b.rank);

  return (
    <main className="px-4 pt-6">
      <header className="mb-4">
        <h1 className="text-2xl font-extrabold">Mixed Relay pathway</h1>
        <p className="text-sm text-ink-faint">
          Top {MR_OQR_PLACES} nations on the Mixed Relay Olympic Qualification Ranking earn relay
          places. Tracked, not simulated in v1.
        </p>
      </header>

      <ul className="space-y-1">
        {nations.map((n) => (
          <Fragment key={n.noc}>
            <li
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm",
                n.rank <= MR_OQR_PLACES ? "bg-good/10" : "bg-white/[0.03]",
              )}
            >
              <span className={cn("tnum w-7 text-center font-bold", n.rank <= MR_OQR_PLACES ? "text-good" : "text-ink-faint")}>
                {n.rank}
              </span>
              <span className="flex-1 font-semibold">{n.noc}</span>
              <span className="tnum font-semibold">{fmtPoints(n.total)}</span>
            </li>
            {n.rank === MR_OQR_PLACES && (
              <li aria-hidden className="relative py-2">
                <div className="qual-line" />
                <span className="absolute -top-1 left-1/2 -translate-x-1/2 rounded-full bg-navy-950 px-2 text-[9px] font-bold uppercase tracking-wider text-la-gold">
                  Relay qualification line · 16 nations
                </span>
              </li>
            )}
          </Fragment>
        ))}
      </ul>

      <div className="card mt-4 p-4 text-xs text-ink-dim">
        <p className="font-bold text-ink">How relay affects individual odds</p>
        <p className="mt-1">
          Relay places are separate from the 21 individual places, but they count against a
          nation&apos;s 2- or 3-athlete cap — so a strong relay nation can qualify more athletes
          overall. Athletes from relay nations see this context on their cockpit.
        </p>
      </div>
    </main>
  );
}
