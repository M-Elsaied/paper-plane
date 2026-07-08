import Link from "next/link";
import { Check, TrendingUp, Ban, MinusCircle } from "lucide-react";
import type { RouteAssessment, RouteStatus } from "@/lib/engine/road";
import { cn } from "@/lib/utils";

const TONE: Record<RouteStatus, { text: string; bg: string; bar: string; label: string; Icon: typeof Check }> = {
  on_track: { text: "text-good", bg: "bg-good/15", bar: "bg-good", label: "ON TRACK", Icon: Check },
  in_contention: { text: "text-electric-bright", bg: "bg-electric/15", bar: "bg-electric", label: "IN CONTENTION", Icon: TrendingUp },
  stretch: { text: "text-la-gold", bg: "bg-la-gold/15", bar: "bg-la-gold", label: "STRETCH", Icon: TrendingUp },
  locked_out: { text: "text-ink-faint", bg: "bg-surface-2", bar: "bg-surface-2", label: "CLOSED", Icon: Ban },
};

export function RouteCard({ route, primary }: { route: RouteAssessment; primary?: boolean }) {
  const t = TONE[route.status];
  return (
    <section className={cn("card p-4", primary && "ring-1 ring-electric/40")}>
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <h3 className="text-sm font-bold">{route.label}</h3>
        <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold", t.bg, t.text)}>
          <t.Icon size={11} strokeWidth={2.5} /> {t.label}
        </span>
      </div>

      {/* realism bar */}
      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-surface">
        <div className={cn("h-full rounded-full", t.bar)} style={{ width: `${route.realism}%` }} />
      </div>

      <p className={cn("text-[13px] font-semibold", t.text)}>{route.headline}</p>
      <p className="mt-1 text-[12px] leading-snug text-ink-dim">{route.detail}</p>
      <p className="mt-2 text-[11px] leading-snug text-ink-faint">{route.mechanic}</p>

      {route.competitors.length > 0 && (
        <div className="mt-3 border-t border-hairline pt-2.5">
          <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wide text-ink-faint">
            Who you&apos;re up against
          </div>
          <div className="flex flex-wrap gap-1.5">
            {route.competitors.map((c, i) => {
              const chip = (
                <span
                  className={cn(
                    "inline-flex items-center gap-1 rounded-full px-2 py-1 text-[11px]",
                    c.ahead ? "bg-bad/10 text-bad" : "bg-surface-2 text-ink-dim",
                  )}
                >
                  {c.ahead && <MinusCircle size={10} />}
                  <span className="font-semibold">{c.name}</span>
                  <span className="text-ink-faint">{c.noc}</span>
                </span>
              );
              return c.athleteId ? (
                <Link key={i} href={`/athlete/${c.athleteId}`} title={c.note}>
                  {chip}
                </Link>
              ) : (
                <span key={i} title={c.note}>
                  {chip}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
