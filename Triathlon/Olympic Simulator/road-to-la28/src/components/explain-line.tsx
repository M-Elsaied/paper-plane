import { Info, ChevronDown } from "lucide-react";
import type { ExplainLine } from "@/lib/explain";
import { PATHWAYS } from "@/config/pathways";
import { cn } from "@/lib/utils";

const TONE: Record<ExplainLine["tone"], string> = {
  good: "border-l-good",
  electric: "border-l-electric",
  warn: "border-l-la-gold",
  muted: "border-l-white/20",
};

/** Plain-language "why are they here" — the deterministic engine in words. */
export function ExplainLineCard({ lines }: { lines: ExplainLine[] }) {
  return (
    <section className="card p-4">
      <div className="mb-3 flex items-center gap-2">
        <Info size={15} className="text-electric-bright" />
        <h2 className="text-sm font-bold">What the line means for them</h2>
      </div>

      <div className="space-y-2.5">
        {lines.map((l, i) => (
          <p key={i} className={cn("border-l-2 pl-3 text-[13px] leading-snug text-ink-dim", TONE[l.tone])}>
            {l.text}
          </p>
        ))}
      </div>

      <details className="group mt-3">
        <summary className="flex cursor-pointer list-none items-center gap-1 text-[11px] font-semibold text-ink-faint">
          <ChevronDown size={13} className="transition-transform group-open:rotate-180" />
          The {PATHWAYS.length} routes into the Games · 55 places per gender
        </summary>
        <ul className="mt-2 space-y-1 pl-1">
          {PATHWAYS.map((p) => (
            <li key={p.key} className="flex items-center justify-between text-[11px] text-ink-dim">
              <span>{p.label}</span>
              <span className="tnum font-semibold text-ink-faint">{p.places}</span>
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
