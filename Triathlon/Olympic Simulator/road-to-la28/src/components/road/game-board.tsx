import Link from "next/link";
import type { Road } from "@/lib/engine/road";
import { cn } from "@/lib/utils";

/**
 * The competitive "game board" for the athlete's primary route: the specific
 * players competing for the same slots, with the subject highlighted and the
 * route's threshold marked — the game-theory picture at a glance.
 */
export function GameBoard({ road }: { road: Road }) {
  const p = road.primary;
  if (!p) return null;

  // Split competitors into those ahead of the subject and those behind.
  const ahead = p.competitors.filter((c) => c.ahead);
  const behind = p.competitors.filter((c) => !c.ahead);

  const threshold =
    p.key === "individual"
      ? `${road.subject.oqrRank ? "" : "the "}qualification line`
      : p.key === "relay" || p.key === "host"
        ? "the 2 relay places"
        : p.key.startsWith("newflag")
          ? "1 place for the continent"
          : "the invitation";

  return (
    <section className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold">The game board</h2>
        <span className="text-[11px] text-ink-faint">{p.label} · {threshold}</span>
      </div>

      <ul className="space-y-1">
        {ahead.map((c, i) => (
          <BoardRow key={`a-${i}`} name={c.name} noc={c.noc} athleteId={c.athleteId} subjectId={road.subject.athleteId} tone="ahead" note="ahead of you" />
        ))}

        {ahead.length > 0 && (
          <li aria-hidden className="relative py-1">
            <div className="qual-line" />
            <span className="absolute -top-0.5 right-2 text-[9px] font-bold uppercase tracking-wider text-la-gold">
              {threshold}
            </span>
          </li>
        )}

        <BoardRow
          name={road.subject.name}
          noc={road.subject.noc}
          athleteId={road.subject.athleteId}
          tone="you"
          note={road.subject.oqrRank ? `#${road.subject.oqrRank} OQR` : "unranked"}
        />

        {behind.map((c, i) => (
          <BoardRow key={`b-${i}`} name={c.name} noc={c.noc} athleteId={c.athleteId} subjectId={road.subject.athleteId} tone="behind" note="chasing you" />
        ))}
      </ul>

      <p className="mt-3 text-[11px] leading-snug text-ink-faint">
        These are the athletes contesting the same slots on your most realistic route — not the whole
        field. Move up this board and the route opens.
      </p>
    </section>
  );
}

function BoardRow({
  name,
  noc,
  athleteId,
  subjectId,
  tone,
  note,
}: {
  name: string;
  noc: string;
  athleteId?: number;
  subjectId?: number;
  tone: "ahead" | "you" | "behind";
  note: string;
}) {
  const inner = (
    <div
      className={cn(
        "flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm",
        tone === "you" ? "bg-electric/15 ring-1 ring-electric/50" : "bg-surface",
      )}
    >
      <span className={cn("min-w-0 flex-1 truncate", tone === "you" && "font-bold")}>{name}</span>
      <span className="text-[11px] text-ink-faint">{noc}</span>
      <span
        className={cn(
          "shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-semibold",
          tone === "you" ? "bg-electric/20 text-electric-bright" : tone === "ahead" ? "bg-bad/10 text-bad" : "bg-good/10 text-good",
        )}
      >
        {note}
      </span>
    </div>
  );
  // Competitor rows link to the head-to-head vs the subject; the "you" row to the cockpit.
  const href =
    tone !== "you" && athleteId && subjectId
      ? `/versus/${subjectId}/${athleteId}`
      : athleteId
        ? `/athlete/${athleteId}`
        : null;
  return <li>{href ? <Link href={href}>{inner}</Link> : inner}</li>;
}
