import Link from "next/link";
import { Users, ShieldCheck, Lock, Swords } from "lucide-react";
import type { NocWarRoom, NocGenderSlots } from "@/lib/war-room";
import { Flag } from "@/components/flag";
import { SlotPips } from "./slot-pips";
import { CONTINENT_LABEL } from "@/config/continents";
import { fmtPoints } from "@/lib/format";
import { cn } from "@/lib/utils";

/** A single nation's slot war: both genders' contender ladders + relay pathway. */
export function NocDetail({ n }: { n: NocWarRoom }) {
  return (
    <div className="space-y-5">
      {/* Identity */}
      <header className="flex items-center gap-3">
        <Flag src={n.flag} noc={n.noc} size={52} />
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-xl font-extrabold leading-tight">{n.name}</h1>
          <div className="text-sm text-ink-dim">
            {n.noc}
            {n.continent ? ` · ${CONTINENT_LABEL[n.continent]}` : ""}
          </div>
        </div>
        <div className="text-right">
          <div className="tnum text-3xl font-black la-gradient-text">
            {n.securedTotal}
            <span className="text-lg text-ink-faint">/{n.capTotal}</span>
          </div>
          <div className="text-[11px] text-ink-faint">places secured</div>
        </div>
      </header>

      {/* Summary chips */}
      <section className="flex flex-wrap gap-2">
        <Chip tone="good" value={n.securedTotal} label="secured" />
        <Chip tone="electric" value={n.openTotal} label="still open" />
        {n.blockedTotal > 0 && <Chip tone="gold" value={n.blockedTotal} label="locked out by cap" icon={<Lock size={12} />} />}
      </section>

      {/* Two ladders */}
      <div className="grid gap-4 lg:grid-cols-2">
        <GenderColumn title="Elite Men" slots={n.men} />
        <GenderColumn title="Elite Women" slots={n.women} />
      </div>

      {/* Mixed Relay pathway */}
      <MrCard n={n} />
    </div>
  );
}

function Chip({
  tone,
  value,
  label,
  icon,
}: {
  tone: "good" | "electric" | "gold";
  value: number;
  label: string;
  icon?: React.ReactNode;
}) {
  const tones = {
    good: "bg-good/10 text-good",
    electric: "bg-electric/10 text-electric-bright",
    gold: "bg-la-gold/15 text-la-gold",
  } as const;
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-bold", tones[tone])}>
      {icon}
      <span className="tnum">{value}</span>
      <span className="text-[11px] font-semibold opacity-80">{label}</span>
    </span>
  );
}

function GenderColumn({ title, slots }: { title: string; slots: NocGenderSlots }) {
  // The cap "line" falls after the last athlete holding an individual place.
  const lastQualified = slots.contenders.reduce((acc, c, i) => (c.qualified ? i : acc), -1);
  const twoTop = slots.contenders.filter((c) => c.qualified).slice(0, 2);

  return (
    <section className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold">{title}</h2>
        <div className="flex items-center gap-2">
          <SlotPips slots={slots} size="sm" />
          <span className="text-[11px] font-semibold text-ink-dim">
            {slots.secured}/{slots.cap}
          </span>
        </div>
      </div>

      {slots.pathwaySecured > 0 && (
        <p className="mb-2 rounded-lg bg-electric/10 px-2.5 py-1.5 text-[11px] font-semibold text-electric-bright">
          +{slots.pathwaySecured} place{slots.pathwaySecured > 1 ? "s" : ""} secured via the host / relay pathway
        </p>
      )}

      {slots.contenders.length === 0 ? (
        <p className="text-[13px] text-ink-faint">No ranked athletes yet.</p>
      ) : (
        <ul className="space-y-1">
          {slots.contenders.map((c, i) => (
            <li key={c.athleteId}>
              <Link
                href={`/athlete/${c.athleteId}`}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition",
                  c.qualified
                    ? "bg-good/10 ring-1 ring-good/30"
                    : c.blockedByCap
                      ? "bg-la-gold/10"
                      : "bg-surface hover:bg-surface-2",
                )}
              >
                <span className="tnum w-9 shrink-0 text-ink-faint">#{c.rank}</span>
                <span className={cn("min-w-0 flex-1 truncate", c.qualified && "font-semibold")}>{c.fullName}</span>
                <span className="tnum text-[13px] text-ink-dim">{fmtPoints(c.total)}</span>
                <StatusPill c={c} />
              </Link>

              {/* Draw the cap line right after the last qualifier. */}
              {i === lastQualified && i < slots.contenders.length - 1 && (
                <div className="relative py-1.5">
                  <div className="qual-line" />
                  <span className="absolute -top-0.5 right-2 text-[9px] font-bold uppercase tracking-wider text-la-gold">
                    nation cap ({slots.cap})
                  </span>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      {/* Internal duel between the top two qualifiers. */}
      {twoTop.length === 2 && (
        <Link
          href={`/versus/${twoTop[0].athleteId}/${twoTop[1].athleteId}`}
          className="mt-2 flex items-center justify-center gap-1.5 rounded-lg bg-surface px-3 py-2 text-[12px] font-semibold text-ink-dim transition hover:bg-surface-2"
        >
          <Swords size={13} className="text-la-gold" /> {twoTop[0].fullName.split(" ").slice(-1)} vs{" "}
          {twoTop[1].fullName.split(" ").slice(-1)}
        </Link>
      )}
    </section>
  );
}

function StatusPill({ c }: { c: NocGenderSlots["contenders"][number] }) {
  if (c.qualified)
    return <span className="shrink-0 rounded-md bg-good/20 px-1.5 py-0.5 text-[10px] font-bold text-good">IN</span>;
  if (c.blockedByCap)
    return (
      <span className="inline-flex shrink-0 items-center gap-1 rounded-md bg-la-gold/20 px-1.5 py-0.5 text-[10px] font-bold text-la-gold">
        <Lock size={9} /> capped
      </span>
    );
  return <span className="shrink-0 rounded-md bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold text-ink-faint">chasing</span>;
}

function MrCard({ n }: { n: NocWarRoom }) {
  const { mr } = n;
  return (
    <section className="card flex items-center gap-3 p-4">
      <Users size={20} className={mr.insideTop16 ? "text-good" : "text-ink-faint"} />
      <div className="flex-1">
        <div className="text-sm font-bold">{n.noc} Mixed Relay pathway</div>
        <div className="text-[11px] text-ink-dim">
          {mr.rank
            ? `MR Olympic rank #${mr.rank} · ${
                mr.insideTop16 ? "inside the top 16 ✓" : `${fmtPoints(mr.gapToTop16 ?? 0)} pts outside`
              }`
            : "not currently in the Mixed Relay Olympic ranking"}
          {mr.worldChampsSlot ? ` · World Champs ${mr.worldChampsSlot} slot` : ""}
        </div>
      </div>
      {mr.insideTop16 && <ShieldCheck size={18} className="text-good" />}
    </section>
  );
}
