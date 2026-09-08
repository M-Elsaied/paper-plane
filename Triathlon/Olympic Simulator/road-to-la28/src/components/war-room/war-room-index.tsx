import Link from "next/link";
import { ChevronRight, Users, Lock } from "lucide-react";
import type { WarRoomModel, NocWarRoom } from "@/lib/war-room";
import { Flag } from "@/components/flag";
import { SlotPips } from "./slot-pips";
import { CONTINENT_LABEL } from "@/config/continents";

/** The nations board: every contending NOC, powerhouses first. */
export function WarRoomIndex({ model }: { model: WarRoomModel }) {
  return (
    <div className="space-y-4">
      {/* Totals strip */}
      <section className="card grid grid-cols-3 gap-2 p-4 text-center">
        <Stat value={model.totals.nations} label="nations in play" />
        <Stat value={model.totals.securedMen + model.totals.securedWomen} label="places secured" />
        <Stat value={model.totals.mrNationsInside} label="relay teams inside" />
      </section>

      <ul className="grid gap-2.5 lg:grid-cols-2">
        {model.nocs.map((n) => (
          <li key={n.noc}>
            <NationCard n={n} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function Stat({ value, label }: { value: number; label: string }) {
  return (
    <div>
      <div className="tnum text-2xl font-black la-gradient-text">{value}</div>
      <div className="text-[11px] leading-tight text-ink-faint">{label}</div>
    </div>
  );
}

function NationCard({ n }: { n: NocWarRoom }) {
  return (
    <Link
      href={`/war-room/${n.noc}`}
      className="card flex items-center gap-3 p-3.5 transition hover:bg-surface active:scale-[0.99]"
    >
      <Flag src={n.flag} noc={n.noc} size={38} />

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate font-bold">{n.name}</span>
          <span className="text-[11px] font-semibold text-ink-faint">{n.noc}</span>
        </div>
        <div className="mt-0.5 text-[11px] text-ink-faint">
          {n.continent ? CONTINENT_LABEL[n.continent] : "—"}
          {" · "}
          <span className="font-semibold text-ink-dim">
            {n.securedTotal}/{n.capTotal} places secured
          </span>
        </div>

        {/* Per-gender slot pips */}
        <div className="mt-2 flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-bold uppercase text-ink-faint">M</span>
            <SlotPips slots={n.men} size="sm" />
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-bold uppercase text-ink-faint">W</span>
            <SlotPips slots={n.women} size="sm" />
          </div>
        </div>
      </div>

      <div className="flex shrink-0 flex-col items-end gap-1.5">
        {n.blockedTotal > 0 && (
          <span className="inline-flex items-center gap-1 rounded-md bg-la-gold/15 px-1.5 py-0.5 text-[10px] font-bold text-la-gold">
            <Lock size={10} /> {n.blockedTotal} locked out
          </span>
        )}
        {n.mr.insideRelayCut && (
          <span className="inline-flex items-center gap-1 rounded-md bg-good/10 px-1.5 py-0.5 text-[10px] font-bold text-good">
            <Users size={10} /> Relay
          </span>
        )}
        <ChevronRight size={16} className="text-ink-faint" />
      </div>
    </Link>
  );
}
