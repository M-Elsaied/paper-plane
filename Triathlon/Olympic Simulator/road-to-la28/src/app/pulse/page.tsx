import { Activity } from "lucide-react";
import { getMovers, getSeedMeta } from "@/lib/data";
import { PulseBoard } from "@/components/pulse-board";

export const revalidate = 300;


export default async function PulsePage() {
  const [men, women] = await Promise.all([getMovers("male"), getMovers("female")]);
  const meta = getSeedMeta();

  return (
    <main className="px-4 pt-6">
      <header className="mb-4">
        <div className="mb-1 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] font-semibold text-good">
          <Activity size={12} /> Updated {meta.menPublished.slice(0, 10)}
        </div>
        <h1 className="text-2xl font-extrabold">The Pulse</h1>
        <p className="text-sm text-ink-faint">
          Who moved on the latest official Olympic ranking. Tap any athlete to open their cockpit.
        </p>
      </header>

      <PulseBoard men={men} women={women} />
    </main>
  );
}
