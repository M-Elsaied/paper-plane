import Link from "next/link";
import { notFound } from "next/navigation";
import { ChevronLeft, Swords } from "lucide-react";
import { buildVersus } from "@/lib/versus";
import { VersusView } from "@/components/versus/versus-view";

export const revalidate = 300;

export async function generateMetadata({ params }: { params: Promise<{ a: string; b: string }> }) {
  const { a, b } = await params;
  const v = await buildVersus(Number(a), Number(b)).catch(() => null);
  if (!v) return { title: "Head-to-head · Road to LA28" };
  const title = `${v.a.name} vs ${v.b.name} · Road to LA28`;
  return { title, description: `Head-to-head: ${v.a.name} vs ${v.b.name} on the road to LA 2028.` };
}

export default async function VersusPage({ params }: { params: Promise<{ a: string; b: string }> }) {
  const { a, b } = await params;
  const v = await buildVersus(Number(a), Number(b)).catch(() => null);
  if (!v) notFound();

  return (
    <main className="mx-auto px-4 pt-6 lg:max-w-2xl">
      <div className="mb-4 flex items-center gap-2">
        <Link
          href={`/athlete/${v.a.athleteId}`}
          className="flex h-9 w-9 items-center justify-center rounded-full border border-hairline bg-surface text-ink-dim"
        >
          <ChevronLeft size={18} />
        </Link>
        <div className="flex items-center gap-1.5">
          <Swords size={18} className="text-la-gold" />
          <h1 className="text-lg font-extrabold leading-tight">Head-to-head</h1>
        </div>
      </div>

      <VersusView v={v} />
    </main>
  );
}
