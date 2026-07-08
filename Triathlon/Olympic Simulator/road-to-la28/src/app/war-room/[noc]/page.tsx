import Link from "next/link";
import { notFound } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { buildNocWarRoom } from "@/lib/war-room";
import { nationName } from "@/config/noc-names";
import { NocDetail } from "@/components/war-room/noc-detail";

export const revalidate = 300;

export async function generateMetadata({ params }: { params: Promise<{ noc: string }> }) {
  const { noc } = await params;
  const name = nationName(noc.toUpperCase());
  return {
    title: `${name} · NOC Slot War Room · Road to LA28`,
    description: `${name}'s fight for its LA28 Olympic triathlon places — secured, open, and capped-out athletes.`,
  };
}

export default async function NocWarRoomPage({ params }: { params: Promise<{ noc: string }> }) {
  const { noc } = await params;
  const n = await buildNocWarRoom(noc).catch(() => null);
  if (!n) notFound();

  return (
    <main className="mx-auto px-4 pt-6 lg:max-w-3xl">
      <Link
        href="/war-room"
        className="mb-4 inline-flex items-center gap-1 text-sm font-semibold text-ink-dim transition hover:text-ink"
      >
        <ChevronLeft size={16} /> War Room
      </Link>
      <NocDetail n={n} />
    </main>
  );
}
