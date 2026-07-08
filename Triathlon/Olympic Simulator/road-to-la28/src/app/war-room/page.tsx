import { buildWarRoom } from "@/lib/war-room";
import { WarRoomIndex } from "@/components/war-room/war-room-index";

export const revalidate = 300;

export const metadata = {
  title: "NOC Slot War Room · Road to LA28",
  description:
    "Every nation's fight for its LA28 triathlon places — who's secured, who's still open, and who's locked out by the 3-per-nation cap.",
};

export default async function WarRoomPage() {
  const model = await buildWarRoom();

  return (
    <main className="px-4 pt-6 lg:max-w-4xl">
      <header className="mb-4">
        <h1 className="text-2xl font-extrabold">NOC Slot War Room</h1>
        <p className="text-sm text-ink-faint">
          Each nation may send at most 3 athletes per gender. Here's who has secured places, who's still
          fighting for them, and who is strong enough to qualify but locked out by their own compatriots.
        </p>
      </header>
      <WarRoomIndex model={model} />
    </main>
  );
}
