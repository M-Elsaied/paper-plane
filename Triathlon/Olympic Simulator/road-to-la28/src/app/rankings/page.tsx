import { buildRanking } from "@/lib/cockpit";
import { RankingsBoard } from "@/components/rankings-board";

export const revalidate = 300;


export default async function RankingsPage() {
  const [men, women] = await Promise.all([buildRanking("male"), buildRanking("female")]);

  return (
    <main className="px-4 pt-6">
      <header className="mb-4">
        <h1 className="text-2xl font-extrabold">Olympic Qualification Ranking</h1>
        <p className="text-sm text-ink-faint">
          Best 12 scores · max 7 per period · the line marks the 21 individual places.
        </p>
      </header>
      <RankingsBoard
        men={men.rows}
        women={women.rows}
        cut={{ male: men.line.cutRank, female: women.line.cutRank }}
      />
    </main>
  );
}
