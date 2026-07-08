/**
 * Athlete search — queries the full World Triathlon athlete database so ANY
 * athlete is findable (not just the ~160 in the Olympic Qualification Ranking).
 * OQR-ranked athletes are annotated with their current rank + sorted first.
 */
import { NextResponse } from "next/server";
import { searchAthletes } from "@/lib/wt-api/athletes";
import { getBothStates } from "@/lib/data";
import { rankAthletes } from "@/lib/engine/qualification";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams.get("q")?.trim() ?? "";
  if (q.length < 2) return NextResponse.json({ results: [] });

  // Build a rank lookup from the current OQR (Neon or seed).
  const rankByAthlete = new Map<number, number>();
  for (const state of await getBothStates()) {
    rankAthletes(state.athletes).forEach((a, i) => rankByAthlete.set(a.athleteId, i + 1));
  }

  let hits;
  try {
    hits = await searchAthletes(q, 25);
  } catch {
    return NextResponse.json({ results: [], error: "search-unavailable" });
  }

  const results = hits
    .map((h) => ({ ...h, rank: rankByAthlete.get(h.athleteId) ?? null }))
    .sort((a, b) => {
      // ranked first (by rank), then everyone else by name
      if (a.rank && b.rank) return a.rank - b.rank;
      if (a.rank) return -1;
      if (b.rank) return 1;
      return a.fullName.localeCompare(b.fullName);
    });

  return NextResponse.json({ results });
}
