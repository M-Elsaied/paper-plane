/**
 * Pick-'Em scoring — pure and deterministic. Score a predicted podium against
 * the actual finishing podium.
 */
import { PICKEM } from "@/config/pickem";

export type SlotOutcome = "exact" | "podium" | "miss";

export interface PickScore {
  score: number;
  perfect: boolean;
  slots: { predicted: number; outcome: SlotOutcome; points: number }[];
  correctCount: number; // predicted athletes who made the podium
}

/**
 * @param predicted length-3 array of athlete ids [1st, 2nd, 3rd]
 * @param actual    length-3 array of athlete ids who actually finished 1-3
 */
export function scorePick(predicted: number[], actual: number[]): PickScore {
  const actualSet = new Set(actual);
  let score = 0;
  let exactHits = 0;
  let correctCount = 0;

  const slots = predicted.slice(0, 3).map((pid, i) => {
    let outcome: SlotOutcome = "miss";
    let points = 0;
    if (actual[i] === pid) {
      outcome = "exact";
      points = PICKEM.exact[i + 1] ?? 0;
      exactHits++;
      correctCount++;
    } else if (actualSet.has(pid)) {
      outcome = "podium";
      points = PICKEM.onPodiumWrongSpot;
      correctCount++;
    }
    score += points;
    return { predicted: pid, outcome, points };
  });

  const perfect = exactHits === 3;
  if (perfect) score += PICKEM.perfectBonus;

  return { score, perfect, slots, correctCount };
}
