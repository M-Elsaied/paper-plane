/**
 * Explain-the-line: turn the deterministic engine output into plain-language
 * sentences a fan or coach can reconcile against the official criteria. This is
 * the trust layer made visible — no black box.
 */
import { fmtPoints, ordinal } from "@/lib/format";
import type { CockpitModel } from "@/lib/cockpit";

export interface ExplainLine {
  tone: "good" | "electric" | "warn" | "muted";
  text: string;
}

export function explainPosition(m: CockpitModel): ExplainLine[] {
  const lines: ExplainLine[] = [];
  const first = m.fullName.split(" ")[0];

  // 1) Headline standing.
  if (m.status.code === "in") {
    lines.push({
      tone: "good",
      text: `${first} sits ${ordinal(m.rank)} on the Olympic Qualification Ranking — inside the 21 individual places by ${fmtPoints(-m.gapToLine)} points. If the ranking closed today, they'd be on the plane to LA.`,
    });
  } else if (m.status.code === "blocked") {
    lines.push({
      tone: "warn",
      text: `${first} is ${ordinal(m.rank)} overall and would make the line on points — but ${m.noc} has already used its cap of ${m.nocUsage.cap} places, with ${m.nocAhead} teammate${m.nocAhead === 1 ? "" : "s"} ranked higher. To qualify, ${first} must out-point a teammate, not just the field.`,
    });
  } else if (m.status.code === "ineligible") {
    lines.push({ tone: "muted", text: `${first} is currently ineligible: ${m.status.detail}.` });
  } else {
    lines.push({
      tone: "electric",
      text: `${first} is ${ordinal(m.rank)}, ${fmtPoints(m.gapToLine)} points below the cut at rank ${m.cutRank}. Closing that gap means banking counting scores worth more than their current weakest (${fmtPoints(m.marginalPoints)}).`,
    });
  }

  // 2) The best-12 / max-7 mechanic.
  const fullPeriod = m.periodFull[1] ? 1 : m.periodFull[2] ? 2 : null;
  if (fullPeriod) {
    lines.push({
      tone: "warn",
      text: `Period ${fullPeriod} is full at 7/7 counting scores — the max the rules allow from one period. Only a result beating ${fmtPoints(m.marginalPoints)} in that period will move the total.`,
    });
  } else {
    lines.push({
      tone: "muted",
      text: `The ranking counts an athlete's best 12 results, no more than 7 from either qualifying period. ${first} has ${m.periodCount[1]} in P1 and ${m.periodCount[2]} in P2 — room to add scores.`,
    });
  }

  // 3) Mixed Relay pathway.
  if (m.mr.rank) {
    lines.push({
      tone: m.mr.insideTop16 ? "good" : "electric",
      text: m.mr.insideTop16
        ? `Separately, ${m.noc}'s Mixed Relay team is ${ordinal(m.mr.rank)} — inside the top 16 that earns relay places, a second route to the Games.`
        : `${m.noc}'s Mixed Relay team is ${ordinal(m.mr.rank)}, ${fmtPoints(m.mr.gapToTop16)} points outside the top-16 relay cut.`,
    });
  }

  return lines;
}
