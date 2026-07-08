/**
 * Rank trajectory — an athlete's Olympic Qualification Ranking position over
 * time, and the pure geometry to draw it as a sparkline.
 *
 * The rank axis is inverted: rank 1 (the best) sits at the TOP of the chart, so
 * a line that rises means the athlete is climbing. All maths here is pure and
 * isomorphic — it runs identically on the server and in the browser, and carries
 * no I/O, so it is cheap to unit-test.
 */

/** One observation of an athlete's rank (oldest first when in a series). */
export interface TrajectoryPoint {
  rank: number;
  /** ISO date of the ranking this rank came from, when known. */
  date?: string;
  /** Human label for endpoints without a date (e.g. "last ranking", "now"). */
  label?: string;
}

export interface SparkPoint {
  x: number;
  y: number;
  rank: number;
}

export interface SparkGeometry {
  points: SparkPoint[];
  /** "x,y x,y …" for a <polyline>. */
  polyline: string;
  /** Closed path filling the area beneath the line, for a soft gradient. */
  area: string;
  bestRank: number;
  worstRank: number;
  /** first rank − last rank: positive = climbed (improved), negative = slipped. */
  improved: number;
}

/**
 * Map a series of ranks to SVG coordinates. Best rank → top (small y), worst →
 * bottom. A single point (or an all-equal series) renders as a flat mid-line.
 */
export function sparkGeometry(ranks: number[], w = 132, h = 34, pad = 4): SparkGeometry {
  const n = ranks.length;
  const best = Math.min(...ranks);
  const worst = Math.max(...ranks);
  const span = worst - best;
  const innerH = h - pad * 2;
  const innerW = w - pad * 2;

  const points: SparkPoint[] = ranks.map((rank, i) => {
    const x = n === 1 ? w / 2 : pad + (i / (n - 1)) * innerW;
    // Inverted: best (min) → top (y = pad); worst (max) → bottom (y = h - pad).
    const y = span === 0 ? h / 2 : pad + ((rank - best) / span) * innerH;
    return { x: round(x), y: round(y), rank };
  });

  const polyline = points.map((p) => `${p.x},${p.y}`).join(" ");
  const first = points[0];
  const last = points[points.length - 1];
  const area = `M ${first.x},${h} L ${points.map((p) => `${p.x},${p.y}`).join(" L ")} L ${last.x},${h} Z`;

  return {
    points,
    polyline,
    area,
    bestRank: best,
    worstRank: worst,
    improved: ranks[0] - ranks[n - 1],
  };
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
