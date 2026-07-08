import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { sparkGeometry, type TrajectoryPoint } from "@/lib/trajectory";
import { cn } from "@/lib/utils";

/**
 * A compact rank-over-time sparkline. The rank axis is inverted, so a rising line
 * means the athlete is climbing the ranking. Trend-coloured via `currentColor`
 * (green climbing, red slipping), so it inherits the theme.
 */
export function RankSparkline({ points, className }: { points: TrajectoryPoint[]; className?: string }) {
  if (!points.length) return null;

  const ranks = points.map((p) => p.rank);
  const w = 132;
  const h = 34;
  const g = sparkGeometry(ranks, w, h);
  const trend = g.improved > 0 ? "up" : g.improved < 0 ? "down" : "flat";
  const color = trend === "up" ? "text-good" : trend === "down" ? "text-bad" : "text-ink-faint";
  const end = g.points[g.points.length - 1];

  const single = points.length === 1;
  const multi = points.length > 2;
  const caption = single
    ? `held #${ranks[0]}`
    : multi
      ? `last ${points.length} rankings`
      : "since last ranking";

  return (
    <div data-testid="rank-sparkline" className={cn("flex items-center gap-2.5", className)}>
      <div className={color}>
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible" aria-hidden>
          {!single && <path d={g.area} fill="currentColor" fillOpacity={0.12} stroke="none" />}
          {!single && (
            <polyline
              points={g.polyline}
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}
          <circle cx={end.x} cy={end.y} r={3.5} fill="currentColor" />
        </svg>
      </div>

      <div className="leading-tight">
        <div className={cn("flex items-center gap-1 text-xs font-bold", color)}>
          {trend === "up" ? <TrendingUp size={13} /> : trend === "down" ? <TrendingDown size={13} /> : <Minus size={13} />}
          {trend === "flat" ? "held" : `${Math.abs(g.improved)} place${Math.abs(g.improved) === 1 ? "" : "s"}`}
        </div>
        <div className="text-[10px] text-ink-faint">{caption}</div>
      </div>
    </div>
  );
}
