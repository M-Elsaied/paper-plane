import { cn } from "@/lib/utils";

/**
 * Horizontal "distance to the line" meter. The glowing gold line is the cut;
 * the marker is the athlete. Inside the line -> marker sits in the qualifying
 * (left) zone; outside -> chasing (right) zone.
 */
export function QualMeter({
  total,
  cutPoints,
  qualified,
}: {
  total: number;
  cutPoints: number | null;
  qualified: boolean;
}) {
  const cut = cutPoints ?? total;
  // Scale: show a window around the cut so movement is legible.
  const window = Math.max(400, cut * 0.8);
  const min = cut - window / 2;
  const max = cut + window / 2;
  const clamp = (v: number) => Math.min(1, Math.max(0, (v - min) / (max - min)));
  const linePos = clamp(cut) * 100;
  const mePos = clamp(total) * 100;

  return (
    <div className="relative h-14">
      {/* track */}
      <div className="absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 overflow-hidden rounded-full bg-surface-2">
        <div
          className="h-full la-gradient opacity-30"
          style={{ width: `${linePos}%` }}
        />
      </div>
      {/* qualification line */}
      <div
        className="absolute top-0 bottom-0 w-px"
        style={{ left: `${linePos}%` }}
      >
        <div className="qual-line absolute inset-y-0 left-0 !h-full !w-0.5" />
        <div className="absolute -top-0.5 left-1/2 -translate-x-1/2 whitespace-nowrap text-[9px] font-bold uppercase tracking-wide text-la-gold">
          Line
        </div>
      </div>
      {/* athlete marker */}
      <div
        className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
        style={{ left: `${mePos}%` }}
      >
        <div
          className={cn(
            "h-5 w-5 rounded-full border-2 border-white shadow-lg",
            qualified ? "bg-good" : "bg-electric",
          )}
        />
      </div>
    </div>
  );
}
