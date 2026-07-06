import { ChevronUp, ChevronDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

/** Rank movement indicator: green up, red down, muted flat. */
export function MovementArrow({
  delta,
  className,
  showZero = false,
}: {
  delta: number;
  className?: string;
  showZero?: boolean;
}) {
  if (delta === 0) {
    return showZero ? (
      <span className={cn("inline-flex items-center text-ink-faint", className)}>
        <Minus size={14} />
      </span>
    ) : null;
  }
  const up = delta > 0;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 font-semibold tabular-nums",
        up ? "text-good" : "text-bad",
        className,
      )}
    >
      {up ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      {Math.abs(delta)}
    </span>
  );
}
