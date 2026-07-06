import { Check, TrendingUp, Ban, MinusCircle } from "lucide-react";
import { STATUS_TONE, type QualStatus, type StatusCode } from "@/lib/engine/status";
import { cn } from "@/lib/utils";

const ICON: Record<StatusCode, typeof Check> = {
  in: Check,
  chasing: TrendingUp,
  blocked: Ban,
  ineligible: MinusCircle,
};

/**
 * The one status pill used everywhere. Colour + icon + label together, so it
 * reads without relying on colour alone (colourblind-safe).
 */
export function StatusBadge({
  status,
  size = "md",
  className,
}: {
  status: QualStatus;
  size?: "sm" | "md";
  className?: string;
}) {
  const tone = STATUS_TONE[status.tone];
  const Icon = ICON[status.code];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full font-bold",
        tone.bg,
        tone.text,
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-[11px]",
        className,
      )}
    >
      <Icon size={size === "sm" ? 11 : 13} strokeWidth={2.5} />
      {status.label}
    </span>
  );
}
