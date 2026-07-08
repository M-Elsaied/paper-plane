import type { NocGenderSlots } from "@/lib/war-room";
import { cn } from "@/lib/utils";

/**
 * A nation's cap rendered as slot boxes: pathway-secured (blue), individually
 * secured (green), then open (hollow). At-a-glance "how many of the cap are gone".
 */
export function SlotPips({ slots, size = "md" }: { slots: NocGenderSlots; size?: "sm" | "md" }) {
  const box = size === "sm" ? "h-3.5 w-3.5" : "h-5 w-6";
  return (
    <div className="flex items-center gap-1">
      {Array.from({ length: slots.cap }).map((_, i) => {
        const isPathway = i < slots.pathwaySecured;
        const isSecured = i < slots.secured;
        return (
          <span
            key={i}
            className={cn(
              "rounded-[5px] border transition-colors",
              box,
              isPathway
                ? "border-electric bg-electric/70"
                : isSecured
                  ? "border-good bg-good/70"
                  : "border-hairline bg-surface",
            )}
            title={isPathway ? "secured via pathway" : isSecured ? "secured" : "open"}
          />
        );
      })}
    </div>
  );
}
