"use client";
import { useState } from "react";
import { Share2, Check } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * One-tap share. Uses the native share sheet (which unfurls the broadcast card
 * via OG meta); falls back to copying the link. The growth loop's trigger.
 */
export function ShareButton({
  athleteId,
  name,
  className,
}: {
  athleteId: number;
  name: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function share() {
    const url =
      typeof window !== "undefined"
        ? `${window.location.origin}/athlete/${athleteId}`
        : `/athlete/${athleteId}`;
    const data = {
      title: `${name} · Road to LA28`,
      text: `${name}'s road to the LA 2028 Olympics — live qualification tracker.`,
      url,
    };
    try {
      if (navigator.share) {
        await navigator.share(data);
        return;
      }
    } catch {
      // user cancelled or share failed — fall through to copy
    }
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* no-op */
    }
  }

  return (
    <button
      onClick={share}
      aria-label={`Share ${name}`}
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-hairline bg-surface px-2.5 py-1 text-[11px] font-semibold text-ink-dim transition active:scale-95",
        className,
      )}
    >
      {copied ? <Check size={12} className="text-good" /> : <Share2 size={12} />}
      {copied ? "Copied" : "Share"}
    </button>
  );
}
