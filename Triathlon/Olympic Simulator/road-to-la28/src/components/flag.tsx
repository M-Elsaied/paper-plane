import { cn } from "@/lib/utils";

/** Turn a 2-letter ISO country code into its flag emoji (offline fallback). */
function isoToEmoji(iso?: string): string | null {
  if (!iso || iso.length !== 2) return null;
  const cp = [...iso.toUpperCase()].map((c) => 0x1f1e6 + (c.charCodeAt(0) - 65));
  if (cp.some((c) => c < 0x1f1e6 || c > 0x1f1ff)) return null;
  return String.fromCodePoint(...cp);
}

/**
 * A country flag. Prefers the World Triathlon circular flag image; falls back to
 * the ISO emoji flag, then a plain NOC chip. Never breaks layout.
 */
export function Flag({
  src,
  iso,
  noc,
  size = 20,
  className,
}: {
  src?: string;
  iso?: string;
  noc?: string;
  size?: number;
  className?: string;
}) {
  if (src) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={src}
        alt={noc ?? "flag"}
        width={size}
        height={size}
        loading="lazy"
        className={cn("inline-block shrink-0 rounded-full object-cover ring-1 ring-white/15", className)}
        style={{ width: size, height: size }}
      />
    );
  }
  const emoji = isoToEmoji(iso);
  if (emoji) {
    return (
      <span className={cn("inline-block shrink-0 leading-none", className)} style={{ fontSize: size * 0.9 }}>
        {emoji}
      </span>
    );
  }
  return null;
}
