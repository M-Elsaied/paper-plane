"use client";
import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";
import { cn } from "@/lib/utils";

type Theme = "dark" | "light";
const KEY = "rtla28:theme";

/** Reads the theme the FOUC script already applied, and lets the user flip it. */
export function ThemeToggle({ className, showLabel }: { className?: string; showLabel?: boolean }) {
  const [theme, setTheme] = useState<Theme>("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const current = (document.documentElement.dataset.theme as Theme) || "dark";
    setTheme(current);
    setMounted(true);
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* no-op */
    }
    setTheme(next);
  }

  if (!mounted) {
    return <span className={cn("inline-block h-9 w-9", className)} aria-hidden />;
  }

  const isDark = theme === "dark";
  return (
    <button
      onClick={toggle}
      aria-label={`Switch to ${isDark ? "light" : "dark"} theme`}
      className={cn(
        "inline-flex items-center gap-2 rounded-xl border border-hairline bg-surface px-3 py-2 text-sm font-semibold text-ink-dim transition hover:bg-surface-2",
        className,
      )}
    >
      {isDark ? <Moon size={16} /> : <Sun size={16} />}
      {showLabel && <span>{isDark ? "Dark" : "Light"}</span>}
    </button>
  );
}

/** Inline, render-blocking script that applies the saved theme before paint
 *  (prevents a flash of the wrong theme). Injected in <head>. */
export const themeScript = `(function(){try{var t=localStorage.getItem('${KEY}');if(t!=='light'&&t!=='dark'){t='dark';}document.documentElement.dataset.theme=t;}catch(e){document.documentElement.dataset.theme='dark';}})();`;
