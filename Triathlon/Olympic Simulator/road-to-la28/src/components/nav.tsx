"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Gauge, ListOrdered, CalendarDays, Activity, Users, Shield, UserCircle2 } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { cn } from "@/lib/utils";

export const TABS = [
  { href: "/", label: "Cockpit", icon: Gauge, match: (p: string) => p === "/" || p.startsWith("/athlete") },
  { href: "/rankings", label: "Rankings", icon: ListOrdered, match: (p: string) => p.startsWith("/rankings") },
  { href: "/race-week", label: "Race Week", icon: CalendarDays, match: (p: string) => p.startsWith("/race-week") || p.startsWith("/race/") },
  { href: "/pulse", label: "Pulse", icon: Activity, match: (p: string) => p.startsWith("/pulse") },
  { href: "/relay", label: "Relay", icon: Users, match: (p: string) => p.startsWith("/relay") },
  { href: "/war-room", label: "War Room", icon: Shield, match: (p: string) => p.startsWith("/war-room") },
];

/** Bottom tab bar — mobile only (lg:hidden). */
export function BottomNav() {
  const pathname = usePathname();
  return (
    <nav
      data-tour="nav"
      className="fixed inset-x-0 bottom-0 z-50 border-t border-hairline bg-elevated/85 backdrop-blur-xl lg:hidden"
    >
      <div className="mx-auto flex max-w-md items-stretch justify-around">
        {TABS.map((t) => {
          const active = t.match(pathname);
          const Icon = t.icon;
          return (
            <Link
              key={t.href}
              href={t.href}
              className={cn(
                "flex flex-1 flex-col items-center gap-1 py-2.5 text-[10px] font-medium transition-colors",
                active ? "text-electric-bright" : "text-ink-faint hover:text-ink-dim",
              )}
            >
              <Icon size={20} strokeWidth={active ? 2.5 : 2} />
              {t.label}
              <span className={cn("h-0.5 w-6 rounded-full transition-all", active ? "la-gradient" : "bg-transparent")} />
            </Link>
          );
        })}
      </div>
    </nav>
  );
}

/** Left sidebar — desktop only (hidden below lg). Fixed, full-height. */
export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="fixed inset-y-0 left-0 z-40 hidden w-60 flex-col border-r border-hairline bg-elevated/40 px-4 py-6 backdrop-blur-xl lg:flex">
      <Link href="/" className="mb-8 flex items-center gap-2 px-2">
        <span className="flex h-9 w-9 items-center justify-center rounded-xl la-gradient text-sm font-black text-navy-950">
          28
        </span>
        <span className="font-[family-name:var(--font-display)] text-lg font-black leading-none">
          ROAD TO
          <span className="block la-gradient-text">LA28</span>
        </span>
      </Link>

      <nav data-tour="nav" className="flex flex-1 flex-col gap-1">
        {TABS.map((t) => {
          const active = t.match(pathname);
          const Icon = t.icon;
          return (
            <Link
              key={t.href}
              href={t.href}
              className={cn(
                "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-semibold transition",
                active ? "bg-surface-2 text-electric-bright" : "text-ink-dim hover:bg-surface hover:text-ink",
              )}
            >
              <Icon size={19} strokeWidth={active ? 2.5 : 2} />
              {t.label}
            </Link>
          );
        })}
      </nav>

      <div className="flex flex-col gap-2 border-t border-hairline pt-4">
        <Link
          href="/account"
          className={cn(
            "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-semibold transition",
            pathname.startsWith("/account") ? "bg-surface-2 text-electric-bright" : "text-ink-dim hover:bg-surface hover:text-ink",
          )}
        >
          <UserCircle2 size={19} /> Your Board
        </Link>
        <ThemeToggle showLabel className="justify-start" />
      </div>
    </aside>
  );
}
