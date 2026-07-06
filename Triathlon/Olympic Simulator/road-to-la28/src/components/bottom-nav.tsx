"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Gauge, ListOrdered, CalendarDays, Activity, Users } from "lucide-react";
import { cn } from "@/lib/utils";

const TABS = [
  { href: "/", label: "Cockpit", icon: Gauge, match: (p: string) => p === "/" || p.startsWith("/athlete") },
  { href: "/rankings", label: "Rankings", icon: ListOrdered, match: (p: string) => p.startsWith("/rankings") },
  { href: "/race-week", label: "Race Week", icon: CalendarDays, match: (p: string) => p.startsWith("/race-week") },
  { href: "/pulse", label: "Pulse", icon: Activity, match: (p: string) => p.startsWith("/pulse") },
  { href: "/relay", label: "Relay", icon: Users, match: (p: string) => p.startsWith("/relay") },
];

export function BottomNav() {
  const pathname = usePathname();
  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 border-t border-white/10 bg-navy-950/80 backdrop-blur-xl">
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
              <span
                className={cn(
                  "h-0.5 w-6 rounded-full transition-all",
                  active ? "la-gradient" : "bg-transparent",
                )}
              />
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
