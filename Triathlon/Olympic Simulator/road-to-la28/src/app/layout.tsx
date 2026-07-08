import type { Metadata, Viewport } from "next";
import { Archivo, Inter } from "next/font/google";
import "./globals.css";
import { BottomNav, Sidebar } from "@/components/nav";
import { ThemeToggle, themeScript } from "@/components/theme-toggle";

const display = Archivo({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["600", "700", "800", "900"],
});
const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "https://road-to-la28.vercel.app"),
  title: "Road to LA28 — Olympic Qualification Cockpit",
  description:
    "Live Olympic triathlon qualification: where every athlete stands on the road to Los Angeles 2028, and what they need to do next.",
  manifest: "/manifest.webmanifest",
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "Road to LA28" },
};

export const viewport: Viewport = {
  themeColor: "#050d1c",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${inter.variable} antialiased`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-[100dvh]">
        <Sidebar />
        {/* Content: full-width narrow on mobile (bottom nav), offset by the
            sidebar and comfortably widened on desktop. */}
        <div className="lg:pl-60">
          {/* Floating theme toggle on mobile (sidebar has its own on desktop). */}
          <div className="fixed right-3 top-3 z-30 lg:hidden">
            <ThemeToggle className="h-9 w-9 justify-center !px-0" />
          </div>
          <div className="mx-auto w-full max-w-md pb-24 lg:max-w-5xl lg:px-8 lg:pb-10">{children}</div>
        </div>
        <BottomNav />
      </body>
    </html>
  );
}
