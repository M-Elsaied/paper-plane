import type { Metadata, Viewport } from "next";
import { Archivo, Inter } from "next/font/google";
import "./globals.css";
import { BottomNav } from "@/components/bottom-nav";

const display = Archivo({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["600", "700", "800", "900"],
});
const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });

export const metadata: Metadata = {
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
    <html lang="en" className={`${display.variable} ${inter.variable} antialiased`}>
      <body className="min-h-[100dvh] pb-20">
        <div className="mx-auto w-full max-w-md">{children}</div>
        <BottomNav />
      </body>
    </html>
  );
}
