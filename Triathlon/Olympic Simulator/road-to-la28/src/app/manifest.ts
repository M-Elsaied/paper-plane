import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Road to LA28 — Olympic Qualification Cockpit",
    short_name: "Road to LA28",
    description:
      "Live Olympic triathlon qualification tracking and what-if simulation for every athlete.",
    start_url: "/",
    display: "standalone",
    orientation: "portrait",
    background_color: "#050d1c",
    theme_color: "#050d1c",
    icons: [
      { src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
      { src: "/icon-maskable.svg", sizes: "any", type: "image/svg+xml", purpose: "maskable" },
    ],
  };
}
