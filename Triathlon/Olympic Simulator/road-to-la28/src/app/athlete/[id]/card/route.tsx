/**
 * Shareable broadcast card — a 1080×1350 OG image rendered from the live
 * qualification math. This is the growth loop: one tap turns an athlete's
 * status into a screenshot-worthy image with the glowing line and their rank.
 */
import { ImageResponse } from "next/og";
import { buildCockpit } from "@/lib/cockpit";
import { fmtPoints } from "@/lib/format";

export const dynamic = "force-dynamic";
export const revalidate = 300;

const TONE_HEX: Record<string, string> = {
  good: "#21d07a",
  electric: "#35c8ff",
  warn: "#f5c518",
  muted: "#9db2d4",
};

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const m = await buildCockpit(Number(id));
  if (!m) return new Response("Not found", { status: 404 });

  const accent = TONE_HEX[m.status.tone] ?? "#35c8ff";
  const inside = m.gapToLine <= 0;

  return new ImageResponse(
    (
      <div
        style={{
          width: "1080px",
          height: "1350px",
          display: "flex",
          flexDirection: "column",
          background: "linear-gradient(160deg, #0b1e3c 0%, #04070f 100%)",
          color: "#eaf2ff",
          fontFamily: "sans-serif",
          position: "relative",
          padding: "72px",
        }}
      >
        {/* wordmark */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontSize: 34, fontWeight: 800, letterSpacing: 2, color: "#9db2d4" }}>
            ROAD TO LA28
          </div>
          <div
            style={{
              display: "flex",
              fontSize: 30,
              fontWeight: 800,
              color: "#04070f",
              background: accent,
              padding: "10px 24px",
              borderRadius: 999,
            }}
          >
            {m.status.label}
          </div>
        </div>

        {/* portrait */}
        <div
          style={{
            display: "flex",
            marginTop: 60,
            alignItems: "flex-end",
            gap: 40,
          }}
        >
          {m.profileImage ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={m.profileImage}
              width={300}
              height={300}
              alt=""
              style={{ borderRadius: 32, objectFit: "cover", border: `4px solid ${accent}` }}
            />
          ) : null}
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: 30, color: "#9db2d4", fontWeight: 700 }}>{m.noc}</div>
            <div style={{ fontSize: 72, fontWeight: 800, lineHeight: 1.05, maxWidth: 560, display: "flex" }}>
              {m.fullName}
            </div>
          </div>
        </div>

        {/* rank + points */}
        <div style={{ display: "flex", marginTop: 80, gap: 80 }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: 30, color: "#9db2d4", fontWeight: 700 }}>OLYMPIC RANK</div>
            <div style={{ fontSize: 200, fontWeight: 800, lineHeight: 1 }}>{`#${m.rank}`}</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: 30, color: "#9db2d4", fontWeight: 700 }}>POINTS</div>
            <div style={{ fontSize: 200, fontWeight: 800, lineHeight: 1, color: "#f5c518" }}>
              {fmtPoints(m.total)}
            </div>
          </div>
        </div>

        {/* the line */}
        <div style={{ display: "flex", marginTop: 90, flexDirection: "column", gap: 20 }}>
          <div
            style={{
              display: "flex",
              height: 8,
              borderRadius: 8,
              background: "linear-gradient(90deg, #ff5c6c, #8b5cf6 55%, #f5c518)",
            }}
          />
          <div style={{ fontSize: 44, fontWeight: 800, color: inside ? "#21d07a" : "#35c8ff", display: "flex" }}>
            {inside
              ? `Inside the line by ${fmtPoints(-m.gapToLine)} pts`
              : `${fmtPoints(m.gapToLine)} pts from the Olympic cut`}
          </div>
        </div>

        <div style={{ display: "flex", marginTop: "auto", fontSize: 28, color: "#5f7699" }}>
          {`${m.daysToDeadline} days to LA 2028 · road-to-la28.vercel.app`}
        </div>
      </div>
    ),
    { width: 1080, height: 1350 },
  );
}
