/**
 * Pick-'Em API.
 *  POST → submit/replace a podium pick { raceId, gender, podium:[3 ids], deviceId }
 *  GET  → ?raceId=&gender=&deviceId=  → { myPick, crowd, total }
 * Owner is the session account when claimed, else the device id.
 */
import { NextResponse } from "next/server";
import { getSessionAccountId } from "@/lib/auth";
import { submitPick, getMyPick, getCrowd } from "@/lib/picks";
import type { Gender } from "@/config/pathways";

export const dynamic = "force-dynamic";

function owner(accountId: string | null, deviceId: string | null): string | null {
  return accountId ?? (deviceId && deviceId.length >= 8 ? `dev:${deviceId}` : null);
}

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as {
    raceId?: number;
    gender?: Gender;
    podium?: number[];
    deviceId?: string;
  };
  const podium = (body.podium ?? []).filter((n) => Number.isFinite(n));
  if (!body.raceId || !body.gender || new Set(podium).size !== 3) {
    return NextResponse.json({ error: "invalid" }, { status: 400 });
  }
  const key = owner(await getSessionAccountId(), body.deviceId ?? null);
  if (!key) return NextResponse.json({ error: "no-owner" }, { status: 400 });
  const ok = await submitPick(key, body.raceId, body.gender, podium.slice(0, 3));
  if (!ok) return NextResponse.json({ error: "no-database" }, { status: 503 });
  return NextResponse.json({ ok: true });
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const raceId = Number(url.searchParams.get("raceId"));
  const gender = (url.searchParams.get("gender") as Gender) || "male";
  const deviceId = url.searchParams.get("deviceId");
  if (!raceId) return NextResponse.json({ error: "no-race" }, { status: 400 });

  const key = owner(await getSessionAccountId(), deviceId);
  const [myPick, crowd] = await Promise.all([
    key ? getMyPick(key, raceId, gender) : Promise.resolve(null),
    getCrowd(raceId, gender),
  ]);
  return NextResponse.json({ myPick, crowd });
}
