/**
 * Account bridge API.
 *  GET  → { claimed, board }
 *  POST → claim: create an account from the posted board, set the session cookie
 *  PUT  → update the claimed account's board (sync)
 */
import { NextResponse } from "next/server";
import { getSessionAccountId, setSession } from "@/lib/auth";
import { createAccount, getBoard, updateBoard, type Board } from "@/lib/account";

export const dynamic = "force-dynamic";

export async function GET() {
  const id = await getSessionAccountId();
  if (!id) return NextResponse.json({ claimed: false, board: null });
  const board = await getBoard(id);
  if (!board) return NextResponse.json({ claimed: false, board: null });
  return NextResponse.json({ claimed: true, board });
}

export async function POST(req: Request) {
  const existing = await getSessionAccountId();
  if (existing && (await getBoard(existing))) {
    return NextResponse.json({ claimed: true });
  }
  const body = (await req.json().catch(() => ({}))) as { board?: Board };
  const board: Board = body.board ?? { myAthlete: null, follows: [] };
  const id = await createAccount(board);
  if (!id) return NextResponse.json({ error: "no-database" }, { status: 503 });
  await setSession(id);
  return NextResponse.json({ claimed: true });
}

export async function PUT(req: Request) {
  const id = await getSessionAccountId();
  if (!id) return NextResponse.json({ error: "not-claimed" }, { status: 401 });
  const body = (await req.json().catch(() => ({}))) as { board?: Board };
  if (!body.board) return NextResponse.json({ error: "no-board" }, { status: 400 });
  await updateBoard(id, body.board);
  return NextResponse.json({ ok: true });
}
