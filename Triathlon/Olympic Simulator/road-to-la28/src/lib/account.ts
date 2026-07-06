/**
 * Account bridge data access — a synced "board" keyed by an opaque account id.
 * No PII: the board is just the user's chosen athlete + follows.
 */
import "server-only";
import { randomUUID } from "node:crypto";
import { eq } from "drizzle-orm";
import { getDb } from "@/db/client";
import { accounts } from "@/db/schema";

export interface StoredAthlete {
  athleteId: number;
  fullName: string;
  noc: string;
  gender: "male" | "female";
}
export interface Board {
  myAthlete: StoredAthlete | null;
  follows: StoredAthlete[];
}

const EMPTY: Board = { myAthlete: null, follows: [] };

export async function createAccount(board: Board): Promise<string | null> {
  const db = getDb();
  if (!db) return null;
  const id = randomUUID();
  await db.insert(accounts).values({ id, board });
  return id;
}

export async function getBoard(id: string): Promise<Board | null> {
  const db = getDb();
  if (!db) return null;
  const [row] = await db.select().from(accounts).where(eq(accounts.id, id)).limit(1);
  if (!row) return null;
  return (row.board as Board) ?? EMPTY;
}

export async function updateBoard(id: string, board: Board): Promise<boolean> {
  const db = getDb();
  if (!db) return false;
  const res = await db
    .update(accounts)
    .set({ board, updatedAt: new Date() })
    .where(eq(accounts.id, id));
  return true;
}

export function followIds(board: Board): number[] {
  const ids = board.follows.map((f) => f.athleteId);
  if (board.myAthlete) ids.push(board.myAthlete.athleteId);
  return [...new Set(ids)];
}
