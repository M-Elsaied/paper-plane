"use client";
/**
 * Client side of the account bridge. Local-first stays the source of truth on
 * this device; when the board is "claimed" we mirror it to the server so it
 * syncs across devices and can power push. A local flag avoids a network round
 * trip on every change.
 */
import { getMyAthlete, getFollows, setMyAthlete, toggleFollow, type StoredAthlete } from "./athlete-store";

const CLAIMED = "rtla28:claimed";

export interface Board {
  myAthlete: StoredAthlete | null;
  follows: StoredAthlete[];
}

export async function localBoard(): Promise<Board> {
  const [myAthlete, follows] = await Promise.all([getMyAthlete(), getFollows()]);
  return { myAthlete, follows };
}

export function isClaimedLocal(): boolean {
  try {
    return !!localStorage.getItem(CLAIMED);
  } catch {
    return false;
  }
}
function setClaimedLocal(v: boolean) {
  try {
    if (v) localStorage.setItem(CLAIMED, "1");
    else localStorage.removeItem(CLAIMED);
  } catch {
    /* no-op */
  }
}

export async function fetchAccount(): Promise<{ claimed: boolean; board: Board | null }> {
  const res = await fetch("/api/account", { cache: "no-store" });
  const data = await res.json();
  setClaimedLocal(!!data.claimed);
  return data;
}

export async function claimBoard(): Promise<boolean> {
  const board = await localBoard();
  const res = await fetch("/api/account", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ board }),
  });
  const ok = res.ok && (await res.json()).claimed;
  if (ok) setClaimedLocal(true);
  return ok;
}

/** Mirror the current local board to the server if claimed. Safe to call often. */
export async function syncBoardIfClaimed(): Promise<void> {
  if (!isClaimedLocal()) return;
  const board = await localBoard();
  await fetch("/api/account", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ board }),
  }).catch(() => {});
}

/** Pull the server board onto this device (after linking a new device). */
export async function pullBoardToLocal(board: Board): Promise<void> {
  if (board.myAthlete) await setMyAthlete(board.myAthlete);
  const current = await getFollows();
  const have = new Set(current.map((f) => f.athleteId));
  for (const f of board.follows) if (!have.has(f.athleteId)) await toggleFollow(f);
  setClaimedLocal(true);
}

export async function getRecoveryLink(): Promise<string | null> {
  const res = await fetch("/api/account/recovery", { cache: "no-store" });
  if (!res.ok) return null;
  return (await res.json()).url ?? null;
}
