/**
 * Local-first personalization store (extractable module — no app imports).
 *
 * Persists the user's "my athlete", their follow list, and the last-seen
 * ranking snapshot entirely on-device via idb-keyval. No accounts, no server
 * state. Everything is namespaced so this module can be lifted into any app.
 */
import { get, set, del } from "idb-keyval";

const NS = "rtla28";
const K = {
  myAthlete: `${NS}:my-athlete`,
  follows: `${NS}:follows`,
  lastSeen: `${NS}:last-seen-snapshot`,
  rivalries: `${NS}:rivalries`,
};

export interface RivalRef {
  athleteId: number;
  name: string;
}
export interface Rivalry {
  a: RivalRef;
  b: RivalRef;
}
/** A stable, order-independent key for a pair of athletes. */
export function rivalryKey(x: number, y: number): string {
  return [x, y].sort((m, n) => m - n).join("-");
}

export interface StoredAthlete {
  athleteId: number;
  fullName: string;
  noc: string;
  gender: "male" | "female";
}

export async function getMyAthlete(): Promise<StoredAthlete | null> {
  return (await get<StoredAthlete>(K.myAthlete)) ?? null;
}
export async function setMyAthlete(a: StoredAthlete): Promise<void> {
  await set(K.myAthlete, a);
}
export async function clearMyAthlete(): Promise<void> {
  await del(K.myAthlete);
}

export async function getFollows(): Promise<StoredAthlete[]> {
  return (await get<StoredAthlete[]>(K.follows)) ?? [];
}
export async function toggleFollow(a: StoredAthlete): Promise<StoredAthlete[]> {
  const list = await getFollows();
  const exists = list.some((x) => x.athleteId === a.athleteId);
  const next = exists
    ? list.filter((x) => x.athleteId !== a.athleteId)
    : [...list, a];
  await set(K.follows, next);
  return next;
}

export async function getLastSeenSnapshot(): Promise<string | null> {
  return (await get<string>(K.lastSeen)) ?? null;
}
export async function setLastSeenSnapshot(id: string): Promise<void> {
  await set(K.lastSeen, id);
}

// ---- rivalries (pinned head-to-head duels) ----
export async function getRivalries(): Promise<Rivalry[]> {
  return (await get<Rivalry[]>(K.rivalries)) ?? [];
}
export async function toggleRivalry(pair: Rivalry): Promise<Rivalry[]> {
  const list = await getRivalries();
  const key = rivalryKey(pair.a.athleteId, pair.b.athleteId);
  const exists = list.some((r) => rivalryKey(r.a.athleteId, r.b.athleteId) === key);
  const next = exists
    ? list.filter((r) => rivalryKey(r.a.athleteId, r.b.athleteId) !== key)
    : [...list, pair];
  await set(K.rivalries, next);
  return next;
}
export async function isRivalry(aId: number, bId: number): Promise<boolean> {
  const list = await getRivalries();
  const key = rivalryKey(aId, bId);
  return list.some((r) => rivalryKey(r.a.athleteId, r.b.athleteId) === key);
}
