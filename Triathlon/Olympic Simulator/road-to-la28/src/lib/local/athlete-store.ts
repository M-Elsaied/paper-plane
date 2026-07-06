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
};

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
