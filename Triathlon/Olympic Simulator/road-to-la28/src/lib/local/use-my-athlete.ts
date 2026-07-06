"use client";
/**
 * SSR-safe React hooks over the local-first store. `hydrated` guards against
 * rendering device state during SSR (which would mismatch), so callers can show
 * a skeleton until the first client read completes.
 */
import { useCallback, useEffect, useState } from "react";
import {
  getMyAthlete,
  setMyAthlete,
  clearMyAthlete,
  getFollows,
  toggleFollow,
  type StoredAthlete,
} from "./athlete-store";

export function useMyAthlete() {
  const [athlete, setAthlete] = useState<StoredAthlete | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    getMyAthlete().then((a) => {
      setAthlete(a);
      setHydrated(true);
    });
  }, []);

  const choose = useCallback(async (a: StoredAthlete) => {
    await setMyAthlete(a);
    setAthlete(a);
  }, []);

  const clear = useCallback(async () => {
    await clearMyAthlete();
    setAthlete(null);
  }, []);

  return { athlete, hydrated, choose, clear };
}

export function useFollows() {
  const [follows, setFollows] = useState<StoredAthlete[]>([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    getFollows().then((f) => {
      setFollows(f);
      setHydrated(true);
    });
  }, []);

  const toggle = useCallback(async (a: StoredAthlete) => {
    const next = await toggleFollow(a);
    setFollows(next);
  }, []);

  const isFollowing = useCallback(
    (id: number) => follows.some((f) => f.athleteId === id),
    [follows],
  );

  return { follows, hydrated, toggle, isFollowing };
}
