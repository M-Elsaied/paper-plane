/** In-memory cookie jar standing in for Next's `cookies()` store. */
export interface FakeCookie {
  name: string;
  value: string;
}
export interface CookieJar {
  store: Map<string, string>;
  get(name: string): FakeCookie | undefined;
  set(name: string, value: string, _opts?: unknown): void;
  delete(name: string): void;
}

export function makeCookieJar(seed?: Map<string, string>): CookieJar {
  const store = seed ?? new Map<string, string>();
  return {
    store,
    get: (name) => (store.has(name) ? { name, value: store.get(name)! } : undefined),
    set: (name, value) => void store.set(name, value),
    delete: (name) => void store.delete(name),
  };
}
