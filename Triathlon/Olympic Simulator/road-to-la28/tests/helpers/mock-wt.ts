/** A fetch stub for the World Triathlon API. Routes are matched by URL
 *  substring; the value is the FULL response envelope the WT API returns.
 *  Unknown URLs throw loudly so no test silently hits the network. */
import { vi } from "vitest";

/** Wrap a `data` payload in the WT envelope shape wtGet expects. */
export function wtOk<T>(data: T) {
  return { code: 200, status: "success", data };
}

export interface WtRoute {
  /** substring the request URL must contain */
  match: string;
  /** either a full envelope, or a function returning one (for status control) */
  response: unknown | ((url: string) => { body: unknown; status?: number });
}

export function makeWtFetch(routes: WtRoute[]) {
  return vi.fn(async (input: string | URL | Request) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    for (const r of routes) {
      if (url.includes(r.match)) {
        if (typeof r.response === "function") {
          const { body, status = 200 } = (r.response as (u: string) => { body: unknown; status?: number })(url);
          return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
        }
        return new Response(JSON.stringify(r.response), { status: 200, headers: { "content-type": "application/json" } });
      }
    }
    throw new Error(`unmocked WT url: ${url}`);
  });
}

/** Install the stub as global fetch for the current test; returns the mock. */
export function installWtFetch(routes: WtRoute[]) {
  const fn = makeWtFetch(routes);
  vi.stubGlobal("fetch", fn);
  return fn;
}
