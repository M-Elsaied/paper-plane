/**
 * World Triathlon API client — polite and resilient.
 * Plain fetch (works in Node scripts, cron route handlers, and RSC). Never
 * imports Next-specific code so the seed script can reuse it verbatim.
 */
import pLimit from "p-limit";
import { WT_API_BASE, WT_CLIENT, wtApiKey } from "@/config/wt-api";

const limit = pLimit(WT_CLIENT.concurrency);
let lastRequestAt = 0;

async function spacing() {
  const now = Date.now();
  const wait = Math.max(0, WT_CLIENT.minDelayMs - (now - lastRequestAt));
  if (wait > 0) await sleep(wait);
  lastRequestAt = Date.now();
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export interface WtResponse<T> {
  code: number;
  status: string;
  data: T;
  total?: number;
  last_page?: number;
  next_page_url?: string | null;
}

export class WtApiError extends Error {
  constructor(
    message: string,
    readonly httpStatus: number,
    readonly path: string,
  ) {
    super(message);
    this.name = "WtApiError";
  }
}

/** GET a path (relative to the API base) and return the parsed `data` payload. */
export async function wtGet<T>(
  path: string,
  params: Record<string, string | number | undefined> = {},
): Promise<WtResponse<T>> {
  return limit(async () => {
    const url = new URL(path.startsWith("http") ? path : `${WT_API_BASE}${path}`);
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }

    let lastErr: unknown;
    for (let attempt = 0; attempt <= WT_CLIENT.retries; attempt++) {
      await spacing();
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), WT_CLIENT.timeoutMs);
      try {
        const res = await fetch(url, {
          headers: { apikey: wtApiKey(), accept: "application/json" },
          signal: controller.signal,
        });
        clearTimeout(timer);
        if (res.status === 429 || res.status >= 500) {
          lastErr = new WtApiError(`HTTP ${res.status}`, res.status, url.pathname);
          await sleep(backoff(attempt));
          continue;
        }
        if (!res.ok) {
          throw new WtApiError(`HTTP ${res.status}`, res.status, url.pathname);
        }
        return (await res.json()) as WtResponse<T>;
      } catch (err) {
        clearTimeout(timer);
        lastErr = err;
        if (attempt < WT_CLIENT.retries) await sleep(backoff(attempt));
      }
    }
    throw lastErr instanceof Error
      ? lastErr
      : new WtApiError("request failed", 0, url.pathname);
  });
}

function backoff(attempt: number): number {
  const base = 400 * 2 ** attempt;
  return base + Math.floor((base / 2) * Math.sin(attempt + 1)); // deterministic jitter
}
