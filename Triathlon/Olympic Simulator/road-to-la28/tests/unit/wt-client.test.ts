import { describe, it, expect, afterEach, vi } from "vitest";
import { wtGet, WtApiError } from "@/lib/wt-api/client";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("wtGet", () => {
  it("sends the apikey header and drops undefined params", async () => {
    const spy = vi.fn(async () => json({ code: 200, status: "ok", data: { ok: 1 } }));
    vi.stubGlobal("fetch", spy);
    await wtGet("/rankings/11", { limit: 1000, foo: undefined });
    const [url, init] = spy.mock.calls[0] as unknown as [URL, RequestInit];
    expect((init.headers as Record<string, string>).apikey).toBeTruthy();
    expect(String(url)).toContain("limit=1000");
    expect(String(url)).not.toContain("foo");
  });

  it("uses WT_API_KEY when the env var is set", async () => {
    vi.stubEnv("WT_API_KEY", "my-key");
    const spy = vi.fn(async () => json({ data: {} }));
    vi.stubGlobal("fetch", spy);
    await wtGet("/x");
    const init = (spy.mock.calls[0] as unknown[])[1] as { headers: Record<string, string> };
    expect(init.headers.apikey).toBe("my-key");
  });

  it("throws WtApiError on 404 without retrying", async () => {
    const spy = vi.fn(async () => new Response("nope", { status: 404 }));
    vi.stubGlobal("fetch", spy);
    await expect(wtGet("/x")).rejects.toBeInstanceOf(WtApiError);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("retries a 429 then succeeds", async () => {
    let n = 0;
    const spy = vi.fn(async () => (++n === 1 ? new Response("busy", { status: 429 }) : json({ data: { ok: 1 } })));
    vi.stubGlobal("fetch", spy);
    const res = await wtGet<{ ok: number }>("/x");
    expect(res.data.ok).toBe(1);
    expect(spy).toHaveBeenCalledTimes(2);
  }, 15000);

  it("throws after exhausting retries on repeated 500s", async () => {
    const spy = vi.fn(async () => new Response("err", { status: 500 }));
    vi.stubGlobal("fetch", spy);
    await expect(wtGet("/x")).rejects.toBeTruthy();
    expect(spy.mock.calls.length).toBeGreaterThanOrEqual(2);
  }, 20000);
});
