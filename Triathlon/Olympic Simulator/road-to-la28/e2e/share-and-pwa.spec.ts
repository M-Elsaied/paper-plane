import { test, expect } from "@playwright/test";

test("share card renders as a 1080x1350 PNG", async ({ request }) => {
  const res = await request.get("/athlete/86042/card");
  expect(res.status()).toBe(200);
  expect(res.headers()["content-type"]).toContain("image/png");
  const buf = await res.body();
  expect(buf.length).toBeGreaterThan(30_000); // catches satori blank-render
  // PNG IHDR: width/height are big-endian uint32 at bytes 16..24
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  expect(width).toBe(1080);
  expect(height).toBe(1350);
});

test("manifest and service worker are served", async ({ request }) => {
  const manifest = await request.get("/manifest.webmanifest");
  expect(manifest.status()).toBe(200);
  const json = await manifest.json();
  expect(json.name).toContain("Road to LA28");
  expect(json.icons.length).toBeGreaterThan(0);

  const sw = await request.get("/sw.js");
  expect(sw.status()).toBe(200);
  expect(sw.headers()["content-type"]).toContain("javascript");
});

test("service worker registers in the browser", async ({ page }) => {
  await page.goto("/");
  const ready = await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) return false;
    await navigator.serviceWorker.register("/sw.js");
    const reg = await navigator.serviceWorker.ready;
    return !!reg;
  });
  expect(ready).toBe(true);
});

test("unknown athlete → 404 page", async ({ page }) => {
  await page.goto("/athlete/999999999");
  await expect(page.getByText("404")).toBeVisible();
});
