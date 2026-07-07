import { test, expect } from "@playwright/test";

// Requires the `db` project. Claim a board, then redeem the recovery link in a
// fresh context and confirm the board synced.
test("account: claim then recover on a fresh device", async ({ browser, baseURL }) => {
  const ctxA = await browser.newContext();
  const pageA = await ctxA.newPage();

  // pick an athlete first so the board isn't empty
  await pageA.goto("/");
  const gotIt = pageA.getByRole("button", { name: /let.?s go/i });
  if (await gotIt.isVisible().catch(() => false)) await gotIt.click();
  await pageA.getByPlaceholder(/search athletes/i).fill("Vilaca");
  await pageA.getByRole("button", { name: /Vasco Vilaca/i }).first().click();
  await expect(pageA).toHaveURL(/\/athlete\/86042/);

  // claim
  await pageA.goto("/account");
  await pageA.getByRole("button", { name: /Claim your board/i }).click();
  await expect(pageA.getByText(/Copy recovery link/i)).toBeVisible();

  // read the recovery link from the API (deterministic)
  const rec = await pageA.request.get("/api/account/recovery");
  const { url } = await rec.json();
  expect(url).toContain("/claim?t=");
  const token = new URL(url).searchParams.get("t")!;

  // fresh context redeems it
  const ctxB = await browser.newContext();
  const pageB = await ctxB.newPage();
  await pageB.goto(`/claim?t=${encodeURIComponent(token)}`);
  await expect(pageB).toHaveURL(/\/account\?linked=1/);
  // the fresh device's home now shows the synced athlete
  await pageB.goto("/");
  await expect(pageB.getByText("Vasco Vilaca")).toBeVisible();

  await ctxA.close();
  await ctxB.close();
});
