import { test, expect } from "@playwright/test";

// Requires the `db` project (TEST_DATABASE_URL). Two contexts make picks; the
// crowd forecast should reflect both.
test("pick-em: lock a podium, second player sees the crowd grow", async ({ browser }) => {
  const a = await browser.newContext();
  const pageA = await a.newPage();
  await pageA.goto("/race/195148");
  await expect(pageA.getByText("Call the podium")).toBeVisible();
  // tap the first three contenders into the podium
  const rows = pageA.locator('button:has-text("+")');
  for (let i = 0; i < 3; i++) await rows.nth(i).click();
  await pageA.getByRole("button", { name: /Lock in your call/i }).click();
  await expect(pageA.getByText(/Locked in/i)).toBeVisible();

  const b = await browser.newContext();
  const pageB = await b.newPage();
  await pageB.goto("/race/195148");
  const rowsB = pageB.locator('button:has-text("+")');
  for (let i = 1; i < 4; i++) await rowsB.nth(i).click();
  await pageB.getByRole("button", { name: /Lock in your call/i }).click();

  await expect(pageB.getByText(/Who the crowd is calling to win/i)).toBeVisible();
  await expect(pageB.getByText(/2 calls/i)).toBeVisible();

  await a.close();
  await b.close();
});
