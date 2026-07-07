import { test, expect } from "@playwright/test";

test("cockpit tour auto-starts once, then is replayable", async ({ page }) => {
  await page.goto("/athlete/86042");
  // auto-start (fresh context has no tour:cockpit flag)
  await expect(page.getByText("Are they going to the Olympics?")).toBeVisible({ timeout: 5000 });
  // step through
  for (let i = 0; i < 5; i++) await page.getByRole("button", { name: /^Next$/ }).click();
  await page.getByRole("button", { name: /Got it/i }).click();
  await expect(page.getByText("Are they going to the Olympics?")).toHaveCount(0);

  // reload → does not auto-start again
  await page.reload();
  await expect(page.getByText("Are they going to the Olympics?")).toHaveCount(0);

  // replay via the button
  await page.getByRole("button", { name: /Take the tour/i }).click();
  await expect(page.getByText("Are they going to the Olympics?")).toBeVisible();
});

test("simulator tour auto-starts", async ({ page }) => {
  await page.goto("/athlete/86042/simulate");
  await expect(page.getByText("Set the finish position")).toBeVisible({ timeout: 5000 });
});
