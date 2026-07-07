import { test, expect } from "@playwright/test";

test("pick an athlete → cockpit → persists across reload", async ({ page }) => {
  await page.goto("/");
  // dismiss the onboarding hint if present
  const gotIt = page.getByRole("button", { name: /let.?s go/i });
  if (await gotIt.isVisible().catch(() => false)) await gotIt.click();

  await page.getByPlaceholder(/search athletes/i).fill("Vilaca");
  await page.getByRole("button", { name: /Vasco Vilaca/i }).first().click();

  await expect(page).toHaveURL(/\/athlete\/86042/);
  await expect(page.getByRole("heading", { name: "Vasco Vilaca" })).toBeVisible();
  await expect(page.getByText("QUALIFYING", { exact: true })).toBeVisible();
  await expect(page.getByText("Olympic Rank", { exact: true })).toBeVisible();

  // idb-keyval persistence: home now offers to resume, linking to the cockpit
  await page.goto("/");
  const resume = page.getByRole("link", { name: /Continue tracking Vasco Vilaca/i });
  await expect(resume).toBeVisible();
  await expect(resume).toHaveAttribute("href", "/athlete/86042");
});
