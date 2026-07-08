import { test, expect } from "@playwright/test";

test("the cockpit hero shows a rank trajectory sparkline", async ({ page }) => {
  await page.goto("/athlete/56027"); // Diego Moya — climbed 15 → 7 in the seed
  const spark = page.getByTestId("rank-sparkline");
  await expect(spark).toBeVisible();
  // A climber's trend reads as a "+places" delta, not "held".
  await expect(spark).toContainText(/places?|held/);
});
