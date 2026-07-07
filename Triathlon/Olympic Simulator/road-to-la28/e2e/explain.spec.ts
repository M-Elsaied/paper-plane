import { test, expect } from "@playwright/test";

test("explain-the-line renders plain-language reasoning + the 8 pathways", async ({ page }) => {
  await page.goto("/athlete/86042");
  await expect(page.getByText(/What the line means for them/i)).toBeVisible();
  // deterministic seed: Vilaca is inside the line
  await expect(page.getByText(/inside the 21 individual places/i)).toBeVisible();
  // expandable pathways explainer
  await page.getByText(/The 8 ways into the Games/i).click();
  await expect(page.getByText(/Individual Olympic Qualification Ranking/i)).toBeVisible();
});
