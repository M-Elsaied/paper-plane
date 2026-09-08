import { test, expect } from "@playwright/test";

test("explain-the-line renders plain-language reasoning + the pathway routes", async ({ page }) => {
  // The cockpit tour auto-starts on a fresh context and its overlay intercepts
  // clicks; mark it seen so the pathways explainer is clickable.
  await page.addInitScript(() => localStorage.setItem("tour:cockpit", "1"));
  await page.goto("/athlete/86042");
  await expect(page.getByText(/What the line means for them/i)).toBeVisible();
  // deterministic seed: Vilaca is inside the line
  await expect(page.getByText(/inside the 21 individual places/i)).toBeVisible();
  // expandable pathways explainer
  await page.getByText(/The \d+ routes into the Games/i).click();
  await expect(page.getByText(/Individual Olympic Qualification Ranking/i)).toBeVisible();
});
