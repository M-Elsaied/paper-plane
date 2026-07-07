import { test, expect } from "@playwright/test";

// Antonio Serrat Seoane (63428) sits just outside the men's line in seed data —
// a win at a top-tier race should vault him across it and fire the celebration.
test("simulator: a strong result crosses the line and celebrates", async ({ page }) => {
  await page.goto("/athlete/63428/simulate");
  await expect(page.getByRole("heading", { name: /What-if simulator/i })).toBeVisible();

  // pick the biggest race tier and slide to a win
  await page.getByRole("button", { name: /WTCS Final/i }).click();
  const slider = page.getByRole("slider");
  await slider.focus();
  for (let i = 0; i < 40; i++) await page.keyboard.press("ArrowLeft"); // drive to 1st

  // outcome banner reflects a big jump + the celebration copy appears
  await expect(page.getByText(/Crosses into the qualifying zone/i)).toBeVisible();
  await expect(page.getByText("QUALIFYING ZONE", { exact: true })).toBeVisible();
});
