import { test, expect } from "@playwright/test";

test("rankings show the qualification line and toggle genders", async ({ page }) => {
  await page.goto("/rankings");
  await expect(page.getByRole("heading", { name: /Olympic Qualification Ranking/i })).toBeVisible();
  // the line separator with its label
  await expect(page.getByText(/Qualification line · 21 individual places/i)).toBeVisible();
  // men's leader present
  await expect(page.getByText("Vasco Vilaca")).toBeVisible();

  // toggle to women — a different board renders
  await page.getByRole("button", { name: "Elite Women" }).click();
  await expect(page.getByText("Vasco Vilaca")).toHaveCount(0);
});

test("bottom nav reaches all five tabs", async ({ page }) => {
  await page.goto("/");
  for (const [name, url] of [
    ["Rankings", /\/rankings/],
    ["Race Week", /\/race-week/],
    ["Pulse", /\/pulse/],
    ["Relay", /\/relay/],
  ] as const) {
    await page.getByRole("link", { name, exact: false }).first().click();
    await expect(page).toHaveURL(url);
  }
});
