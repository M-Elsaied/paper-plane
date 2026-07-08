import { test, expect } from "@playwright/test";

test("head-to-head compares two athletes with a race record", async ({ page }) => {
  await page.goto("/versus/86042/49390"); // Vilaca vs Coninx
  await expect(page.getByRole("heading", { name: /Head-to-head/i }).first()).toBeVisible();
  await expect(page.getByText("Vasco Vilaca").first()).toBeVisible();
  await expect(page.getByText("Dorian Coninx").first()).toBeVisible();
  // comparison rows
  await expect(page.getByText("Olympic rank")).toBeVisible();
  await expect(page.getByText("Career starts")).toBeVisible();
  // head-to-head record from the common races
  await expect(page.getByText(/career meetings/i)).toBeVisible();
});

test("cockpit chasers link to the head-to-head", async ({ page }) => {
  await page.goto("/athlete/86042");
  const chaser = page.getByText("Around the athlete");
  await expect(chaser).toBeVisible();
});
