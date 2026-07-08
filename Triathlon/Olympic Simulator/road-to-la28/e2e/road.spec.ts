import { test, expect } from "@playwright/test";

test("Your Road shows a ranked athlete's route portfolio + game board", async ({ page }) => {
  await page.goto("/athlete/86042/road"); // Vasco Vilaca (POR), ranked #1
  await expect(page.getByRole("heading", { name: /Your Road to LA28/i })).toBeVisible();
  await expect(page.getByText(/Every route, ranked by realism/i)).toBeVisible();
  // the individual route is his clearest, on track
  await expect(page.getByText("Individual Olympic Ranking").first()).toBeVisible();
  await expect(page.getByText("ON TRACK").first()).toBeVisible();
  // the game board renders
  await expect(page.getByText("The game board")).toBeVisible();
});

test("Your Road gives an unranked athlete honest New Flag routes", async ({ page }) => {
  await page.goto("/athlete/70338/road"); // Mohamed Elsaied (EGY), unranked
  await expect(page.getByRole("heading", { name: "Mohamed Elsaied" })).toBeVisible();
  await expect(page.getByText(/New Flag/).first()).toBeVisible();
  await expect(page.getByText("CLOSED").first()).toBeVisible(); // individual route closed
  await expect(page.getByText(/Africa/).first()).toBeVisible();
});
