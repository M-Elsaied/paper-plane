import { test, expect } from "@playwright/test";

// Hamburg (195148) — fixture start list served by the WT fixture server.
test("race companion loads the official start list and projects a ranking", async ({ page }) => {
  await page.goto("/race/195148");
  await expect(page.getByRole("heading", { name: /Live Race Companion/i })).toBeVisible();
  await expect(page.getByText(/Official start list/i)).toBeVisible();
  await expect(page.getByText("Projected finish")).toBeVisible();
  await expect(page.getByText(/Projected ranking · Monday/i)).toBeVisible();
  // the qualification line is drawn in the projected board
  await expect(page.getByText(/line at #\d+/i)).toBeVisible();
});
