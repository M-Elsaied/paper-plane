import { test, expect } from "@playwright/test";

test("rankings show the qualification line and toggle genders", async ({ page }) => {
  await page.goto("/rankings");
  await expect(page.getByRole("heading", { name: /Olympic Qualification Ranking/i })).toBeVisible();
  // scope to the mobile board (desktop 2-col grid duplicates names, hidden here)
  const board = page.getByTestId("rankings-mobile");
  await expect(board.getByText(/Qualification line/i)).toBeVisible();
  await expect(board.getByText("Vasco Vilaca")).toBeVisible();

  // toggle to women — the men's leader is no longer in the mobile list
  await board.getByRole("button", { name: "Elite Women" }).click();
  await expect(board.getByText("Vasco Vilaca")).toHaveCount(0);
});

test("bottom nav reaches all tabs", async ({ page }) => {
  await page.goto("/");
  for (const [name, url] of [
    ["Rankings", /\/rankings/],
    ["Race Week", /\/race-week/],
    ["Pulse", /\/pulse/],
    ["Relay", /\/relay/],
    ["War Room", /\/war-room/],
  ] as const) {
    await page.getByRole("link", { name, exact: false }).first().click();
    await expect(page).toHaveURL(url);
  }
});
