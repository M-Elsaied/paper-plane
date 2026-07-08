import { test, expect } from "@playwright/test";

test("war room lists nations and opens a nation's slot detail", async ({ page }) => {
  await page.goto("/war-room");
  await expect(page.getByRole("heading", { name: "NOC Slot War Room" })).toBeVisible();
  await expect(page.getByText("nations in play")).toBeVisible();

  // Open France's detail from its card.
  await page.getByRole("link", { name: /France/ }).first().click();
  await expect(page).toHaveURL(/\/war-room\/FRA$/);
  await expect(page.getByRole("heading", { name: "France" })).toBeVisible();
  await expect(page.getByText("Elite Men")).toBeVisible();
  await expect(page.getByText("Elite Women")).toBeVisible();
  await expect(page.getByText(/Mixed Relay pathway/)).toBeVisible();
  // The cap bites: France fields more contenders than places, so a cap line shows.
  await expect(page.getByText(/nation cap/i).first()).toBeVisible();
});

test("a cockpit links to its nation's war room", async ({ page }) => {
  await page.goto("/athlete/86042"); // Vilaca, POR
  await page.getByRole("link", { name: /POR/ }).first().click();
  await expect(page).toHaveURL(/\/war-room\/POR$/);
  await expect(page.getByRole("heading", { name: "Portugal" })).toBeVisible();
});
