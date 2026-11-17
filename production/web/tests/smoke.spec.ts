import { test, expect } from "@playwright/test";

for (const [path, heading] of [
  ["/?date=2026-11-16", "Market Overview"],
  ["/screener?date=2026-11-16", "Liquidity Screener"],
  ["/stock?date=2026-11-16&ticker=RELIANCE.NS", "RELIANCE"],
  ["/anomalies?date=2026-11-16", "Anomaly Surveillance"],
  ["/events?date=2020-03-23", "Event Intelligence"],
  ["/status", "Data & System Status"],
]) {
  test("production page " + heading, async ({ page }, info) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));

    await page.goto(path);
    await expect(page.locator("h1")).toHaveText(heading, { timeout: 30000 });
    await expect(page.locator(".loading-panel")).toHaveCount(0, {
      timeout: 30000,
    });
    await expect(page.locator(".error-panel")).toHaveCount(0);
    await page.screenshot({
      path: `test-results/${info.project.name}-${path.split("?")[0].replaceAll("/", "") || "overview"}.png`,
      fullPage: true,
      animations: "disabled",
    });
    const width = await page.evaluate(
      () => document.documentElement.scrollWidth,
    );
    expect(width).toBeLessThanOrEqual((page.viewportSize()?.width || 1440) + 2);
    expect(errors).toEqual([]);
  });
}

test("historical selection and filtered screener use requested dates", async ({
  page,
  request,
}) => {
  const response = await request.get(
    "/api/screener?date=2020-03-23&cap=small&limit=10&sort=probability",
  );
  expect(response.ok()).toBeTruthy();
  const result = await response.json();
  expect(result.meta.selected_date).toBe("2020-03-23");
  expect(result.rows.length).toBeLessThanOrEqual(10);
  expect(
    result.rows.every((row: any) => row.cap_segment === "small"),
  ).toBeTruthy();
  for (const date of ["2026-02-30", "2026-11-08"]) {
    expect((await request.get("/api/overview?date=" + date)).status()).toBe(
      400,
    );
  }
  await page.goto("/screener?date=2026-11-16");
  await expect(page.locator("table tbody tr").first()).toBeVisible({
    timeout: 30000,
  });
  await page
    .getByRole("combobox", { name: "Capitalisation filter" })
    .selectOption("small");
  await expect(
    page.locator("table tbody tr").first().locator(".segment"),
  ).toHaveText("small", { timeout: 30000 });
  const expectedTicker = String(result.rows[0].ticker);
  await page
    .getByRole("textbox", { name: "Search stocks" })
    .fill(expectedTicker);
  await expect(page.locator("table tbody tr")).toHaveCount(1, {
    timeout: 30000,
  });
  await expect(page.locator("table tbody tr").first()).toContainText(
    expectedTicker.replace(".NS", ""),
  );
});
