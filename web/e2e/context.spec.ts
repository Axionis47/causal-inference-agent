import { test, expect } from "@playwright/test";

test("Phase 4: open saved column context, follow relationships and inspect exact source after refresh", async ({ page, request }) => {
  const datasets = (await (await request.get("/api/datasets")).json()).items;
  const record = datasets.find((d: { title: string }) => d.title === "Browser study 1");
  expect(record).toBeTruthy();
  const path = `/datasets/${record.dataset_id}`;
  await page.goto(path);
  await page.getByRole("button", { name: "View context", exact: true }).click();
  const panel = page.getByRole("complementary", { name: "Organized context" });
  await panel.getByRole("button", { name: "note", exact: true, expanded: false }).click();
  const column = panel.getByLabel("Context for note");
  await expect(column).toContainText("note contains a supplied note.");
  await column.getByRole("button", { name: "View source 1", exact: true }).first().click();
  await expect(panel.locator("mark")).toHaveText("note contains a supplied note.");
  await column.getByRole("button", { name: "id", exact: true }).click();
  await expect(panel.getByLabel("Context for id")).toContainText("id identifies a participant.");
  await page.reload();
  await page.getByRole("button", { name: "View context", exact: true }).click();
  await panel.getByRole("button", { name: "note", exact: true, expanded: false }).click();
  await expect(panel.getByLabel("Context for note")).toContainText("note contains a supplied note.");
  await panel.getByRole("button", { name: "Close context", exact: true }).click();
  await expect(panel).toHaveCount(0);
});
