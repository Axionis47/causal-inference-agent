import { test, expect } from "@playwright/test";

test("question, explicit review, warm opening, column focus and durable correction", async ({ page, request }) => {
  const datasets = (await (await request.get("/api/datasets")).json()).items;
  const dataset = datasets.find((d: { title: string }) => d.title === "Browser study 1");
  const extracted = (await (await request.get(`/api/datasets/${dataset.dataset_id}/context`)).json()).pack;
  await page.goto(`/datasets/${dataset.dataset_id}`);
  const start = page.getByRole("button", { name: "Review context for this question" });
  await expect(start).toBeDisabled();
  await page.getByLabel("What question would you like to explore?").fill("What do these notes describe?");
  await start.click();
  await expect(page.getByRole("heading", { name: "Review context before chat" })).toBeVisible();
  const review = await (await request.get(`/api${new URL(page.url()).pathname}`)).json();
  expect(review.starting_revision_id).toBe(extracted.revision_id);
  await expect(page.getByRole("button", { name: "Confirm context and start chat" })).toBeDisabled();
  await page.getByRole("button", { name: "Leave remaining columns unresolved" }).click();
  await page.getByRole("button", { name: "Confirm context and start chat" }).click();
  await expect(page.getByRole("heading", { name: "What do these notes describe?" })).toBeVisible();
  await expect(page.getByLabel("Conversation")).toContainText("Your reviewed context is saved");
  const select = page.getByLabel("Inspect", { exact: true });
  await select.selectOption({ label: "note" });
  await expect(page.getByText("note contains a supplied note.", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Clear focus" })).toHaveCount(0);
  await page.getByRole("button", { name: "Discuss this column" }).click();
  await expect(page.getByRole("button", { name: "Clear focus" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Edit statement" }).first()).toBeEnabled();
  await page
    .locator("article")
    .filter({ has: page.getByText("note contains a supplied note.", { exact: true }) })
    .getByRole("button", { name: "Edit statement" })
    .click();
  await page.getByLabel("Description", { exact: true }).fill("A supplied note about the participant.");
  await page.getByLabel("Reason for the change").fill("Clarified the supplied meaning.");
  await page.getByRole("button", { name: "Save correction" }).click();
  await expect(page.getByLabel("Conversation")).toContainText("Context updated to revision");
  await page.reload();
  await select.selectOption({ label: "note" });
  await expect(page.getByText("A supplied note about the participant.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Clear focus" })).toBeVisible();
  await page.getByRole("button", { name: "Artifacts", exact: true }).click();
  await expect(page.getByText("No artifacts in this chat yet. Ask for a visualization.")).toBeVisible();
  const chatId = new URL(page.url()).pathname.split("/").pop();
  const chat = await (await request.get(`/api/conversations/${chatId}`)).json();
  const current = (await (await request.get(`/api/datasets/${dataset.dataset_id}/context`)).json()).pack;
  expect(current.revision_id).toBe(chat.revision_id);
  await page.goto(`/datasets/${dataset.dataset_id}`);
  await page.getByRole("button", { name: "View context", exact: true }).click();
  const panel = page.getByRole("complementary", { name: "Organized context" });
  await panel.getByRole("button", { name: "note", exact: true, expanded: false }).click();
  await expect(panel.getByLabel("Context for note")).toContainText("A supplied note about the participant.");
});
