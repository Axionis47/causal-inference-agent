import { test, expect } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const fixture = resolve(
  import.meta.dirname,
  "../../.runtime/separation-chat-acceptance.json",
);
const record = existsSync(fixture)
  ? JSON.parse(readFileSync(fixture, "utf8"))
  : null;
test.skip(!record, "Run the saved real-model Chat acceptance journey first.");

test.beforeEach(async ({ page }) => {
  await page.goto("/login");
  await page
    .getByLabel("Email address", { exact: true })
    .fill(process.env.BOOTSTRAP_EMAIL!);
  await page
    .getByLabel("Password", { exact: true })
    .fill(process.env.BOOTSTRAP_PASSWORD!);
  await page.getByRole("button", { name: "Log in", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Dataset library." }),
  ).toBeVisible();
});

test("workspace switching preserves drafts and context returns to the same saved chat", async ({
  page,
}) => {
  const path = `/datasets/${record.dataset.id}/chat/${record.dataset.version.id}`;
  const before = await (
    await page.request.get(`/api/conversations/${record.conversation.id}`)
  ).json();
  await page.goto(`${path}?conversation=${record.conversation.id}`);
  await expect(
    page.getByRole("region", { name: "Conversation", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("complementary", { name: "Analysis workspace" }),
  ).toBeHidden();
  const message = page.getByLabel("Your message", { exact: true });
  await message.fill("Keep this unsent draft while I inspect analysis.");
  const dialogs: string[] = [];
  page.on("dialog", async (dialog) => {
    dialogs.push(dialog.message());
    await dialog.dismiss();
  });
  const tabs = page.getByRole("navigation", { name: "Dataset workspace" });
  await tabs.getByRole("link", { name: "Analysis", exact: true }).click();
  await expect(
    page.getByRole("complementary", { name: "Analysis workspace" }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Conversation", exact: true }),
  ).toBeHidden();
  await tabs.getByRole("link", { name: "Chat", exact: true }).click();
  await expect(message).toHaveValue(
    "Keep this unsent draft while I inspect analysis.",
  );
  expect(dialogs).toEqual([]);
  await message.fill("");
  await page
    .getByRole("link", { name: "Inspect selected context", exact: true })
    .click();
  await expect(
    page.getByRole("link", { name: "Back to conversation", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Columns", exact: true }).click();
  await page
    .getByLabel("Column", { exact: true })
    .selectOption({ label: "lunch" });
  await page.reload();
  expect(new URL(page.url()).searchParams.get("fromConversation")).toBe(
    record.conversation.id,
  );
  await page
    .getByRole("link", { name: "Back to conversation", exact: true })
    .click();
  await expect(message).toBeVisible();
  expect(new URL(page.url()).searchParams.get("conversation")).toBe(
    record.conversation.id,
  );
  const after = await (
    await page.request.get(`/api/conversations/${record.conversation.id}`)
  ).json();
  expect(after).toEqual(before);
  for (const width of [1440, 1024, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    if (width === 1440)
      expect(
        (await page.locator(".chat-thread").boundingBox())!.width,
      ).toBeGreaterThan(900);
    const nav = page.getByRole("navigation", { name: "Dataset workspace" });
    await expect(
      nav.getByRole("link", { name: "Data", exact: true }),
    ).toBeVisible();
    await expect(
      nav.getByRole("link", { name: "Context", exact: true }),
    ).toBeVisible();
    await expect(
      nav.getByRole("link", { name: "Chat", exact: true }),
    ).toBeVisible();
  }
});

test("analysis deep links keep diagnostic failures behind technical details", async ({
  page,
}) => {
  const analysisId = "21047ae3-4f67-5e41-956b-fbd4c2546702";
  const response = await page.request.get(`/api/analyses/${analysisId}`);
  test.skip(
    response.status() === 404,
    "The preserved contract-19 failure fixture is not present.",
  );
  expect(response.ok()).toBe(true);
  const before = await response.json();
  await page.goto(
    `/datasets/${before.dataset_id}/chat/${before.version_id}?conversation=${before.conversation_id}&analysis=${analysisId}`,
  );
  await expect(
    page.getByText("Analysis interrupted", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Resume saved work", exact: true }),
  ).toBeHidden();
  await expect(
    page.getByText(/This analysis uses a historical contract and is read-only/),
  ).toBeVisible();
  const checkpoints = page.getByRole("heading", {
    name: "Scientific checkpoints",
    exact: true,
  });
  await expect(checkpoints).toBeHidden();
  await expect(
    page.getByRole("heading", {
      name: "Shared provider attempts",
      exact: true,
    }),
  ).toBeHidden();
  await page.getByText("Technical details", { exact: true }).click();
  await expect(checkpoints).toBeVisible();
  await page
    .getByRole("button", {
      name: "03 Establish treatments failed",
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("region", {
      name: "Establish treatments inspector",
      exact: true,
    }),
  ).toContainText("Analysis interrupted (ClientError)");
  await page.getByText("Technical details", { exact: true }).click();
  await expect(checkpoints).toBeHidden();
  const after = await (
    await page.request.get(`/api/analyses/${analysisId}`)
  ).json();
  expect(after).toEqual(before);
});
