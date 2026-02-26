import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const records = JSON.parse(
  readFileSync(
    resolve(import.meta.dirname, "../../.runtime/context-acceptance.json"),
    "utf8",
  ),
);
const record = records.find(
  (r: { title: string }) => r.title === "Students Performance",
);
const root = `/datasets/${record.dataset.id}/context/${record.dataset.version.id}`;

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

test(
  "source notes upload, original download, and reopening",
  { tag: "@sources" },
  async ({ page }) => {
    await page.getByRole("button", { name: "Upload CSV", exact: true }).click();
    await page
      .getByLabel("Dataset title")
      .fill("Browser acceptance / source notes");
    await page.getByLabel("CSV file").setInputFiles({
      name: "browser.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("id,score\n01,12\n02,15\n"),
    });
    await page.getByRole("button", { name: "Save dataset" }).click();
    await page.getByRole("link", { name: "Open context workspace" }).click();
    await page.getByRole("button", { name: "Add source", exact: true }).click();
    await page.getByLabel("Source title").fill("Unicode notes");
    await page.getByLabel("Source format").selectOption("note");
    await page
      .getByLabel("Source notes")
      .fill(
        "Each row is one pupil. Café 🧪.\nThe score is recorded after the exam.",
      );
    await page
      .getByRole("button", { name: "Save source", exact: true })
      .click();
    await expect(
      page.getByRole("button", { name: /Unicode notes notes.txt/ }),
    ).toBeVisible();
    await page.reload();
    await page.getByRole("button", { name: /Unicode notes notes.txt/ }).click();
    await expect(page.getByRole("dialog").locator("pre")).toContainText(
      "Café 🧪.",
    );
    await page.getByRole("button", { name: "Close", exact: true }).click();
    const downloaded = page.waitForEvent("download");
    await page.getByRole("link", { name: "Original", exact: true }).click();
    const path = await (await downloaded).path();
    expect(readFileSync(path!, "utf8")).toBe(
      "Each row is one pupil. Café 🧪.\nThe score is recorded after the exam.",
    );
    await page.getByRole("button", { name: "Add source", exact: true }).click();
    await page.getByLabel("Source title").fill("Markdown codebook");
    const markdown = Buffer.from(
      "# Codebook\r\nScore: marks from 0 to 100. Café 🧪.\r\n",
    );
    await page.getByLabel("Text or Markdown file").setInputFiles({
      name: "codebook.md",
      mimeType: "text/markdown",
      buffer: markdown,
    });
    await page
      .getByRole("button", { name: "Save source", exact: true })
      .click();
    const item = page
      .locator(".source-list li")
      .filter({ hasText: "Markdown codebook" });
    await expect(item).toBeVisible();
    const fileDownload = page.waitForEvent("download");
    await item.getByRole("link", { name: "Original", exact: true }).click();
    expect(readFileSync((await (await fileDownload).path())!)).toEqual(
      markdown,
    );
  },
);

test("real prepared revision, citation, failed save, correction, history and mobile", async ({
  page,
}) => {
  await page.goto(root + "?section=dataset");
  await page.locator(".context-history-tool > summary").click();
  const latest = page.getByRole("link", {
    name: /Revision \d+ · .* · current/,
  });
  await latest.click();
  await expect(
    page.getByRole("heading", { name: "Dataset", exact: true }),
  ).toBeVisible();
  const originalUrl = page.url();
  const base = `/api/versions/${record.dataset.version.id}/context/revisions/`;
  const originalRevision = await (
    await page.request.get(
      base + new URL(originalUrl).searchParams.get("revision"),
    )
  ).json();
  await page
    .getByRole("button", { name: "Inspect Observations", exact: true })
    .click();
  const cards = page.getByTestId("claim-card");
  const first = cards.filter({ hasText: "Observations" }).first();
  await first
    .getByRole("button", { name: /study account: line/ })
    .first()
    .click();
  await expect(page.getByRole("dialog").locator("mark")).toContainText("row");
  await page.getByRole("button", { name: "Close", exact: true }).click();
  await first.getByRole("button", { name: "Correct", exact: true }).click();
  await page
    .getByLabel("Statement", { exact: true })
    .fill(
      "Each row records one student who sat the May 2026 exam at one school.",
    );
  await page
    .getByLabel("Reason for correction")
    .fill("Clarify the supplied observational unit during browser review.");
  await page
    .getByLabel("These original passages also support my correction.")
    .check();
  await page.getByRole("button", { name: "Stage change", exact: true }).click();
  await page
    .getByRole("dialog", { name: "Observations", exact: true })
    .getByRole("button", { name: "Close", exact: true })
    .click();
  await expect(
    page.getByText("1 unsaved statement changes.", { exact: true }),
  ).toBeVisible();
  page.once("dialog", (dialog) => dialog.dismiss());
  await page
    .getByRole("link", { name: "Dataset preview", exact: true })
    .click();
  await expect(page).toHaveURL(originalUrl);
  await expect(
    page.getByText("1 unsaved statement changes.", { exact: true }),
  ).toBeVisible();
  // Exercise a failed response without mutating the backend, then send the real save.
  await page.route("**/context/revisions", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({
        detail: "Temporary save failure for browser verification.",
      }),
    }),
  );
  await page.getByRole("button", { name: "Save draft", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Temporary save failure");
  await expect(
    page.getByText("1 unsaved statement changes.", { exact: true }),
  ).toBeVisible();
  await page.unroute("**/context/revisions");
  await page.getByLabel("Leave all remaining proposals unresolved").check();
  await page
    .getByRole("button", { name: "Save reviewed revision", exact: true })
    .click();
  await expect(page).not.toHaveURL(originalUrl);
  await expect(page.getByText(/All displayed changes are saved/)).toBeVisible();
  await expect(
    page.getByRole("link", { name: /Revision \d+ · reviewed · current/ }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page
      .getByText(
        "Each row records one student who sat the May 2026 exam at one school.",
        { exact: true },
      )
      .first(),
  ).toBeVisible();
  const reviewedUrl = page.url();
  await page.goto(originalUrl);
  await expect(
    page.getByText(
      "Historical revision. Open the current revision to make changes.",
    ),
  ).toBeVisible();
  expect(
    await (await page.request.get(base + originalRevision.id)).json(),
  ).toEqual(originalRevision);
  await page.goto(reviewedUrl);
  await page
    .getByRole("navigation", { name: "Context sections" })
    .getByRole("link", { name: "Relationships", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Relationships", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Inspect Relationship 1", exact: true })
    .click();
  const relationship = page.getByTestId("claim-card").first();
  const beforeRelationshipUrl = page.url();
  const relationshipId = (
    (await relationship.locator(".context-key").textContent()) ?? ""
  )
    .trim()
    .split(".")
    .at(-1)!;
  const before = await (
    await page.request.get(
      base + new URL(page.url()).searchParams.get("revision"),
    )
  ).json();
  expect(before.content.relationships).toHaveProperty(relationshipId);
  await relationship
    .getByRole("button", { name: "Correct", exact: true })
    .click();
  await page
    .getByLabel("Reason for correction")
    .fill(
      "Reviewed participants and relationship details against the cited account.",
    );
  await page
    .getByLabel("These original passages also support my correction.")
    .check();
  await page.getByRole("button", { name: "Stage change", exact: true }).click();
  await page
    .getByRole("dialog", { name: "Relationship details", exact: true })
    .getByRole("button", { name: "Close", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Save reviewed revision", exact: true })
    .click();
  await expect(page).not.toHaveURL(beforeRelationshipUrl);
  await expect(
    page.getByText("All displayed changes are saved.", { exact: true }),
  ).toBeVisible();
  const after = await (
    await page.request.get(
      base + new URL(page.url()).searchParams.get("revision"),
    )
  ).json();
  const replacement = Object.values(after.content.relationships).find(
    (r) => (r as { replaces: string }).replaces === relationshipId,
  ) as Record<string, unknown>;
  for (const key of [
    "value",
    "kind",
    "participants",
    "qualifications",
    "timing",
    "conditions",
    "directness",
    "absence_scope",
    "citations",
  ])
    expect(replacement[key]).toEqual(
      before.content.relationships[relationshipId][key],
    );
  await page
    .getByRole("navigation", { name: "Context sections" })
    .getByRole("link", { name: "Columns", exact: true })
    .click();
  await page.getByRole("combobox", { name: "Column", exact: true }).selectOption({
    label: "test preparation course",
  });
  await expect(
    page.getByRole("heading", { name: "test preparation course", exact: true }),
  ).toBeVisible();
  for (const group of [
    "Meaning",
    "Values",
    "Measurement",
    "Construction",
    "Missingness",
    "Qualifications",
  ])
    await expect(
      page.getByRole("region", { name: group, exact: true }),
    ).toBeVisible();
  // Overview shows compact cards, with full claims available on demand.
  await expect(page.getByTestId("claim-card")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Inspect Meaning", exact: true })
    .press("Enter");
  await expect(
    page.getByRole("dialog", { name: "Meaning", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close", exact: true }).click();
  await page
    .getByRole("region", { name: "Applicable study context" })
    .getByRole("button", { name: "Inspect", exact: true })
    .click();
  await expect(page.getByTestId("claim-card").first()).toBeVisible();
  await page.getByRole("button", { name: "Close", exact: true }).click();
  const related = page.getByRole("region", { name: "Related to", exact: true });
  await expect(related.locator(".participant-node").first()).toBeVisible();
  await page
    .getByRole("button", { name: "Participant definitions", exact: true })
    .click();
  await expect(
    page.getByRole("dialog", { name: "Participant definitions", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Save reviewed revision" }),
  ).toBeDisabled();
  // Correct a relationship in the column view, including optional classification.
  await related
    .getByRole("button", { name: "Inspect relationship 1", exact: true })
    .click();
  const relatedCard = page
    .getByRole("dialog", { name: "Relationship details", exact: true })
    .getByTestId("claim-card")
    .first();
  const originalRelatedValue = (await relatedCard
    .locator(".claim-value")
    .textContent())!.trim();
  await relatedCard
    .getByRole("button", { name: "Correct", exact: true })
    .click();
  await page.getByLabel("Relationship kind (optional)").selectOption("");
  await page
    .getByRole("textbox", { name: "Qualifications", exact: true })
    .fill("Preserve this account without resolving its classification.");
  await page
    .getByLabel("Reason for correction")
    .fill("Keep the source description; classification remains open.");
  await page
    .getByLabel("These original passages also support my correction.")
    .check();
  await page.getByRole("button", { name: "Stage change", exact: true }).click();
  await expect(relatedCard).toContainText(
    "Preserve this account without resolving its classification.",
  );
  await expect(relatedCard.locator(".eyebrow")).toHaveText("Relationship");
  const parent = new URL(page.url()).searchParams.get("revision");
  const beforeColumn = await (await page.request.get(base + parent)).json();
  await page
    .getByRole("dialog", { name: "Relationship details", exact: true })
    .getByRole("button", { name: "Close", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Save reviewed revision", exact: true })
    .click();
  await expect
    .poll(() => new URL(page.url()).searchParams.get("revision"))
    .not.toBe(parent);
  await page.reload();
  await related
    .locator(".relationship-summary")
    .filter({ hasText: originalRelatedValue })
    .click();
  const savedCard = page
    .getByRole("dialog", { name: "Relationship details", exact: true })
    .getByTestId("claim-card")
    .filter({
      hasText: "Preserve this account without resolving its classification.",
    })
    .filter({ hasText: originalRelatedValue });
  await expect(savedCard.locator(".claim-value")).toHaveText(
    originalRelatedValue,
  );
  const afterColumn = await (
    await page.request.get(
      base + new URL(page.url()).searchParams.get("revision"),
    )
  ).json();
  expect(afterColumn.content.columns).toEqual(beforeColumn.content.columns);
  expect(
    Object.values(afterColumn.content.relationships).some(
      (r) =>
        (r as { kind: string | null; qualifications: string }).kind === null &&
        (r as { qualifications: string }).qualifications ===
          "Preserve this account without resolving its classification.",
    ),
  ).toBe(true);
  const participantLink = savedCard
    .locator(".relationship-participants a")
    .first();
  const destination = await participantLink.getAttribute("href");
  await participantLink.focus();
  await page.keyboard.press("Enter");
  await expect
    .poll(() => new URL(page.url()).searchParams.get("section"))
    .toBe(new URL(destination!, page.url()).searchParams.get("section"));
  await page
    .getByRole("navigation", { name: "Context sections" })
    .getByRole("link", { name: "Columns", exact: true })
    .click();
  await page.getByRole("combobox", { name: "Column", exact: true }).selectOption({
    label: "test preparation course",
  });
  await page.screenshot({
    path: resolve(import.meta.dirname, "../../.runtime/context-review.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: resolve(import.meta.dirname, "../../.runtime/context-mobile.png"),
    fullPage: true,
  });
});
