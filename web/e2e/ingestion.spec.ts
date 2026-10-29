import { test, expect } from "@playwright/test";

test("Phases 1-3: save datasets, return, profile columns and retrieve exact originals", async ({ page, request }) => {
  const saved: { id: string; title: string; csv: Buffer; context: string }[] = [];
  for (const n of [1, 2]) {
    const csv = Buffer.from(`id,note\r\n${n},"hello, world"\r\n`);
    const context = ` Original study account ${n}\nid identifies a participant. note contains a supplied note.\nid and note describe the same participant.\nunchanged `;
    const title = `Browser study ${n}`;
    await page.goto("/new");
    await page.getByLabel("Dataset title").fill(title);
    await page.getByLabel("Original study context (optional)").fill(context);
    await page.getByLabel("CSV file").setInputFiles({ name: "browser.csv", mimeType: "text/csv", buffer: csv });
    const response = page.waitForResponse((r) => r.url().endsWith("/api/datasets") && r.request().method() === "POST");
    await page.getByRole("button", { name: "Save dataset" }).click();
    const received = await response;
    expect(received.status()).toBe(201);
    const record = await received.json();
    expect(record.initial_context).toBe(context);
    await expect(page).toHaveURL(new RegExp(`/datasets/${record.dataset_id}$`));
    await expect(page.getByRole("heading", { name: title })).toBeVisible();
    saved.push({ id: record.dataset_id, title, csv, context });
  }
  for (const record of saved) {
    await page.goto("/");
    await page.locator(".main").getByRole("link", { name: record.title, exact: true }).click();
    await page.reload();
    await expect(page.getByRole("heading", { name: record.title })).toBeVisible();
    await page.getByRole("button", { name: "Profile dataset", exact: true }).click();
    await expect(page.getByText("Profile status:")).toContainText("complete");
    await page.getByRole("button", { name: "note", exact: true }).click();
    await expect(page.locator(".dataset-profile table")).toContainText("hello, world");
    const profileResponse = await request.get(`/api/datasets/${record.id}/profile`);
    const profile = (await profileResponse.json()).profile;
    expect(profile.dataset.rows).toBe(1);
    await page.reload();
    await expect(page.getByText("Profile status:")).toContainText("complete");
    const afterRefresh = await request.get(`/api/datasets/${record.id}/profile`);
    expect((await afterRefresh.json()).profile).toEqual(profile);
    const link = page.getByRole("link", { name: "Download original CSV" });
    const downloaded = await request.get((await link.getAttribute("href"))!);
    expect(downloaded.status()).toBe(200);
    expect(await downloaded.body()).toEqual(record.csv);
    const metadata = await request.get(`/api/datasets/${record.id}`);
    expect((await metadata.json()).initial_context).toBe(record.context);
  }
});
