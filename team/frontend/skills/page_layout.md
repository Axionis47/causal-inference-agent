# Skill: Page Layout

Load when: building pages or layout structure.

## Current pages

```
frontend/src/pages/
  HomePage.tsx       <- Job creation form, dataset URL input
  JobPage.tsx        <- Single job view: progress + results
  JobsListPage.tsx   <- List of all jobs with status
```

## Layout pattern

```tsx
// React + TypeScript + Tailwind
export default function MyPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <Header />
      <main>
        {/* Page content */}
      </main>
    </div>
  );
}
```

## Rules

1. All pages import Header from components/common/Header.tsx
2. Use Tailwind for styling — no CSS modules or styled-components
3. Use React Router for navigation
4. State management through Zustand hooks (useJob, useCreateJob, etc.)
5. Every page handles: loading, error, empty, and populated states
