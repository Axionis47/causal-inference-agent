import type { components } from "@/lib/api/schema";

type Task = components["schemas"]["AnalysisTaskView"];

export function taskProgress(tasks: readonly Task[]) {
  const roots = tasks.filter((task) => !task.scope.parent);
  const children = new Map<string, Task[]>();
  for (const task of tasks) {
    const parent = task.scope.parent;
    if (typeof parent !== "string" || !parent) continue;
    const siblings = children.get(parent) ?? [];
    siblings.push(task);
    children.set(parent, siblings);
  }
  const resolvedRoots = new Set(
    roots
      .filter((task) => {
        const widened = children.get(task.id) ?? [];
        return (
          (task.status === "accepted" && widened.length === 0) ||
          (task.status === "expanded" &&
            widened.length === 1 &&
            widened[0]?.status === "accepted")
        );
      })
      .map((task) => task.id),
  );
  // Scientific coverage belongs to root keys. Widening can use different budget
  // keys while its accepted child resolves the original scientific scope.
  const obligationOwners = new Map<string, Task[]>();
  const screenColumns = new Map<string, Task[]>();
  for (const task of roots) {
    for (const key of task.obligation_keys) {
      obligationOwners.set(key, [...(obligationOwners.get(key) ?? []), task]);
    }
    if (task.kind === "screen") {
      for (const column of task.column_ids) {
        screenColumns.set(column, [...(screenColumns.get(column) ?? []), task]);
      }
    }
  }
  const fullyResolved = (owners: Task[]) =>
    owners.every((task) => resolvedRoots.has(task.id));
  return {
    roots: roots.length,
    resolvedRoots: resolvedRoots.size,
    unresolvedRoots: roots.length - resolvedRoots.size,
    obligations: obligationOwners.size,
    resolvedObligations: [...obligationOwners.values()].filter(fullyResolved)
      .length,
    screening: {
      columns: screenColumns.size,
      resolved: [...screenColumns.values()].filter(fullyResolved).length,
    },
    widened: roots.filter((task) => task.status === "expanded").length,
    tasks: tasks.length,
    attempts: tasks.reduce(
      (count, task) => count + task.attempt_details.length,
      0,
    ),
  };
}
