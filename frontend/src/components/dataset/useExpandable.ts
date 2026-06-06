/**
 * useExpandable — persist a per-section expanded boolean in localStorage
 * so a user who always wants the schema visible doesn't have to re-expand
 * on every job. Keyed by section name, not by job id.
 */
import { useCallback, useEffect, useState } from 'react';

const STORAGE_PREFIX = 'data-panel.expanded.';

export function useExpandable(sectionKey: string, defaultExpanded = false): {
  expanded: boolean;
  toggle: () => void;
} {
  const storageKey = STORAGE_PREFIX + sectionKey;
  const [expanded, setExpanded] = useState<boolean>(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      return raw === null ? defaultExpanded : raw === 'true';
    } catch {
      return defaultExpanded;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, String(expanded));
    } catch {
      // Quota / disabled storage — silently no-op.
    }
  }, [storageKey, expanded]);

  const toggle = useCallback(() => setExpanded((e) => !e), []);
  return { expanded, toggle };
}
