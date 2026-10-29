import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { DatasetListItem } from "../types";

export function useDatasetLibrary() {
  const [items, setItems] = useState<DatasetListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const generation = useRef(0);
  const pending = useRef(false);
  const load = useCallback(async (cursor?: string) => {
    if (cursor && pending.current) return;
    const request = ++generation.current;
    pending.current = true;
    setLoading(true);
    setError(null);
    if (!cursor) {
      setItems(null);
      setNextCursor(null);
    }
    try {
      const page = await api.listDatasets(cursor);
      if (request !== generation.current) return;
      setItems((previous) => (cursor ? [...(previous ?? []), ...page.items] : page.items));
      setNextCursor(page.next_cursor);
    } catch (e) {
      if (request === generation.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (request === generation.current) {
        pending.current = false;
        setLoading(false);
      }
    }
  }, []);
  const refresh = useCallback(() => load(), [load]);
  const loadMore = useCallback(() => (nextCursor ? load(nextCursor) : Promise.resolve()), [load, nextCursor]);
  useEffect(() => {
    void refresh();
    return () => {
      generation.current++;
      pending.current = false;
    };
  }, [refresh]);
  return { items, error, loading, nextCursor, refresh, loadMore };
}
