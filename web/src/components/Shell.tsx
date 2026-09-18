import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { api, ApiError } from "../api";
import { useNarrow } from "../hooks/useNarrow";
import { useStoredState } from "../hooks/useStoredState";
import { parseSidebar, serialiseSidebar, SIDEBAR_KEY } from "../layout";
import type { DatasetSummary } from "../types";
import Sidebar from "./Sidebar";

interface Datasets {
  items: DatasetSummary[] | null;
  error: string | null;
  refresh: () => Promise<void>;
}

const Ctx = createContext<Datasets>({ items: null, error: null, refresh: async () => {} });

/** The dataset list, shared by the sidebar and the pages. */
export const useDatasets = () => useContext(Ctx);

/** Sidebar on the left, the page in the middle. Every route renders inside this. */
export default function Shell() {
  const [items, setItems] = useState<DatasetSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    try {
      setItems(await api.listDatasets());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, []);
  const { pathname } = useLocation();
  useEffect(() => {
    void refresh();
  }, [refresh, pathname]);

  const narrow = useNarrow();
  const [pref, setPref] = useStoredState(SIDEBAR_KEY, parseSidebar, serialiseSidebar);
  const [overlay, setOverlay] = useState(false);
  const open = narrow ? overlay : pref;
  const toggle = () => (narrow ? setOverlay((o) => !o) : setPref((p) => !p));
  const onNav = () => narrow && setOverlay(false);

  const current = pathname.startsWith("/d/") ? decodeURIComponent(pathname.slice(3).split("/")[0]) : null;

  return (
    <div className="shell" data-side={open ? "open" : "closed"}>
      <Sidebar items={items} current={current} open={open} onToggle={toggle} onNav={onNav} />
      <div className="main">
        <Ctx.Provider value={{ items, error, refresh }}>
          <Outlet />
        </Ctx.Provider>
      </div>
    </div>
  );
}
