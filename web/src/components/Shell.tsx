import { createContext, useContext, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { useDatasetLibrary } from "../hooks/useDatasetLibrary";
import { useNarrow } from "../hooks/useNarrow";
import { useStoredState } from "../hooks/useStoredState";
import { parseSidebar, serialiseSidebar, SIDEBAR_KEY } from "../layout";
import Sidebar from "./Sidebar";

const Ctx = createContext<ReturnType<typeof useDatasetLibrary> | null>(null);

/** The dataset list, shared by the sidebar and the pages. */
export const useDatasets = () => {
  const value = useContext(Ctx);
  if (!value) throw new Error("Dataset library requires Shell");
  return value;
};

/** Sidebar on the left, the page in the middle. Every route renders inside this. */
export default function Shell() {
  const library = useDatasetLibrary();
  const { pathname } = useLocation();
  const narrow = useNarrow();
  const [pref, setPref] = useStoredState(SIDEBAR_KEY, parseSidebar, serialiseSidebar);
  const [overlay, setOverlay] = useState(false);
  const open = narrow ? overlay : pref;
  const toggle = () => (narrow ? setOverlay((o) => !o) : setPref((p) => !p));
  const onNav = () => narrow && setOverlay(false);

  const current = pathname.startsWith("/datasets/") ? decodeURIComponent(pathname.slice(10).split("/")[0]) : null;

  return (
    <div className="shell" data-side={open ? "open" : "closed"}>
      <Sidebar library={library} current={current} open={open} onToggle={toggle} onNav={onNav} />
      <div className="main">
        <Ctx.Provider value={library}>
          <Outlet />
        </Ctx.Provider>
      </div>
    </div>
  );
}
