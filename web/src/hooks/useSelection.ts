import { useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { parseSelection, serialiseSelection, type Selection } from "../selection";

/** The inspector's selection lives in the URL hash, so reload and back keep it and polling cannot reset it. */
export function useSelection() {
  const { hash, pathname } = useLocation();
  const navigate = useNavigate();
  const sel = parseSelection(hash);
  const set = useCallback((s: Selection) => navigate({ pathname, hash: serialiseSelection(s) }, { replace: true }), [navigate, pathname]);
  const close = useCallback(() => set(null), [set]);
  return { sel, set, close };
}
