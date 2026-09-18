// Layout preferences: the sidebar's open state and the inspector's width, both remembered in localStorage.
export const SIDEBAR_KEY = "desk.sidebar";
export const PANE_KEY = "desk.pane";
export const PANE_MIN = 360;
export const PANE_MAX_FRAC = 0.7;
export const PANE_DEFAULT_FRAC = 0.46;
export const NARROW = 960;

export function parseSidebar(raw: string | null): boolean {
  return raw !== "closed";
}

export function serialiseSidebar(open: boolean): string {
  return open ? "open" : "closed";
}

export function parsePaneWidth(raw: string | null): number | null {
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.round(n) : null;
}

/** Keep the pane between PANE_MIN and PANE_MAX_FRAC of the desk. On a desk too narrow for PANE_MIN, the max wins. */
export function clampPaneWidth(px: number, deskWidth: number): number {
  const max = Math.floor(deskWidth * PANE_MAX_FRAC);
  const min = Math.min(PANE_MIN, max);
  return Math.round(Math.min(Math.max(px, min), max));
}
