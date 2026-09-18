import type { KeyboardEvent, PointerEvent, RefObject } from "react";
import { useRef } from "react";
import { clampPaneWidth, PANE_KEY, PANE_MAX_FRAC, PANE_MIN, parsePaneWidth } from "../layout";
import { useStoredState } from "./useStoredState";

const serialise = (v: number | null) => (v === null ? "" : String(v));

export interface SplitterProps {
  onPointerDown: (e: PointerEvent<HTMLElement>) => void;
  onPointerMove: (e: PointerEvent<HTMLElement>) => void;
  onPointerUp: (e: PointerEvent<HTMLElement>) => void;
  onPointerCancel: (e: PointerEvent<HTMLElement>) => void;
  onKeyDown: (e: KeyboardEvent<HTMLElement>) => void;
}

/** Drag the splitter to size the inspector. During the drag the width goes straight to a CSS variable; React hears about it once, at the end. */
export function usePaneResize(deskRef: RefObject<HTMLElement | null>) {
  const [width, setWidth] = useStoredState<number | null>(PANE_KEY, parsePaneWidth, serialise);
  const drag = useRef<{ startX: number; startW: number; deskW: number } | null>(null);

  const pane = () => deskRef.current?.querySelector<HTMLElement>(".inspector") ?? null;
  const proposed = (clientX: number) => {
    const d = drag.current;
    return d ? clampPaneWidth(d.startW + (d.startX - clientX), d.deskW) : null;
  };

  const onPointerDown = (e: PointerEvent<HTMLElement>) => {
    const desk = deskRef.current;
    const p = pane();
    if (!desk || !p) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { startX: e.clientX, startW: p.getBoundingClientRect().width, deskW: desk.clientWidth };
    desk.classList.add("dragging");
  };
  const onPointerMove = (e: PointerEvent<HTMLElement>) => {
    const px = proposed(e.clientX);
    if (px !== null) deskRef.current?.style.setProperty("--pane-pref", `${px}px`);
  };
  const onPointerUp = (e: PointerEvent<HTMLElement>) => {
    const px = proposed(e.clientX);
    if (px === null) return;
    drag.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
    deskRef.current?.classList.remove("dragging");
    setWidth(px);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    const desk = deskRef.current;
    const p = pane();
    if (!desk || !p) return;
    const cur = p.getBoundingClientRect().width;
    const deskW = desk.clientWidth;
    let next: number;
    if (e.key === "ArrowLeft") next = cur + 24;
    else if (e.key === "ArrowRight") next = cur - 24;
    else if (e.key === "Home") next = PANE_MIN;
    else if (e.key === "End") next = deskW * PANE_MAX_FRAC;
    else return;
    e.preventDefault();
    setWidth(clampPaneWidth(next, deskW));
  };

  const splitterProps: SplitterProps = { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp, onKeyDown };
  return { width, splitterProps };
}
