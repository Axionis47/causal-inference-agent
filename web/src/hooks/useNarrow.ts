import { useEffect, useState } from "react";
import { NARROW } from "../layout";

const QUERY = `(max-width: ${NARROW}px)`;

/** True when the viewport is too narrow for the sidebar and the inspector to sit beside the chat. */
export function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => (typeof window !== "undefined" && "matchMedia" in window ? window.matchMedia(QUERY).matches : false));
  useEffect(() => {
    if (!("matchMedia" in window)) return;
    const mq = window.matchMedia(QUERY);
    const on = () => setNarrow(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return narrow;
}
