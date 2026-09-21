import type { SplitterProps } from "../hooks/usePaneResize";
import { PANE_MIN } from "../layout";

/** The handle between the chat and the inspector. Drag it, or focus it and use the arrow keys. */
export default function Splitter({ width, ...props }: SplitterProps & { width: number | null }) {
  return (
    <div
      className="splitter"
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize the inspector"
      aria-valuemin={PANE_MIN}
      aria-valuenow={width ?? undefined}
      tabIndex={0}
      {...props}
    />
  );
}
