import type { ReactNode } from "react";

/** The row above a page: where you are on the left, actions on the right. The brand lives in the sidebar. */
export default function TopBar({ crumb, right }: { crumb?: ReactNode; right?: ReactNode }) {
  return (
    <header className="top">
      <div className="crumb">{crumb}</div>
      <div>{right}</div>
    </header>
  );
}
