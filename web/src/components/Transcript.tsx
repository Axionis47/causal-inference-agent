import { useEffect, useRef } from "react";
import type { Turn } from "../types";
import Message from "./Message";

export default function Transcript({ turns }: { turns: Turn[] }) {
  const box = useRef<HTMLDivElement>(null);
  const count = turns.length;
  useEffect(() => {
    const el = box.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [count]);
  return (
    <div className="thread" ref={box}>
      {turns.map((t, i) => (
        <Message key={`${t.at}-${i}`} t={t} />
      ))}
    </div>
  );
}
