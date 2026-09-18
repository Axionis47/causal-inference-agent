import { useRef, useState } from "react";

export default function UploadDrop({ onFile, busy }: { onFile: (f: File) => void; busy: boolean }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const pick = (files: FileList | null) => {
    const f = files?.[0];
    if (f) onFile(f);
  };
  return (
    <div
      className={`drop${over ? " over" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        pick(e.dataTransfer.files);
      }}
    >
      <input ref={input} type="file" accept=".csv,text/csv" onChange={(e) => pick(e.target.files)} />
      <div className="big">{busy ? "Reading the file…" : "Drop a CSV here, or click to choose one"}</div>
      <div>One table, one row per unit. The columns are detected next.</div>
    </div>
  );
}
