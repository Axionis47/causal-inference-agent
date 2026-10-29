import { useRef, useState } from "react";

export default function UploadDrop({ onFile, busy }: { onFile: (f: File) => void; busy: boolean }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const pick = (files: FileList | null) => {
    const f = files?.[0];
    if (f && !busy) onFile(f);
  };
  return (
    <div
      className={`drop${over ? " over" : ""}`}
      role="button"
      aria-disabled={busy}
      tabIndex={0}
      onClick={() => !busy && input.current?.click()}
      onKeyDown={(e) => !busy && (e.key === "Enter" || e.key === " ") && input.current?.click()}
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
      <input ref={input} type="file" disabled={busy} aria-label="CSV file" accept=".csv,text/csv" onChange={(e) => pick(e.target.files)} />
      <div className="big">{busy ? "Reading the file…" : "Drop a CSV here, or click to choose one"}</div>
      <div>Choose one CSV. Your original bytes are preserved.</div>
    </div>
  );
}
