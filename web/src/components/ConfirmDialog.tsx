import { useEffect, useRef } from "react";

export default function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel,
  danger,
  busy,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  body: string;
  confirmLabel: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} onClose={onCancel} onCancel={onCancel}>
      <h3>{title}</h3>
      <p>{body}</p>
      <div className="row">
        <button className="btn" onClick={onCancel} disabled={busy}>
          Keep it
        </button>
        <button className={`btn ${danger ? "danger" : "primary"}`} onClick={onConfirm} disabled={busy} autoFocus>
          {busy ? "Working…" : confirmLabel}
        </button>
      </div>
    </dialog>
  );
}
