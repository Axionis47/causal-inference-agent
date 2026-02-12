import { Terminal } from "lucide-react";

export function Brand() {
  return (
    <div className="brand">
      <span className="brand-mark">
        <Terminal size={19} aria-hidden="true" />
      </span>
      causal<span className="text-muted-foreground font-normal">/</span>
    </div>
  );
}
