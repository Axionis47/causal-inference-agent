import { useEffect, useState } from "react";

/** State mirrored in localStorage. Storage may be missing or blocked; then it is plain state. */
export function useStoredState<T>(key: string, parse: (raw: string | null) => T, serialise: (v: T) => string): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      return parse(window.localStorage.getItem(key));
    } catch {
      return parse(null);
    }
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(key, serialise(value));
    } catch {
      /* storage unavailable */
    }
  }, [key, value, serialise]);
  return [value, setValue];
}
