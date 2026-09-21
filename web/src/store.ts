import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { SessionView } from "./types";

const POLL_MS = 1500;

export function useSession(name: string) {
  const [view, setView] = useState<SessionView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const timer = useRef<number | null>(null);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const v = await api.session(name);
      if (!alive.current) return v;
      setView(v);
      setError(null);
      return v;
    } catch (e) {
      if (alive.current) setError(e instanceof ApiError ? e.message : String(e));
      return null;
    }
  }, [name]);

  useEffect(() => {
    alive.current = true;
    void refresh();
    return () => {
      alive.current = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [refresh]);

  useEffect(() => {
    if (timer.current) window.clearTimeout(timer.current);
    if (view?.stage === "busy") {
      timer.current = window.setTimeout(() => void refresh(), POLL_MS);
    }
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [view, refresh]);

  const act = useCallback(async (fn: () => Promise<SessionView>) => {
    setSending(true);
    try {
      const v = await fn();
      if (alive.current) {
        setView(v);
        setError(null);
      }
    } catch (e) {
      if (alive.current) setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      if (alive.current) setSending(false);
    }
  }, []);

  return {
    view,
    error,
    sending,
    refresh,
    send: (text: string) => act(() => api.send(name, text)),
    resume: () => act(() => api.resume(name)),
    restart: () => act(() => api.restart(name)),
  };
}
