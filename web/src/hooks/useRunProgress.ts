/**
 * Subscribes to a run's server-sent event stream.
 *
 * Two details matter and both come from having watched this fail:
 *
 * 1. `EventSource` reconnects automatically on error, including after the
 *    server closes a finished stream. Left alone it reopens the connection
 *    forever, so the socket is closed explicitly on the terminal event.
 *
 * 2. The API emits a named `done` event after the last `progress` event. Only
 *    listening for `message` would miss both, because named events do not fire
 *    the default handler.
 */

import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { RunProgress } from "../api/types";

export interface RunProgressState {
  progress: RunProgress | null;
  finished: boolean;
  error: string | null;
}

export function useRunProgress(runId: string | null): RunProgressState {
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const [finished, setFinished] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Mirrors `finished` for the error handler to read synchronously. Reading the
  // state variable there would close over a stale value, and deciding inside a
  // state updater would run the decision twice under StrictMode.
  const terminalRef = useRef(false);

  useEffect(() => {
    if (!runId) return;

    setProgress(null);
    setFinished(false);
    setError(null);
    terminalRef.current = false;

    const source = new EventSource(api.eventsUrl(runId));

    const finish = () => {
      terminalRef.current = true;
      setFinished(true);
      source.close();
    };

    const close = () => source.close();

    source.addEventListener("progress", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as RunProgress;
      setProgress(payload);
      if (payload.status === "completed" || payload.status === "failed") {
        finish();
      }
    });

    source.addEventListener("done", finish);

    source.onerror = () => {
      // Fires both on a genuine failure and on the normal close of a finished
      // stream, so it is only an error if the run had not reached a terminal
      // state. Reporting it unconditionally would put a red banner at the end
      // of every successful run.
      if (terminalRef.current) return;
      if (source.readyState === EventSource.CLOSED) {
        setError("Lost connection to the run stream.");
        finish();
      }
    };

    return close;
  }, [runId]);

  return { progress, finished, error };
}
