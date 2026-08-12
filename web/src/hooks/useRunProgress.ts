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
