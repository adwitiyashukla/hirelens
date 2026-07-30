/**
 * Shell and routing.
 *
 * No router library. There are three destinations and one of them takes an id,
 * so a discriminated union in state does the whole job in a dozen lines and
 * keeps the dependency list to React alone. If this grew a fourth or fifth
 * view, `react-router` would earn its place.
 */

import { useEffect, useState } from "react";
import { api } from "./api/client";
import type { Health } from "./api/types";
import { CandidateDetail } from "./components/CandidateDetail";
import { MethodPage } from "./components/MethodPage";
import { ScreenPage } from "./components/ScreenPage";

type View =
  | { name: "screen" }
  | { name: "method" }
  | { name: "candidate"; assessmentId: string };

export default function App() {
  const [view, setView] = useState<View>({ name: "screen" });
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealthError(true));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          HireLens <span>evidence-grounded screening</span>
        </div>
        <nav>
          <button
            onClick={() => setView({ name: "screen" })}
            aria-current={view.name !== "method" ? "page" : undefined}
          >
            Screen
          </button>
          <button
            onClick={() => setView({ name: "method" })}
            aria-current={view.name === "method" ? "page" : undefined}
          >
            How it works
          </button>
        </nav>
        <div className="spacer" />
        {health && (
          <>
            {health.blind_mode && (
              <span className="badge neutral" title="Names and contact details are redacted before the model reads a resume">
                blind mode
              </span>
            )}
            <span className="tag mono">{health.model}</span>
          </>
        )}
      </header>

      {healthError && (
        <div className="page" style={{ paddingBottom: 0 }}>
          <div className="banner error">
            Cannot reach the API. Start it with{" "}
            <code>uvicorn hirelens.api.app:app --reload</code> and reload this
            page.
          </div>
        </div>
      )}

      {health && !health.provider_configured && (
        <div className="page" style={{ paddingBottom: 0 }}>
          <div className="banner warn">
            No model credential is configured, so screening will fail. Set{" "}
            <code>HIRELENS_GEMINI_API_KEY</code> in your <code>.env</code> file,
            or switch to a local model with{" "}
            <code>HIRELENS_LLM_PROVIDER=ollama</code>.
          </div>
        </div>
      )}

      {/*
        Hidden rather than unmounted. Opening a candidate and coming back must
        not discard the compiled rubric and the shortlist, and keeping the state
        here would mean lifting the whole screening flow into the shell.
      */}
      <div style={{ display: view.name === "screen" ? "contents" : "none" }}>
        <ScreenPage
          onOpenCandidate={(assessmentId) =>
            setView({ name: "candidate", assessmentId })
          }
        />
      </div>

      {view.name === "candidate" && (
        <CandidateDetail
          assessmentId={view.assessmentId}
          onBack={() => setView({ name: "screen" })}
        />
      )}

      {view.name === "method" && <MethodPage />}
    </div>
  );
}
