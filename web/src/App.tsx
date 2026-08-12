import { useEffect, useState } from "react";
import { api } from "./api/client";
import type { Health } from "./api/types";
import { CandidateDetail } from "./components/CandidateDetail";
import { Cpu, Logo, Shield } from "./components/Icons";
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
          <span className="mark">
            <Logo />
          </span>
          <span className="wordmark">
            <strong>HireLens</strong>
            <span>evidence-grounded screening</span>
          </span>
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
            {}
            {health.blind_mode && (
              <span
                className="status-chip on"
                title="Names, emails, phone numbers, addresses and institutions are masked before the resume reaches the model. The masks are the same length as the text they replace, so character offsets stay valid and citations still point at the right lines."
              >
                <Shield />
                Names hidden from model
              </span>
            )}
            <span className="status-chip" title="The model performing the screening">
              <Cpu />
              <span className="value">{health.model}</span>
            </span>
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

      {}
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
