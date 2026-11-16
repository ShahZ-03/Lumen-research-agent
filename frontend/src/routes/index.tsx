import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { Sparkles } from "lucide-react";
import TopicInput from "@/components/TopicInput";
import Progress from "@/components/Progress";
import Report from "@/components/Report";
import JobSidebar from "@/components/JobSidebar";

const apiBase = (import.meta as any).env?.VITE_API_URL || "";
const ACTIVE_JOB_KEY = "lumen.activeJobId";

interface StatusShape {
  job_id?: string;
  topic?: string;
  conversation_id?: string;
  parent_job_id?: string | null;
  turn_index?: number;
  status?: string;
  step?: string;
  iteration?: number | null;
  source_domains_count?: number | null;
  error?: string | null;
  report?: string | null;
  grounding?: any;
}

export const Route = createFileRoute("/")({ ssr: false, component: Index });

function rememberJob(id: string | null) {
  try {
    if (id) localStorage.setItem(ACTIVE_JOB_KEY, id);
    else localStorage.removeItem(ACTIVE_JOB_KEY);
  } catch { /* History remains available from the server when storage is blocked. */ }
}

function Index() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<StatusShape[]>([]);
  const [status, setStatus] = useState<StatusShape | null>(null);
  const [polling, setPolling] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [streamedReport, setStreamedReport] = useState("");
  const [jobsRefresh, setJobsRefresh] = useState(0);
  const activeJobRef = useRef<string | null>(null);
  const loadVersion = useRef(0);

  const onJobStarted = useCallback((id: string, topic: string, threadId: string) => {
    loadVersion.current += 1;
    activeJobRef.current = id;
    const initial = { job_id: id, topic, conversation_id: threadId, status: "processing", step: "starting" };
    setJobId(id);
    setConversationId(threadId);
    setTurns((previous) => [...previous, initial]);
    setStreamedReport("");
    setStatus(initial);
    setPolling(true);
    setLoadError(null);
    setJobsRefresh((value) => value + 1);
    rememberJob(id);
  }, []);

  const onStatus = useCallback((next: StatusShape) => {
    if (next.job_id !== activeJobRef.current) return;
    setStatus(next);
    setTurns((previous) => previous.map((turn) => turn.job_id === next.job_id ? { ...turn, ...next } : turn));
    if (next.status === "complete" || next.status === "error") {
      setPolling(false);
      setJobsRefresh((value) => value + 1);
    }
  }, []);

  const loadJob = useCallback(async (id: string) => {
    const version = ++loadVersion.current;
    activeJobRef.current = null;
    setLoading(true);
    setPolling(false);
    setLoadError(null);
    setStatus(null);
    setStreamedReport("");
    try {
      const response = await fetch(`${apiBase}/status/${id}`);
      if (!response.ok) throw new Error("Could not load this research conversation.");
      const selected: StatusShape = await response.json();
      const threadId = selected.conversation_id || id;
      const history = await fetch(`${apiBase}/conversations/${threadId}`);
      if (!history.ok) throw new Error("Could not load conversation history.");
      const data = await history.json();
      if (version !== loadVersion.current) return;
      const messages: StatusShape[] = data.turns;
      const latest = messages[messages.length - 1];
      if (!latest?.job_id) throw new Error("This conversation has no messages.");
      activeJobRef.current = latest.job_id;
      setJobId(latest.job_id);
      setConversationId(threadId);
      setTurns(messages);
      setStatus(latest);
      setPolling(latest.status === "processing");
      rememberJob(latest.job_id);
    } catch (error) {
      if (version === loadVersion.current) {
        setJobId(null);
        setConversationId(null);
        setTurns([]);
        setLoadError(error instanceof Error ? error.message : "Could not load conversation.");
        rememberJob(null);
      }
    } finally {
      if (version === loadVersion.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    let saved: string | null = null;
    try { saved = localStorage.getItem(ACTIVE_JOB_KEY); } catch { /* Optional persistence. */ }
    if (saved) void loadJob(saved);
  }, [loadJob]);

  const startNew = useCallback(() => {
    loadVersion.current += 1;
    activeJobRef.current = null;
    setJobId(null);
    setConversationId(null);
    setTurns([]);
    setStatus(null);
    setStreamedReport("");
    setPolling(false);
    setLoading(false);
    setLoadError(null);
    rememberJob(null);
  }, []);

  useEffect(() => {
    if (!jobId || status?.job_id !== jobId || status.status !== "processing") return;
    const controller = new AbortController();
    void (async () => {
      try {
        const response = await fetch(`${apiBase}/stream/report/${jobId}`, { signal: controller.signal });
        if (!response.ok || !response.body) return;
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let text = "";
        while (!controller.signal.aborted) {
          const { done, value } = await reader.read();
          if (done) break;
          text += decoder.decode(value, { stream: true });
          if (activeJobRef.current === jobId) setStreamedReport(text);
        }
      } catch { /* Status polling supplies the final persisted report or error. */ }
    })();
    return () => controller.abort();
  }, [jobId, status?.job_id, status?.status]);

  return (
    <div className="flex min-h-screen bg-background">
      <JobSidebar selectedJobId={jobId} onSelectJob={loadJob} onNewJob={startNew} refreshKey={jobsRefresh} />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-6 py-12 md:px-10 md:py-16">
          <header className="mb-10">
            <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1">
              <Sparkles className="h-3 w-3 text-accent" />
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Autonomous research</span>
            </div>
            <h1 className="font-serif text-4xl leading-[1.05] tracking-tight text-ink md:text-5xl">
              Ask anything.<br /><span className="italic text-muted-foreground">Get a sourced answer.</span>
            </h1>
            <p className="mt-4 max-w-xl text-base text-muted-foreground">
              Lumen searches the open web, retrieves relevant sources, and synthesises a structured
              report - every claim grounded in evidence.
            </p>
          </header>

          {loading && <p className="mb-6 text-sm text-muted-foreground" role="status">Loading conversation...</p>}
          {loadError && <p className="mb-6 text-sm text-destructive" role="alert">{loadError}</p>}

          <div className="space-y-8">
            {turns.map((turn) => {
              const active = turn.job_id === jobId;
              const text = turn.report || (active ? streamedReport : "");
              return (
                <section key={turn.job_id} className="space-y-4">
                  <div className="ml-auto max-w-[90%] rounded-2xl border border-border bg-surface-elevated px-5 py-4 text-sm text-ink whitespace-pre-wrap">
                    {turn.topic}
                  </div>
                  {active && status && <Progress jobId={jobId} status={status} onStatus={onStatus} polling={polling} />}
                  {turn.status === "error" && <p className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive" role="alert">{turn.error || "Research failed. You can try a follow-up below."}</p>}
                  {text && <Report markdown={text} jobId={turn.job_id} streaming={active && turn.status === "processing"} grounding={turn.status === "complete" ? turn.grounding : null} />}
                </section>
              );
            })}
          </div>

          {/* The original composer remains available after every answer. */}
          <section className={turns.length ? "mt-8" : "mb-8"}>
            <TopicInput key={conversationId || "new"} onJobStarted={onJobStarted}
              conversationId={conversationId} disabled={loading || status?.status === "processing"} />
          </section>

          {turns.length > 0 && status?.status !== "processing" && (
            <div className="mt-6 flex justify-center">
              <button type="button" onClick={startNew} className="rounded-full border border-border bg-surface px-5 py-2 text-sm text-muted-foreground transition-colors hover:border-border-strong hover:text-ink">
                Start new research
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
