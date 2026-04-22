import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";
import { Sparkles } from "lucide-react";
import TopicInput from "@/components/TopicInput";
import Progress from "@/components/Progress";
import Report from "@/components/Report";
import JobSidebar from "@/components/JobSidebar";

const apiBase = (import.meta as any).env?.VITE_API_URL || "";

interface StatusShape {
  job_id?: string;
  status?: string;
  step?: string;
  iteration?: number | null;
  source_domains_count?: number | null;
  error?: string | null;
  report?: string | null;
  grounding?: any;
}

export const Route = createFileRoute("/")({
  ssr: false,
  component: Index,
});

function Index() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusShape | null>(null);
  const [polling, setPolling] = useState(false);
  const [streamedReport, setStreamedReport] = useState("");
  const [jobsRefresh, setJobsRefresh] = useState(0);

  const onJobStarted = useCallback((id: string) => {
    setJobId(id);
    setStreamedReport("");
    setStatus({ job_id: id, status: "processing", step: "starting", iteration: 0 });
    setPolling(true);
    setJobsRefresh((k) => k + 1);
  }, []);

  const onStatus = useCallback((s: StatusShape) => {
    setStatus(s);
    if (s.status === "complete" || s.status === "error") {
      setPolling(false);
      setJobsRefresh((k) => k + 1);
    }
  }, []);

  const loadJob = useCallback(async (id: string) => {
    setStreamedReport("");
    setJobId(id);
    try {
      const res = await fetch(`${apiBase}/status/${id}`);
      if (!res.ok) return;
      const data = await res.json();
      setStatus(data);
      setPolling(data.status === "processing");
      // For completed jobs loaded from history, populate streamedReport
      // so the report section renders without waiting for a stream.
      if (data.status === "complete" && data.report) {
        setStreamedReport(data.report);
      }
    } catch {
      /* ignore */
    }
  }, []);

  const startNew = useCallback(() => {
    setJobId(null);
    setStatus(null);
    setStreamedReport("");
    setPolling(false);
  }, []);

  useEffect(() => {
    if (!jobId) return;
    // If we already have a complete report from status (e.g. loaded from history),
    // don't open a stream connection — it would block forever waiting on the queue.
    if (status?.status === "complete" && status?.report) return;

    let cancelled = false;
    const ac = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${apiBase}/stream/report/${jobId}`, { signal: ac.signal });
        if (!res.ok || !res.body) return;
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let acc = "";
        while (!cancelled) {
          const { done, value } = await reader.read();
          if (done) break;
          acc += dec.decode(value, { stream: true });
          if (!cancelled) setStreamedReport(acc);
        }
      } catch (e: any) {
        if (e?.name !== "AbortError") {
          /* ignore */
        }
      }
    })();

    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [jobId, status?.status, status?.report]);

  const displayReport =
    status?.status === "complete" && status?.report ? status.report : streamedReport || "";

  const showReport =
    (status?.status === "complete" && status?.report) ||
    (status?.status === "processing" && streamedReport.length > 0);

  const idle = !jobId && !status;

  return (
    <div className="flex min-h-screen bg-background">
      <JobSidebar
        selectedJobId={jobId}
        onSelectJob={loadJob}
        onNewJob={startNew}
        refreshKey={jobsRefresh}
      />

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-6 py-12 md:px-10 md:py-16">
          {/* Header */}
          <header className="mb-10">
            <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1">
              <Sparkles className="h-3 w-3 text-accent" />
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Autonomous research
              </span>
            </div>
            <h1 className="font-serif text-4xl leading-[1.05] tracking-tight text-ink md:text-5xl">
              Ask anything.
              <br />
              <span className="italic text-muted-foreground">Get a sourced answer.</span>
            </h1>
            <p className="mt-4 max-w-xl text-base text-muted-foreground">
              Lumen searches the open web, retrieves relevant sources, and synthesises a structured
              report — every claim grounded in evidence.
            </p>
          </header>

          {/* Input */}
          {(idle || !showReport) && (
            <section className="mb-8">
              <TopicInput
                onJobStarted={onJobStarted}
                disabled={polling && status?.status === "processing"}
              />
            </section>
          )}

          {/* Progress */}
          {jobId && status && (
            <section className="mb-6">
              <Progress jobId={jobId} status={status} onStatus={onStatus} polling={polling} />
            </section>
          )}

          {/* Error */}
          {status?.status === "error" && status?.error && (
            <div className="mb-6 rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
              {status.error}
            </div>
          )}

          {/* Report */}
          {showReport && displayReport && (
            <section>
              <Report
                markdown={displayReport}
                jobId={jobId}
                streaming={status?.status === "processing"}
                grounding={
                  status?.status === "complete" && status?.grounding != null
                    ? status.grounding
                    : null
                }
              />

              {status?.status === "complete" && (
                <div className="mt-8 flex justify-center">
                  <button
                    type="button"
                    onClick={startNew}
                    className="rounded-full border border-border bg-surface px-5 py-2 text-sm text-muted-foreground transition-colors hover:border-border-strong hover:text-ink"
                  >
                    Start new research
                  </button>
                </div>
              )}
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
