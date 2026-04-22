import { useCallback, useEffect, useState } from "react";
import { Plus, History, AlertCircle, X } from "lucide-react";

const apiBase = import.meta.env?.VITE_API_URL || "";

function statusDot(status) {
  if (status === "complete") return "bg-success";
  if (status === "error") return "bg-destructive";
  if (status === "processing") return "bg-accent pulse-dot";
  return "bg-muted-foreground";
}

export default function JobSidebar({ selectedJobId, onSelectJob, onNewJob, refreshKey }) {
  const [jobs, setJobs] = useState([]);
  const [err, setErr] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      const res = await fetch(`${apiBase}/jobs`);
      if (!res.ok) throw new Error(await res.text());
      setJobs(await res.json());
    } catch (e) {
      setErr(e.message || String(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  const deleteJob = useCallback(
    async (jobId) => {
      const ok = window.confirm("Delete this research job permanently? This cannot be undone.");
      if (!ok) return;
      setErr(null);
      setDeletingId(jobId);
      try {
        const res = await fetch(`${apiBase}/jobs/${jobId}`, { method: "DELETE" });
        if (!res.ok) throw new Error(await res.text());
        if (selectedJobId === jobId) {
          onNewJob();
        }
        await load();
      } catch (e) {
        setErr(e.message || String(e));
      } finally {
        setDeletingId(null);
      }
    },
    [load, onNewJob, selectedJobId],
  );

  return (
    <aside className="flex h-screen w-72 shrink-0 flex-col border-r border-border bg-surface-elevated">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-5 pt-6 pb-5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-ink">
          <div className="h-2 w-2 rounded-full bg-accent" />
        </div>
        <span className="font-serif text-xl tracking-tight text-ink">Lumen</span>
      </div>

      {/* New */}
      <div className="px-3">
        <button
          type="button"
          onClick={onNewJob}
          className="flex w-full items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-sm font-medium text-ink shadow-[var(--shadow-soft)] transition-all hover:border-border-strong"
        >
          <Plus className="h-4 w-4" />
          New research
        </button>
      </div>

      {/* History */}
      <div className="mt-6 flex items-center gap-2 px-5 pb-2">
        <History className="h-3 w-3 text-muted-foreground" />
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          History
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-3 pb-6">
        {err && (
          <div className="mx-2 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />
            <span className="break-words">{err}</span>
          </div>
        )}
        {!err && jobs.length === 0 && (
          <p className="px-2 py-3 text-xs text-muted-foreground">No past jobs yet.</p>
        )}
        <ul className="space-y-0.5">
          {jobs.map((j) => {
            const active = selectedJobId === j.job_id;
            return (
              <li key={j.job_id}>
                <div
                  className={`group flex items-start gap-2 rounded-lg px-2 py-1 transition-colors ${
                    active ? "bg-surface shadow-[var(--shadow-soft)]" : "hover:bg-surface"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => onSelectJob(j.job_id)}
                    title={j.topic}
                    className="flex min-w-0 flex-1 items-start gap-2.5 rounded-md px-1 py-1 text-left"
                  >
                    <span
                      className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${statusDot(j.status)}`}
                    />
                    <span className="min-w-0 flex-1">
                      <span
                        className={`block truncate text-sm ${
                          active ? "text-ink" : "text-foreground"
                        }`}
                      >
                        {j.topic || "Untitled research"}
                      </span>
                      <span className="mt-0.5 block font-mono text-[10px] text-muted-foreground">
                        {j.status} · {j.job_id.slice(0, 8)}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => deleteJob(j.job_id)}
                    disabled={deletingId === j.job_id}
                    className="mt-1 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:opacity-40"
                    aria-label="Delete job"
                    title="Delete job"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="border-t border-border px-5 py-3">
        <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          Lumen · v0.1
        </p>
      </div>
    </aside>
  );
}
