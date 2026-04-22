import { useEffect, useRef } from "react";
import { CheckCircle2, AlertCircle, Loader2 } from "lucide-react";

const apiBase = import.meta.env?.VITE_API_URL || "";
const POLL_MS = 3000;

export default function Progress({ jobId, status, onStatus, polling }) {
  const timerRef = useRef(null);
  const onStatusRef = useRef(onStatus);
  useEffect(() => { onStatusRef.current = onStatus; }, [onStatus]);

  useEffect(() => {
    if (!jobId || !polling) return;
    let cancelled = false;

    async function tick() {
      try {
        const res = await fetch(`${apiBase}/status/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        onStatusRef.current(data);
        if (data.status === "complete" || data.status === "error") {
          if (timerRef.current) clearInterval(timerRef.current);
          timerRef.current = null;
        }
      } catch {
        /* transient */
      }
    }

    tick();
    timerRef.current = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [jobId, polling]);

  if (!jobId || !status) return null;

  const processing = status.status === "processing";
  const complete = status.status === "complete";
  const errored = status.status === "error";

  return (
    <section className="rounded-2xl border border-border bg-surface px-5 py-4 shadow-[var(--shadow-soft)]">
      <div className="flex items-start gap-4">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
          {processing && <Loader2 className="h-4 w-4 animate-spin text-foreground" />}
          {complete && <CheckCircle2 className="h-4 w-4 text-success" />}
          {errored && <AlertCircle className="h-4 w-4 text-destructive" />}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
            <span>Job</span>
            <span className="text-ink">{jobId.slice(0, 8)}</span>
            <span className="text-border-strong">•</span>
            <span>{processing ? "running" : status.status}</span>
          </div>

          <div className="mt-1.5 text-[15px] text-ink">
            {processing && (
              <span className="flex items-center gap-2">
                <span className="capitalize">{status.step ?? "Working"}</span>
                {status.iteration != null && (
                  <span className="font-mono text-xs text-muted-foreground">
                    iter {status.iteration}
                  </span>
                )}
              </span>
            )}
            {complete && (
              <span>
                Research complete
                {status.source_domains_count != null && (
                  <span className="ml-2 text-sm text-muted-foreground">
                    · {status.source_domains_count} unique sources
                  </span>
                )}
              </span>
            )}
            {errored && <span className="text-destructive">{status.error || "Error"}</span>}
          </div>

          {processing && (
            <div className="mt-3 h-[2px] w-full overflow-hidden rounded-full bg-muted">
              <div className="shimmer h-full w-full" />
            </div>
          )}
        </div>
      </div>
    </section>
  );
}