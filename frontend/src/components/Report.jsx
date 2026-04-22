import { useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { Copy, Download, ChevronDown, ExternalLink, ShieldCheck, ShieldAlert } from "lucide-react";

function scoreColor(overall) {
  if (overall >= 0.8) return "text-success";
  if (overall >= 0.6) return "text-warning";
  return "text-destructive";
}

export default function Report({ markdown, jobId, streaming, grounding }) {
  const copy = useCallback(async () => {
    if (!markdown) return;
    try {
      await navigator.clipboard.writeText(markdown);
    } catch {
      /* ignore */
    }
  }, [markdown]);

  const download = useCallback(() => {
    if (!markdown) return;
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `research-${jobId?.slice(0, 8) || "report"}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }, [markdown, jobId]);

  if (!markdown) return null;

  const showGrounding = !streaming && grounding && Array.isArray(grounding.claims);
  const totalClaims = showGrounding ? grounding.claims.length : 0;
  const verifiedN = showGrounding ? grounding.verified_count : 0;
  const overall = showGrounding ? grounding.overall_score : 0;

  return (
    <article className="rounded-2xl border border-border bg-surface shadow-[var(--shadow-soft)]">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
            Report
          </span>
          {streaming && (
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-accent" />
              Drafting…
            </span>
          )}
          {showGrounding && totalClaims > 0 && (
            <span className={`flex items-center gap-1.5 text-xs font-medium ${scoreColor(overall)}`}>
              <ShieldCheck className="h-3.5 w-3.5" />
              {verifiedN}/{totalClaims} verified · {overall.toFixed(2)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={copy}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-ink"
          >
            <Copy className="h-3.5 w-3.5" />
            Copy
          </button>
          <button
            type="button"
            onClick={download}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-ink"
          >
            <Download className="h-3.5 w-3.5" />
            Download
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="px-6 py-6 md:px-10 md:py-8">
        <div className="prose-research">
          <ReactMarkdown>{markdown}</ReactMarkdown>
        </div>

        {showGrounding && totalClaims > 0 && (
          <details className="group mt-10 border-t border-border pt-6">
            <summary className="flex cursor-pointer list-none items-center justify-between text-sm font-medium text-ink">
              <span className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4" />
                Claim verification
                <span className="font-mono text-xs text-muted-foreground">
                  {grounding.claims.length}
                </span>
              </span>
              <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
            </summary>
            <ul className="mt-4 space-y-2">
              {grounding.claims.map((c, i) => (
                <li key={i} className="rounded-xl border border-border bg-surface-elevated p-4">
                  <div className="flex items-start gap-3">
                    <span
                      className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full ${
                        c.verified
                          ? "bg-success/10 text-success"
                          : "bg-destructive/10 text-destructive"
                      }`}
                    >
                      {c.verified ? (
                        <ShieldCheck className="h-3 w-3" />
                      ) : (
                        <ShieldAlert className="h-3 w-3" />
                      )}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-ink">{c.claim}</p>
                      {c.best_source_url && (
                        <a
                          href={c.best_source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="mt-1.5 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-ink"
                        >
                          <ExternalLink className="h-3 w-3" />
                          <span className="max-w-[40ch] truncate">
                            {c.best_source_title || c.best_source_url}
                          </span>
                          {c.score != null && (
                            <span className="font-mono text-[10px] text-border-strong">
                              {c.score}
                            </span>
                          )}
                        </a>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </details>
        )}

        {showGrounding && totalClaims === 0 && (
          <p className="mt-6 text-xs text-muted-foreground">
            No factual claims extracted for grounding check.
          </p>
        )}
      </div>
    </article>
  );
}
