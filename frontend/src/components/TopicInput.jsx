import { useState } from "react";
import { ArrowUpRight, Loader2 } from "lucide-react";

const apiBase = import.meta.env?.VITE_API_URL || "";

const SUGGESTIONS = [
  "Impact of CRISPR on rare disease treatment",
  "State of small modular nuclear reactors in 2025",
  "How transformer attention mechanisms evolved",
];

export default function TopicInput({ onJobStarted, disabled }) {
  const [topic, setTopic] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  async function submit(value) {
    const t = value.trim();
    if (!t || busy || disabled) return;
    setErr(null);
    setBusy(true);
    try {
      const res = await fetch(`${apiBase}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: t }),
      });
      if (!res.ok) throw new Error((await res.text()) || res.statusText);
      const data = await res.json();
      onJobStarted(data.job_id);
      setTopic("");
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    submit(topic);
  }

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="group relative">
        <div className="relative flex items-center rounded-2xl border border-border bg-surface shadow-[var(--shadow-soft)] transition-all focus-within:border-border-strong focus-within:shadow-[var(--shadow-elevated)]">
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="What would you like to research?"
            disabled={busy || disabled}
            className="flex-1 bg-transparent px-5 py-5 text-base text-ink placeholder:text-muted-foreground focus:outline-none disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={busy || disabled || !topic.trim()}
            className="mr-2 flex h-10 w-10 items-center justify-center rounded-xl bg-ink text-background transition-all hover:scale-[1.03] disabled:scale-100 disabled:cursor-not-allowed disabled:opacity-30"
            aria-label="Run research"
          >
            {busy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <ArrowUpRight className="h-4 w-4" strokeWidth={2.5} />
            )}
          </button>
        </div>
      </form>

      {err && <p className="mt-3 text-sm text-destructive">{err}</p>}

      <div className="mt-4 flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => submit(s)}
            disabled={busy || disabled}
            className="rounded-full border border-border bg-surface px-3.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border-strong hover:text-ink disabled:opacity-40"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
