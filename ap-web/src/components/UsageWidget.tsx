import { useUsage, type ProviderUsage } from "@/hooks/useUsage";

function colorFor(p: number | null | undefined): string {
  if (p == null) return "text-muted-foreground";
  if (p >= 80) return "text-red-500";
  if (p >= 50) return "text-amber-500";
  return "text-emerald-600 dark:text-emerald-400";
}

function fmt(p: number | null | undefined): string {
  return p == null ? "–" : `${Math.round(p)}%`;
}

function tooltip(label: string, u: ProviderUsage | undefined): string {
  if (!u?.available) {
    return `${label}: usage unavailable${u?.error ? ` (${u.error})` : ""}`;
  }
  const parts: string[] = [];
  if (u.plan_type) parts.push(`plan: ${u.plan_type}`);
  if (u.session) {
    parts.push(
      `session/5h: ${fmt(u.session.used_percent)}${u.session.resets ? ` (resets ${u.session.resets})` : ""}`,
    );
  }
  if (u.week) {
    parts.push(
      `week: ${fmt(u.week.used_percent)}${u.week.resets ? ` (resets ${u.week.resets})` : ""}`,
    );
  }
  return `${label} — ${parts.join(" · ")}`;
}

/**
 * Compact header widget showing weekly subscription usage for Claude + Codex
 * (the limit you're most likely to hit). Hover for session/5h + reset detail.
 */
export function UsageWidget() {
  const { data } = useUsage();
  if (!data) return null;

  const claudeWeek = data.claude?.available ? data.claude.week?.used_percent : null;
  const codexWeek = data.codex?.available ? data.codex.week?.used_percent : null;

  return (
    <div
      className="hidden items-center gap-1.5 text-xs tabular-nums sm:flex"
      aria-label="Subscription usage"
    >
      <span title={tooltip("Claude", data.claude)} className="whitespace-nowrap">
        <span className="text-muted-foreground">Claude</span>{" "}
        <span className={colorFor(claudeWeek)}>{fmt(claudeWeek)}</span>
      </span>
      <span className="text-muted-foreground/40">·</span>
      <span title={tooltip("Codex", data.codex)} className="whitespace-nowrap">
        <span className="text-muted-foreground">Codex</span>{" "}
        <span className={colorFor(codexWeek)}>{fmt(codexWeek)}</span>
      </span>
    </div>
  );
}
