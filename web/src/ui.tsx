import React from "react";
import { Loader2 } from "lucide-react";

export const cx = (...c: (string | false | undefined)[]) => c.filter(Boolean).join(" ");

export function Button({ variant = "primary", size = "md", className, children, loading, ...p }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "sec" | "ghost" | "danger"; size?: "sm" | "md"; loading?: boolean }) {
  const base = "inline-flex items-center justify-center gap-2 font-semibold rounded-lg transition active:scale-[.98] disabled:opacity-50 disabled:pointer-events-none";
  const sizes = { sm: "text-xs px-2.5 py-1.5", md: "text-sm px-3.5 py-2" }[size];
  const variants = {
    primary: "text-white bg-gradient-to-b from-accent to-[#5a68e6] hover:brightness-110 shadow-[0_4px_20px_rgba(109,124,255,.25)]",
    sec: "bg-card2 border border-line hover:border-line2 text-fg",
    ghost: "bg-transparent border border-line hover:bg-card text-fg",
    danger: "bg-transparent border border-bad/60 text-bad hover:bg-bad/10",
  }[variant];
  return <button className={cx(base, sizes, variants, className)} {...p}>{loading && <Loader2 className="w-4 h-4 animate-spin" />}{children}</button>;
}

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cx("bg-gradient-to-b from-card to-card2 border border-line rounded-2xl p-5 shadow-soft", className)}>{children}</div>;
}
export function CardTitle({ children, sub }: { children: React.ReactNode; sub?: React.ReactNode }) {
  return <div className="mb-4"><h3 className="text-[15px] font-bold tracking-tight">{children}</h3>{sub && <p className="text-[13px] text-muted mt-0.5">{sub}</p>}</div>;
}

export function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return <label className="block"><span className="block text-[11px] font-semibold uppercase tracking-wide text-muted mb-1.5">{label}</span>{children}{hint && <span className="block text-[11px] text-muted mt-1">{hint}</span>}</label>;
}

const inputCls = "w-full bg-bg border border-line rounded-lg px-3 py-2 text-sm text-fg outline-none focus:border-accent transition";
export const Input = (p: React.InputHTMLAttributes<HTMLInputElement>) => <input {...p} className={cx(inputCls, p.className)} />;
export const Textarea = (p: React.TextareaHTMLAttributes<HTMLTextAreaElement>) => <textarea {...p} className={cx(inputCls, "resize-y min-h-[80px]", p.className)} />;
export const Select = (p: React.SelectHTMLAttributes<HTMLSelectElement>) => <select {...p} className={cx(inputCls, "appearance-none", p.className)} />;

export function Badge({ kind = "default", children }: { kind?: string; children: React.ReactNode }) {
  const map: Record<string, string> = {
    default: "text-muted border-line",
    ai: "text-indigo-300 border-indigo-500/40 bg-indigo-500/10",
    connected: "text-indigo-300 border-indigo-500/40 bg-indigo-500/10",
    ready: "text-ok border-ok/40 bg-ok/10",
    active: "text-indigo-300 border-indigo-500/40 bg-indigo-500/10",
    human: "text-ok border-ok/40 bg-ok/10",
    needs_human: "text-warn border-warn/40 bg-warn/10",
    pending: "text-warn border-warn/40 bg-warn/10",
    processing: "text-warn border-warn/40 bg-warn/10",
    error: "text-bad border-bad/40 bg-bad/10",
    closed: "text-bad border-bad/40 bg-bad/10",
  };
  return <span className={cx("inline-flex items-center text-[11px] font-semibold px-2 py-0.5 rounded-full border", map[kind] || map.default)}>{children}</span>;
}

export function Msg({ type, children }: { type?: "err" | "ok"; children: React.ReactNode }) {
  if (!children) return <div className="min-h-[18px]" />;
  return <div className={cx("text-[13px] mt-2 min-h-[18px]", type === "err" ? "text-bad" : "text-ok")}>{children}</div>;
}

export function Table({ head, children }: { head: string[]; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto -mx-1">
      <table className="w-full border-collapse">
        <thead><tr>{head.map((h, i) => <th key={i} className="text-left text-[11px] uppercase tracking-wide text-muted font-semibold px-2 py-2 border-b border-line">{h}</th>)}</tr></thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}
export const Td = (p: React.TdHTMLAttributes<HTMLTableCellElement>) => <td {...p} className={cx("px-2 py-2.5 border-b border-line/60 text-sm align-top", p.className)} />;

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-sm text-muted py-6 text-center">{children}</div>;
}
export function Kpi({ n, l }: { n: React.ReactNode; l: string }) {
  return <div className="bg-gradient-to-b from-card to-card2 border border-line rounded-xl p-4"><div className="text-2xl font-extrabold tracking-tight">{n}</div><div className="text-xs text-muted mt-1">{l}</div></div>;
}
export function Spinner() { return <div className="flex items-center gap-2 text-muted text-sm py-6"><Loader2 className="w-4 h-4 animate-spin" /> Đang tải…</div>; }
