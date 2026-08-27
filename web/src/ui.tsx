import React, { useEffect } from "react";
import { Loader2, HelpCircle, X } from "lucide-react";

export const cx = (...c: (string | false | undefined)[]) => c.filter(Boolean).join(" ");

export function Button({ variant = "primary", size = "md", className, children, loading, ...p }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "sec" | "ghost" | "danger"; size?: "sm" | "md"; loading?: boolean }) {
  const base = "inline-flex items-center justify-center gap-2 font-semibold rounded-lg transition active:scale-[.98] disabled:opacity-50 disabled:pointer-events-none";
  const sizes = { sm: "text-xs px-2.5 py-1.5", md: "text-sm px-3.5 py-2" }[size];
  const variants = {
    primary: "text-[#0b0e1a] font-bold bg-pastel hover:brightness-105 shadow-[0_6px_24px_rgba(129,140,248,.35)]",
    sec: "bg-card2 border border-line hover:border-line2 text-fg",
    ghost: "bg-transparent border border-line hover:bg-card text-fg",
    danger: "bg-transparent border border-bad/60 text-bad hover:bg-bad/10",
  }[variant];
  return <button className={cx(base, sizes, variants, className)} {...p}>{loading && <Loader2 className="w-4 h-4 animate-spin" />}{children}</button>;
}

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cx("relative bg-gradient-to-b from-card to-card2 border border-line/90 rounded-2xl p-5 shadow-soft before:absolute before:inset-x-0 before:top-0 before:h-px before:rounded-t-2xl before:bg-gradient-to-r before:from-transparent before:via-white/12 before:to-transparent", className)}>{children}</div>;
}
export function CardTitle({ children, sub, right }: { children: React.ReactNode; sub?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 mb-4">
      <div><h3 className="text-[14px] font-semibold tracking-tight">{children}</h3>{sub && <p className="text-[12.5px] text-muted mt-0.5 font-normal leading-snug">{sub}</p>}</div>
      {right}
    </div>
  );
}

export function Info({ text }: { text: string }) {
  return (
    <span className="relative inline-flex group align-middle ml-1">
      <HelpCircle className="w-3.5 h-3.5 text-muted cursor-help" />
      <span role="tooltip" className="pointer-events-none absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-60 rounded-lg border border-line2 bg-card2 px-3 py-2 text-[12px] leading-snug text-fg font-normal opacity-0 group-hover:opacity-100 transition-opacity duration-150 shadow-soft z-50 text-left normal-case tracking-normal">
        {text}
      </span>
    </span>
  );
}

export function Field({ label, children, info }: { label: string; children: React.ReactNode; info?: string }) {
  return (
    <label className="block">
      <span className="flex items-center text-[11px] font-semibold uppercase tracking-wide text-muted mb-1.5">{label}{info && <Info text={info} />}</span>
      {children}
    </label>
  );
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
    active: "text-indigo-300 border-indigo-500/40 bg-indigo-500/10",
    paid: "text-ok border-ok/40 bg-ok/10",
    ready: "text-ok border-ok/40 bg-ok/10",
    human: "text-ok border-ok/40 bg-ok/10",
    needs_human: "text-warn border-warn/40 bg-warn/10",
    pending: "text-warn border-warn/40 bg-warn/10",
    processing: "text-warn border-warn/40 bg-warn/10",
    error: "text-bad border-bad/40 bg-bad/10",
    void: "text-bad border-bad/40 bg-bad/10",
    closed: "text-muted border-line",
  };
  return <span className={cx("inline-flex items-center text-[11px] font-semibold px-2 py-0.5 rounded-full border", map[kind] || map.default)}>{children}</span>;
}

export function Msg({ type, children }: { type?: "err" | "ok"; children: React.ReactNode }) {
  if (!children) return <div className="min-h-[18px]" />;
  return <div className={cx("text-[13px] mt-2 min-h-[18px] font-normal", type === "err" ? "text-bad" : "text-ok")}>{children}</div>;
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
export const Td = (p: React.TdHTMLAttributes<HTMLTableCellElement>) => <td {...p} className={cx("px-2 py-2.5 border-b border-line/60 text-sm align-top font-normal", p.className)} />;

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-sm text-muted py-8 text-center font-normal">{children}</div>;
}
export function Kpi({ n, l, info }: { n: React.ReactNode; l: string; info?: string }) {
  return (
    <div className="relative overflow-hidden bg-gradient-to-b from-card to-card2 border border-line rounded-xl p-4">
      <div className="absolute -right-6 -top-8 w-24 h-24 rounded-full bg-pastel-soft blur-2xl" />
      <div className="relative text-[22px] font-extrabold tracking-tight leading-none">{n}</div>
      <div className="relative text-xs text-muted mt-1.5 flex items-center font-normal">{l}{info && <Info text={info} />}</div>
    </div>
  );
}
export function Spinner() { return <div className="flex items-center gap-2 text-muted text-sm py-8 justify-center font-normal"><Loader2 className="w-4 h-4 animate-spin" /> Đang tải</div>; }

export function Modal({ open, onClose, title, sub, children, footer, size = "md" }: {
  open: boolean; onClose: () => void; title: string; sub?: string; children: React.ReactNode; footer?: React.ReactNode; size?: "sm" | "md" | "lg";
}) {
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  const w = { sm: "max-w-sm", md: "max-w-md", lg: "max-w-lg" }[size];
  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div className={cx("w-full bg-card border border-line2 rounded-2xl shadow-soft", w)} onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-3 px-5 pt-5">
          <div><h3 className="text-[15px] font-bold">{title}</h3>{sub && <p className="text-[13px] text-muted mt-0.5 font-normal">{sub}</p>}</div>
          <button onClick={onClose} className="text-muted hover:text-fg transition"><X className="w-5 h-5" /></button>
        </div>
        <div className="px-5 py-4">{children}</div>
        {footer && <div className="flex justify-end gap-2 px-5 py-4 border-t border-line">{footer}</div>}
      </div>
    </div>
  );
}
