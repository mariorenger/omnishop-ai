import React, { useEffect, useState } from "react";
import { Loader2, HelpCircle, X, CheckCircle2, AlertTriangle, Info as InfoIcon } from "lucide-react";

export const cx = (...c: (string | false | undefined)[]) => c.filter(Boolean).join(" ");

// ---- shared toast notifications (replaces native alert) --------------------
type Toast = { id: number; type: "ok" | "err" | "info"; text: string };
let _tid = 0;
const _subs = new Set<(t: Toast[]) => void>();
let _toasts: Toast[] = [];
const _emit = () => _subs.forEach((f) => f(_toasts));
export function notify(text: string, type: "ok" | "err" | "info" = "info", ms = 6000) {
  const t: Toast = { id: ++_tid, type, text };
  _toasts = [..._toasts, t];
  _emit();
  if (ms > 0) setTimeout(() => { _toasts = _toasts.filter((x) => x.id !== t.id); _emit(); }, ms);
}
export function Toaster() {
  const [ts, setTs] = useState<Toast[]>([]);
  useEffect(() => { _subs.add(setTs); return () => { _subs.delete(setTs); }; }, []);
  const dismiss = (id: number) => { _toasts = _toasts.filter((x) => x.id !== id); _emit(); };
  const style = { ok: "border-ok/40 bg-ok/10 text-ok", err: "border-bad/40 bg-bad/10 text-bad", info: "border-line2 bg-card2 text-fg" };
  const Icon = { ok: CheckCircle2, err: AlertTriangle, info: InfoIcon };
  return (
    <div className="fixed z-[100] bottom-4 right-4 flex flex-col gap-2 w-[calc(100%-2rem)] max-w-sm">
      {ts.map((t) => { const Ic = Icon[t.type]; return (
        <div key={t.id} className={cx("rounded-xl border px-3.5 py-3 text-[13px] font-normal shadow-soft backdrop-blur flex items-start gap-2.5", style[t.type])}>
          <Ic className="w-4 h-4 shrink-0 mt-0.5" />
          <span className="flex-1 leading-snug">{t.text}</span>
          <button onClick={() => dismiss(t.id)} className="shrink-0 opacity-70 hover:opacity-100"><X className="w-4 h-4" /></button>
        </div>
      ); })}
    </div>
  );
}

// ---- shared confirm dialog (replaces native confirm) ----------------------
type ConfirmOpts = { title?: string; message: React.ReactNode; confirmText?: string; cancelText?: string; danger?: boolean };
type ConfirmState = ConfirmOpts & { resolve: (v: boolean) => void };
let _confirmSub: ((c: ConfirmState | null) => void) | null = null;
export function confirmDialog(opts: ConfirmOpts): Promise<boolean> {
  return new Promise((resolve) => {
    if (!_confirmSub) { resolve(typeof window !== "undefined" ? window.confirm(typeof opts.message === "string" ? opts.message : (opts.title || "Xác nhận?")) : false); return; }
    _confirmSub({ ...opts, resolve });
  });
}
export function ConfirmHost() {
  const [c, setC] = useState<ConfirmState | null>(null);
  useEffect(() => { _confirmSub = setC; return () => { _confirmSub = null; }; }, []);
  const close = (v: boolean) => { setC((cur) => { cur?.resolve(v); return null; }); };
  return (
    <Modal open={!!c} onClose={() => close(false)} title={c?.title || "Xác nhận"} size="sm" z="z-[70]"
      footer={<><Button variant="sec" onClick={() => close(false)}>{c?.cancelText || "Huỷ"}</Button>
        <Button variant={c?.danger ? "danger" : "primary"} onClick={() => close(true)}>{c?.confirmText || "Đồng ý"}</Button></>}>
      <div className="text-[13.5px] text-fg font-normal leading-relaxed whitespace-pre-line">{c?.message}</div>
    </Modal>
  );
}

export function Button({ variant = "primary", size = "md", className, children, loading, ...p }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "sec" | "ghost" | "danger"; size?: "sm" | "md"; loading?: boolean }) {
  const base = "inline-flex items-center justify-center gap-2 font-semibold rounded-lg transition active:scale-[.98] disabled:opacity-50 disabled:pointer-events-none outline-none focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-2 focus-visible:ring-offset-bg";
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

const inputCls = "w-full bg-bg border border-line rounded-lg px-3 py-2 text-sm text-fg outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/25 disabled:opacity-60 disabled:cursor-not-allowed";
export const Input = (p: React.InputHTMLAttributes<HTMLInputElement>) => <input {...p} className={cx(inputCls, p.className)} />;
export const Textarea = (p: React.TextareaHTMLAttributes<HTMLTextAreaElement>) => <textarea {...p} className={cx(inputCls, "resize-y min-h-[80px]", p.className)} />;
export const Select = (p: React.SelectHTMLAttributes<HTMLSelectElement>) => <select {...p} className={cx(inputCls, "appearance-none ui-select cursor-pointer", p.className)} />;

export function Badge({ kind = "default", children }: { kind?: string; children: React.ReactNode }) {
  const map: Record<string, string> = {
    default: "text-muted border-line",
    ai: "text-indigo-300 border-indigo-500/40 bg-indigo-500/10",
    connected: "text-ok border-ok/40 bg-ok/10",
    active: "text-indigo-300 border-indigo-500/40 bg-indigo-500/10",
    paid: "text-ok border-ok/40 bg-ok/10",
    ready: "text-ok border-ok/40 bg-ok/10",
    human: "text-ok border-ok/40 bg-ok/10",
    needs_human: "text-warn border-warn/40 bg-warn/10",
    pending: "text-warn border-warn/40 bg-warn/10",
    queued: "text-warn border-warn/40 bg-warn/10",
    processing: "text-warn border-warn/40 bg-warn/10",
    degraded: "text-bad border-bad/40 bg-bad/10",
    disconnected: "text-bad border-bad/40 bg-bad/10",
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
    <div className="relative bg-gradient-to-b from-card to-card2 border border-line rounded-xl p-4">
      {/* clip the glow to the card WITHOUT clipping the tooltip that lives above */}
      <div className="absolute inset-0 overflow-hidden rounded-xl pointer-events-none"><div className="absolute -right-6 -top-8 w-24 h-24 rounded-full bg-pastel-soft blur-2xl" /></div>
      <div className="relative text-[22px] font-extrabold tracking-tight leading-none">{n}</div>
      <div className="relative text-xs text-muted mt-1.5 flex items-center font-normal">{l}{info && <Info text={info} />}</div>
    </div>
  );
}
export function Spinner() { return <div className="flex items-center gap-2 text-muted text-sm py-8 justify-center font-normal"><Loader2 className="w-4 h-4 animate-spin" /> Đang tải</div>; }

// "Load more" row for paginated lists. Shows a count when known, hides itself
// when there is nothing more to load.
export function LoadMore({ show, loading, onClick, shown, total }:
  { show: boolean; loading?: boolean; onClick: () => void; shown?: number; total?: number }) {
  if (!show) return total != null && (shown ?? 0) > 0
    ? <div className="text-[11px] text-muted text-center py-2 font-normal">Đã hiển thị tất cả {total}</div> : null;
  return (
    <div className="flex items-center justify-center gap-3 py-2">
      <Button variant="ghost" size="sm" loading={loading} onClick={onClick}>Xem thêm</Button>
      {total != null && <span className="text-[11px] text-muted font-normal">{shown ?? 0} / {total}</span>}
    </div>
  );
}

export function Modal({ open, onClose, title, sub, children, footer, size = "md", z = "z-50" }: {
  open: boolean; onClose: () => void; title: string; sub?: string; children: React.ReactNode; footer?: React.ReactNode; size?: "sm" | "md" | "lg" | "xl"; z?: string;
}) {
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  const w = { sm: "max-w-sm", md: "max-w-md", lg: "max-w-lg", xl: "max-w-5xl" }[size];
  // Backdrop clicks do NOT close the dialog on purpose — an accidental outside
  // click must never wipe a half-filled form. Close only via the X / Huỷ / Esc.
  return (
    <div className={cx("fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4", z)}>
      <div className={cx("w-full bg-card border border-line2 rounded-2xl shadow-soft max-h-[92vh] flex flex-col", w)}>
        <div className="flex items-start justify-between gap-3 px-5 pt-5 shrink-0">
          <div><h3 className="text-[15px] font-bold">{title}</h3>{sub && <p className="text-[13px] text-muted mt-0.5 font-normal">{sub}</p>}</div>
          <button onClick={onClose} className="text-muted hover:text-fg transition"><X className="w-5 h-5" /></button>
        </div>
        <div className="px-5 py-4 overflow-y-auto">{children}</div>
        {footer && <div className="flex justify-end gap-2 px-5 py-4 border-t border-line shrink-0">{footer}</div>}
      </div>
    </div>
  );
}
