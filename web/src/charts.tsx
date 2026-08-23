import React from "react";

// Colors: AI = indigo, Nhân viên = amber. Blue/amber is a colorblind-safe pair;
// identity is also carried by the legend + labels (never color alone). Text stays
// in muted ink, marks are thin with rounded tops and a 2px gap between segments.
const AI = "#6d7cff";
const HUMAN = "#f59e0b";

export function StackedBars({ data }: { data: { day: string; ai: number; human: number }[] }) {
  const W = 720, H = 190, padL = 28, padB = 26, padT = 8, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const max = Math.max(1, ...data.map((d) => d.ai + d.human));
  const n = data.length || 1;
  const step = cw / n;
  const bw = Math.min(22, step * 0.6);
  const y = (v: number) => (v / max) * ch;
  const ticks = [0, Math.round(max / 2), max];
  return (
    <div>
      <div className="flex items-center gap-4 mb-2 text-xs text-muted font-normal">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: AI }} /> AI tự trả lời</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: HUMAN }} /> Nhân viên</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Tin nhắn theo ngày">
        {ticks.map((t, i) => {
          const yy = padT + ch - y(t);
          return <g key={i}><line x1={padL} y1={yy} x2={W - padR} y2={yy} stroke="#232a3b" strokeWidth={1} /><text x={0} y={yy + 3} fill="#8b95ad" fontSize={9}>{t}</text></g>;
        })}
        {data.map((d, i) => {
          const x = padL + i * step + (step - bw) / 2;
          const hAi = y(d.ai), hHu = y(d.human);
          const yAi = padT + ch - hAi;
          const yHu = yAi - hHu - (hHu > 0 && hAi > 0 ? 2 : 0);
          return (
            <g key={i}>
              {d.ai > 0 && <rect x={x} y={yAi} width={bw} height={hAi} rx={2} fill={AI}><title>{d.day}: AI {d.ai}</title></rect>}
              {d.human > 0 && <rect x={x} y={yHu} width={bw} height={hHu} rx={2} fill={HUMAN}><title>{d.day}: Nhân viên {d.human}</title></rect>}
              {i % 2 === 0 && <text x={x + bw / 2} y={H - 8} fill="#8b95ad" fontSize={9} textAnchor="middle">{d.day}</text>}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function IntentBars({ data }: { data: { intent: string; count: number }[] }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  const label: Record<string, string> = { product: "Hỏi sản phẩm", knowledge: "Hỏi thông tin", order: "Hỏi đơn hàng", "khác": "Khác" };
  if (!data.length) return <div className="text-sm text-muted py-6 text-center font-normal">Chưa có dữ liệu.</div>;
  return (
    <div className="space-y-2.5">
      {data.map((d, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="w-28 text-[13px] text-muted font-normal shrink-0">{label[d.intent] || d.intent}</div>
          <div className="flex-1 h-2.5 rounded-full bg-bg overflow-hidden"><div className="h-full rounded-full" style={{ width: `${(d.count / max) * 100}%`, background: AI }} /></div>
          <div className="w-8 text-right text-[13px] font-semibold">{d.count}</div>
        </div>
      ))}
    </div>
  );
}
