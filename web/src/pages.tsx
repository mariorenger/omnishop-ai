import React, { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Badge, Button, Card, CardTitle, Empty, Field, Info, Input, Kpi, Modal, Msg, Select, Spinner, Table, Td, Textarea } from "./ui";
import { StackedBars, IntentBars } from "./charts";
import { RefreshCw, Upload, Plug, Send, UserPlus, CheckCircle2, ArrowUpRight, Bot, MessageSquare, Plus } from "lucide-react";
import QRCode from "qrcode";

const fmt = (n: number) => n.toLocaleString("vi-VN");

// ============================================================ Overview
export function Overview({ shopId }: { shopId: string }) {
  const [a, setA] = useState<any>(null); const [sub, setSub] = useState<any>(null);
  const [convs, setConvs] = useState<any[]>([]); const [err, setErr] = useState("");
  useEffect(() => {
    setA(null);
    api.get(`/api/analytics/overview?shop_id=${shopId}`).then(setA).catch((e) => setErr(e.message));
    api.get("/api/subscription").then(setSub).catch(() => {});
    api.get(`/api/conversations?shop_id=${shopId}`).then((d) => setConvs(d.slice(0, 6))).catch(() => {});
  }, [shopId]);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!a) return <Spinner />;
  const t = a.totals;
  return (
    <div className="space-y-4">
      <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
        <Kpi n={fmt(t.conversations)} l="Hội thoại" info="Tổng số cuộc trò chuyện với khách trên cửa hàng này." />
        <Kpi n={`${t.ai_rate}%`} l="Tỷ lệ AI tự xử lý" info="Phần trăm câu trả lời do AI đảm nhận, phần còn lại do nhân viên." />
        <Kpi n={`${fmt(t.ai_messages)}${sub ? " / " + fmt(sub.quota.limit) : ""}`} l="Tin nhắn AI tháng này" info="Số câu trả lời AI đã dùng trong tháng so với hạn mức của gói." />
        <Kpi n={`$${t.cost_month.toFixed(2)}`} l="Chi phí AI ước tính" info="Chi phí suy luận ước tính trong tháng, dùng để theo dõi biên lợi nhuận." />
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardTitle sub="14 ngày gần nhất">Lưu lượng tin nhắn</CardTitle>
          <StackedBars data={a.series} />
        </Card>
        <Card>
          <CardTitle sub="30 ngày gần nhất">Loại câu hỏi phổ biến</CardTitle>
          <IntentBars data={a.intents} />
        </Card>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardTitle>Chỉ số vận hành</CardTitle>
          <div className="space-y-3 text-sm">
            <Row l="Chuyển nhân viên" v={fmt(t.handoff)} info="Số hội thoại đã chuyển cho nhân viên hỗ trợ." />
            <Row l="Nhân viên đã trả lời" v={fmt(t.human_replies)} />
            <Row l="Độ trễ trung bình" v={`${t.avg_latency_ms} ms`} info="Thời gian AI tạo câu trả lời trung bình." />
            <Row l="Khách đã nhắn" v={fmt(t.customer_messages)} />
          </div>
        </Card>
        <Card className="lg:col-span-2">
          <CardTitle sub="Mới nhất" right={<a className="text-accent text-xs font-semibold" href="#">Xem tất cả</a>}>Hội thoại gần đây</CardTitle>
          {convs.length === 0 ? <Empty>Chưa có hội thoại nào.</Empty> :
            <Table head={["Khách", "Trạng thái", "Nội dung gần nhất"]}>
              {convs.map((c) => <tr key={c.id}><Td className="font-semibold">{c.customer_ref}</Td><Td><Badge kind={c.status}>{statusLabel(c.status)}</Badge></Td><Td className="text-muted">{(c.last_message || "").slice(0, 60)}</Td></tr>)}
            </Table>}
        </Card>
      </div>
    </div>
  );
}
function Row({ l, v, info }: { l: string; v: string; info?: string }) {
  return <div className="flex items-center justify-between border-b border-line/50 pb-2.5"><span className="text-muted flex items-center font-normal">{l}{info && <Info text={info} />}</span><span className="font-semibold">{v}</span></div>;
}
const statusLabel = (s: string) => ({ ai: "AI xử lý", needs_human: "Cần hỗ trợ", human: "Nhân viên", closed: "Đã đóng" } as any)[s] || s;

// ============================================================ Products
export function Products({ shopId }: { shopId: string }) {
  const [items, setItems] = useState<any[] | null>(null); const [bots, setBots] = useState<any[]>([]);
  const [q, setQ] = useState(""); const [open, setOpen] = useState(false);
  const [f, setF] = useState<any>({ name: "", price: "", currency: "VND", sku: "", category: "", description: "", variants: "", bot_id: "" });
  const [busy, setBusy] = useState(false); const [err, setErr] = useState("");
  const load = () => { api.get(`/api/products?shop_id=${shopId}`).then(setItems).catch((e) => setErr(e.message)); api.get(`/api/bots?shop_id=${shopId}`).then(setBots).catch(() => {}); };
  useEffect(() => { setItems(null); load(); }, [shopId]);
  const add = async () => {
    setBusy(true); setErr("");
    const variants = (f.variants || "").split(",").map((s: string) => s.trim()).filter(Boolean).map((s: string) => { const [n, st] = s.split(":"); return { name: (n || "").trim(), stock: parseInt(st || "0") || 0 }; });
    try {
      await api.post("/api/products", { shop_id: shopId, name: f.name, price: f.price ? parseFloat(f.price) : null, currency: f.currency, sku: f.sku, description: f.description, attributes: f.category ? { category: f.category } : {}, variants, bot_id: f.bot_id || null });
      setF({ name: "", price: "", currency: "VND", sku: "", category: "", description: "", variants: "", bot_id: "" }); setOpen(false); load();
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  const totalStock = (p: any) => (p.variants || []).reduce((s: number, v: any) => s + (v.stock || 0), 0);
  const botName = (id: string) => bots.find((b) => b.id === id)?.name;
  const rows = (items || []).filter((p) => !q || p.name.toLowerCase().includes(q.toLowerCase()) || (p.sku || "").toLowerCase().includes(q.toLowerCase()));
  return (
    <Card>
      <CardTitle sub="Trợ lý AI dùng dữ liệu này để trả lời về giá, tồn kho và biến thể."
        right={<div className="flex gap-2"><Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Tìm tên hoặc SKU" className="w-44" /><Button size="sm" onClick={() => setOpen(true)}><Plus className="w-4 h-4" /> Thêm</Button></div>}>Sản phẩm</CardTitle>
      {!items ? <Spinner /> : rows.length === 0 ? <Empty>{q ? "Không tìm thấy sản phẩm." : "Chưa có sản phẩm nào."}</Empty> :
        <Table head={["Sản phẩm", "SKU", "Giá", "Tồn kho", "Biến thể", "Áp dụng"]}>
          {rows.map((p) => (
            <tr key={p.id}>
              <Td><div className="font-medium">{p.name}</div><div className="text-xs text-muted line-clamp-1">{p.description || ""}</div></Td>
              <Td className="text-muted">{p.sku || "—"}</Td>
              <Td className="whitespace-nowrap">{p.price != null ? `${fmt(p.price)} ${p.currency}` : "Liên hệ"}</Td>
              <Td>{p.variants && p.variants.length ? <span className={totalStock(p) === 0 ? "text-bad" : ""}>{totalStock(p)}</span> : "—"}</Td>
              <Td><div className="flex flex-wrap gap-1">{(p.variants || []).slice(0, 4).map((v: any, i: number) => <span key={i} className={"text-[11px] border rounded px-1.5 py-0.5 font-normal " + (v.stock === 0 ? "border-bad/40 text-bad" : "border-line text-muted")}>{v.name}·{v.stock}</span>)}{(p.variants || []).length > 4 && <span className="text-[11px] text-muted">+{p.variants.length - 4}</span>}</div></Td>
              <Td className="text-muted text-xs">{p.bot_id ? botName(p.bot_id) || "1 trợ lý" : "Tất cả"}</Td>
            </tr>
          ))}
        </Table>}
      <Msg type="err">{err}</Msg>
      <Modal open={open} onClose={() => setOpen(false)} title="Thêm sản phẩm" size="lg"
        footer={<><Button variant="sec" onClick={() => setOpen(false)}>Huỷ</Button><Button loading={busy} onClick={add}>Lưu sản phẩm</Button></>}>
        <div className="grid sm:grid-cols-2 gap-3">
          <Field label="Tên sản phẩm"><Input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="Áo thun cotton" /></Field>
          <Field label="SKU"><Input value={f.sku} onChange={(e) => setF({ ...f, sku: e.target.value })} placeholder="AO-001" /></Field>
        </div>
        <div className="grid sm:grid-cols-3 gap-3 mt-3">
          <Field label="Giá"><Input type="number" value={f.price} onChange={(e) => setF({ ...f, price: e.target.value })} /></Field>
          <Field label="Tiền tệ"><Input value={f.currency} onChange={(e) => setF({ ...f, currency: e.target.value })} /></Field>
          <Field label="Danh mục"><Input value={f.category} onChange={(e) => setF({ ...f, category: e.target.value })} placeholder="Áo" /></Field>
        </div>
        <div className="mt-3"><Field label="Mô tả"><Textarea value={f.description} onChange={(e) => setF({ ...f, description: e.target.value })} /></Field></div>
        <div className="mt-3"><Field label="Biến thể và tồn kho" info="Dạng Tên:Số lượng, cách nhau bởi dấu phẩy. Ví dụ: Size M:10, Size L:3."><Input value={f.variants} onChange={(e) => setF({ ...f, variants: e.target.value })} placeholder="Size M:10, Size L:3" /></Field></div>
        <div className="mt-3"><Field label="Áp dụng cho" info="Chọn một trợ lý để chỉ trợ lý đó dùng sản phẩm này, hoặc Tất cả trợ lý."><Select value={f.bot_id} onChange={(e) => setF({ ...f, bot_id: e.target.value })}><option value="">Tất cả trợ lý</option>{bots.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}</Select></Field></div>
        <Msg type="err">{err}</Msg>
      </Modal>
    </Card>
  );
}

// ============================================================ Knowledge
export function Knowledge({ shopId }: { shopId: string }) {
  const [docs, setDocs] = useState<any[] | null>(null);
  const [title, setTitle] = useState(""); const [text, setText] = useState(""); const [botId, setBotId] = useState("");
  const [bots, setBots] = useState<any[]>([]);
  const [msg, setMsg] = useState(""); const [err, setErr] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const load = () => { api.get(`/api/knowledge/documents?shop_id=${shopId}`).then((d) => { setDocs(d); if (d.some((x: any) => x.status !== "ready" && x.status !== "error")) setTimeout(load, 2000); }).catch((e) => setErr(e.message)); api.get(`/api/bots?shop_id=${shopId}`).then(setBots).catch(() => {}); };
  useEffect(() => { setDocs(null); load(); }, [shopId]);
  const addText = async () => { setErr(""); setMsg(""); try { await api.post("/api/knowledge/documents", { shop_id: shopId, title, text, bot_id: botId || null }); setTitle(""); setText(""); setMsg("Đã thêm tài liệu, đang xử lý nội dung."); load(); } catch (e: any) { setErr(e.message); } };
  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return; setErr(""); setMsg(`Đang xử lý ${file.name}`);
    const fd = new FormData(); fd.append("shop_id", shopId); fd.append("file", file); if (botId) fd.append("bot_id", botId);
    try { const r = await api.upload("/api/knowledge/upload", fd); setMsg(`Đã tải ${file.name}: ${fmt(r.extracted_chars)} ký tự, ${r.chunks} đoạn.`); load(); }
    catch (ex: any) { setErr(ex.message); setMsg(""); } finally { if (fileRef.current) fileRef.current.value = ""; }
  };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Nhập nội dung hoặc tải tệp. Nội dung được xử lý và lập chỉ mục để trợ lý AI tra cứu.">Tài liệu kiến thức</CardTitle>
        <div onClick={() => fileRef.current?.click()} className="border border-dashed border-line rounded-xl p-6 text-center text-muted cursor-pointer hover:border-accent hover:text-fg transition flex flex-col items-center gap-2">
          <Upload className="w-6 h-6" />
          <div className="text-sm font-medium text-fg">Tải tệp lên</div>
          <div className="text-xs font-normal">Hỗ trợ PDF, Word, PowerPoint, Excel, CSV, HTML, văn bản và hình ảnh. Ảnh và PDF scan được nhận dạng bằng OCR.</div>
        </div>
        <input ref={fileRef} type="file" className="hidden" onChange={onFile} />
        <div className="mt-4 grid gap-3">
          <div className="grid sm:grid-cols-2 gap-3">
            <Field label="Tiêu đề"><Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Chính sách đổi trả" /></Field>
            <Field label="Áp dụng cho" info="Chọn một trợ lý để chỉ trợ lý đó dùng tài liệu này, hoặc Tất cả trợ lý."><Select value={botId} onChange={(e) => setBotId(e.target.value)}><option value="">Tất cả trợ lý</option>{bots.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}</Select></Field>
          </div>
          <Field label="Nội dung"><Textarea value={text} onChange={(e) => setText(e.target.value)} className="min-h-[110px]" /></Field>
        </div>
        <div className="mt-3"><Button variant="sec" onClick={addText}>Thêm nội dung</Button></div>
        <Msg type="ok">{msg}</Msg><Msg type="err">{err}</Msg>
      </Card>
      <Card>
        <CardTitle>Tài liệu đã tải</CardTitle>
        {!docs ? <Spinner /> : docs.length === 0 ? <Empty>Chưa có tài liệu nào.</Empty> :
          <Table head={["Tiêu đề", "Nguồn", "Trạng thái", "Số đoạn"]}>
            {docs.map((d) => <tr key={d.id}><Td className="font-semibold">{d.title}</Td><Td className="text-muted">{d.source || "Nhập tay"}</Td><Td><Badge kind={d.status}>{docStatus(d.status)}</Badge></Td><Td>{d.chunks}</Td></tr>)}
          </Table>}
      </Card>
    </div>
  );
}
const docStatus = (s: string) => ({ pending: "Đang chờ", processing: "Đang xử lý", ready: "Sẵn sàng", error: "Lỗi" } as any)[s] || s;

// ============================================================ Channels
const channelStatus = (s: string) => ({ connected: "Đang hoạt động", degraded: "Cần kiểm tra", pending: "Chờ kích hoạt", disconnected: "Đã ngắt" } as any)[s] || s;

export function Channels({ shopId }: { shopId: string }) {
  const [items, setItems] = useState<any[] | null>(null); const [kinds, setKinds] = useState<any[]>([]);
  const [err, setErr] = useState(""); const [open, setOpen] = useState(false);
  const [kind, setKind] = useState("website"); const [name, setName] = useState(""); const [greeting, setGreeting] = useState("Xin chào! Mình có thể giúp gì cho bạn?");
  const [creds, setCreds] = useState<Record<string, string>>({}); const [busy, setBusy] = useState(false); const [ferr, setFerr] = useState("");
  const [editing, setEditing] = useState<any>(null); const [bots, setBots] = useState<any[]>([]); const [botId, setBotId] = useState("");
  const load = () => { api.get(`/api/channels?shop_id=${shopId}`).then(setItems).catch((e) => setErr(e.message)); api.get("/api/channels/kinds").then(setKinds).catch(() => {}); api.get(`/api/bots?shop_id=${shopId}`).then(setBots).catch(() => {}); };
  useEffect(() => { setItems(null); load(); }, [shopId]);
  const spec = kinds.find((k) => k.kind === kind);
  const connect = async () => {
    setBusy(true); setFerr("");
    try { await api.post("/api/channels", { shop_id: shopId, kind, name, greeting, credentials: creds, bot_id: botId || null }); setOpen(false); setCreds({}); setName(""); setBotId(""); load(); }
    catch (e: any) { setFerr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Kết nối các kênh bán hàng để trợ lý AI trả lời khách trên mọi nơi."
          right={<div className="flex gap-2">
            <Button size="sm" variant="sec" onClick={async () => { try { const r = await api.get(`/api/channels/oauth/meta/start?shop_id=${shopId}`); location.href = r.url; } catch (e: any) { alert(e.message); } }}>Kết nối Facebook</Button>
            <Button size="sm" onClick={() => setOpen(true)}><Plug className="w-4 h-4" /> Kết nối kênh</Button>
          </div>}>Kênh kết nối</CardTitle>
        <Msg type="err">{err}</Msg>
        {!items ? <Spinner /> : items.length === 0 ? <Empty>Chưa có kênh nào. Bấm Kết nối kênh để bắt đầu.</Empty> :
          <div className="space-y-3">
            {items.map((ch) => {
              const url = ch.public_key ? `${location.origin}/widget.html?key=${ch.public_key}` : "";
              const isMeta = ch.kind === "messenger" || ch.kind === "instagram";
              return (
                <div key={ch.id} className="border border-line rounded-xl p-4">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-semibold">{ch.name}</span>
                    <Badge kind={ch.status}>{channelStatus(ch.status)}</Badge>
                    {ch.bot_name && <span className="text-xs text-muted inline-flex items-center gap-1"><Bot className="w-3.5 h-3.5" />{ch.bot_name}</span>}
                    <span className="flex-1" />
                    {isMeta && <Button size="sm" variant="ghost" onClick={async () => { await api.post(`/api/channels/${ch.id}/verify`); load(); }}>Kiểm tra</Button>}
                    {ch.kind !== "website" && <Button size="sm" variant="ghost" onClick={() => setEditing(ch)}>Chỉnh sửa</Button>}
                    <Button size="sm" variant="danger" onClick={async () => { if (confirm("Ngắt kết nối kênh này?")) { await api.del(`/api/channels/${ch.id}`); load(); } }}>Ngắt kết nối</Button>
                  </div>
                  {url && <div className="mt-3 flex gap-4 flex-col sm:flex-row">
                    <div className="flex-1 min-w-0">
                      <Field label="Mã tích hợp website" info="Dán đoạn mã này vào website của bạn để hiển thị khung chat.">
                        <pre className="bg-bg border border-line rounded-lg p-3 text-xs overflow-x-auto font-normal">{`<iframe src="${url}" style="border:0;width:380px;height:560px"></iframe>`}</pre>
                      </Field>
                      <a className="text-accent text-sm font-semibold inline-flex items-center gap-1 mt-2" href={url} target="_blank">Xem thử tiện ích <ArrowUpRight className="w-3.5 h-3.5" /></a>
                    </div>
                    <WidgetQR url={url} />
                  </div>}
                </div>
              );
            })}
          </div>}
      </Card>
      <EditChannelModal channel={editing} kinds={kinds} bots={bots} onClose={() => setEditing(null)} onSaved={() => { setEditing(null); load(); }} />
      <Modal open={open} onClose={() => setOpen(false)} title="Kết nối kênh" size="md"
        footer={<><Button variant="sec" onClick={() => setOpen(false)}>Huỷ</Button><Button loading={busy} disabled={spec && !spec.allowed} onClick={connect}>Kết nối</Button></>}>
        <Field label="Loại kênh"><Select value={kind} onChange={(e) => { setKind(e.target.value); setCreds({}); }}>
          {kinds.map((k) => <option key={k.kind} value={k.kind} disabled={!k.allowed}>{k.label}{k.allowed ? "" : " — không có trong gói"}</option>)}
        </Select></Field>
        {spec?.note && <p className="text-[12px] text-muted mt-2 font-normal">{spec.note}</p>}
        <div className="mt-3"><Field label="Trợ lý xử lý" info="Chọn trợ lý AI sẽ trả lời trên kênh này. Để trống sẽ dùng trợ lý mặc định.">
          <Select value={botId} onChange={(e) => setBotId(e.target.value)}><option value="">Trợ lý mặc định</option>{bots.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}</Select>
        </Field></div>
        {kind === "website" ? (
          <><div className="mt-3"><Field label="Tên hiển thị"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Tiện ích website" /></Field></div>
            <div className="mt-3"><Field label="Lời chào"><Input value={greeting} onChange={(e) => setGreeting(e.target.value)} /></Field></div></>
        ) : (
          <>
            <div className="mt-3"><Field label="Tên hiển thị"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder={spec?.label} /></Field></div>
            {(spec?.fields || []).map((f: any) => (
              <div className="mt-3" key={f.key}><Field label={f.label + (f.required ? "" : " (tuỳ chọn)")}>
                <Input type={f.secret ? "password" : "text"} value={creds[f.key] || ""} onChange={(e) => setCreds({ ...creds, [f.key]: e.target.value })} />
              </Field></div>
            ))}
          </>
        )}
        <Msg type="err">{ferr}</Msg>
      </Modal>
    </div>
  );
}

function WidgetQR({ url }: { url: string }) {
  const [src, setSrc] = useState("");
  useEffect(() => { QRCode.toDataURL(url, { margin: 1, width: 128, color: { dark: "#0a0c11", light: "#ffffff" } }).then(setSrc).catch(() => {}); }, [url]);
  if (!src) return null;
  return (
    <div className="text-center shrink-0">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-muted mb-1.5">Mã QR</div>
      <img src={src} alt="QR" className="w-28 h-28 rounded-lg border border-line bg-white p-1" />
      <div className="text-[11px] text-muted mt-1 font-normal">Quét để mở khung chat</div>
    </div>
  );
}

function EditChannelModal({ channel, kinds, bots, onClose, onSaved }: { channel: any; kinds: any[]; bots: any[]; onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState(""); const [creds, setCreds] = useState<Record<string, string>>({}); const [botId, setBotId] = useState("");
  const [busy, setBusy] = useState(false); const [err, setErr] = useState("");
  useEffect(() => { if (channel) { setName(channel.name || ""); setCreds({}); setBotId(channel.bot_id || ""); setErr(""); } }, [channel]);
  if (!channel) return null;
  const spec = kinds.find((k) => k.kind === channel.kind);
  const save = async () => {
    setBusy(true); setErr("");
    try { await api.put(`/api/channels/${channel.id}`, { name, credentials: creds, bot_id: botId }); onSaved(); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <Modal open={!!channel} onClose={onClose} title="Chỉnh sửa kênh" sub={spec?.label}
      footer={<><Button variant="sec" onClick={onClose}>Huỷ</Button><Button loading={busy} onClick={save}>Lưu</Button></>}>
      <Field label="Tên hiển thị"><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>
      <div className="mt-3"><Field label="Trợ lý xử lý"><Select value={botId} onChange={(e) => setBotId(e.target.value)}><option value="">Trợ lý mặc định</option>{bots.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}</Select></Field></div>
      {(spec?.fields || []).map((f: any) => (
        <div className="mt-3" key={f.key}><Field label={f.label} info={f.secret ? "Để trống nếu giữ giá trị hiện tại." : undefined}>
          <Input type={f.secret ? "password" : "text"} placeholder={f.secret ? "••• (giữ nguyên)" : ""} value={creds[f.key] || ""} onChange={(e) => setCreds({ ...creds, [f.key]: e.target.value })} />
        </Field></div>
      ))}
      <Msg type="err">{err}</Msg>
    </Modal>
  );
}

// ============================================================ Inbox
export function Inbox({ shopId }: { shopId: string }) {
  const [convs, setConvs] = useState<any[] | null>(null);
  const [active, setActive] = useState<any>(null); const [msgs, setMsgs] = useState<any[]>([]); const [reply, setReply] = useState("");
  const loadList = () => api.get(`/api/conversations?shop_id=${shopId}`).then(setConvs);
  useEffect(() => { setConvs(null); setActive(null); loadList(); }, [shopId]);
  const open = async (c: any) => { setActive(c); setMsgs(await api.get(`/api/conversations/${c.id}/messages`)); };
  const send = async () => { if (!reply.trim()) return; await api.post(`/api/conversations/${active.id}/reply`, { text: reply }); setReply(""); open(active); };
  return (
    <Card>
      <CardTitle sub="Toàn bộ hội thoại từ mọi kênh, có thể tiếp quản khi AI chuyển cho nhân viên.">Hộp thư hợp nhất</CardTitle>
      <div className="flex gap-4 items-start flex-col md:flex-row">
        <div className="flex-1 min-w-[220px] w-full space-y-2">
          {!convs ? <Spinner /> : convs.length === 0 ? <Empty>Chưa có hội thoại. Hãy thử nhắn qua tiện ích website.</Empty> :
            convs.map((c) => (
              <div key={c.id} onClick={() => open(c)} className={"cursor-pointer rounded-xl border p-3 transition " + (active?.id === c.id ? "border-accent bg-card2" : "border-line hover:bg-card2")}>
                <div className="flex items-center gap-2"><span className="font-semibold text-sm">{c.customer_ref}</span><Badge kind={c.status}>{statusLabel(c.status)}</Badge></div>
                <div className="text-xs text-muted mt-1 line-clamp-1 font-normal">{(c.last_message || "").slice(0, 70)}</div>
              </div>
            ))}
        </div>
        <div className="flex-[1.4] min-w-[260px] w-full">
          {!active ? <Empty>Chọn một hội thoại để xem chi tiết.</Empty> : (
            <div>
              <div className="flex items-center gap-2 mb-3"><span className="font-semibold text-sm">Khách: {active.customer_ref}</span><Badge kind={active.status}>{statusLabel(active.status)}</Badge></div>
              <div className="max-h-[52vh] overflow-auto pr-1 space-y-2">
                {msgs.map((m, i) => (
                  <div key={i} className={"px-3 py-2 rounded-xl text-sm max-w-[85%] whitespace-pre-wrap font-normal " +
                    (m.role === "customer" ? "bg-bg border border-line ml-auto" : m.role === "ai" ? "bg-indigo-900/40" : m.role === "agent" ? "bg-emerald-900/40" : "bg-amber-900/30 text-xs")}>{m.content}</div>
                ))}
              </div>
              <div className="flex gap-2 mt-3">
                <Input value={reply} onChange={(e) => setReply(e.target.value)} placeholder="Nhập câu trả lời của nhân viên" onKeyDown={(e) => e.key === "Enter" && send()} />
                <Button onClick={send}><Send className="w-4 h-4" /></Button>
                <Button variant="sec" onClick={async () => { await api.post(`/api/conversations/${active.id}/close`); loadList(); setActive(null); }}>Đóng</Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

// ============================================================ Members
export function Members({ role }: { role: string }) {
  const [rows, setRows] = useState<any[] | null>(null); const [open, setOpen] = useState(false);
  const [email, setEmail] = useState(""); const [r, setR] = useState("agent"); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  const canManage = role === "owner" || role === "admin";
  const load = () => api.get("/api/members").then(setRows).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);
  const invite = async () => { setBusy(true); setErr(""); try { await api.post("/api/members", { email, role: r }); setOpen(false); setEmail(""); load(); } catch (e: any) { setErr(e.message); } finally { setBusy(false); } };
  return (
    <Card>
      <CardTitle sub="Phân quyền theo vai trò: Chủ sở hữu, Quản trị, Nhân viên và Người xem."
        right={canManage ? <Button size="sm" onClick={() => setOpen(true)}><UserPlus className="w-4 h-4" /> Thêm thành viên</Button> : undefined}>Thành viên</CardTitle>
      {!rows ? <Spinner /> : rows.length === 0 ? <Empty>Chưa có thành viên.</Empty> :
        <Table head={["Email", "Vai trò"]}>
          {rows.map((m, i) => <tr key={i}><Td className="font-semibold">{m.email}</Td><Td><Badge kind="active">{roleLabel(m.role)}</Badge></Td></tr>)}
        </Table>}
      <Msg type="err">{!open ? err : ""}</Msg>
      <Modal open={open} onClose={() => setOpen(false)} title="Thêm thành viên" sub="Người dùng cần đã có tài khoản OmniShop AI."
        footer={<><Button variant="sec" onClick={() => setOpen(false)}>Huỷ</Button><Button loading={busy} onClick={invite}>Gửi lời mời</Button></>}>
        <Field label="Email"><Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="nhanvien@example.com" /></Field>
        <div className="mt-3"><Field label="Vai trò" info="Quản trị cấu hình cửa hàng; Nhân viên trả lời hội thoại; Người xem chỉ đọc.">
          <Select value={r} onChange={(e) => setR(e.target.value)}><option value="admin">Quản trị</option><option value="agent">Nhân viên</option><option value="viewer">Người xem</option></Select></Field></div>
        <Msg type="err">{err}</Msg>
      </Modal>
    </Card>
  );
}
const roleLabel = (r: string) => ({ owner: "Chủ sở hữu", admin: "Quản trị", agent: "Nhân viên", viewer: "Người xem" } as any)[r] || r;

// ============================================================ Billing
export function Billing({ role }: { role: string }) {
  const [plans, setPlans] = useState<any[] | null>(null); const [sub, setSub] = useState<any>(null);
  const [invoices, setInvoices] = useState<any[]>([]); const [err, setErr] = useState("");
  const [checkout, setCheckout] = useState<any>(null); const [busy, setBusy] = useState(false);
  const isOwner = role === "owner";
  const load = () => Promise.all([api.get("/api/plans"), api.get("/api/subscription"), api.get("/api/billing/invoices")])
    .then(([p, s, i]) => { setPlans(p); setSub(s); setInvoices(i); }).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!plans || !sub) return <Spinner />;
  const cur = sub.entitlements._plan;
  const pick = async (p: any) => {
    if (p.price_month === 0) { await api.post("/api/subscription", { plan_code: p.code }); await load(); return; }
    const co = await api.post("/api/billing/checkout", { plan_code: p.code }); setCheckout(co);
  };
  const confirm = async () => { setBusy(true); try { await api.post(`/api/billing/checkout/${checkout.invoice_id}/confirm`); setCheckout(null); await load(); } finally { setBusy(false); } };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub={`Hạn mức tin nhắn AI: ${fmt(sub.quota.used)} / ${fmt(sub.quota.limit)} trong tháng.`}>Gói hiện tại: {sub.entitlements._plan_name}</CardTitle>
        {!isOwner && <p className="text-[13px] text-muted font-normal">Chỉ Chủ sở hữu mới thay đổi được gói dịch vụ.</p>}
      </Card>
      <div className="grid md:grid-cols-3 gap-4">
        {plans.map((p) => (
          <Card key={p.code} className={cur === p.code ? "border-accent shadow-glow" : ""}>
            <div className="flex items-baseline justify-between"><span className="font-bold text-[15px]">{p.name}</span><span className="text-muted text-sm font-normal">${p.price_month}<span className="text-xs">/tháng</span></span></div>
            <ul className="text-[13px] text-muted mt-3 space-y-1.5 font-normal">
              <li>{fmt(p.entitlements.ai_messages_month)} tin nhắn AI mỗi tháng</li>
              <li>{p.entitlements.shops} cửa hàng</li>
              <li>Kênh: {(p.entitlements.channels_allowed || []).join(", ")}</li>
            </ul>
            <div className="mt-4">{cur === p.code ? <Badge kind="active">Đang sử dụng</Badge> :
              isOwner ? <Button variant={p.price_month ? "primary" : "sec"} onClick={() => pick(p)}>{p.price_month ? "Nâng cấp" : "Chuyển gói"}</Button> : <span className="text-xs text-muted">—</span>}</div>
          </Card>
        ))}
      </div>
      <Card>
        <CardTitle>Hoá đơn</CardTitle>
        {invoices.length === 0 ? <Empty>Chưa có hoá đơn nào.</Empty> :
          <Table head={["Ngày", "Gói", "Số tiền", "Trạng thái"]}>
            {invoices.map((iv) => <tr key={iv.id}><Td className="font-normal">{new Date(iv.created_at).toLocaleDateString("vi-VN")}</Td><Td className="font-semibold">{iv.plan}</Td><Td>${iv.amount}</Td><Td><Badge kind={iv.status}>{iv.status === "paid" ? "Đã thanh toán" : iv.status === "pending" ? "Chờ thanh toán" : "Đã huỷ"}</Badge></Td></tr>)}
          </Table>}
      </Card>
      <Modal open={!!checkout} onClose={() => setCheckout(null)} title="Xác nhận thanh toán"
        sub={checkout ? `Nâng cấp lên gói ${checkout.plan} — $${checkout.amount}/tháng.` : ""}
        footer={<><Button variant="sec" onClick={() => setCheckout(null)}>Huỷ</Button><Button loading={busy} onClick={confirm}>Xác nhận & kích hoạt</Button></>}>
        <div className="flex items-start gap-3 text-sm">
          <CheckCircle2 className="w-5 h-5 text-ok shrink-0 mt-0.5" />
          <p className="text-muted font-normal">Đây là luồng thanh toán demo (chuyển khoản thủ công). Cổng thanh toán thực tế như Stripe hoặc VNPay sẽ được tích hợp qua cùng một giao diện.</p>
        </div>
      </Modal>
    </div>
  );
}

// ============================================================ LLM form (reused)
export function LlmForm({ initial, providers, endpoints }: {
  initial: any; providers: { id: string; label: string; base_url?: string }[];
  endpoints: { save: string; test?: string; models?: string; del?: string };
}) {
  const findIdx = () => { if (!initial) return 0; const i = providers.findIndex((p) => p.id === initial.provider && (!p.base_url || p.base_url === initial.base_url)); return i >= 0 ? i : Math.max(0, providers.findIndex((p) => p.id === initial.provider)); };
  const [idx, setIdx] = useState(findIdx());
  const [model, setModel] = useState(initial?.model || "");
  const [baseUrl, setBaseUrl] = useState(initial?.base_url || "");
  const [apiKey, setApiKey] = useState("");
  const [maxTokens, setMaxTokens] = useState(initial?.extra?.max_tokens ? String(initial.extra.max_tokens) : "");
  const [models, setModels] = useState<string[]>([]);
  const [busy, setBusy] = useState(""); const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  const body = () => ({ provider: providers[idx].id, model, base_url: baseUrl, api_key: apiKey || null, max_tokens: maxTokens ? parseInt(maxTokens) : null });
  const onProvider = (v: number) => { setIdx(v); const p = providers[v]; if (p.base_url && !baseUrl) setBaseUrl(p.base_url); setModels([]); };
  const loadModels = async () => { if (!endpoints.models) return; setBusy("models"); setErr(""); try { const r = await api.post(endpoints.models, body()); setModels(r.models || []); if (!r.ok && r.error) setErr("Không lấy được danh sách model: " + r.error); } catch (e: any) { setErr(e.message); } finally { setBusy(""); } };
  const test = async () => { if (!endpoints.test) return; setBusy("test"); setOk(""); setErr(""); try { const r = await api.post(endpoints.test, body()); if (r.ok) setOk("Kết nối thành công " + (r.model ? `· ${r.model}` : r.dim ? `· ${r.dim} chiều` : "")); else setErr(r.error); } catch (e: any) { setErr(e.message); } finally { setBusy(""); } };
  const save = async () => { setBusy("save"); setOk(""); setErr(""); try { await api.put(endpoints.save, body()); setOk("Đã lưu cấu hình."); setApiKey(""); } catch (e: any) { setErr(e.message); } finally { setBusy(""); } };
  const del = async () => { if (!endpoints.del) return; try { await api.del(endpoints.del); setOk("Đã xoá cấu hình riêng, quay lại mặc định của hệ thống."); } catch (e: any) { setErr(e.message); } };
  return (
    <div>
      <Field label="Nhà cung cấp"><Select value={idx} onChange={(e) => onProvider(parseInt(e.target.value))}>{providers.map((p, i) => <option key={i} value={i}>{p.label}</option>)}</Select></Field>
      <div className="grid md:grid-cols-2 gap-3 mt-3">
        <Field label="Model" info="Bấm biểu tượng làm mới để tải danh sách model từ nhà cung cấp, hoặc nhập trực tiếp.">
          <div className="flex gap-2">
            <Input value={model} onChange={(e) => setModel(e.target.value)} placeholder="Chọn hoặc nhập model" />
            {endpoints.models && <Button variant="sec" size="sm" loading={busy === "models"} onClick={loadModels}><RefreshCw className="w-3.5 h-3.5" /></Button>}
          </div>
          {models.length > 0 && <Select className="mt-2" onChange={(e) => setModel(e.target.value)} value=""><option value="">Chọn từ {models.length} model có sẵn</option>{models.map((m) => <option key={m} value={m}>{m}</option>)}</Select>}
        </Field>
        <Field label="Base URL" info="Chỉ cần khi dùng máy chủ tự host hoặc endpoint tuỳ chỉnh."><Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="Mặc định theo nhà cung cấp" /></Field>
      </div>
      <div className="grid md:grid-cols-2 gap-3 mt-3">
        <Field label="API key" info="Khoá được mã hoá khi lưu. Để trống nếu muốn giữ khoá hiện tại."><Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="••••••••" /></Field>
        <Field label="Số token tối đa"><Input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} placeholder="1024" /></Field>
      </div>
      <div className="flex gap-2 mt-4 flex-wrap">
        {endpoints.test && <Button loading={busy === "test"} onClick={test}><Plug className="w-4 h-4" /> Kiểm tra kết nối</Button>}
        <Button variant="sec" loading={busy === "save"} onClick={save}>Lưu</Button>
        {endpoints.del && <Button variant="danger" onClick={del}>Dùng mặc định</Button>}
      </div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </div>
  );
}

// ============================================================ Settings
export function Settings() {
  const [llm, setLlm] = useState<any>(null); const [ocr, setOcr] = useState<any>(null); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/settings/llm").then(setLlm).catch((e) => setErr(e.message)); api.get("/api/settings/ocr").then(setOcr).catch(() => {}); }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!llm) return <Spinner />;
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub={llm.can_edit ? `Đang dùng ${llm.effective.provider}${llm.effective.model ? " · " + llm.effective.model : ""}.` : "Quản trị hệ thống đã khoá tuỳ chọn này. Đang dùng cấu hình mặc định."}>Mô hình ngôn ngữ (LLM)</CardTitle>
        {llm.can_edit
          ? <LlmForm initial={llm.org_config} providers={llm.providers} endpoints={{ save: "/api/settings/llm", test: "/api/settings/llm/test", models: "/api/settings/llm/models", del: llm.org_config ? "/api/settings/llm" : undefined }} />
          : <Empty>Bạn không có quyền chỉnh mô hình. Vui lòng liên hệ quản trị hệ thống.</Empty>}
      </Card>
      {ocr && <OcrCard ocr={ocr} />}
      <DangerZone />
    </div>
  );
}
function DangerZone() {
  const [confirm, setConfirm] = useState(""); const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  const del = async () => {
    setOk(""); setErr("");
    try { await api.del(`/api/org?confirm=${encodeURIComponent(confirm)}`); setOk("Đã xoá tổ chức. Vui lòng đăng xuất."); } catch (e: any) { setErr(e.message); }
  };
  return (
    <Card className="border-bad/40">
      <CardTitle sub="Xoá vĩnh viễn toàn bộ dữ liệu của tổ chức (cửa hàng, trợ lý, hội thoại, hoá đơn). Không thể hoàn tác. Chỉ Chủ sở hữu thực hiện được.">Dữ liệu & quyền riêng tư</CardTitle>
      <Field label="Nhập đúng tên tổ chức để xác nhận"><Input value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="Tên tổ chức" /></Field>
      <div className="mt-3"><Button variant="danger" onClick={del}>Xoá tổ chức</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}
function OcrCard({ ocr }: { ocr: any }) {
  const [idx, setIdx] = useState(Math.max(0, ocr.providers.findIndex((p: any) => p.id === (ocr.org_config?.provider || ocr.effective.provider))));
  const [model, setModel] = useState(ocr.org_config?.model || ""); const [lang, setLang] = useState("");
  const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  const save = async () => { setOk(""); setErr(""); try { await api.put("/api/settings/ocr", { provider: ocr.providers[idx].id, model, lang }); setOk("Đã lưu cấu hình OCR."); } catch (e: any) { setErr(e.message); } };
  return (
    <Card>
      <CardTitle sub={`Nhận dạng chữ trong ảnh và PDF scan. Đang dùng ${ocr.effective.provider}.`}>Nhận dạng ký tự (OCR)</CardTitle>
      <Field label="Bộ máy OCR"><Select value={idx} onChange={(e) => setIdx(parseInt(e.target.value))}>{ocr.providers.map((p: any, i: number) => <option key={i} value={i}>{p.label}</option>)}</Select></Field>
      <div className="grid md:grid-cols-2 gap-3 mt-3">
        <Field label="Model" info="Dùng cho chế độ VLM. Để trống sẽ dùng chính mô hình ngôn ngữ đang cấu hình."><Input value={model} onChange={(e) => setModel(e.target.value)} /></Field>
        <Field label="Ngôn ngữ" info="Dùng cho Tesseract. Ví dụ vie+eng cho tiếng Việt và tiếng Anh."><Input value={lang} onChange={(e) => setLang(e.target.value)} placeholder="vie+eng" /></Field>
      </div>
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

// ============================================================ Admin
export function Admin() {
  const [ov, setOv] = useState<any>(null); const [s, setS] = useState<any>(null); const [tenants, setTenants] = useState<any[]>([]);
  const [an, setAn] = useState<any>(null); const [pol, setPol] = useState<any>({}); const [polMsg, setPolMsg] = useState(""); const [err, setErr] = useState("");
  useEffect(() => {
    api.get("/api/admin/overview").then(setOv).catch((e) => setErr(e.message));
    api.get("/api/admin/settings").then((d) => { setS(d); setPol(d.policy); }).catch((e) => setErr(e.message));
    api.get("/api/admin/tenants").then(setTenants).catch(() => {});
    api.get("/api/admin/analytics").then(setAn).catch(() => {});
  }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!ov || !s) return <Spinner />;
  const embProviders = [{ id: "local", label: "Cục bộ (không cần khoá)" }, { id: "openai_compatible", label: "OpenAI-compatible", base_url: "https://api.openai.com/v1" }, { id: "gemini", label: "Gemini" }];
  const savePolicy = async () => { await api.put("/api/admin/settings/policy", pol); setPolMsg("Đã lưu chính sách."); };
  const series = an ? an.series.map((x: any) => ({ day: x.day, ai: x.ai_messages, human: 0 })) : [];
  return (
    <div className="space-y-4">
      <div className="grid gap-3 grid-cols-2 lg:grid-cols-5">
        <Kpi n={fmt(ov.tenants)} l="Khách hàng" /><Kpi n={fmt(ov.shops)} l="Cửa hàng" /><Kpi n={fmt(ov.conversations)} l="Hội thoại" />
        <Kpi n={fmt(ov.ai_messages_month)} l="Tin AI tháng này" /><Kpi n={`$${ov.cost_month.toFixed(2)}`} l="Chi phí tháng này" />
      </div>
      {an && <Card><CardTitle sub="14 ngày gần nhất, toàn nền tảng">Tin nhắn AI theo ngày</CardTitle><StackedBars data={series} /></Card>}
      <Card>
        <CardTitle sub="Cho phép khách hàng tự chọn nhà cung cấp AI hay không.">Chính sách nền tảng</CardTitle>
        <label className="flex items-center gap-2 text-sm mb-2 font-normal"><input type="checkbox" checked={!!pol.allow_tenant_llm} onChange={(e) => setPol({ ...pol, allow_tenant_llm: e.target.checked })} /> Cho phép khách hàng tự cấu hình mô hình ngôn ngữ</label>
        <label className="flex items-center gap-2 text-sm font-normal"><input type="checkbox" checked={!!pol.allow_tenant_ocr} onChange={(e) => setPol({ ...pol, allow_tenant_ocr: e.target.checked })} /> Cho phép khách hàng tự cấu hình OCR</label>
        <div className="mt-3"><Button variant="sec" onClick={savePolicy}>Lưu chính sách</Button></div><Msg type="ok">{polMsg}</Msg>
      </Card>
      <PaymentCard />
      <MetaAppCard />
      <Card><CardTitle sub="Áp dụng khi khách hàng không cấu hình riêng.">Mô hình mặc định của nền tảng</CardTitle>
        <LlmForm initial={s.llm} providers={s.llm_providers} endpoints={{ save: "/api/admin/settings/llm", test: "/api/admin/settings/llm/test", models: "/api/admin/settings/llm/models" }} /></Card>
      <Card><CardTitle sub="Dùng chung toàn nền tảng. Đổi model yêu cầu lập chỉ mục lại; số chiều cố định 384.">Mô hình embedding</CardTitle>
        <LlmForm initial={s.embedding} providers={embProviders} endpoints={{ save: "/api/admin/settings/embedding", test: "/api/admin/settings/embedding/test", models: "/api/admin/settings/embedding/models" }} /></Card>
      <Card>
        <CardTitle>Danh sách khách hàng</CardTitle>
        {tenants.length === 0 ? <Empty>Chưa có khách hàng.</Empty> :
          <Table head={["Tổ chức", "Gói", "Cửa hàng", "Tin AI tháng", "Chi phí"]}>
            {tenants.map((t) => <tr key={t.id}><Td className="font-semibold">{t.name}</Td><Td><Badge kind="active">{t.plan}</Badge></Td><Td>{t.shops}</Td><Td>{fmt(t.ai_messages)}</Td><Td>${t.cost_month.toFixed(2)}</Td></tr>)}
          </Table>}
      </Card>
    </div>
  );
}

function PaymentCard() {
  const [cfg, setCfg] = useState<any>(null); const [providers, setProviders] = useState<any[]>([]);
  const [provider, setProvider] = useState("manual"); const [apiKey, setApiKey] = useState("");
  const [f, setF] = useState({ publishable_key: "", webhook_secret: "", success_url: "", cancel_url: "", currency: "USD" });
  const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/admin/settings/payment").then((d) => { setProviders(d.providers); setCfg(d.config); if (d.config) setProvider(d.config.provider); }).catch((e) => setErr(e.message)); }, []);
  const save = async () => {
    setOk(""); setErr("");
    try { await api.put("/api/admin/settings/payment", { provider, api_key: apiKey || null, ...f }); setOk("Đã lưu cấu hình thanh toán."); setApiKey(""); } catch (e: any) { setErr(e.message); }
  };
  return (
    <Card>
      <CardTitle sub="Chọn cổng thanh toán và nhập khoá. Khoá được mã hoá khi lưu. Khách hàng sẽ thanh toán qua cổng này.">Cấu hình thanh toán</CardTitle>
      <Field label="Cổng thanh toán"><Select value={provider} onChange={(e) => setProvider(e.target.value)}>{providers.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}</Select></Field>
      {provider === "stripe" && (
        <>
          <div className="grid md:grid-cols-2 gap-3 mt-3">
            <Field label="Secret key" info="sk_live_… hoặc sk_test_… — được mã hoá khi lưu."><Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="••••••••" /></Field>
            <Field label="Publishable key"><Input value={f.publishable_key} onChange={(e) => setF({ ...f, publishable_key: e.target.value })} placeholder="pk_..." /></Field>
          </div>
          <div className="grid md:grid-cols-2 gap-3 mt-3">
            <Field label="Webhook secret" info="whsec_… để xác thực webhook kích hoạt gói tự động."><Input type="password" value={f.webhook_secret} onChange={(e) => setF({ ...f, webhook_secret: e.target.value })} /></Field>
            <Field label="Tiền tệ"><Input value={f.currency} onChange={(e) => setF({ ...f, currency: e.target.value })} placeholder="USD" /></Field>
          </div>
          <div className="grid md:grid-cols-2 gap-3 mt-3">
            <Field label="Success URL"><Input value={f.success_url} onChange={(e) => setF({ ...f, success_url: e.target.value })} placeholder="https://app.cua-ban.com/?paid=1" /></Field>
            <Field label="Cancel URL"><Input value={f.cancel_url} onChange={(e) => setF({ ...f, cancel_url: e.target.value })} placeholder="https://app.cua-ban.com/?canceled=1" /></Field>
          </div>
        </>
      )}
      {provider === "manual" && <p className="text-[13px] text-muted mt-2 font-normal">Chế độ thủ công: khách xác nhận trong ứng dụng (phù hợp demo hoặc chuyển khoản).</p>}
      {(provider === "vnpay" || provider === "momo") && <p className="text-[13px] text-warn mt-2 font-normal">Cổng này sẽ sớm được hỗ trợ. Lưu thông tin trước để kích hoạt sau.</p>}
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

// ============================================================ Bots
export function Bots({ shopId, role }: { shopId: string; role: string }) {
  const [bots, setBots] = useState<any[] | null>(null); const [err, setErr] = useState("");
  const [selected, setSelected] = useState<any>(null);
  const canManage = role === "owner" || role === "admin";
  const load = () => api.get(`/api/bots?shop_id=${shopId}`).then(setBots).catch((e) => setErr(e.message));
  useEffect(() => { setBots(null); setSelected(null); load(); }, [shopId]);
  if (selected) return <BotDetail bot={selected} shopId={shopId} canManage={canManage} onBack={() => { setSelected(null); load(); }} onSaved={(b) => setSelected(b)} />;
  return (
    <Card>
      <CardTitle sub="Mỗi trợ lý có prompt riêng, lời chào và giao diện riêng, gắn vào từng kênh hoặc trang."
        right={canManage ? <Button size="sm" onClick={() => setSelected({ shop_id: shopId, name: "", persona: "", greeting: "Xin chào! Mình có thể giúp gì cho bạn?", avatar_url: "", accent_color: "#6d7cff", config: {} })}><Plus className="w-4 h-4" /> Tạo trợ lý</Button> : undefined}>Trợ lý AI</CardTitle>
      {!bots ? <Spinner /> : bots.length === 0 ? <Empty>Chưa có trợ lý nào. Bấm Tạo trợ lý để bắt đầu.</Empty> :
        <div className="grid md:grid-cols-2 gap-3">
          {bots.map((b) => (
            <button key={b.id} onClick={() => setSelected(b)} className="text-left border border-line rounded-xl p-4 hover:border-line2 hover:bg-card2 transition">
              <div className="flex items-center gap-3">
                <span className="w-9 h-9 rounded-lg flex items-center justify-center text-white shrink-0 overflow-hidden" style={{ background: b.accent_color }}>
                  {b.avatar_url ? <img src={b.avatar_url} className="w-9 h-9 object-cover" /> : <Bot className="w-5 h-5" />}
                </span>
                <div className="min-w-0"><div className="font-semibold truncate">{b.name}</div><div className="text-xs text-muted">{b.channels} kênh · bấm để cấu hình</div></div>
              </div>
              <p className="text-[13px] text-muted mt-3 font-normal line-clamp-2">{b.persona || "Chưa đặt prompt tuỳ chỉnh."}</p>
            </button>
          ))}
        </div>}
      <Msg type="err">{err}</Msg>
    </Card>
  );
}

function BotDetail({ bot, shopId, canManage, onBack, onSaved }: { bot: any; shopId: string; canManage: boolean; onBack: () => void; onSaved: (b: any) => void }) {
  const [f, setF] = useState<any>({ config: {}, ...bot }); const [busy, setBusy] = useState(false); const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  const isNew = !f.id; const fileRef = useRef<HTMLInputElement>(null);
  const cfg = f.config || {}; const bh = cfg.business_hours || {};
  const setCfg = (patch: any) => setF({ ...f, config: { ...cfg, ...patch } });
  const setBH = (patch: any) => setCfg({ business_hours: { ...bh, ...patch } });
  const save = async () => {
    setBusy(true); setOk(""); setErr("");
    try {
      const body = { shop_id: f.shop_id || shopId, name: f.name, persona: f.persona, greeting: f.greeting, avatar_url: f.avatar_url, accent_color: f.accent_color, config: f.config || {} };
      if (isNew) { const r = await api.post("/api/bots", body); onSaved({ ...body, id: r.id, config: body.config }); setF({ ...f, id: r.id }); }
      else await api.put(`/api/bots/${f.id}`, body);
      setOk("Đã lưu.");
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  const upload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    const fd = new FormData(); fd.append("file", file);
    try { const r = await api.upload("/api/uploads", fd); setF((s: any) => ({ ...s, avatar_url: r.url })); } catch (ex: any) { setErr(ex.message); }
    finally { if (fileRef.current) fileRef.current.value = ""; }
  };
  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <Button variant="ghost" size="sm" onClick={onBack}>← Trợ lý</Button>
        <span className="text-muted text-sm">/</span>
        <span className="font-semibold">{f.name || "Trợ lý mới"}</span>
      </div>
      <div className="grid lg:grid-cols-2 gap-4 items-start">
        {/* settings */}
        <Card>
          <CardTitle>Cấu hình trợ lý</CardTitle>
          <div className="flex items-center gap-3 mb-3">
            <span className="w-14 h-14 rounded-xl flex items-center justify-center text-white overflow-hidden shrink-0" style={{ background: f.accent_color }}>
              {f.avatar_url ? <img src={f.avatar_url} className="w-14 h-14 object-cover" /> : <Bot className="w-7 h-7" />}
            </span>
            <div>
              <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={upload} />
              <Button size="sm" variant="sec" onClick={() => fileRef.current?.click()}><Upload className="w-3.5 h-3.5" /> Tải ảnh đại diện</Button>
              <div className="text-[11px] text-muted mt-1 font-normal">PNG/JPG, tối đa 2MB</div>
            </div>
          </div>
          <Field label="Tên trợ lý"><Input value={f.name || ""} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="Trợ lý cửa hàng" /></Field>
          <div className="mt-3"><Field label="Prompt tuỳ chỉnh" info="Cách trợ lý xưng hô, giọng điệu, điều nên/không nên làm. Để trống sẽ dùng mặc định.">
            <Textarea value={f.persona || ""} onChange={(e) => setF({ ...f, persona: e.target.value })} className="min-h-[130px]"
              placeholder="Ví dụ: Xưng mình, gọi khách là bạn; tư vấn size ngắn gọn; luôn gợi ý thêm một sản phẩm phù hợp." /></Field></div>
          <div className="grid sm:grid-cols-2 gap-3 mt-3">
            <Field label="Lời chào"><Input value={f.greeting || ""} onChange={(e) => setF({ ...f, greeting: e.target.value })} /></Field>
            <Field label="Màu widget"><div className="flex gap-2 items-center"><input type="color" value={f.accent_color || "#6d7cff"} onChange={(e) => setF({ ...f, accent_color: e.target.value })} className="w-10 h-9 rounded-lg border border-line bg-bg p-0.5" /><Input value={f.accent_color || ""} onChange={(e) => setF({ ...f, accent_color: e.target.value })} /></div></Field>
          </div>
          <div className="border-t border-line/60 mt-4 pt-3">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted mb-2">Chuyển nhân viên & giờ làm việc</div>
            <label className="flex items-center gap-2 text-sm font-normal"><input type="checkbox" checked={cfg.handoff_no_context !== false} onChange={(e) => setCfg({ handoff_no_context: e.target.checked })} /> Chuyển nhân viên khi không đủ thông tin</label>
            <label className="flex items-center gap-2 text-sm font-normal mt-2"><input type="checkbox" checked={!!bh.enabled} onChange={(e) => setBH({ enabled: e.target.checked })} /> Giới hạn giờ làm việc</label>
            {bh.enabled && <div className="grid sm:grid-cols-3 gap-2 mt-2">
              <Field label="Từ giờ"><Input type="number" value={bh.start ?? 8} onChange={(e) => setBH({ start: parseInt(e.target.value) })} /></Field>
              <Field label="Đến giờ"><Input type="number" value={bh.end ?? 22} onChange={(e) => setBH({ end: parseInt(e.target.value) })} /></Field>
              <Field label="Tin ngoài giờ"><Input value={bh.off_message || ""} onChange={(e) => setBH({ off_message: e.target.value })} placeholder="Ngoài giờ làm việc…" /></Field>
            </div>}
          </div>
          {canManage && <div className="mt-4 flex gap-2"><Button loading={busy} onClick={save}>Lưu</Button>
            {!isNew && <Button variant="danger" onClick={async () => { if (confirm("Xoá trợ lý này?")) { try { await api.del(`/api/bots/${f.id}`); onBack(); } catch (e: any) { setErr(e.message); } } }}>Xoá</Button>}</div>}
          <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
        </Card>
        {/* test chat (side panel, with memory) */}
        <BotTestPanel botId={f.id} accent={f.accent_color} />
      </div>
    </div>
  );
}

function BotTestPanel({ botId, accent }: { botId?: string; accent?: string }) {
  const [msgs, setMsgs] = useState<{ role: string; text: string }[]>([]); const [text, setText] = useState(""); const [busy, setBusy] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight; }, [msgs, busy]);
  const send = async () => {
    if (!text.trim() || !botId) return; const q = text; setText("");
    const history = msgs.map((m) => ({ role: m.role === "customer" ? "customer" : "ai", content: m.text }));
    setMsgs((m) => [...m, { role: "customer", text: q }]); setBusy(true);
    try { const r = await api.post(`/api/bots/${botId}/test`, { text: q, history }); setMsgs((m) => [...m, { role: "ai", text: r.reply }]); }
    catch (e: any) { setMsgs((m) => [...m, { role: "system", text: e.message }]); } finally { setBusy(false); }
  };
  return (
    <Card className="lg:sticky lg:top-20 flex flex-col" >
      <CardTitle sub="Trò chuyện nhiều lượt với dữ liệu thật, có ghi nhớ hội thoại.">Chạy thử</CardTitle>
      <div ref={boxRef} className="flex-1 min-h-[300px] max-h-[52vh] overflow-auto space-y-2 pr-1">
        {!botId ? <Empty>Lưu trợ lý để bắt đầu trò chuyện thử.</Empty> :
          msgs.length === 0 ? <Empty>Nhập câu hỏi, ví dụ: "Áo thun size M còn không?"</Empty> :
            msgs.map((m, i) => <div key={i} className={"px-3 py-2 rounded-xl text-sm max-w-[88%] whitespace-pre-wrap font-normal " + (m.role === "customer" ? "text-white ml-auto" : m.role === "ai" ? "bg-card2 border border-line" : "bg-amber-900/30 text-xs")} style={m.role === "customer" ? { background: accent || "#6d7cff" } : undefined}>{m.text}</div>)}
        {busy && <div className="text-xs text-muted px-1">Đang trả lời…</div>}
      </div>
      <div className="flex gap-2 mt-3">
        <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Nhập câu hỏi của khách" disabled={!botId} onKeyDown={(e) => e.key === "Enter" && send()} />
        <Button onClick={send} loading={busy} disabled={!botId}><Send className="w-4 h-4" /></Button>
      </div>
    </Card>
  );
}

function MetaAppCard() {
  const [c, setC] = useState<any>(null); const [appId, setAppId] = useState(""); const [secret, setSecret] = useState(""); const [vt, setVt] = useState("");
  const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/admin/settings/meta").then((d) => { setC(d); setAppId(d.app_id || ""); setVt(d.verify_token || "omnishop-verify"); }).catch((e) => setErr(e.message)); }, []);
  const save = async () => { setOk(""); setErr(""); try { await api.put("/api/admin/settings/meta", { app_id: appId, app_secret: secret || null, verify_token: vt }); setOk("Đã lưu Facebook App."); setSecret(""); } catch (e: any) { setErr(e.message); } };
  return (
    <Card>
      <CardTitle sub="Cấu hình Facebook App của nền tảng để bật nút Kết nối Facebook và webhook Messenger/Instagram cho khách hàng.">Facebook App (Meta)</CardTitle>
      <div className="grid md:grid-cols-2 gap-3">
        <Field label="App ID"><Input value={appId} onChange={(e) => setAppId(e.target.value)} /></Field>
        <Field label="App Secret" info="Dùng cho xác thực webhook và đổi token. Mã hoá khi lưu."><Input type="password" value={secret} onChange={(e) => setSecret(e.target.value)} placeholder={c && c.has_secret ? "•••• (giữ nguyên)" : ""} /></Field>
      </div>
      <div className="mt-3"><Field label="Verify token (webhook)"><Input value={vt} onChange={(e) => setVt(e.target.value)} /></Field></div>
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

// ============================================================ Help
export function Help() {
  const steps = [
    ["Tạo cửa hàng", "Thêm cửa hàng đầu tiên trong thanh chọn ở đầu trang."],
    ["Nhập sản phẩm & kiến thức", "Thêm sản phẩm, biến thể, tồn kho và tài liệu chính sách để trợ lý AI có dữ liệu trả lời."],
    ["Kết nối kênh", "Tạo tiện ích website và dán mã tích hợp vào trang của bạn."],
    ["Cấu hình AI", "Chọn nhà cung cấp mô hình trong mục Cài đặt và kiểm tra kết nối."],
    ["Theo dõi & hỗ trợ", "Xem thống kê ở Tổng quan và tiếp quản hội thoại trong Hộp thư khi cần."],
  ];
  const faqs = [
    ["Trợ lý AI lấy thông tin từ đâu?", "Từ sản phẩm, biến thể và tài liệu kiến thức bạn đã nhập cho cửa hàng. Nếu không đủ thông tin, hội thoại sẽ được chuyển cho nhân viên."],
    ["Tôi có thể dùng mô hình nào?", "Anthropic Claude, OpenAI, Google Gemini, hoặc máy chủ tself-host tương thích OpenAI như vLLM. Cấu hình trong mục Cài đặt."],
    ["Dữ liệu giữa các cửa hàng có tách biệt không?", "Có. Mỗi tổ chức được cô lập ở nhiều lớp, khách hàng này không thể truy cập dữ liệu của khách hàng khác."],
    ["Chi phí AI được tính thế nào?", "Mỗi câu trả lời được ghi nhận token và chi phí ước tính, hiển thị ở Tổng quan để theo dõi biên lợi nhuận."],
  ];
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Năm bước để đưa trợ lý AI vào hoạt động.">Bắt đầu nhanh</CardTitle>
        <ol className="space-y-3">
          {steps.map(([t, d], i) => (
            <li key={i} className="flex gap-3">
              <span className="w-6 h-6 rounded-full bg-accent/20 text-accent text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">{i + 1}</span>
              <div><div className="font-semibold text-sm">{t}</div><div className="text-[13px] text-muted font-normal">{d}</div></div>
            </li>
          ))}
        </ol>
      </Card>
      <Card>
        <CardTitle>Câu hỏi thường gặp</CardTitle>
        <div className="space-y-4">
          {faqs.map(([q, a], i) => <div key={i}><div className="font-semibold text-sm">{q}</div><div className="text-[13px] text-muted font-normal mt-0.5">{a}</div></div>)}
        </div>
      </Card>
    </div>
  );
}
