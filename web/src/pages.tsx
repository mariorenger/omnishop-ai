import React, { useEffect, useRef, useState } from "react";
import { api, clearAuth } from "./api";
import { Badge, Button, Card, CardTitle, Empty, Field, Info, Input, Kpi, LoadMore, Modal, Msg, notify, Select, Spinner, Table, Td, Textarea } from "./ui";
import { StackedBars, IntentBars, BarList } from "./charts";
import { RefreshCw, Upload, Plug, Send, UserPlus, CheckCircle2, ArrowUpRight, Bot, MessageSquare, Plus, Pencil, ChevronRight, Lock, ShieldCheck, UserRound, AlertTriangle } from "lucide-react";
import QRCode from "qrcode";

const fmt = (n: number) => n.toLocaleString("vi-VN");

// ============================================================ Overview
export function Overview({ shopId, onGoInbox }: { shopId: string; onGoInbox?: () => void }) {
  const [a, setA] = useState<any>(null); const [sub, setSub] = useState<any>(null);
  const [convs, setConvs] = useState<any[]>([]); const [err, setErr] = useState("");
  useEffect(() => {
    setA(null);
    api.get(`/api/analytics/overview?shop_id=${shopId}`).then(setA).catch((e) => setErr(e.message));
    api.get("/api/subscription").then(setSub).catch(() => {});
    api.get(`/api/conversations?shop_id=${shopId}&limit=6`).then((d) => setConvs((d.items || d).slice(0, 6))).catch(() => {});
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
        {sub?.quota?.llm_mode === "byok"
          ? <Kpi n={`$${t.cost_month.toFixed(2)}`} l="Chi phí AI (khoá của bạn)" info="Ước tính chi phí token bằng chính khoá AI của bạn trong tháng này." />
          : <Kpi n={fmt(t.customer_messages)} l="Khách đã nhắn" info="Số tin nhắn khách gửi đến cửa hàng trong kỳ." />}
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
          <CardTitle sub="Mới nhất" right={onGoInbox ? <button className="text-accent text-xs font-semibold" onClick={onGoInbox}>Xem tất cả</button> : undefined}>Hội thoại gần đây</CardTitle>
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
export function Products({ shopId, role }: { shopId: string; role?: string }) {
  const [items, setItems] = useState<any[] | null>(null); const [bots, setBots] = useState<any[]>([]);
  const [q, setQ] = useState(""); const [editId, setEditId] = useState<string | null>(null);   // null=closed, ""=new, id=edit
  const [f, setF] = useState<any>({ name: "", price: "", currency: "VND", sku: "", category: "", description: "", variants: "", bot_id: "" });
  const [busy, setBusy] = useState(false); const [err, setErr] = useState("");
  const [total, setTotal] = useState(0); const [more, setMore] = useState(false); const [loadingMore, setLoadingMore] = useState(false);
  const canManage = role === "owner" || role === "admin";
  const PAGE = 50;
  const load = (count = PAGE) => {
    api.get(`/api/products?shop_id=${shopId}&limit=${Math.max(count, PAGE)}&offset=0`)
      .then((d) => { setItems(d.items); setTotal(d.total); setMore(d.has_more); }).catch((e) => setErr(e.message));
    api.get(`/api/bots?shop_id=${shopId}`).then(setBots).catch(() => {});
  };
  const loadMore = async () => {
    if (!items) return; setLoadingMore(true);
    try { const d = await api.get(`/api/products?shop_id=${shopId}&limit=${PAGE}&offset=${items.length}`);
      setItems([...items, ...d.items]); setTotal(d.total); setMore(d.has_more); }
    finally { setLoadingMore(false); }
  };
  useEffect(() => { setItems(null); load(); }, [shopId]);
  const blank = { name: "", price: "", currency: "VND", sku: "", category: "", description: "", variants: "", bot_id: "" };
  const openNew = () => { setF(blank); setEditId(""); setErr(""); };
  const openEdit = (p: any) => {
    setF({ name: p.name, price: p.price ?? "", currency: p.currency || "VND", sku: p.sku || "",
           category: (p.attributes || {}).category || "", description: p.description || "",
           variants: (p.variants || []).map((v: any) => `${v.name}:${v.stock}`).join(", "), bot_id: p.bot_id || "" });
    setEditId(p.id); setErr("");
  };
  const save = async () => {
    setBusy(true); setErr("");
    const variants = (f.variants || "").split(",").map((s: string) => s.trim()).filter(Boolean).map((s: string) => { const [n, st] = s.split(":"); return { name: (n || "").trim(), stock: parseInt(st || "0") || 0 }; });
    const body = { shop_id: shopId, name: f.name, price: f.price ? parseFloat(f.price) : null, currency: f.currency, sku: f.sku, description: f.description, attributes: f.category ? { category: f.category } : {}, variants, bot_id: f.bot_id || null };
    try {
      if (editId) await api.put(`/api/products/${editId}`, body); else await api.post("/api/products", body);
      setEditId(null); load(items?.length); notify(editId ? "Đã cập nhật sản phẩm." : "Đã thêm sản phẩm.", "ok");
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  const del = async (p: any) => { if (!confirm(`Xoá sản phẩm "${p.name}"? Không thể hoàn tác.`)) return; try { await api.del(`/api/products/${p.id}`); load(items?.length); notify("Đã xoá sản phẩm.", "ok"); } catch (e: any) { notify(e.message, "err"); } };
  const totalStock = (p: any) => (p.variants || []).reduce((s: number, v: any) => s + (v.stock || 0), 0);
  const botName = (id: string) => bots.find((b) => b.id === id)?.name;
  const rows = (items || []).filter((p) => !q || p.name.toLowerCase().includes(q.toLowerCase()) || (p.sku || "").toLowerCase().includes(q.toLowerCase()));
  return (
    <Card>
      <CardTitle sub="Trợ lý AI dùng dữ liệu này để trả lời về giá, tồn kho và biến thể."
        right={<div className="flex gap-2"><Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Tìm tên hoặc SKU" className="w-44" />{canManage && <Button size="sm" onClick={openNew}><Plus className="w-4 h-4" /> Thêm</Button>}</div>}>Sản phẩm</CardTitle>
      {!items ? <Spinner /> : rows.length === 0 ? <Empty>{q ? "Không tìm thấy sản phẩm." : "Chưa có sản phẩm nào."}</Empty> :
        <Table head={["Sản phẩm", "SKU", "Giá", "Tồn kho", "Biến thể", "Áp dụng", ""]}>
          {rows.map((p) => (
            <tr key={p.id}>
              <Td><div className="font-medium">{p.name}</div><div className="text-xs text-muted line-clamp-1">{p.description || ""}</div></Td>
              <Td className="text-muted">{p.sku || "—"}</Td>
              <Td className="whitespace-nowrap">{p.price != null ? `${fmt(p.price)} ${p.currency}` : "Liên hệ"}</Td>
              <Td>{p.variants && p.variants.length ? <span className={totalStock(p) === 0 ? "text-bad" : ""}>{totalStock(p)}</span> : "—"}</Td>
              <Td><div className="flex flex-wrap gap-1">{(p.variants || []).slice(0, 4).map((v: any, i: number) => <span key={i} className={"text-[11px] border rounded px-1.5 py-0.5 font-normal " + (v.stock === 0 ? "border-bad/40 text-bad" : "border-line text-muted")}>{v.name}·{v.stock}</span>)}{(p.variants || []).length > 4 && <span className="text-[11px] text-muted">+{p.variants.length - 4}</span>}</div></Td>
              <Td className="text-muted text-xs">{p.bot_id ? botName(p.bot_id) || "1 trợ lý" : "Tất cả"}</Td>
              <Td className="text-right whitespace-nowrap">{canManage && <span className="inline-flex gap-1">
                <Button size="sm" variant="ghost" onClick={() => openEdit(p)}><Pencil className="w-3.5 h-3.5" /></Button>
                <Button size="sm" variant="danger" onClick={() => del(p)}>Xoá</Button>
              </span>}</Td>
            </tr>
          ))}
        </Table>}
      {items && items.length > 0 && !q && <LoadMore show={more} loading={loadingMore} onClick={loadMore} shown={items.length} total={total} />}
      <Msg type="err">{!editId ? err : ""}</Msg>
      <Modal open={editId !== null} onClose={() => setEditId(null)} title={editId ? "Sửa sản phẩm" : "Thêm sản phẩm"} size="lg"
        footer={<><Button variant="sec" onClick={() => setEditId(null)}>Huỷ</Button><Button loading={busy} onClick={save}>Lưu sản phẩm</Button></>}>
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

// A small on/off switch used for enabling/disabling knowledge documents.
function Toggle({ on, onChange, disabled }: { on: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button type="button" disabled={disabled} onClick={() => onChange(!on)}
      title={on ? "Đang dùng cho AI — bấm để tắt" : "Đã tắt — bấm để dùng cho AI"}
      className={"relative inline-flex h-5 w-9 items-center rounded-full transition shrink-0 " + (on ? "bg-accent" : "bg-line") + (disabled ? " opacity-50" : "")}>
      <span className={"inline-block h-4 w-4 transform rounded-full bg-white transition " + (on ? "translate-x-4" : "translate-x-0.5")} />
    </button>
  );
}

// A read-only value with a copy button — for callback/webhook URLs to paste
// into Facebook / Google / Zalo consoles.
function CopyField({ label, value, info }: { label: string; value?: string; info?: string }) {
  if (!value) return null;
  const copy = async () => { try { await navigator.clipboard.writeText(value); notify("Đã sao chép.", "ok"); } catch { notify("Không sao chép được — hãy bôi đen và copy thủ công.", "err"); } };
  return (
    <Field label={label} info={info}>
      <div className="flex gap-2">
        <input readOnly value={value} onFocus={(e) => e.target.select()}
          className="flex-1 min-w-0 bg-bg border border-line rounded-lg px-3 py-2 text-[12.5px] font-mono text-fg outline-none focus:border-accent" />
        <Button variant="sec" size="sm" onClick={copy}>Sao chép</Button>
      </div>
    </Field>
  );
}

// ============================================================ Knowledge
export function Knowledge({ shopId, role }: { shopId: string; role?: string }) {
  const canManage = role === "owner" || role === "admin";
  const [docs, setDocs] = useState<any[] | null>(null);
  const [title, setTitle] = useState(""); const [text, setText] = useState(""); const [botId, setBotId] = useState("");
  const [bots, setBots] = useState<any[]>([]); const [kb, setKb] = useState<any>(null); const [kbName, setKbName] = useState(""); const [editKb, setEditKb] = useState(false);
  const [msg, setMsg] = useState(""); const [err, setErr] = useState(""); const [open, setOpen] = useState<string | null>(null);
  const [total, setTotal] = useState(0); const [more, setMore] = useState(false); const [loadingMore, setLoadingMore] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const PAGE = 50;
  const load = (count = PAGE) => {
    const lim = Math.max(count, PAGE);
    api.get(`/api/knowledge/documents?shop_id=${shopId}&limit=${lim}&offset=0`).then((d) => {
      setDocs(d.items); setTotal(d.total); setMore(d.has_more);
      if (d.items.some((x: any) => x.status !== "ready" && x.status !== "error")) setTimeout(() => load(lim), 2000);
    }).catch((e) => setErr(e.message));
  };
  const loadMore = async () => {
    if (!docs) return; setLoadingMore(true);
    try { const d = await api.get(`/api/knowledge/documents?shop_id=${shopId}&limit=${PAGE}&offset=${docs.length}`);
      setDocs([...docs, ...d.items]); setTotal(d.total); setMore(d.has_more); }
    finally { setLoadingMore(false); }
  };
  useEffect(() => { setDocs(null); load(); api.get(`/api/bots?shop_id=${shopId}`).then(setBots).catch(() => {}); api.get(`/api/knowledge/kb?shop_id=${shopId}`).then((k) => { setKb(k); setKbName(k.name); }).catch(() => {}); }, [shopId]);
  const addText = async () => { setErr(""); setMsg(""); if (!text.trim()) { setErr("Nội dung trống"); return; } try { await api.post("/api/knowledge/documents", { shop_id: shopId, title, text, bot_id: botId || null }); setTitle(""); setText(""); setMsg("Đã thêm tài liệu, đang lập chỉ mục."); load(docs?.length); } catch (e: any) { setErr(e.message); } };
  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return; setErr(""); setMsg(`Đang tải ${file.name}…`);
    const fd = new FormData(); fd.append("shop_id", shopId); fd.append("file", file); if (botId) fd.append("bot_id", botId);
    try { await api.upload("/api/knowledge/upload", fd); setMsg(`Đã nhận ${file.name} — đang trích xuất & lập chỉ mục ở nền.`); load(docs?.length); }
    catch (ex: any) { setErr(ex.message); setMsg(""); } finally { if (fileRef.current) fileRef.current.value = ""; }
  };
  const saveKb = async () => { try { await api.put("/api/knowledge/kb", { shop_id: shopId, name: kbName }); setKb({ ...kb, name: kbName }); setEditKb(false); } catch (e: any) { setErr(e.message); } };
  const del = async (id: string) => { if (!confirm("Xoá tài liệu này?")) return; try { await api.del(`/api/knowledge/documents/${id}`); setOpen(null); load(docs?.length); } catch (e: any) { setErr(e.message); } };
  const reprocess = async (id: string) => { try { await api.post(`/api/knowledge/documents/${id}/reprocess`, {}); setOpen(null); load(docs?.length); } catch (e: any) { setErr(e.message); } };
  const setActive = async (id: string, active: boolean) => { try { await api.put(`/api/knowledge/documents/${id}/active`, { active }); load(docs?.length); } catch (e: any) { setErr(e.message); } };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Nhập nội dung hoặc tải tệp. Nội dung được trích xuất, chunk và nhúng vector để trợ lý AI tra cứu (RAG)."
          right={kb && (editKb
            ? <span className="flex items-center gap-2"><Input value={kbName} onChange={(e) => setKbName(e.target.value)} className="w-44 h-8 py-1" /><Button size="sm" onClick={saveKb}>Lưu</Button></span>
            : <button onClick={() => setEditKb(true)} className="text-[12px] text-muted hover:text-fg font-semibold inline-flex items-center gap-1"><Pencil className="w-3.5 h-3.5" /> {kb.name}</button>)}>Kho kiến thức</CardTitle>
        <div onClick={() => fileRef.current?.click()} className="border border-dashed border-line rounded-xl p-6 text-center text-muted cursor-pointer hover:border-accent hover:text-fg transition flex flex-col items-center gap-2">
          <Upload className="w-6 h-6" />
          <div className="text-sm font-medium text-fg">Tải tệp lên</div>
          <div className="text-xs font-normal">PDF, Word, PowerPoint, Excel, CSV, JSON, HTML, văn bản và hình ảnh. Ảnh/PDF scan nhận dạng bằng OCR. Tối đa 25MB.</div>
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
        <CardTitle sub="Bấm một tài liệu để xem văn bản đã trích xuất, trạng thái xử lý và thao tác.">Tài liệu đã tải</CardTitle>
        {!docs ? <Spinner /> : docs.length === 0 ? <Empty>Chưa có tài liệu nào.</Empty> :
          <Table head={["Tiêu đề", "Nguồn", "Trạng thái", "Ký tự", "Số đoạn", "Dùng cho AI", ""]}>
            {docs.map((d) => <tr key={d.id} onClick={() => setOpen(d.id)} className={"cursor-pointer hover:bg-card2/60 transition " + (d.active === false ? "opacity-55" : "")}>
              <Td className="font-semibold">{d.title}</Td>
              <Td className="text-muted">{d.source || "Nhập tay"}</Td>
              <Td><Badge kind={d.status}>{docStatus(d.status)}</Badge>{d.status === "error" && d.error ? <span className="block text-[11px] text-bad mt-0.5">{d.error}</span> : null}</Td>
              <Td>{d.char_count ? fmt(d.char_count) : "—"}</Td>
              <Td>{d.chunks}</Td>
              <Td onClick={(e) => e.stopPropagation()}><Toggle on={d.active !== false} onChange={(v) => setActive(d.id, v)} /></Td>
              <Td className="text-right text-muted"><ChevronRight className="w-4 h-4 inline" /></Td>
            </tr>)}
          </Table>}
        {docs && docs.length > 0 && <LoadMore show={more} loading={loadingMore} onClick={loadMore} shown={docs.length} total={total} />}
      </Card>
      <DocDetail id={open} canEdit={canManage} onClose={() => setOpen(null)} onDelete={del} onReprocess={reprocess} onSetActive={setActive} onSaved={() => load(docs?.length)} />
    </div>
  );
}
const docStatus = (s: string) => ({ queued: "Trong hàng đợi", pending: "Đang chờ", processing: "Đang xử lý", ready: "Sẵn sàng", error: "Lỗi" } as any)[s] || s;

function DocDetail({ id, canEdit, onClose, onDelete, onReprocess, onSetActive, onSaved }: { id: string | null; canEdit?: boolean; onClose: () => void; onDelete: (id: string) => void; onReprocess: (id: string) => void; onSetActive: (id: string, active: boolean) => void; onSaved?: () => void }) {
  const [d, setD] = useState<any>(null);
  const [edit, setEdit] = useState(false); const [title, setTitle] = useState(""); const [text, setText] = useState(""); const [busy, setBusy] = useState(false);
  useEffect(() => { setD(null); setEdit(false); if (id) api.get(`/api/knowledge/documents/${id}`).then(setD).catch(() => {}); }, [id]);
  const toggle = async () => { if (!d) return; await onSetActive(d.id, d.active === false); setD({ ...d, active: !(d.active !== false) }); };
  const startEdit = () => { setTitle(d.title || ""); setText(d.text || ""); setEdit(true); };
  const save = async () => {
    setBusy(true);
    try {
      await api.put(`/api/knowledge/documents/${d.id}`, { title, text });
      const fresh = await api.get(`/api/knowledge/documents/${d.id}`);
      setD(fresh); setEdit(false); onSaved && onSaved();
      notify("Đã lưu và lập chỉ mục lại tài liệu.", "ok");
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };
  const footer = d ? (edit
    ? <><Button variant="sec" onClick={() => setEdit(false)}>Huỷ</Button><Button loading={busy} onClick={save}>Lưu văn bản</Button></>
    : <><div className="flex-1" />{canEdit && <Button variant="sec" onClick={startEdit}><Pencil className="w-3.5 h-3.5" /> Sửa văn bản</Button>}<Button variant="sec" onClick={() => onReprocess(d.id)}>Xử lý lại</Button><Button variant="danger" onClick={() => onDelete(d.id)}>Xoá</Button></>
  ) : undefined;
  return (
    <Modal open={!!id} onClose={onClose} size="lg" title={d?.title || "Tài liệu"} sub={d ? `${d.source || "Nhập tay"} · ${d.mime || ""}` : ""} footer={footer}>
      {!d ? <Spinner /> : edit ? (
        <div className="space-y-3">
          <Field label="Tiêu đề"><Input value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
          <Field label="Văn bản" info="Sửa để chỉnh lại lỗi trích xuất/OCR. Khi lưu, tài liệu sẽ được chia đoạn và lập chỉ mục lại."><Textarea value={text} onChange={(e) => setText(e.target.value)} className="min-h-[46vh] text-[12.5px] leading-relaxed" /></Field>
          {d.source && d.source !== "text" ? <div className="text-[12px] text-warn font-normal">Lưu ý: nếu bấm “Xử lý lại” trên tài liệu từ tệp, hệ thống sẽ trích xuất lại từ tệp gốc và ghi đè phần sửa tay.</div> : null}
        </div>
      ) : (
        <div>
          <div className="flex flex-wrap items-center gap-4 text-[13px] mb-3">
            <span className="flex items-center gap-1.5"><Badge kind={d.status}>{docStatus(d.status)}</Badge></span>
            <span className="text-muted font-normal">Ký tự: <b className="text-fg">{d.char_count ? fmt(d.char_count) : "—"}</b></span>
            <span className="text-muted font-normal">Số đoạn: <b className="text-fg">{d.chunks}</b></span>
            <span className="flex items-center gap-2 font-normal"><Toggle on={d.active !== false} onChange={toggle} /><span className="text-muted">{d.active === false ? "Đã tắt — AI không dùng tài liệu này" : "Đang dùng cho AI"}</span></span>
          </div>
          {d.error && <Msg type="err">{d.error}</Msg>}
          <div className="text-[11px] font-semibold uppercase tracking-wide text-muted mb-1.5">Văn bản đã trích xuất</div>
          <pre className="bg-bg border border-line rounded-lg p-3 text-[12.5px] text-fg font-normal whitespace-pre-wrap max-h-[46vh] overflow-auto leading-relaxed">{d.text || "(trống)"}</pre>
        </div>
      )}
    </Modal>
  );
}

// ============================================================ Channels
const channelStatus = (s: string) => ({ connected: "Đang hoạt động", degraded: "Cần kiểm tra", pending: "Chờ kích hoạt", disconnected: "Đã ngắt" } as any)[s] || s;

const VERIFIABLE = ["messenger", "instagram", "telegram", "zalo", "whatsapp"];
const KIND_VI: Record<string, string> = { website: "Website", messenger: "Messenger", instagram: "Instagram", telegram: "Telegram", zalo: "Zalo OA", whatsapp: "WhatsApp", tiktok: "TikTok Shop", shopee: "Shopee" };

// Official brand marks (single-path SVGs) + brand colours, so each channel is
// instantly recognisable instead of relying on OS-dependent emoji.
const BRAND: Record<string, { c: string; d: string }> = {
  messenger: { c: "#0084FF", d: "M12 2C6.36 2 2 6.13 2 11.7c0 2.91 1.19 5.44 3.14 7.19.16.14.26.35.27.57l.05 1.78c.02.57.6.94 1.12.71l1.98-.87c.17-.08.36-.09.54-.04 .91.25 1.88.38 2.83.38 5.64 0 10-4.13 10-9.7C22 6.13 17.64 2 12 2zm6 7.46l-2.93 4.65c-.47.74-1.47.93-2.18.4l-2.33-1.75a.6.6 0 00-.72 0l-3.16 2.4c-.42.32-.97-.18-.68-.62l2.93-4.65c.47-.74 1.47-.93 2.18-.4l2.33 1.75c.21.16.51.16.72 0l3.16-2.39c.42-.32.97.18.68.62z" },
  instagram: { c: "#E4405F", d: "M12 2.16c3.2 0 3.58.01 4.85.07 1.17.05 1.8.25 2.23.41.56.22.96.48 1.38.9.42.42.68.82.9 1.38.16.42.36 1.06.41 2.23.06 1.27.07 1.65.07 4.85s-.01 3.58-.07 4.85c-.05 1.17-.25 1.8-.41 2.23-.22.56-.48.96-.9 1.38-.42.42-.82.68-1.38.9-.42.16-1.06.36-2.23.41-1.27.06-1.65.07-4.85.07s-3.58-.01-4.85-.07c-1.17-.05-1.8-.25-2.23-.41a3.7 3.7 0 01-1.38-.9 3.7 3.7 0 01-.9-1.38c-.16-.42-.36-1.06-.41-2.23-.06-1.27-.07-1.65-.07-4.85s.01-3.58.07-4.85c.05-1.17.25-1.8.41-2.23.22-.56.48-.96.9-1.38.42-.42.82-.68 1.38-.9.42-.16 1.06-.36 2.23-.41 1.27-.06 1.65-.07 4.85-.07M12 0C8.74 0 8.33.01 7.05.07 5.78.13 4.9.33 4.14.63c-.79.3-1.46.72-2.12 1.38C1.35 2.67.94 3.35.63 4.14.33 4.9.13 5.78.07 7.05.01 8.33 0 8.74 0 12s.01 3.67.07 4.95c.06 1.27.26 2.15.56 2.91.3.79.72 1.46 1.38 2.12.66.66 1.33 1.08 2.12 1.38.76.3 1.64.5 2.91.56C8.33 23.99 8.74 24 12 24s3.67-.01 4.95-.07c1.27-.06 2.15-.26 2.91-.56.79-.3 1.46-.72 2.12-1.38.66-.66 1.08-1.33 1.38-2.12.3-.76.5-1.64.56-2.91.06-1.28.07-1.69.07-4.95s-.01-3.67-.07-4.95c-.06-1.27-.26-2.15-.56-2.91a5.7 5.7 0 00-1.38-2.12A5.7 5.7 0 0019.86.63c-.76-.3-1.64-.5-2.91-.56C15.67.01 15.26 0 12 0zm0 5.84A6.16 6.16 0 105.84 12 6.16 6.16 0 0012 5.84zm0 10.16A4 4 0 1116 12a4 4 0 01-4 4zm6.41-11.85a1.44 1.44 0 11-1.44 1.44 1.44 1.44 0 011.44-1.44z" },
  telegram: { c: "#26A5E4", d: "M11.94 2C6.44 2 2 6.48 2 12s4.44 10 9.94 10c5.52 0 9.98-4.48 9.98-10S17.46 2 11.94 2zm4.64 6.8l-1.55 7.33c-.12.52-.42.65-.85.4l-2.35-1.73-1.13 1.09c-.13.13-.24.24-.48.24l.17-2.43 4.42-3.99c.19-.17-.04-.27-.3-.1L9.36 13.1l-2.35-.73c-.51-.16-.52-.51.11-.76l9.17-3.54c.42-.16.79.1.65.73z" },
  whatsapp: { c: "#25D366", d: "M17.47 14.38c-.3-.15-1.75-.86-2.02-.96-.27-.1-.47-.15-.66.15-.2.3-.77.96-.94 1.16-.17.2-.35.22-.64.07-.3-.15-1.25-.46-2.38-1.47-.88-.79-1.47-1.75-1.64-2.05-.17-.3-.02-.46.13-.6.13-.13.3-.35.44-.52.15-.17.2-.3.3-.5.1-.2.05-.37-.02-.52-.07-.15-.66-1.6-.9-2.19-.24-.57-.48-.5-.66-.5-.17 0-.37-.02-.56-.02s-.52.07-.79.37c-.27.3-1.04 1.01-1.04 2.47s1.06 2.87 1.21 3.07c.15.2 2.09 3.2 5.07 4.49.71.3 1.26.49 1.69.62.71.22 1.36.19 1.87.12.57-.09 1.75-.72 2-1.41.25-.69.25-1.28.17-1.41-.07-.13-.27-.2-.56-.35zM12.05 21.5h-.01a9.42 9.42 0 01-4.8-1.31l-.34-.2-3.57.93.96-3.48-.22-.36a9.4 9.4 0 01-1.44-5.01c0-5.2 4.24-9.44 9.46-9.44 2.53 0 4.9.99 6.69 2.78a9.38 9.38 0 012.76 6.67c0 5.2-4.24 9.45-9.45 9.45zM20.52 3.49A11.36 11.36 0 0012.05 0C5.5 0 .16 5.34.16 11.9c0 2.1.55 4.15 1.6 5.96L.06 24l6.3-1.65a11.9 11.9 0 005.69 1.45h.01c6.56 0 11.9-5.34 11.9-11.9 0-3.18-1.24-6.17-3.49-8.41z" },
  tiktok: { c: "#ffffff", d: "M12.53 1.5h3.36c.06 1.03.35 2.06 1.06 2.94.7.88 1.72 1.44 2.8 1.66v3.4a7.63 7.63 0 01-3.85-1.13v6.9a5.94 5.94 0 11-5.94-5.94c.3 0 .6.03.89.08v3.44a2.55 2.55 0 00-.89-.16 2.58 2.58 0 102.58 2.58V1.5z" },
  shopee: { c: "#EE4D2D", d: "M12 1.5c-2.3 0-4.16 1.86-4.16 4.15v.35H4.2c-.5 0-.9.42-.87.92l.6 12.2A2.9 2.9 0 006.82 22h10.36a2.9 2.9 0 002.9-2.78l.6-12.2a.87.87 0 00-.88-.92h-3.64v-.35c0-2.29-1.86-4.15-4.16-4.15zm0 1.6c1.41 0 2.56 1.14 2.56 2.55v.35H9.44v-.35c0-1.41 1.15-2.55 2.56-2.55zm.02 6.3c1.9 0 3.13.9 3.32 2.3l-1.6.37c-.12-.78-.77-1.2-1.74-1.2-.9 0-1.5.4-1.5.98 0 .6.55.86 1.86 1.16 1.9.43 3.06 1 3.06 2.5 0 1.65-1.4 2.62-3.45 2.62-2.03 0-3.4-.94-3.6-2.55l1.64-.35c.14.9.9 1.4 2 1.4.98 0 1.6-.4 1.6-1 0-.63-.5-.9-1.9-1.22-1.77-.4-3-.94-3-2.44 0-1.55 1.35-2.57 3.31-2.57z" },
  zalo: { c: "#0068FF", d: "M12 2C6.48 2 2 5.94 2 10.8c0 2.7 1.4 5.1 3.6 6.7-.15.9-.6 2.1-1.5 3.1-.2.2 0 .5.3.45 1.9-.35 3.35-1.05 4.3-1.65.98.27 2.02.4 3.1.4 5.52 0 10-3.94 10-8.8S17.52 2 12 2z" },
};
function ChannelMark({ kind }: { kind: string }) {
  const b = BRAND[kind];
  if (!b) return <Plug className="w-4 h-4 text-muted" />;  // website & unknowns
  return (
    <span className="w-5 h-5 rounded-md flex items-center justify-center shrink-0" style={{ background: b.c }}>
      <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill={kind === "tiktok" ? "#000" : "#fff"}><path d={b.d} /></svg>
    </span>
  );
}

export function Channels({ shopId }: { shopId: string }) {
  const [items, setItems] = useState<any[] | null>(null); const [kinds, setKinds] = useState<any[]>([]);
  const [err, setErr] = useState(""); const [open, setOpen] = useState(false); const [msg, setMsg] = useState("");
  const [kind, setKind] = useState("website"); const [name, setName] = useState(""); const [greeting, setGreeting] = useState("Xin chào! Mình có thể giúp gì cho bạn?");
  const [creds, setCreds] = useState<Record<string, string>>({}); const [busy, setBusy] = useState(false); const [ferr, setFerr] = useState("");
  const [editing, setEditing] = useState<any>(null); const [bots, setBots] = useState<any[]>([]); const [botId, setBotId] = useState(""); const [verifying, setVerifying] = useState("");
  const load = () => { api.get(`/api/channels?shop_id=${shopId}`).then(setItems).catch((e) => setErr(e.message)); api.get("/api/channels/kinds").then(setKinds).catch(() => {}); api.get(`/api/bots?shop_id=${shopId}`).then(setBots).catch(() => {}); };
  useEffect(() => { setItems(null); load(); }, [shopId]);
  const spec = kinds.find((k) => k.kind === kind);
  const kindLabel = (k: string) => kinds.find((x) => x.kind === k)?.label || k;
  const connect = async () => {
    setBusy(true); setFerr("");
    try {
      const r = await api.post("/api/channels", { shop_id: shopId, kind, name, greeting, credentials: creds, bot_id: botId || null });
      setOpen(false); setCreds({}); setName(""); setBotId(""); load();
      const s = channelStatus(r.status);
      setMsg(r.status === "connected" ? `Đã kết nối ${kindLabel(kind)} — ${s}.${r.note ? " " + r.note : ""}`
        : r.status === "pending" ? `Đã lưu ${kindLabel(kind)} — ${s} (chờ phê duyệt đối tác).`
        : `Đã lưu ${kindLabel(kind)} nhưng ${s.toLowerCase()}: ${r.note || "kiểm tra lại thông tin đăng nhập."}`);
    }
    catch (e: any) { setFerr(e.message); } finally { setBusy(false); }
  };
  const verify = async (ch: any) => {
    setVerifying(ch.id); setMsg("");
    try { const r = await api.post(`/api/channels/${ch.id}/verify`, {}); load(); setMsg(`${ch.name}: ${r.status === "connected" ? "kết nối thông" : "chưa thông"} — ${r.note || channelStatus(r.status)}`); }
    catch (e: any) { setErr(e.message); } finally { setVerifying(""); }
  };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Kết nối các kênh bán hàng để trợ lý AI trả lời khách trên mọi nơi. Webhook nhận tin (Telegram, Messenger, Zalo…) cần địa chỉ HTTPS công khai."
          right={<div className="flex gap-2">
            <Button size="sm" variant="sec" onClick={async () => { try { const r = await api.get(`/api/channels/oauth/meta/start?shop_id=${shopId}`); location.href = r.url; } catch (e: any) { notify(e.message, "err"); } }}>Kết nối Facebook</Button>
            <Button size="sm" onClick={() => setOpen(true)}><Plug className="w-4 h-4" /> Kết nối kênh</Button>
          </div>}>Kênh kết nối</CardTitle>
        <Msg type="err">{err}</Msg><Msg type="ok">{msg}</Msg>
        {items && items.some((c) => VERIFIABLE.includes(c.kind) && c.status !== "connected") && (
          <div className="mb-3 flex items-start gap-2 rounded-xl border border-bad/50 bg-bad/10 px-3 py-2.5 text-[13px] text-bad font-normal">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span><b>Có kênh chưa thông.</b> {items.filter((c) => VERIFIABLE.includes(c.kind) && c.status !== "connected").map((c) => c.name).join(", ")} có thể không nhận/trả lời tin của khách. Bấm “Kiểm tra kết nối” để kiểm tra lại.</span>
          </div>
        )}
        {!items ? <Spinner /> : items.length === 0 ? <Empty>Chưa có kênh nào. Bấm Kết nối kênh để bắt đầu.</Empty> :
          <div className="space-y-3">
            {items.map((ch) => {
              const url = (ch.kind === "website" && ch.public_key) ? `${location.origin}/widget.html?key=${ch.public_key}` : "";
              const canVerify = VERIFIABLE.includes(ch.kind);
              // Green ring when live, red when a verifiable channel isn't connected,
              // amber while pending — so tenants spot a broken channel instantly.
              const tone = !canVerify ? "border-line"
                : ch.status === "connected" ? "border-ok/60 bg-ok/5 shadow-[0_0_0_1px_rgba(52,211,153,0.25)]"
                : ch.status === "pending" ? "border-warn/60 bg-warn/5"
                : "border-bad/70 bg-bad/5 shadow-[0_0_0_1px_rgba(251,113,133,0.25)]";
              return (
                <div key={ch.id} className={"border-2 rounded-xl p-4 transition " + tone}>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[11px] font-semibold border border-line rounded-full pl-1 pr-2.5 py-0.5 text-muted inline-flex items-center gap-1.5"><ChannelMark kind={ch.kind} /> {kindLabel(ch.kind)}</span>
                    <span className="font-semibold">{ch.name}</span>
                    <Badge kind={ch.status}>{channelStatus(ch.status)}</Badge>
                    {ch.bot_name && <span className="text-xs text-muted inline-flex items-center gap-1"><Bot className="w-3.5 h-3.5" />{ch.bot_name}</span>}
                    <span className="flex-1" />
                    {canVerify && <Button size="sm" variant="ghost" loading={verifying === ch.id} onClick={() => verify(ch)}>Kiểm tra kết nối</Button>}
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
        {spec && !spec.live && <p className="text-[12px] text-warn mt-2 font-normal">Kênh này cần phê duyệt đối tác trước khi hoạt động. Bạn vẫn lưu được thông tin để kích hoạt sau.</p>}
        {spec && VERIFIABLE.includes(kind) && <p className="text-[12px] text-muted mt-2 font-normal flex gap-1.5"><Lock className="w-3.5 h-3.5 shrink-0 mt-0.5" /><span>Để nhận tin từ kênh này, hệ thống cần một địa chỉ <b>HTTPS công khai</b>. Sau khi kết nối, bấm <b>Kiểm tra kết nối</b> để xác nhận token hợp lệ.</span></p>}
        {spec?.note && <p className="text-[12px] text-muted mt-2 font-normal leading-relaxed">{spec.note}</p>}
        {spec?.guide?.length > 0 && (
          <details className="mt-2 rounded-lg border border-line bg-card2/50 overflow-hidden" open>
            <summary className="cursor-pointer select-none px-3 py-2 text-[12.5px] font-semibold text-fg">Hướng dẫn lấy thông tin</summary>
            <ol className="list-decimal ml-5 pr-3 pb-3 space-y-1 text-[12px] text-muted font-normal leading-relaxed">
              {spec.guide.map((g: string, i: number) => <li key={i}>{g}</li>)}
            </ol>
          </details>
        )}
        {spec?.docs && <a href={spec.docs} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[12px] text-accent font-semibold mt-2">Tài liệu chính thức của hãng ↗</a>}
        {spec?.webhook_url && <div className="mt-3"><CopyField label="Webhook URL (dán vào console của kênh)" value={spec.webhook_url}
          info={kind === "zalo" ? "Zalo OA/Developer Console → Webhook URL." : "Dùng chung webhook Meta App (quản trị đã cấu hình)."} /></div>}
        <div className="mt-3"><Field label="Trợ lý xử lý" info="Chọn trợ lý AI sẽ trả lời trên kênh này. Để trống sẽ dùng trợ lý mặc định.">
          <Select value={botId} onChange={(e) => setBotId(e.target.value)}><option value="">Trợ lý mặc định</option>{bots.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}</Select>
        </Field></div>
        {kind === "website" ? (
          <><div className="mt-3"><Field label="Tên hiển thị"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Tiện ích website" /></Field></div>
            <div className="mt-3"><Field label="Lời chào"><Input value={greeting} onChange={(e) => setGreeting(e.target.value)} /></Field></div></>
        ) : (
          <>
            <p className="text-[12px] text-muted mt-3 font-normal rounded-lg border border-line bg-card2 px-3 py-2 flex gap-1.5"><UserRound className="w-3.5 h-3.5 shrink-0 mt-0.5" /><span>Token và ID bên dưới lấy từ tài khoản của <b className="text-fg">chính cửa hàng bạn</b> trên nền tảng tương ứng. Cấu hình dùng chung như Facebook App đã do quản trị hệ thống thiết lập sẵn.</span></p>
            <div className="mt-3"><Field label="Tên hiển thị"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder={spec?.label} /></Field></div>
            {(spec?.fields || []).map((f: any) => (
              <div className="mt-3" key={f.key}><Field label={f.label + (f.required ? "" : " (tuỳ chọn)")} info={f.hint}>
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
export function Inbox({ shopId, me, role }: { shopId: string; me?: { id: string; email: string }; role?: string }) {
  const PAGE = 30, MSG_PAGE = 50;
  const [convs, setConvs] = useState<any[] | null>(null); const [convTotal, setConvTotal] = useState(0); const [convMore, setConvMore] = useState(false); const [loadingMore, setLoadingMore] = useState(false);
  const [active, setActive] = useState<any>(null); const [msgs, setMsgs] = useState<any[]>([]); const [msgMore, setMsgMore] = useState(false); const [loadingOld, setLoadingOld] = useState(false); const [reply, setReply] = useState("");
  const [members, setMembers] = useState<any[]>([]); const [busy, setBusy] = useState(false);
  const canManage = role === "owner" || role === "admin";
  const loadList = async (count = PAGE) => {
    const d = await api.get(`/api/conversations?shop_id=${shopId}&limit=${count}&offset=0`);
    setConvs(d.items); setConvTotal(d.total); setConvMore(d.has_more);
  };
  const loadMore = async () => {
    if (!convs) return; setLoadingMore(true);
    try { const d = await api.get(`/api/conversations?shop_id=${shopId}&limit=${PAGE}&offset=${convs.length}`);
      setConvs([...convs, ...d.items]); setConvTotal(d.total); setConvMore(d.has_more); }
    finally { setLoadingMore(false); }
  };
  useEffect(() => { setConvs(null); setActive(null); loadList(); if (canManage) api.get("/api/members").then(setMembers).catch(() => {}); }, [shopId]);
  const open = async (c: any) => {
    setActive(c);
    const d = await api.get(`/api/conversations/${c.id}/messages?limit=${MSG_PAGE}`);
    setMsgs(d.items); setMsgMore(d.has_more);
  };
  const loadOlder = async () => {
    if (!active || !msgs.length) return; setLoadingOld(true);
    try { const d = await api.get(`/api/conversations/${active.id}/messages?limit=${MSG_PAGE}&before=${encodeURIComponent(msgs[0].at)}`);
      setMsgs([...d.items, ...msgs]); setMsgMore(d.has_more); }
    finally { setLoadingOld(false); }
  };
  const refresh = async () => {
    const keep = Math.max(PAGE, convs?.length || 0);
    await loadList(keep);
    if (active) { const c = (await api.get(`/api/conversations?shop_id=${shopId}&limit=${keep}&offset=0`)).items.find((x: any) => x.id === active.id); if (c) setActive(c); }
  };
  const send = async () => {
    if (!reply.trim()) return; setBusy(true);
    try {
      const r = await api.post(`/api/conversations/${active.id}/reply`, { text: reply });
      setReply(""); await open(active); await refresh();
      if (r.delivered === false) notify("Đã lưu câu trả lời nhưng CHƯA gửi được tới khách: " + (r.note || "kiểm tra kết nối kênh"), "err");
      else if (active.channel_kind && active.channel_kind !== "website") notify("Đã gửi câu trả lời tới khách trên kênh.", "ok");
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };
  const claim = async (userId?: string) => {
    try { await api.post(`/api/conversations/${active.id}/assign`, userId ? { user_id: userId } : {}); await refresh(); notify(userId ? "Đã gán hội thoại." : "Bạn đã nhận xử lý hội thoại này.", "ok"); }
    catch (e: any) { notify(e.message, "err"); }
  };
  const assignedToMe = active && me && active.assigned_user_id === me.id;
  return (
    <Card>
      <CardTitle sub="Toàn bộ hội thoại từ mọi kênh. Câu trả lời của nhân viên được gửi thẳng về kênh của khách. Có thể phân công ai xử lý hội thoại nào.">Hộp thư hợp nhất</CardTitle>
      <div className="flex gap-4 items-start flex-col md:flex-row">
        <div className="flex-1 min-w-[220px] w-full space-y-2">
          {!convs ? <Spinner /> : convs.length === 0 ? <Empty>Chưa có hội thoại. Hãy thử nhắn qua tiện ích website.</Empty> :
            convs.map((c) => (
              <div key={c.id} onClick={() => open(c)} className={"cursor-pointer rounded-xl border p-3 transition " + (active?.id === c.id ? "border-accent bg-card2" : "border-line hover:bg-card2")}>
                <div className="flex items-center gap-2 flex-wrap"><ChannelMark kind={c.channel_kind} /><span className="font-semibold text-sm truncate">{c.customer_name || c.customer_ref}</span><Badge kind={c.status}>{statusLabel(c.status)}</Badge></div>
                <div className="text-xs text-muted mt-1 line-clamp-1 font-normal">{(c.last_message || "").slice(0, 70)}</div>
                <div className="text-[11px] mt-1 font-normal flex items-center gap-1.5 flex-wrap">
                  <span className="text-muted">{KIND_VI[c.channel_kind] || c.channel_kind || "—"}</span><span className="text-muted opacity-50">·</span>
                  {c.assignee ? <span className="text-accent">● {c.assignee === me?.email ? "Bạn xử lý" : c.assignee}</span> : <span className="text-muted">○ Chưa nhận</span>}
                </div>
              </div>
            ))}
          {convs && convs.length > 0 && <LoadMore show={convMore} loading={loadingMore} onClick={loadMore} shown={convs.length} total={convTotal} />}
        </div>
        <div className="flex-[1.4] min-w-[260px] w-full">
          {!active ? <Empty>Chọn một hội thoại để xem chi tiết.</Empty> : (
            <div>
              <div className="mb-3 rounded-xl border border-line bg-card2/50 p-3">
                <div className="flex items-center gap-2 flex-wrap">
                  <ChannelMark kind={active.channel_kind} />
                  <span className="font-semibold">{active.customer_name || active.customer_ref}</span>
                  <Badge kind={active.status}>{statusLabel(active.status)}</Badge>
                </div>
                <div className="grid sm:grid-cols-2 gap-x-4 gap-y-1 mt-2.5 text-[12px] font-normal">
                  <div className="text-muted">Kênh: <span className="text-fg font-medium">{KIND_VI[active.channel_kind] || active.channel_kind || "—"}{active.channel_name ? ` · ${active.channel_name}` : ""}</span></div>
                  <div className="text-muted">Mã khách: <span className="text-fg font-mono">{active.customer_ref}</span></div>
                  <div className="text-muted">Bắt đầu: <span className="text-fg">{active.created_at ? new Date(active.created_at).toLocaleString("vi-VN") : "—"}</span></div>
                  <div className="text-muted">Số tin nhắn: <span className="text-fg">{active.messages ?? "—"}</span></div>
                </div>
                <div className="flex items-center gap-2 mt-2.5 flex-wrap text-[12px] border-t border-line/60 pt-2.5">
                  <span className="text-muted font-normal">Phụ trách:</span>
                  <span className={active.assignee ? "font-semibold" : "text-muted font-normal"}>{active.assignee ? (assignedToMe ? "Bạn" : active.assignee) : "Chưa ai nhận"}</span>
                  {!assignedToMe && <Button size="sm" variant="ghost" onClick={() => claim()}>Nhận xử lý</Button>}
                  {canManage && members.length > 0 && (
                    <Select className="w-auto h-9 py-0 leading-none text-[12px]" value={active.assigned_user_id || ""} onChange={(e) => claim(e.target.value || undefined)}>
                      <option value="">— Gán cho —</option>
                      {members.map((m) => <option key={m.email} value={m.user_id || ""}>{m.email}</option>)}
                    </Select>
                  )}
                </div>
              </div>
              <div className="max-h-[54vh] overflow-auto pr-1 space-y-3 py-1">
                {msgMore && <div className="text-center"><Button variant="ghost" size="sm" loading={loadingOld} onClick={loadOlder}>Tải tin cũ hơn</Button></div>}
                {msgs.map((m, i) => {
                  if (m.role === "system") return <div key={i} className="mx-auto text-[11.5px] text-warn bg-warn/10 border border-warn/30 rounded-full px-3 py-1 font-normal">{m.content}</div>;
                  const mine = m.role === "ai" || m.role === "agent";
                  const label = m.role === "customer" ? "Khách" : m.role === "ai" ? "AI" : (m.sender || "Nhân viên");
                  const t = m.at ? new Date(m.at).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" }) : "";
                  return (
                    <div key={i} className={"flex flex-col max-w-[82%] " + (mine ? "ml-auto items-end" : "items-start")}>
                      <div className="text-[10.5px] text-muted mb-1 px-1 flex items-center gap-1.5">
                        <span className="font-semibold">{label}</span>{t && <span className="opacity-60">· {t}</span>}
                      </div>
                      <div className={"px-3.5 py-2.5 rounded-2xl text-sm whitespace-pre-wrap font-normal shadow-soft " +
                        (m.role === "customer" ? "bg-card2 border border-line text-fg rounded-tl-sm"
                          : m.role === "ai" ? "bg-indigo-500/15 border border-indigo-400/30 text-fg rounded-tr-sm"
                          : "bg-pastel text-[#0b0e1a] font-medium rounded-tr-sm")}>
                        {m.content}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="flex gap-2 mt-3">
                <Input value={reply} onChange={(e) => setReply(e.target.value)} placeholder="Nhập câu trả lời của nhân viên" onKeyDown={(e) => e.key === "Enter" && send()} />
                <Button onClick={send} loading={busy}><Send className="w-4 h-4" /></Button>
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
const modeLabel = (llm: string, bill: string) => bill === "payg" ? "Trả theo dùng" : llm === "managed" ? "Trọn gói AI" : "Tự nhập khoá AI";

function UsageBar({ used, total, unit, note }: { used: number; total: number; unit: string; note?: string }) {
  const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;
  const over = total > 0 && used >= total;
  return (
    <div>
      <div className="flex justify-between text-[13px] font-normal mb-1.5"><span className="text-muted">Đã dùng {unit} tháng này</span><span className="font-semibold">{fmt(used)}{total > 0 ? ` / ${fmt(total)}` : ""}</span></div>
      <div className="h-2 rounded-full bg-bg border border-line overflow-hidden"><div className={"h-full rounded-full " + (over ? "bg-bad" : "bg-pastel")} style={{ width: `${total > 0 ? pct : (used > 0 ? 100 : 0)}%` }} /></div>
      {note && <div className="text-[12px] text-muted font-normal mt-1.5">{note}</div>}
    </div>
  );
}

export function Billing({ role }: { role: string }) {
  const [plans, setPlans] = useState<any[] | null>(null); const [sub, setSub] = useState<any>(null);
  const [invoices, setInvoices] = useState<any[]>([]); const [err, setErr] = useState(""); const [customers, setCustomers] = useState<any[]>([]);
  const [checkout, setCheckout] = useState<any>(null); const [busy, setBusy] = useState(false);
  const isOwner = role === "owner";
  const load = () => Promise.all([api.get("/api/plans"), api.get("/api/subscription"), api.get("/api/billing/invoices"), api.get("/api/usage/by-customer")])
    .then(([p, s, i, c]) => { setPlans(p); setSub(s); setInvoices(i.items || i); setCustomers(c); }).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!plans || !sub) return <Spinner />;
  const cur = sub.entitlements._plan;
  const pick = async (p: any) => {
    if (p.price_month === 0) { await api.post("/api/subscription", { plan_code: p.code }); await load(); return; }
    const co = await api.post("/api/billing/checkout", { plan_code: p.code });
    if (co.redirect_url) { window.location.href = co.redirect_url; return; }  // VNPay/MoMo/Stripe
    setCheckout(co);  // VietQR (qr_image_url) or manual (instructions)
  };
  const reportTransfer = async () => { setBusy(true); try { await api.post(`/api/billing/checkout/${checkout.invoice_id}/submitted`); setCheckout(null); await load(); notify("Đã ghi nhận. Gói sẽ kích hoạt sau khi quản trị xác nhận khoản chuyển.", "ok"); } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); } };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle right={<Badge kind="ai">{modeLabel(sub.quota.llm_mode, sub.quota.billing_mode)}</Badge>}>Gói hiện tại: {sub.entitlements._plan_name}</CardTitle>
        {sub.quota.billing_mode === "payg" ? (
          <div className="text-[13px] font-normal">
            <div className="flex justify-between"><span className="text-muted">Đã dùng tháng này</span><span className="font-semibold">{fmt(sub.quota.tokens_used)} token · {fmt(sub.quota.messages_used)} tin nhắn</span></div>
            <div className="flex justify-between mt-1"><span className="text-muted">Ước tính chi phí</span><span className="font-semibold">${sub.quota.cost.toFixed(4)}</span></div>
          </div>
        ) : sub.quota.llm_mode === "managed" ? (
          <UsageBar used={sub.quota.tokens_used} total={sub.quota.tokens_included} unit="token" note={sub.quota.overage_per_1k > 0 ? `Vượt hạn mức: $${sub.quota.overage_per_1k}/1k token` : "Hết hạn mức sẽ tạm dừng trả lời tự động"} />
        ) : (
          <div><UsageBar used={sub.quota.messages_used} total={sub.quota.messages_limit} unit="tin nhắn" note="Bạn dùng khoá AI của mình nên chỉ tính phí phần mềm, không tính token." />
            <div className="text-[12px] text-muted font-normal mt-1.5">Đã tiêu thụ {fmt(sub.quota.tokens_used)} token (bằng khoá của bạn).</div></div>
        )}
        {!isOwner && <p className="text-[13px] text-muted font-normal mt-2">Chỉ Chủ sở hữu mới thay đổi được gói dịch vụ.</p>}
      </Card>
      <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
        {plans.map((p) => { const e = p.entitlements; return (
          <Card key={p.code} className={cur === p.code ? "border-accent shadow-glow" : ""}>
            <div className="flex items-start justify-between gap-2"><span className="font-bold text-[15px]">{p.name}</span><Badge kind={e.llm_mode === "managed" ? "ai" : "default"}>{modeLabel(e.llm_mode, e.billing_mode)}</Badge></div>
            <div className="mt-2 mb-1">{e.billing_mode === "payg"
              ? <span className="text-2xl font-extrabold">${e.payg_per_1k}<span className="text-xs text-muted font-semibold">/1k token</span></span>
              : <span className="text-2xl font-extrabold">${p.price_month}<span className="text-xs text-muted font-semibold">/tháng</span></span>}
              {e.billing_mode !== "payg" && p.price_month > 0 && sub.usd_vnd ? <div className="text-[12px] text-muted font-normal mt-0.5">≈ {fmt(Math.round(p.price_month * sub.usd_vnd))} đ/tháng</div> : null}</div>
            <ul className="text-[13px] text-muted mt-3 space-y-1.5 font-normal">
              {e.llm_mode === "managed"
                ? <li className="text-fg">{e.billing_mode === "payg" ? "Trả theo token thực dùng" : `${fmt(e.ai_tokens_month)} token AI/tháng`}</li>
                : <li className="text-fg">Tự nhập khoá OpenAI / Gemini / Claude</li>}
              {e.llm_mode === "byok" && <li>{e.ai_messages_month ? `${fmt(e.ai_messages_month)} tin nhắn/tháng` : "Không giới hạn tin nhắn"}</li>}
              {e.overage_per_1k > 0 && <li>Vượt hạn mức ${e.overage_per_1k}/1k token</li>}
              <li>{e.shops} cửa hàng · {(e.channels_allowed || []).length} loại kênh</li>
            </ul>
            <div className="mt-4">{cur === p.code ? <Badge kind="active">Đang sử dụng</Badge> :
              isOwner ? <Button variant={p.price_month || e.billing_mode === "payg" ? "primary" : "sec"} onClick={() => pick(p)}>{p.price_month || e.billing_mode === "payg" ? "Chọn gói" : "Chuyển gói"}</Button> : <span className="text-xs text-muted">—</span>}</div>
          </Card>
        ); })}
      </div>
      <Card>
        <CardTitle sub="Token và chi phí ước tính theo từng khách hàng trong tháng.">Sử dụng theo khách</CardTitle>
        {customers.length === 0 ? <Empty>Chưa có dữ liệu sử dụng tháng này.</Empty> :
          <Table head={["Khách", "Tin nhắn", "Token", "Chi phí ước tính"]}>
            {customers.map((c) => <tr key={c.customer_ref}><Td className="font-semibold">{c.customer_ref}</Td><Td>{fmt(c.messages)}</Td><Td>{fmt(c.tokens)}</Td><Td className="text-muted">${c.cost.toFixed(5)}</Td></tr>)}
          </Table>}
      </Card>
      <Card>
        <CardTitle>Hoá đơn</CardTitle>
        {invoices.length === 0 ? <Empty>Chưa có hoá đơn nào.</Empty> :
          <Table head={["Ngày", "Gói", "Số tiền", "Trạng thái"]}>
            {invoices.map((iv) => <tr key={iv.id}><Td className="font-normal">{new Date(iv.created_at).toLocaleDateString("vi-VN")}</Td><Td className="font-semibold">{iv.plan}</Td><Td>${iv.amount}</Td><Td><Badge kind={iv.status === "submitted" ? "pending" : iv.status}>{iv.status === "paid" ? "Đã thanh toán" : iv.status === "submitted" ? "Chờ xác nhận" : iv.status === "pending" ? "Chờ thanh toán" : "Đã huỷ"}</Badge></Td></tr>)}
          </Table>}
      </Card>
      <Modal open={!!checkout} onClose={() => setCheckout(null)} title={checkout?.qr_image_url ? "Quét mã để thanh toán" : "Chuyển khoản thanh toán"}
        sub={checkout ? `Nâng cấp gói ${checkout.plan}: ${checkout.amount_vnd ? fmt(checkout.amount_vnd) + " đ" : "$" + checkout.amount}/tháng.` : ""}
        footer={<><Button variant="sec" onClick={() => setCheckout(null)}>Huỷ</Button>{!checkout?.error && <Button loading={busy} onClick={reportTransfer}>Tôi đã chuyển khoản</Button>}</>}>
        {checkout?.error ? (
          <Msg type="err">{checkout.error}</Msg>
        ) : checkout?.qr_image_url ? (
          <div className="flex flex-col items-center text-center">
            <div className="bg-white rounded-xl p-3"><img src={checkout.qr_image_url} alt="VietQR" className="w-52 h-52 object-contain" /></div>
            <p className="text-[13px] text-muted font-normal mt-3">Mở app ngân hàng, quét mã VietQR và giữ nguyên nội dung chuyển khoản
              {checkout.transfer_note ? <> <b className="text-fg">{checkout.transfer_note}</b></> : null}.</p>
          </div>
        ) : (
          <div className="flex items-start gap-3 text-sm">
            <CheckCircle2 className="w-5 h-5 text-ok shrink-0 mt-0.5" />
            <p className="text-muted font-normal">{checkout?.instructions || "Chuyển khoản theo hướng dẫn rồi bấm “Tôi đã chuyển khoản”."}</p>
          </div>
        )}
        <div className="mt-3 text-[12px] text-muted font-normal border-t border-line/60 pt-2.5">{checkout?.auto
          ? "Chuyển khoản đúng nội dung, gói sẽ TỰ ĐỘNG kích hoạt trong giây lát sau khi hệ thống nhận được tiền. Bạn có thể tải lại trang để kiểm tra."
          : "Sau khi bạn báo đã chuyển khoản, quản trị viên sẽ đối chiếu và kích hoạt gói. Bạn sẽ thấy trạng thái “Chờ xác nhận” ở mục Hoá đơn."}</div>
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
  // Reset Base URL to the newly-selected provider's default so a stale/empty URL
  // from another provider never leaks across (e.g. OpenAI URL left when switching
  // to Gemini, or an empty URL causing "missing http/https").
  const onProvider = (v: number) => { setIdx(v); setBaseUrl(providers[v].base_url || ""); setModels([]); };
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
  const [llm, setLlm] = useState<any>(null); const [ocr, setOcr] = useState<any>(null); const [sub, setSub] = useState<any>(null); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/settings/llm").then(setLlm).catch((e) => setErr(e.message)); api.get("/api/settings/ocr").then(setOcr).catch(() => {}); api.get("/api/subscription").then(setSub).catch(() => {}); }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!llm) return <Spinner />;
  const byok = sub?.quota?.llm_mode === "byok";
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub={llm.can_edit ? `Đang dùng ${llm.effective.provider}${llm.effective.model ? " · " + llm.effective.model : ""}.` : "Quản trị hệ thống đã khoá tuỳ chọn này. Đang dùng cấu hình mặc định."}>Mô hình ngôn ngữ (LLM)</CardTitle>
        {llm.can_edit && sub && (
          <div className={"text-[12.5px] font-normal rounded-lg px-3 py-2 mb-3 border " + (byok ? "border-accent/40 bg-accent/10 text-fg" : "border-line bg-card2 text-muted")}>
            {byok
              ? "Gói của bạn dùng khoá AI riêng. Hãy nhập API key của bạn ở bên dưới; hệ thống chỉ tính phí phần mềm, không tính token."
              : "Gói của bạn đã bao gồm AI của hệ thống. Bạn có thể nhập khoá riêng nếu muốn dùng nhà cung cấp của mình."}
          </div>
        )}
        {llm.can_edit
          ? <LlmForm initial={llm.org_config} providers={llm.providers} endpoints={{ save: "/api/settings/llm", test: "/api/settings/llm/test", models: "/api/settings/llm/models", del: llm.org_config ? "/api/settings/llm" : undefined }} />
          : <Empty>Bạn không có quyền chỉnh mô hình. Vui lòng liên hệ quản trị hệ thống. (Quản trị có thể bật ở: Quản trị hệ thống → Chính sách nền tảng → “Cho phép khách hàng tự cấu hình mô hình”.)</Empty>}
      </Card>
      {ocr && <OcrCard ocr={ocr} />}
      <SecurityCard />
      <DangerZone />
    </div>
  );
}

function SecurityCard() {
  const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  const logoutAll = async () => {
    if (!confirm("Đăng xuất khỏi tất cả thiết bị? Bạn sẽ cần đăng nhập lại.")) return;
    try { await api.post("/api/auth/logout-all", {}); clearAuth(); location.reload(); } catch (e: any) { setErr(e.message); }
  };
  return (
    <Card>
      <CardTitle sub="Thu hồi mọi phiên đăng nhập hiện có trên mọi thiết bị.">Bảo mật phiên</CardTitle>
      <Button variant="ghost" onClick={logoutAll}>Đăng xuất tất cả thiết bị</Button>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
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
export function Admin({ role = "admin" }: { role?: string }) {
  const isAdmin = role === "admin";
  const [ov, setOv] = useState<any>(null); const [s, setS] = useState<any>(null); const [tenants, setTenants] = useState<any[]>([]);
  const [an, setAn] = useState<any>(null); const [pol, setPol] = useState<any>({}); const [polMsg, setPolMsg] = useState(""); const [err, setErr] = useState("");
  const [sub, setSub] = useState("monitor");
  useEffect(() => {
    api.get("/api/admin/overview").then(setOv).catch((e) => setErr(e.message));
    api.get("/api/admin/tenants").then(setTenants).catch(() => {});
    api.get("/api/admin/analytics").then(setAn).catch(() => {});
    if (isAdmin) api.get("/api/admin/settings").then((d) => { setS(d); setPol(d.policy); }).catch((e) => setErr(e.message));
  }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!ov || (isAdmin && !s)) return <Spinner />;
  const embProviders = [{ id: "local", label: "Cục bộ (không cần khoá)" }, { id: "openai_compatible", label: "OpenAI-compatible", base_url: "https://api.openai.com/v1" }, { id: "gemini", label: "Gemini" }];
  const savePolicy = async () => { await api.put("/api/admin/settings/policy", pol); setPolMsg("Đã lưu chính sách."); };
  const series = an ? an.series.map((x: any) => ({ day: x.day, ai: x.ai_messages, human: 0 })) : [];
  const topTenants = [...tenants].sort((a, b) => b.cost_month - a.cost_month).slice(0, 8)
    .map((t) => ({ label: t.name, value: t.cost_month, hint: `$${t.cost_month.toFixed(2)}` }));

  // sub-sections: monitoring (giám sát) vs configuration (cài đặt) split by purpose
  const TABS: [string, string][] = isAdmin
    ? [["monitor", "Giám sát"], ["finance", "Tài chính"], ["customers", "Khách hàng"], ["config", "Cấu hình hệ thống"], ["plans", "Gói & định giá"], ["branding", "Thương hiệu"], ["staff", "Nhân sự & nhật ký"]]
    : [["monitor", "Giám sát"], ["finance", "Tài chính"], ["audit", "Nhật ký"]];
  const cur = TABS.some((t) => t[0] === sub) ? sub : "monitor";

  return (
    <div className="space-y-4">
      {!isAdmin && <div className="text-[13px] text-muted font-normal">Vai trò <b className="text-fg">Quản lý</b>: xem thống kê và xuất báo cáo. Không có quyền chỉnh cấu hình hay khách hàng.</div>}
      {isAdmin && cur === "config" && <div className="text-[13px] rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 font-normal flex gap-2"><ShieldCheck className="w-4 h-4 shrink-0 mt-0.5 text-accent" /><span>Mọi cấu hình ở đây do bạn thiết lập và áp dụng cho toàn bộ khách hàng. Khách hàng không nhìn thấy hay chỉnh sửa các mục này.</span></div>}

      <div className="flex gap-1.5 overflow-x-auto border-b border-line pb-2">
        {TABS.map(([id, lb]) => (
          <button key={id} onClick={() => setSub(id)}
            className={"px-3 py-1.5 rounded-lg text-[13px] font-semibold whitespace-nowrap transition " + (cur === id ? "bg-accent/20 text-fg border border-line" : "text-muted hover:text-fg hover:bg-card")}>{lb}</button>
        ))}
      </div>

      {cur === "monitor" && <>
        <div className="grid gap-3 grid-cols-2 lg:grid-cols-5">
          <Kpi n={fmt(ov.tenants)} l="Khách hàng" /><Kpi n={fmt(ov.shops)} l="Cửa hàng" /><Kpi n={fmt(ov.conversations)} l="Hội thoại" />
          <Kpi n={fmt(ov.ai_messages_month)} l="Tin AI tháng này" /><Kpi n={`$${ov.cost_month.toFixed(2)}`} l="Chi phí tháng này" />
        </div>
        {an && <Card><CardTitle sub="14 ngày gần nhất, toàn nền tảng">Tin nhắn AI theo ngày</CardTitle><StackedBars data={series} /></Card>}
        <Card><CardTitle sub="Chi phí AI tháng này theo khách hàng (top 8)">Khách hàng dùng nhiều nhất</CardTitle><BarList data={topTenants} empty="Chưa có dữ liệu sử dụng." /></Card>
        <Card>
          <CardTitle sub="Xuất số liệu tình trạng hệ thống ra CSV để làm báo cáo." right={<div className="flex flex-wrap gap-2">
            <Button variant="sec" size="sm" onClick={() => api.download("/api/admin/reports/tenants.csv", "omnishop-tenants.csv")}>Khách hàng (CSV)</Button>
            <Button variant="sec" size="sm" onClick={() => api.download("/api/admin/reports/usage.csv", "omnishop-usage.csv")}>Sử dụng 30 ngày (CSV)</Button>
          </div>}>Danh sách khách hàng</CardTitle>
          {tenants.length === 0 ? <Empty>Chưa có khách hàng.</Empty> :
            <Table head={["Tổ chức", "Gói", "Cửa hàng", "Tin AI tháng", "Chi phí"]}>
              {tenants.map((t) => <tr key={t.id}><Td className="font-semibold">{t.name}</Td><Td><Badge kind="active">{t.plan}</Badge></Td><Td>{t.shops}</Td><Td>{fmt(t.ai_messages)}</Td><Td>${t.cost_month.toFixed(2)}</Td></tr>)}
            </Table>}
        </Card>
      </>}

      {cur === "finance" && <FinanceCard />}
      {isAdmin && cur === "customers" && <TenantBillingCard />}
      {cur === "audit" && !isAdmin && <AuditCard />}

      {isAdmin && cur === "config" && <>
        <Card>
          <CardTitle sub="Cho phép khách hàng tự chọn nhà cung cấp AI hay không.">Chính sách nền tảng</CardTitle>
          <label className="flex items-center gap-2 text-sm mb-2 font-normal"><input type="checkbox" checked={!!pol.allow_tenant_llm} onChange={(e) => setPol({ ...pol, allow_tenant_llm: e.target.checked })} /> Cho phép khách hàng tự cấu hình mô hình ngôn ngữ</label>
          <label className="flex items-center gap-2 text-sm font-normal"><input type="checkbox" checked={!!pol.allow_tenant_ocr} onChange={(e) => setPol({ ...pol, allow_tenant_ocr: e.target.checked })} /> Cho phép khách hàng tự cấu hình OCR</label>
          <div className="mt-3"><Button variant="sec" onClick={savePolicy}>Lưu chính sách</Button></div><Msg type="ok">{polMsg}</Msg>
        </Card>
        <Card><CardTitle sub="Áp dụng khi khách hàng không cấu hình riêng.">Mô hình mặc định của nền tảng</CardTitle>
          <LlmForm initial={s.llm} providers={s.llm_providers} endpoints={{ save: "/api/admin/settings/llm", test: "/api/admin/settings/llm/test", models: "/api/admin/settings/llm/models" }} /></Card>
        <Card><CardTitle sub="Dùng chung toàn nền tảng. Đổi model yêu cầu lập chỉ mục lại; số chiều cố định 384.">Mô hình embedding</CardTitle>
          <LlmForm initial={s.embedding} providers={embProviders} endpoints={{ save: "/api/admin/settings/embedding", test: "/api/admin/settings/embedding/test", models: "/api/admin/settings/embedding/models" }} /></Card>
        <RuntimeCard />
        <PaymentCard />
        <MetaAppCard />
        <GoogleCard />
        <EmailCard />
      </>}

      {isAdmin && cur === "plans" && <><PlansCard /><CostCard /></>}
      {isAdmin && cur === "branding" && <BrandingCard />}
      {isAdmin && cur === "staff" && <><StaffCard /><AuditCard /></>}
    </div>
  );
}

function TenantBillingCard() {
  const PAGE = 25;
  const [rows, setRows] = useState<any[] | null>(null); const [total, setTotal] = useState(0); const [more, setMore] = useState(false);
  const [q, setQ] = useState(""); const [plans, setPlans] = useState<any[]>([]); const [pend, setPend] = useState<any[]>([]);
  const [err, setErr] = useState(""); const [loadingMore, setLoadingMore] = useState(false);
  const loadPage = async (count = PAGE) => {
    const d = await api.get(`/api/admin/billing/tenants?limit=${Math.max(count, PAGE)}&offset=0&q=${encodeURIComponent(q)}`);
    setRows(d.items); setTotal(d.total); setMore(d.has_more);
  };
  const loadPending = () => api.get("/api/admin/billing/pending?limit=50").then((d) => setPend(d.items)).catch(() => {});
  useEffect(() => { api.get("/api/plans").then(setPlans).catch(() => {}); }, []);
  useEffect(() => { setRows(null); loadPage().catch((e) => setErr(e.message)); loadPending(); }, [q]);
  const loadMore = async () => { if (!rows) return; setLoadingMore(true); try { const d = await api.get(`/api/admin/billing/tenants?limit=${PAGE}&offset=${rows.length}&q=${encodeURIComponent(q)}`); setRows([...rows, ...d.items]); setTotal(d.total); setMore(d.has_more); } finally { setLoadingMore(false); } };
  const setPlan = async (org: any, code: string) => {
    if (!code || code === org.plan) return;
    if (!confirm(`Cấp gói "${code}" cho ${org.name}? Đây là cấp thủ công (không tính vào doanh thu).`)) return;
    try { await api.put(`/api/admin/tenants/${org.id}/plan`, { plan_code: code }); notify("Đã cấp gói (admin_manual).", "ok"); loadPage(rows?.length); } catch (e: any) { notify(e.message, "err"); }
  };
  const confirmInvoice = async (iv: any) => {
    if (!confirm(`Xác nhận đã nhận tiền hoá đơn của ${iv.tenant} (gói ${iv.plan}, $${iv.amount})? Sẽ kích hoạt gói và tính vào doanh thu.`)) return;
    try { await api.post(`/api/admin/invoices/${iv.id}/confirm`, {}); notify("Đã xác nhận thanh toán.", "ok"); loadPending(); loadPage(rows?.length); } catch (e: any) { notify(e.message, "err"); }
  };
  if (err) return <Msg type="err">{err}</Msg>;
  return (
    <div className="space-y-4">
      {pend.length > 0 && (
        <Card>
          <CardTitle sub="Khách hàng đã báo chuyển khoản (QR/thủ công) — đối chiếu ngân hàng rồi xác nhận để kích hoạt gói và ghi nhận doanh thu.">Chờ xác nhận thanh toán</CardTitle>
          <Table head={["Khách hàng", "Gói", "Số tiền", "Kênh", "Trạng thái", ""]}>
            {pend.map((iv) => <tr key={iv.id}>
              <Td className="font-semibold">{iv.tenant}</Td><Td>{iv.plan}</Td><Td>${iv.amount}</Td>
              <Td className="text-muted">{iv.provider}</Td>
              <Td><Badge kind={iv.status === "submitted" ? "pending" : "default"}>{iv.status === "submitted" ? "Đã báo CK" : "Chờ CK"}</Badge></Td>
              <Td className="text-right"><Button size="sm" onClick={() => confirmInvoice(iv)}>Xác nhận</Button></Td>
            </tr>)}
          </Table>
        </Card>
      )}
      <Card>
        <CardTitle sub="Quản lý gói của từng khách hàng. Cấp gói ở đây là cấp thủ công (admin_manual) và không tính vào doanh thu/lợi nhuận."
          right={<Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Tìm khách hàng" className="w-48" />}>Khách hàng &amp; Gói</CardTitle>
        {!rows ? <Spinner /> : rows.length === 0 ? <Empty>Không có khách hàng.</Empty> :
          <Table head={["Khách hàng", "Gói hiện tại", "Doanh thu tháng", "Chờ xác nhận", "Cấp gói thủ công"]}>
            {rows.map((t) => <tr key={t.id}>
              <Td className="font-semibold">{t.name}</Td>
              <Td><Badge kind={t.plan_provider === "admin_manual" ? "default" : "active"}>{t.plan}{t.plan_provider === "admin_manual" ? " · cấp tay" : ""}</Badge></Td>
              <Td>${t.revenue_month.toFixed(2)}</Td>
              <Td>{t.pending > 0 ? <span className="text-warn font-semibold">{t.pending}</span> : <span className="text-muted">—</span>}</Td>
              <Td><Select className="w-auto h-9 py-0 leading-none text-[12px]" value="" onChange={(e) => setPlan(t, e.target.value)}>
                <option value="">— Chọn gói —</option>
                {plans.map((p) => <option key={p.code} value={p.code}>{p.name}</option>)}
              </Select></Td>
            </tr>)}
          </Table>}
        {rows && rows.length > 0 && <LoadMore show={more} loading={loadingMore} onClick={loadMore} shown={rows.length} total={total} />}
      </Card>
    </div>
  );
}

function FinanceCard() {
  const [f, setF] = useState<any>(null); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/admin/finance").then(setF).catch((e) => setErr(e.message)); }, []);
  if (err) return null;
  if (!f) return <Card><CardTitle>Doanh thu & lợi nhuận</CardTitle><Spinner /></Card>;
  const profitPos = f.profit_month >= 0;
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="So sánh doanh thu đã thu với chi phí AI trong tháng để biết lãi hay lỗ."
          right={<Button variant="sec" size="sm" onClick={() => api.download("/api/admin/reports/finance.csv", "omnishop-finance-by-model.csv")}>Xuất theo model (CSV)</Button>}>Doanh thu & lợi nhuận</CardTitle>
        <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
          <Kpi n={`$${fmt(Math.round(f.revenue_month))}`} l="Doanh thu tháng" info="Tổng hoá đơn đã thanh toán trong tháng này." />
          <Kpi n={`$${f.cost_month.toFixed(2)}`} l="Chi phí AI tháng" info="Chi phí suy luận ước tính (token vào/ra) toàn nền tảng tháng này." />
          <Kpi n={<span className={profitPos ? "text-ok" : "text-bad"}>{profitPos ? "" : "−"}${fmt(Math.abs(Math.round(f.profit_month)))}</span>} l="Lợi nhuận gộp" info="Doanh thu tháng trừ chi phí AI tháng." />
          <Kpi n={f.margin_month == null ? "—" : `${f.margin_month}%`} l="Biên lợi nhuận" info="Lợi nhuận / doanh thu. '—' khi chưa có doanh thu." />
        </div>
        <div className="text-[12.5px] text-muted font-normal mt-3">
          Chờ thu: <b className="text-fg">${fmt(Math.round(f.pending))}</b> · Đã thu luỹ kế: <b className="text-fg">${fmt(Math.round(f.revenue_all))}</b> · Chi phí luỹ kế: <b className="text-fg">${f.cost_all.toFixed(2)}</b>{f.comped_month ? <> · Gói cấp tay (không tính DT): <b className="text-fg">${fmt(Math.round(f.comped_month))}</b></> : null}
        </div>
      </Card>
      <Card>
        <CardTitle sub="Token vào, token ra và chi phí theo từng model trong tháng.">Token & chi phí theo model</CardTitle>
        {f.by_model.length > 0 && <div className="mb-4"><BarList data={f.by_model.map((m: any) => ({ label: m.model, value: m.cost, hint: `$${m.cost.toFixed(4)}` }))} /></div>}
        {f.by_model.length === 0 ? <Empty>Chưa có lượt gọi AI nào tháng này.</Empty> :
          <Table head={["Model", "Tin nhắn", "Token vào", "Token ra", "Chi phí ước tính"]}>
            {f.by_model.map((m: any) => <tr key={m.model}>
              <Td className="font-semibold">{m.model}</Td><Td>{fmt(m.messages)}</Td>
              <Td>{fmt(m.input_tokens)}</Td><Td>{fmt(m.output_tokens)}</Td><Td className="text-muted">${m.cost.toFixed(4)}</Td>
            </tr>)}
          </Table>}
      </Card>
      <Card>
        <CardTitle sub="Doanh thu đã thu và chi phí AI theo từng khách hàng trong tháng.">Lãi/lỗ theo khách hàng</CardTitle>
        {f.by_tenant.length === 0 ? <Empty>Chưa có khách hàng.</Empty> :
          <Table head={["Khách", "Gói", "Doanh thu", "Chi phí AI", "Token (vào/ra)", "Lợi nhuận"]}>
            {f.by_tenant.map((t: any) => <tr key={t.id}>
              <Td className="font-semibold">{t.name}</Td><Td><Badge kind="active">{t.plan}</Badge></Td>
              <Td>${fmt(Math.round(t.revenue))}</Td><Td className="text-muted">${t.cost.toFixed(4)}</Td>
              <Td className="text-muted">{fmt(t.input_tokens)} / {fmt(t.output_tokens)}</Td>
              <Td className={t.profit >= 0 ? "text-ok font-semibold" : "text-bad font-semibold"}>{t.profit >= 0 ? "" : "−"}${fmt(Math.abs(Math.round(t.profit * 100) / 100))}</Td>
            </tr>)}
          </Table>}
      </Card>
    </div>
  );
}

function AuditCard() {
  const [rows, setRows] = useState<any[] | null>(null);
  useEffect(() => { api.get("/api/admin/audit?limit=60").then(setRows).catch(() => setRows([])); }, []);
  return (
    <Card>
      <CardTitle sub="Ai đã đăng nhập, đổi cấu hình, cấp quyền hay thu hồi — mới nhất trước.">Nhật ký hoạt động</CardTitle>
      {!rows ? <Spinner /> : rows.length === 0 ? <Empty>Chưa có hoạt động.</Empty> :
        <div className="max-h-[360px] overflow-auto"><Table head={["Thời gian", "Người thực hiện", "Hành động", "Đối tượng"]}>
          {rows.map((r, i) => <tr key={i}>
            <Td className="text-muted whitespace-nowrap">{new Date(r.created_at).toLocaleString("vi-VN")}</Td>
            <Td className="font-normal">{r.actor || "—"}</Td>
            <Td className="font-semibold">{r.action}</Td>
            <Td className="text-muted">{r.target || ""}</Td>
          </tr>)}
        </Table></div>}
    </Card>
  );
}

function StaffCard() {
  const [rows, setRows] = useState<any[]>([]); const [email, setEmail] = useState(""); const [role, setRole] = useState("manager");
  const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  const load = () => api.get("/api/auth/staff").then(setRows).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);
  const grant = async () => { setOk(""); setErr(""); try { await api.put("/api/auth/staff", { email, platform_role: role }); setEmail(""); setOk("Đã cập nhật vai trò."); load(); } catch (e: any) { setErr(e.message); } };
  const revoke = async (e2: string) => { try { await api.put("/api/auth/staff", { email: e2, platform_role: "none" }); load(); } catch (e: any) { setErr(e.message); } };
  return (
    <Card>
      <CardTitle sub="Cấp quyền Quản trị (toàn quyền) hoặc Quản lý (chỉ xem & xuất báo cáo). Người dùng phải đã có tài khoản.">Nhân sự vận hành</CardTitle>
      <div className="grid md:grid-cols-[1fr_180px_auto] gap-2 items-end">
        <Field label="Email"><Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="nguoivanhanh@congty.com" /></Field>
        <Field label="Vai trò"><Select value={role} onChange={(e) => setRole(e.target.value)}><option value="manager">Quản lý (xem + báo cáo)</option><option value="admin">Quản trị (toàn quyền)</option></Select></Field>
        <Button variant="sec" onClick={grant}>Cấp quyền</Button>
      </div>
      {rows.length > 0 && <div className="mt-3"><Table head={["Email", "Vai trò", ""]}>
        {rows.map((r) => <tr key={r.email}><Td className="font-semibold">{r.email}</Td><Td><Badge kind={r.platform_role === "admin" ? "ai" : "active"}>{r.platform_role === "admin" ? "Quản trị" : "Quản lý"}</Badge></Td><Td className="text-right"><button className="text-[12px] text-bad font-semibold" onClick={() => revoke(r.email)}>Thu hồi</button></Td></tr>)}
      </Table></div>}
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

function GoogleCard() {
  const [c, setC] = useState<any>(null); const [cid, setCid] = useState(""); const [secret, setSecret] = useState(""); const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/admin/settings/google").then((d) => { setC(d); setCid(d.client_id || ""); }).catch((e) => setErr(e.message)); }, []);
  const save = async () => { setOk(""); setErr(""); try { await api.put("/api/admin/settings/google", { client_id: cid, client_secret: secret || null }); setOk("Đã lưu Google Sign-In."); setSecret(""); } catch (e: any) { setErr(e.message); } };
  return (
    <Card>
      <CardTitle sub="Bật đăng nhập bằng Google cho mọi người dùng. Redirect URI cần đăng ký ở Google Cloud: <base>/api/auth/google/callback.">Đăng nhập Google (OAuth)</CardTitle>
      <div className="grid md:grid-cols-2 gap-3">
        <Field label="Client ID"><Input value={cid} onChange={(e) => setCid(e.target.value)} placeholder="...apps.googleusercontent.com" /></Field>
        <Field label="Client Secret" info="Mã hoá khi lưu."><Input type="password" value={secret} onChange={(e) => setSecret(e.target.value)} placeholder={c && c.has_secret ? "•••• (giữ nguyên)" : ""} /></Field>
      </div>
      <div className="mt-3"><CopyField label="Authorized redirect URI (dán vào Google Cloud Console)" value={c?.redirect_uri}
        info="Google Cloud → APIs & Services → Credentials → OAuth client → Authorized redirect URIs. Phải khớp chính xác." /></div>
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

function RuntimeCard() {
  const [c, setC] = useState<any>(null); const [base, setBase] = useState(""); const [stale, setStale] = useState("600");
  const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/admin/settings/runtime").then((d) => { setC(d); setBase(d.public_base || ""); setStale(String(d.stale_seconds ?? 600)); }).catch((e) => setErr(e.message)); }, []);
  const save = async () => { setOk(""); setErr(""); try { const d = await api.put("/api/admin/settings/runtime", { public_base: base.trim(), stale_seconds: Number(stale) || 0 }); setC({ ...c, ...d }); setOk("Đã lưu cấu hình vận hành."); } catch (e: any) { setErr(e.message); } };
  const badBase = base && !base.startsWith("https://");
  return (
    <Card>
      <CardTitle sub="Tên miền công khai để nhận webhook và chuyển hướng đăng nhập. Đặt đúng domain HTTPS thật thì các kênh mới nhận được tin và đăng nhập Google/Facebook mới quay về đúng chỗ.">Vận hành</CardTitle>
      <div className="grid md:grid-cols-2 gap-3">
        <Field label="Tên miền công khai (Public URL)" info="Ví dụ: https://chat.cuahang.vn — không có dấu / ở cuối."><Input value={base} onChange={(e) => setBase(e.target.value)} placeholder="https://ten-mien-cua-ban" /></Field>
        <Field label="Bỏ qua tin cũ hơn (giây)" info="Tin đến cũ hơn ngưỡng này sẽ được lưu nhưng không trả lời tự động, tránh tốn token khi kênh vừa kết nối lại. 0 = tắt."><Input type="number" value={stale} onChange={(e) => setStale(e.target.value)} /></Field>
      </div>
      {badBase ? <div className="mt-2 text-[12px] text-warn font-normal">Nên dùng địa chỉ bắt đầu bằng https:// — webhook của các nền tảng yêu cầu HTTPS công khai.</div> : null}
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

function EmailCard() {
  const [c, setC] = useState<any>(null); const [provider, setProvider] = useState("console"); const [from, setFrom] = useState("");
  const [secret, setSecret] = useState(""); const [host, setHost] = useState(""); const [port, setPort] = useState("587"); const [user, setUser] = useState("");
  const [to, setTo] = useState(""); const [ok, setOk] = useState(""); const [err, setErr] = useState(""); const [testing, setTesting] = useState(false);
  useEffect(() => { api.get("/api/admin/settings/email").then((d) => { setC(d); setProvider(d.provider || "console"); setFrom(d.from || ""); setHost(d.smtp_host || ""); setPort(String(d.smtp_port || 587)); setUser(d.smtp_user || ""); }).catch((e) => setErr(e.message)); }, []);
  const body = () => ({ provider, from_addr: from, secret: secret || null, smtp_host: host, smtp_port: Number(port) || 587, smtp_user: user });
  const save = async () => { setOk(""); setErr(""); try { await api.put("/api/admin/settings/email", body()); setOk("Đã lưu cấu hình email."); setSecret(""); } catch (e: any) { setErr(e.message); } };
  const test = async () => { setOk(""); setErr(""); setTesting(true); try { const r = await api.post("/api/admin/settings/email/test", { ...body(), to }); if (r.ok) setOk(`Đã gửi email thử tới ${to || "email của bạn"}.`); else setErr("Gửi thất bại: " + (r.error || "")); } catch (e: any) { setErr(e.message); } finally { setTesting(false); } };
  const secretLabel = provider === "resend" ? "Resend API Key" : provider === "smtp" ? "Mật khẩu SMTP" : "Khoá bí mật";
  return (
    <Card>
      <CardTitle sub="Dùng để gửi email cảnh báo khi kênh gặp sự cố và các thông báo khác. Console chỉ ghi log; chọn SMTP hoặc Resend để gửi thật.">Email thông báo</CardTitle>
      <div className="grid md:grid-cols-2 gap-3">
        <Field label="Nhà cung cấp"><Select value={provider} onChange={(e) => setProvider(e.target.value)}>{(c?.providers || []).map((p: any) => <option key={p.id} value={p.id}>{p.label}</option>)}</Select></Field>
        <Field label="Địa chỉ người gửi (From)" info="Ví dụ: OmniShop <no-reply@cuahang.vn>"><Input value={from} onChange={(e) => setFrom(e.target.value)} placeholder="Tên cửa hàng <no-reply@domain>" /></Field>
      </div>
      {provider !== "console" && (
        <div className="grid md:grid-cols-2 gap-3 mt-3">
          {provider === "smtp" && <>
            <Field label="SMTP Host"><Input value={host} onChange={(e) => setHost(e.target.value)} placeholder="smtp.gmail.com" /></Field>
            <Field label="SMTP Port"><Input type="number" value={port} onChange={(e) => setPort(e.target.value)} /></Field>
            <Field label="SMTP User"><Input value={user} onChange={(e) => setUser(e.target.value)} placeholder="tài khoản đăng nhập SMTP" /></Field>
          </>}
          <Field label={secretLabel} info="Mã hoá khi lưu."><Input type="password" value={secret} onChange={(e) => setSecret(e.target.value)} placeholder={c && c.has_secret ? "•••• (giữ nguyên)" : ""} /></Field>
        </div>
      )}
      <div className="mt-3 flex items-end gap-2 flex-wrap">
        <Field label="Gửi thử tới"><Input value={to} onChange={(e) => setTo(e.target.value)} placeholder="email nhận thử (mặc định email của bạn)" /></Field>
        <Button variant="ghost" size="sm" loading={testing} onClick={test}>Gửi email thử</Button>
      </div>
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

function PlansCard() {
  const [plans, setPlans] = useState<any[] | null>(null); const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/admin/plans").then(setPlans).catch((e) => setErr(e.message)); }, []);
  const setField = (i: number, patch: any) => setPlans((ps) => ps!.map((p, j) => j === i ? { ...p, ...patch } : p));
  const setEnt = (i: number, patch: any) => setPlans((ps) => ps!.map((p, j) => j === i ? { ...p, entitlements: { ...p.entitlements, ...patch } } : p));
  const save = async (p: any) => { setOk(""); setErr(""); try { await api.put(`/api/admin/plans/${p.code}`, { name: p.name, price_month: Number(p.price_month), entitlements: p.entitlements }); setOk(`Đã lưu gói ${p.name}.`); } catch (e: any) { setErr(e.message); } };
  const num = (v: any) => v === "" || v === null || v === undefined ? 0 : Number(v);
  return (
    <Card>
      <CardTitle sub="Toàn quyền định giá: điều chỉnh tên gói, giá, chế độ AI, hạn mức token và đơn giá vượt/PAYG cho từng gói ngay tại đây.">Gói dịch vụ & định giá</CardTitle>
      {!plans ? <Spinner /> : (
        <div className="space-y-3">
          {plans.map((p, i) => (
            <div key={p.code} className="border border-line rounded-xl p-3">
              <div className="grid md:grid-cols-4 gap-2 items-end">
                <Field label={`Gói · ${p.code}`}><Input value={p.name} onChange={(e) => setField(i, { name: e.target.value })} /></Field>
                <Field label="Giá / tháng ($)"><Input type="number" value={p.price_month} onChange={(e) => setField(i, { price_month: e.target.value })} /></Field>
                <Field label="Kiểu AI"><Select value={p.entitlements.llm_mode || "byok"} onChange={(e) => setEnt(i, { llm_mode: e.target.value })}><option value="byok">Tự nhập khoá</option><option value="managed">Trọn gói (nền tảng)</option></Select></Field>
                <Field label="Thanh toán"><Select value={p.entitlements.billing_mode || "subscription"} onChange={(e) => setEnt(i, { billing_mode: e.target.value })}><option value="subscription">Thuê bao</option><option value="payg">Trả theo dùng</option></Select></Field>
              </div>
              <div className="grid md:grid-cols-4 gap-2 items-end mt-2">
                <Field label="Token/tháng" info="Gói trọn gói: token AI kèm theo."><Input type="number" value={num(p.entitlements.ai_tokens_month)} onChange={(e) => setEnt(i, { ai_tokens_month: num(e.target.value) })} /></Field>
                <Field label="Vượt hạn mức ($/1k)"><Input type="number" step="0.001" value={num(p.entitlements.overage_per_1k)} onChange={(e) => setEnt(i, { overage_per_1k: num(e.target.value) })} /></Field>
                <Field label="PAYG ($/1k)"><Input type="number" step="0.001" value={num(p.entitlements.payg_per_1k)} onChange={(e) => setEnt(i, { payg_per_1k: num(e.target.value) })} /></Field>
                <Field label="Trần tin nhắn" info="Gói tự nhập khoá: 0 = không giới hạn."><Input type="number" value={num(p.entitlements.ai_messages_month)} onChange={(e) => setEnt(i, { ai_messages_month: num(e.target.value) })} /></Field>
              </div>
              <div className="mt-2"><Button size="sm" variant="sec" onClick={() => save(p)}>Lưu gói</Button></div>
            </div>
          ))}
        </div>
      )}
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

function CostCard() {
  const [c, setC] = useState<any>(null); const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/admin/settings/cost").then(setC).catch((e) => setErr(e.message)); }, []);
  const save = async () => { setOk(""); setErr(""); try { const d = await api.put("/api/admin/settings/cost", { cost_input_per_m: Number(c.input), cost_output_per_m: Number(c.output), cost_embedding_per_m: Number(c.embedding) }); setC(d); setOk("Đã lưu đơn giá."); } catch (e: any) { setErr(e.message); } };
  return (
    <Card>
      <CardTitle sub="Đơn giá token ($/1 triệu token) dùng để ước tính chi phí và tính PAYG. Admin chỉnh trực tiếp, không cần deploy.">Đơn giá token (COGS)</CardTitle>
      {!c ? <Spinner /> : (
        <div className="grid md:grid-cols-3 gap-3">
          <Field label="Đầu vào ($/1M)"><Input type="number" step="0.01" value={c.input} onChange={(e) => setC({ ...c, input: e.target.value })} /></Field>
          <Field label="Đầu ra ($/1M)"><Input type="number" step="0.01" value={c.output} onChange={(e) => setC({ ...c, output: e.target.value })} /></Field>
          <Field label="Embedding ($/1M)"><Input type="number" step="0.001" value={c.embedding} onChange={(e) => setC({ ...c, embedding: e.target.value })} /></Field>
        </div>
      )}
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

function BrandingCard() {
  const [b, setB] = useState<any>(null); const [name, setName] = useState(""); const [accent, setAccent] = useState("#818cf8");
  const [ok, setOk] = useState(""); const [err, setErr] = useState(""); const logoRef = useRef<HTMLInputElement>(null);
  const load = () => api.get("/api/admin/branding").then((d) => { setB(d); setName(d.app_name); setAccent(d.accent_color || "#818cf8"); }).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);
  const save = async () => { setOk(""); setErr(""); try { const d = await api.put("/api/admin/branding", { app_name: name, accent_color: accent }); setB(d); setOk("Đã lưu. Tải lại trang để áp dụng toàn hệ thống."); } catch (e: any) { setErr(e.message); } };
  const onLogo = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return; setOk(""); setErr("");
    const fd = new FormData(); fd.append("file", file);
    try { const d = await api.upload("/api/admin/branding/logo", fd); setB(d); setOk("Đã cập nhật logo. Tải lại trang để áp dụng."); } catch (ex: any) { setErr(ex.message); }
    finally { if (logoRef.current) logoRef.current.value = ""; }
  };
  return (
    <Card>
      <CardTitle sub="Đặt tên và logo hiển thị cho toàn hệ thống. Áp dụng cho mọi khách hàng.">Thương hiệu hệ thống</CardTitle>
      <div className="flex items-center gap-3 mb-3">
        <span className="w-14 h-14 rounded-xl bg-pastel flex items-center justify-center overflow-hidden shrink-0">
          {b?.logo_url ? <img src={b.logo_url} className="w-14 h-14 object-contain" /> : <Bot className="w-7 h-7 text-[#0b0e1a]" />}
        </span>
        <div>
          <input ref={logoRef} type="file" accept="image/*" className="hidden" onChange={onLogo} />
          <Button size="sm" variant="sec" onClick={() => logoRef.current?.click()}><Upload className="w-3.5 h-3.5" /> Tải logo</Button>
          <div className="text-[11px] text-muted mt-1 font-normal">PNG/SVG nền trong suốt, tối đa 2MB</div>
        </div>
      </div>
      <div className="grid md:grid-cols-2 gap-3">
        <Field label="Tên hệ thống"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="OmniShop AI" /></Field>
        <Field label="Màu nhấn"><div className="flex gap-2 items-center"><input type="color" value={accent} onChange={(e) => setAccent(e.target.value)} className="w-10 h-9 rounded-lg border border-line bg-bg p-0.5" /><Input value={accent} onChange={(e) => setAccent(e.target.value)} /></div></Field>
      </div>
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

function PaymentCard() {
  const [cfg, setCfg] = useState<any>(null); const [providers, setProviders] = useState<any[]>([]);
  const [provider, setProvider] = useState("manual"); const [apiKey, setApiKey] = useState("");
  const [f, setF] = useState<any>({ publishable_key: "", webhook_secret: "", success_url: "", cancel_url: "", currency: "USD",
    bank_bin: "", account_no: "", account_name: "", template: "compact2", usd_vnd: "25000",
    tmn_code: "", pay_url: "", return_url: "", partner_code: "", access_key: "", redirect_url: "", ipn_url: "", endpoint: "" });
  const [ok, setOk] = useState(""); const [err, setErr] = useState(""); const [hookUrl, setHookUrl] = useState(""); const [txns, setTxns] = useState<any[]>([]);
  const set = (k: string, v: string) => setF((s: any) => ({ ...s, [k]: v }));
  useEffect(() => { api.get("/api/admin/settings/payment").then((d) => { setProviders(d.providers); setCfg(d.config); setHookUrl(d.sepay_webhook_url || ""); if (d.config) { setProvider(d.config.provider); if (d.config.extra) setF((s: any) => ({ ...s, ...d.config.extra })); } }).catch((e) => setErr(e.message)); api.get("/api/admin/billing/sepay?limit=8").then((d) => setTxns(d.items)).catch(() => {}); }, []);
  const save = async () => {
    setOk(""); setErr("");
    try { await api.put("/api/admin/settings/payment", { provider, api_key: apiKey || null, ...f }); setOk("Đã lưu cấu hình thanh toán."); setApiKey(""); } catch (e: any) { setErr(e.message); }
  };
  return (
    <Card>
      <CardTitle sub="Chọn cổng thanh toán và nhập khoá. Khoá được mã hoá khi lưu. Khách hàng sẽ thanh toán qua cổng này.">Cấu hình thanh toán</CardTitle>
      <Field label="Cổng thanh toán"><Select value={provider} onChange={(e) => setProvider(e.target.value)}>{providers.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}</Select></Field>
      {({ vietqr: "https://www.vietqr.io/en/danh-sach-api/link-tao-ma-nhanh/", vnpay: "https://sandbox.vnpayment.vn/apis/docs/thanh-toan-pay/pay.html", momo: "https://developers.momo.vn/v3/docs/payment/api/payment-api/init", stripe: "https://stripe.com/docs/payments/checkout" } as any)[provider] &&
        <a href={({ vietqr: "https://www.vietqr.io/en/danh-sach-api/link-tao-ma-nhanh/", vnpay: "https://sandbox.vnpayment.vn/apis/docs/thanh-toan-pay/pay.html", momo: "https://developers.momo.vn/v3/docs/payment/api/payment-api/init", stripe: "https://stripe.com/docs/payments/checkout" } as any)[provider]} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[12px] text-accent font-semibold mt-1.5">Xem tài liệu tích hợp của cổng ↗</a>}
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
      {["sepay", "vietqr", "vnpay", "momo"].includes(provider) && (
        <div className="mt-3"><Field label="Tỷ giá USD → VND" info="Giá gói lưu bằng USD; các cổng Việt Nam thu bằng VND nên được quy đổi theo tỷ giá này. Ví dụ 25000 nghĩa là 1 USD = 25.000đ.">
          <Input type="number" value={f.usd_vnd} onChange={(e) => set("usd_vnd", e.target.value)} placeholder="25000" className="max-w-[220px]" /></Field></div>
      )}
      {provider === "sepay" && (
        <>
          <div className="grid md:grid-cols-2 gap-3 mt-3">
            <Field label="Ngân hàng (mã/tên SePay)" info="Mã ngân hàng theo SePay, ví dụ: TPBank, VCB, MBBank, ACB… hoặc mã BIN."><Input value={f.bank_bin} onChange={(e) => set("bank_bin", e.target.value)} placeholder="TPBank" /></Field>
            <Field label="Số tài khoản nhận"><Input value={f.account_no} onChange={(e) => set("account_no", e.target.value)} placeholder="0123456789" /></Field>
            <Field label="Tên chủ tài khoản"><Input value={f.account_name} onChange={(e) => set("account_name", e.target.value)} placeholder="CONG TY OMNISHOP" /></Field>
            <Field label="API Key Webhook" info="SePay gửi header Authorization: Apikey <key>. Nhập đúng key bạn đặt trong SePay. Mã hoá khi lưu."><Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder={cfg && cfg.has_key ? "•••• (giữ nguyên)" : "khoá webhook SePay"} /></Field>
          </div>
          <div className="mt-3"><CopyField label="Webhook URL (dán vào SePay → Tích hợp → Webhooks)" value={hookUrl}
            info="Phương thức POST, xác thực bằng API Key ở trên. Khi có tiền vào khớp nội dung, gói tự kích hoạt." /></div>
          <div className="mt-4">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted mb-1.5">Giao dịch SePay gần đây</div>
            {txns.length === 0 ? <div className="text-[13px] text-muted font-normal py-2">Chưa nhận giao dịch nào từ SePay.</div> :
              <Table head={["Thời gian", "Ngân hàng", "Số tiền", "Nội dung", "Kết quả"]}>
                {txns.map((t, i) => <tr key={i}>
                  <Td className="text-muted whitespace-nowrap">{new Date(t.created_at).toLocaleString("vi-VN")}</Td>
                  <Td>{t.gateway || "—"}</Td>
                  <Td className="font-semibold whitespace-nowrap">{fmt(Math.round(t.amount))}</Td>
                  <Td className="text-muted max-w-[200px] truncate" title={t.content}>{t.content || "—"}</Td>
                  <Td>{t.matched ? <Badge kind="connected">{t.tenant || "Đã khớp"}</Badge> : <Badge kind="default">Chưa khớp</Badge>}</Td>
                </tr>)}
              </Table>}
          </div>
        </>
      )}
      {provider === "vietqr" && (
        <div className="grid md:grid-cols-2 gap-3 mt-3">
          <Field label="Mã ngân hàng (BIN)" info="Ví dụ Vietcombank 970436, Techcombank 970407, MBBank 970422."><Input value={f.bank_bin} onChange={(e) => set("bank_bin", e.target.value)} placeholder="970436" /></Field>
          <Field label="Số tài khoản"><Input value={f.account_no} onChange={(e) => set("account_no", e.target.value)} placeholder="0123456789" /></Field>
          <Field label="Tên chủ tài khoản"><Input value={f.account_name} onChange={(e) => set("account_name", e.target.value)} placeholder="CONG TY OMNISHOP" /></Field>
          <Field label="Mẫu QR" info="compact2 (mặc định), compact, qr_only, print."><Input value={f.template} onChange={(e) => set("template", e.target.value)} placeholder="compact2" /></Field>
        </div>
      )}
      {provider === "vnpay" && (
        <div className="grid md:grid-cols-2 gap-3 mt-3">
          <Field label="TMN Code"><Input value={f.tmn_code} onChange={(e) => set("tmn_code", e.target.value)} placeholder="OMNI0001" /></Field>
          <Field label="Hash Secret" info="Chuỗi bí mật để ký HMAC-SHA512 — mã hoá khi lưu."><Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="••••••••" /></Field>
          <Field label="Pay URL" info="Sandbox mặc định; go-live đổi sang https://pay.vnpay.vn/vpcpay.html."><Input value={f.pay_url} onChange={(e) => set("pay_url", e.target.value)} placeholder="https://sandbox.vnpayment.vn/paymentv2/vpcpay.html" /></Field>
          <Field label="Return URL"><Input value={f.return_url} onChange={(e) => set("return_url", e.target.value)} placeholder="https://app.cua-ban.com/api/billing/return/vnpay" /></Field>
        </div>
      )}
      {provider === "momo" && (
        <div className="grid md:grid-cols-2 gap-3 mt-3">
          <Field label="Partner Code"><Input value={f.partner_code} onChange={(e) => set("partner_code", e.target.value)} placeholder="MOMOXXXX" /></Field>
          <Field label="Access Key"><Input value={f.access_key} onChange={(e) => set("access_key", e.target.value)} placeholder="..." /></Field>
          <Field label="Secret Key" info="Ký HMAC-SHA256 — mã hoá khi lưu."><Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="••••••••" /></Field>
          <Field label="Endpoint" info="Sandbox mặc định; đổi sang payment.momo.vn khi go-live."><Input value={f.endpoint} onChange={(e) => set("endpoint", e.target.value)} placeholder="https://test-payment.momo.vn/v2/gateway/api/create" /></Field>
          <Field label="Redirect URL"><Input value={f.redirect_url} onChange={(e) => set("redirect_url", e.target.value)} placeholder="https://app.cua-ban.com/?paid=1" /></Field>
          <Field label="IPN URL"><Input value={f.ipn_url} onChange={(e) => set("ipn_url", e.target.value)} placeholder="https://app.cua-ban.com/api/billing/ipn/momo" /></Field>
        </div>
      )}
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
  const [llmInfo, setLlmInfo] = useState<any>(null); const [models, setModels] = useState<string[]>([]); const [loadingModels, setLoadingModels] = useState(false);
  useEffect(() => { api.get("/api/settings/llm").then(setLlmInfo).catch(() => {}); }, []);
  const loadModels = async () => {
    if (!llmInfo?.effective) return; setLoadingModels(true);
    try { const r = await api.post("/api/settings/llm/models", { provider: llmInfo.effective.provider, base_url: llmInfo.effective.base_url || "" }); setModels(r.models || []); } catch { /* ignore */ } finally { setLoadingModels(false); }
  };
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
    <div className="flex flex-col h-[calc(100vh-118px)]">
      <div className="flex items-center gap-2 mb-3 shrink-0">
        <Button variant="ghost" size="sm" onClick={onBack}>← Trợ lý</Button>
        <span className="text-muted text-sm">/</span>
        <span className="font-semibold">{f.name || "Trợ lý mới"}</span>
      </div>
      <div className="grid lg:grid-cols-[1fr_430px] gap-4 flex-1 min-h-0">
        {/* test chat — big, full height, left */}
        <BotTestPanel botId={f.id} accent={f.accent_color} />
        {/* settings — full height, right, scrolls */}
        <Card className="h-full overflow-auto">
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
          <div className="border-t border-line/60 mt-4 pt-3">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted mb-2">Cấu hình nâng cao</div>
            <Field label="Mô hình AI riêng cho trợ lý này" info="Để trống sẽ dùng mô hình mặc định. Trường này chỉ thay tên model, vẫn dùng nhà cung cấp và khoá AI đã cấu hình ở Cài đặt.">
              <div className="flex gap-2">
                <Input value={cfg.model || ""} onChange={(e) => setCfg({ model: e.target.value })} placeholder={llmInfo?.effective?.model ? `Mặc định: ${llmInfo.effective.model}` : "Mặc định của hệ thống"} />
                <Button variant="sec" size="sm" loading={loadingModels} onClick={loadModels}><RefreshCw className="w-3.5 h-3.5" /></Button>
              </div>
            </Field>
            {models.length > 0 && <Select className="mt-2" value="" onChange={(e) => e.target.value && setCfg({ model: e.target.value })}><option value="">Chọn từ {models.length} model có sẵn</option>{models.map((m) => <option key={m} value={m}>{m}</option>)}</Select>}
            <div className="grid sm:grid-cols-2 gap-3 mt-3">
              <Field label="Số đoạn ngữ cảnh RAG (top-k)" info="Số sản phẩm/tài liệu liên quan nhất được đưa vào ngữ cảnh mỗi câu hỏi. Cao hơn = nhiều thông tin hơn nhưng tốn token hơn. Mặc định 3.">
                <Input type="number" min={1} max={8} value={cfg.rag_top_k ?? 3} onChange={(e) => setCfg({ rag_top_k: Math.max(1, Math.min(8, parseInt(e.target.value) || 3)) })} />
              </Field>
            </div>
          </div>
          {canManage && <div className="mt-4 flex gap-2"><Button loading={busy} onClick={save}>Lưu</Button>
            {!isNew && <Button variant="danger" onClick={async () => { if (confirm("Xoá trợ lý này?")) { try { await api.del(`/api/bots/${f.id}`); onBack(); } catch (e: any) { setErr(e.message); } } }}>Xoá</Button>}</div>}
          <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
        </Card>
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
    <Card className="h-full flex flex-col min-h-0">
      <CardTitle sub="Trò chuyện nhiều lượt với dữ liệu thật, có ghi nhớ hội thoại.">Chạy thử</CardTitle>
      <div ref={boxRef} className="flex-1 min-h-0 overflow-auto space-y-2 pr-1">
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
      <div className="mt-4 border-t border-line/60 pt-3 space-y-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-muted">Dán các URL này vào Facebook App</div>
        <CopyField label="Valid OAuth Redirect URI" value={c?.redirect_uri}
          info="Facebook Login → Settings → Valid OAuth Redirect URIs. Khắc phục lỗi “URL không thuộc domain”." />
        <CopyField label="Webhook Callback URL" value={c?.webhook_url}
          info="Meta App → Webhooks → callback URL (kèm Verify token ở trên), subscribe field messages." />
        <p className="text-[12px] text-muted font-normal">Thêm domain vào mục App Domains trong Settings → Basic. Các URL sinh theo <code>OAUTH_REDIRECT_BASE</code>; nếu đang là localhost, hãy đặt domain thật rồi khởi động lại.</p>
      </div>
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

// ============================================================ Help / Docs
function StepList({ steps }: { steps: string[] }) {
  return (
    <ol className="space-y-2.5">
      {steps.map((d, i) => (
        <li key={i} className="flex gap-3">
          <span className="w-5 h-5 rounded-full bg-accent/20 text-accent text-[11px] font-bold flex items-center justify-center shrink-0 mt-0.5">{i + 1}</span>
          <div className="text-[13px] text-fg font-normal leading-relaxed">{d}</div>
        </li>
      ))}
    </ol>
  );
}

export function Help() {
  const [kinds, setKinds] = useState<any[]>([]);
  useEffect(() => { api.get("/api/channels/kinds").then(setKinds).catch(() => {}); }, []);
  const quick = [
    "Tạo cửa hàng ở thanh chọn đầu trang (nếu chưa có).",
    "Vào Trợ lý → Tạo trợ lý: đặt tên, prompt, lời chào, ảnh đại diện.",
    "Nạp dữ liệu: thêm Sản phẩm (giá, tồn kho, biến thể) và Kiến thức (tài liệu/FAQ).",
    "Chạy thử trợ lý ngay trong trang cấu hình để kiểm tra câu trả lời.",
    "Kết nối kênh (Website, Messenger, Telegram, Zalo…) để đưa trợ lý ra khách hàng.",
    "Theo dõi ở Tổng quan và tiếp quản hội thoại trong Hộp thư khi cần.",
  ];
  const botSteps = [
    "Vào menu Trợ lý → “Tạo trợ lý”.",
    "Đặt Tên, Lời chào và tải Ảnh đại diện; chọn màu widget.",
    "Viết Prompt tuỳ chỉnh: cách xưng hô, giọng điệu, điều nên/không nên làm.",
    "Cấu hình nâng cao (tuỳ chọn): chọn model AI riêng và số đoạn ngữ cảnh RAG (top-k).",
    "Bật Chuyển nhân viên khi thiếu thông tin và Giới hạn giờ làm việc nếu cần.",
    "Dùng khung “Chạy thử” bên trái để trò chuyện thử với dữ liệu thật trước khi kết nối kênh.",
  ];
  const productSteps = [
    "Vào Sản phẩm → “Thêm”.",
    "Nhập Tên, SKU, Giá, Danh mục và Mô tả.",
    "Nhập Biến thể & tồn kho dạng Tên:Số lượng, cách nhau bởi dấu phẩy (VD: Size M:10, Size L:3).",
    "Chọn “Áp dụng cho” một trợ lý cụ thể hoặc tất cả trợ lý.",
    "Trợ lý sẽ dùng dữ liệu này để trả lời chính xác về giá, tồn kho, biến thể.",
  ];
  const knowledgeSteps = [
    "Vào Kiến thức. Dán nội dung văn bản, hoặc tải tệp (PDF, Word, Excel, PowerPoint, CSV, ảnh…).",
    "Ảnh/PDF scan sẽ được nhận dạng bằng OCR. Tối đa 25MB mỗi tệp.",
    "Hệ thống trích xuất → chia đoạn → nhúng vector ở nền; theo dõi trạng thái (đang xử lý/sẵn sàng).",
    "Bấm một tài liệu để xem văn bản đã trích xuất; có thể Xử lý lại hoặc Xoá.",
    "Dùng công tắc “Dùng cho AI” để bật/tắt một tài liệu mà không cần xoá.",
  ];
  const faqs = [
    ["Trợ lý AI lấy thông tin từ đâu?", "Từ Sản phẩm (giá/tồn/biến thể) và tài liệu Kiến thức bạn đã nhập. Nếu không đủ thông tin, hội thoại được chuyển cho nhân viên."],
    ["Tôi có thể dùng mô hình nào?", "Anthropic Claude, OpenAI, Google Gemini, hoặc máy chủ tự host tương thích OpenAI như vLLM (nếu quản trị cho phép chọn model)."],
    ["Cần HTTPS để kết nối kênh không?", "Các kênh nhận tin qua webhook như Telegram, Messenger, Zalo, WhatsApp cần địa chỉ HTTPS công khai. Tiện ích website thì không cần."],
    ["Dữ liệu giữa các cửa hàng có tách biệt không?", "Có. Mỗi tổ chức được cô lập nhiều lớp; khách hàng này không truy cập được dữ liệu của khách hàng khác."],
  ];
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Sáu bước để đưa trợ lý AI vào hoạt động.">Bắt đầu nhanh</CardTitle>
        <StepList steps={quick} />
      </Card>
      <div className="grid lg:grid-cols-2 gap-4">
        <Card><CardTitle sub="Tạo và huấn luyện trợ lý cho cửa hàng.">Tạo trợ lý AI</CardTitle><StepList steps={botSteps} /></Card>
        <Card><CardTitle sub="Dữ liệu có cấu trúc để trả lời giá & tồn kho.">Thêm sản phẩm</CardTitle><StepList steps={productSteps} /></Card>
      </div>
      <Card><CardTitle sub="Nạp tài liệu/FAQ cho trợ lý tra cứu (RAG).">Thêm kiến thức (RAG)</CardTitle><StepList steps={knowledgeSteps} /></Card>

      <Card>
        <CardTitle sub="Cách lấy thông tin để kết nối từng loại kênh. Bấm để mở hướng dẫn chi tiết + tài liệu chính thức của hãng.">Kết nối kênh — hướng dẫn theo từng loại</CardTitle>
        <div className="space-y-2">
          {kinds.filter((k) => (k.guide || []).length).map((k) => (
            <details key={k.kind} className="rounded-lg border border-line bg-card2/40 overflow-hidden">
              <summary className="cursor-pointer select-none px-3 py-2.5 text-sm font-semibold flex items-center gap-2"><ChannelMark kind={k.kind} /> {k.label}{!k.live && <Badge kind="pending">Chờ duyệt đối tác</Badge>}</summary>
              <div className="px-3 pb-3">
                <StepList steps={k.guide} />
                {k.webhook_url && <div className="mt-3"><CopyField label="Webhook URL (dán vào console của kênh)" value={k.webhook_url} /></div>}
                {k.docs && <a href={k.docs} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[12px] text-accent font-semibold mt-2">Tài liệu chính thức của hãng ↗</a>}
              </div>
            </details>
          ))}
        </div>
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
