import React, { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Badge, Button, Card, CardTitle, Empty, Field, Info, Input, Kpi, Modal, Msg, Select, Spinner, Table, Td, Textarea } from "./ui";
import { StackedBars, IntentBars } from "./charts";
import { RefreshCw, Upload, Plug, Send, UserPlus, CheckCircle2, ArrowUpRight } from "lucide-react";

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
  const [items, setItems] = useState<any[] | null>(null);
  const [f, setF] = useState({ name: "", price: "", description: "", variants: "" });
  const [err, setErr] = useState("");
  const load = () => api.get(`/api/products?shop_id=${shopId}`).then(setItems).catch((e) => setErr(e.message));
  useEffect(() => { setItems(null); load(); }, [shopId]);
  const add = async () => {
    setErr("");
    const variants = f.variants.split(",").map((s) => s.trim()).filter(Boolean).map((s) => { const [n, st] = s.split(":"); return { name: (n || "").trim(), stock: parseInt(st || "0") || 0 }; });
    try { await api.post("/api/products", { shop_id: shopId, name: f.name, price: f.price ? parseFloat(f.price) : null, description: f.description, variants }); setF({ name: "", price: "", description: "", variants: "" }); load(); }
    catch (e: any) { setErr(e.message); }
  };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Trợ lý AI dùng dữ liệu này để trả lời về giá, tồn kho và biến thể.">Thêm sản phẩm</CardTitle>
        <div className="grid md:grid-cols-2 gap-3">
          <Field label="Tên sản phẩm"><Input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="Áo thun cotton" /></Field>
          <Field label="Giá" info="Đơn vị VND. Bỏ trống nếu giá liên hệ."><Input type="number" value={f.price} onChange={(e) => setF({ ...f, price: e.target.value })} /></Field>
        </div>
        <div className="mt-3"><Field label="Mô tả"><Textarea value={f.description} onChange={(e) => setF({ ...f, description: e.target.value })} /></Field></div>
        <div className="mt-3"><Field label="Biến thể và tồn kho" info="Mỗi biến thể theo dạng Tên:Số lượng, cách nhau bởi dấu phẩy. Ví dụ: Size M:10, Size L:3."><Input value={f.variants} onChange={(e) => setF({ ...f, variants: e.target.value })} placeholder="Size M:10, Size L:3" /></Field></div>
        <div className="mt-4"><Button onClick={add}>Lưu sản phẩm</Button></div>
        <Msg type="err">{err}</Msg>
      </Card>
      <Card>
        <CardTitle>Danh sách sản phẩm</CardTitle>
        {!items ? <Spinner /> : items.length === 0 ? <Empty>Chưa có sản phẩm nào.</Empty> :
          <Table head={["Tên", "Giá", "Biến thể và tồn kho"]}>
            {items.map((p) => (
              <tr key={p.id}>
                <Td><div className="font-semibold">{p.name}</div><div className="text-xs text-muted">{(p.description || "").slice(0, 60)}</div></Td>
                <Td>{p.price != null ? `${fmt(p.price)} ${p.currency}` : "Liên hệ"}</Td>
                <Td><div className="flex flex-wrap gap-1">{(p.variants || []).map((v: any, i: number) => <span key={i} className="text-xs bg-card2 border border-line rounded px-2 py-0.5 font-normal">{v.name} · {v.stock}</span>)}</div></Td>
              </tr>
            ))}
          </Table>}
      </Card>
    </div>
  );
}

// ============================================================ Knowledge
export function Knowledge({ shopId }: { shopId: string }) {
  const [docs, setDocs] = useState<any[] | null>(null);
  const [title, setTitle] = useState(""); const [text, setText] = useState("");
  const [msg, setMsg] = useState(""); const [err, setErr] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const load = () => api.get(`/api/knowledge/documents?shop_id=${shopId}`).then((d) => { setDocs(d); if (d.some((x: any) => x.status !== "ready" && x.status !== "error")) setTimeout(load, 2000); }).catch((e) => setErr(e.message));
  useEffect(() => { setDocs(null); load(); }, [shopId]);
  const addText = async () => { setErr(""); setMsg(""); try { await api.post("/api/knowledge/documents", { shop_id: shopId, title, text }); setTitle(""); setText(""); setMsg("Đã thêm tài liệu, đang xử lý nội dung."); load(); } catch (e: any) { setErr(e.message); } };
  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return; setErr(""); setMsg(`Đang xử lý ${file.name}`);
    const fd = new FormData(); fd.append("shop_id", shopId); fd.append("file", file);
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
          <Field label="Tiêu đề"><Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Chính sách đổi trả" /></Field>
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
  const load = () => { api.get(`/api/channels?shop_id=${shopId}`).then(setItems).catch((e) => setErr(e.message)); api.get("/api/channels/kinds").then(setKinds).catch(() => {}); };
  useEffect(() => { setItems(null); load(); }, [shopId]);
  const spec = kinds.find((k) => k.kind === kind);
  const connect = async () => {
    setBusy(true); setFerr("");
    try { await api.post("/api/channels", { shop_id: shopId, kind, name, greeting, credentials: creds }); setOpen(false); setCreds({}); setName(""); load(); }
    catch (e: any) { setFerr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Kết nối các kênh bán hàng để trợ lý AI trả lời khách trên mọi nơi."
          right={<Button size="sm" onClick={() => setOpen(true)}><Plug className="w-4 h-4" /> Kết nối kênh</Button>}>Kênh kết nối</CardTitle>
        <Msg type="err">{err}</Msg>
        {!items ? <Spinner /> : items.length === 0 ? <Empty>Chưa có kênh nào. Bấm Kết nối kênh để bắt đầu.</Empty> :
          <div className="space-y-3">
            {items.map((ch) => {
              const url = ch.public_key ? `${location.origin}/widget.html?key=${ch.public_key}` : "";
              return (
                <div key={ch.id} className="border border-line rounded-xl p-4">
                  <div className="flex items-center gap-2"><span className="font-semibold">{ch.name}</span><Badge kind={ch.status}>{channelStatus(ch.status)}</Badge></div>
                  {url && <div className="mt-3">
                    <Field label="Mã tích hợp website" info="Dán đoạn mã này vào website của bạn để hiển thị khung chat.">
                      <pre className="bg-bg border border-line rounded-lg p-3 text-xs overflow-x-auto font-normal">{`<iframe src="${url}" style="border:0;width:380px;height:560px"></iframe>`}</pre>
                    </Field>
                    <a className="text-accent text-sm font-semibold inline-flex items-center gap-1 mt-2" href={url} target="_blank">Xem thử tiện ích <ArrowUpRight className="w-3.5 h-3.5" /></a>
                  </div>}
                </div>
              );
            })}
          </div>}
      </Card>
      <Modal open={open} onClose={() => setOpen(false)} title="Kết nối kênh" size="md"
        footer={<><Button variant="sec" onClick={() => setOpen(false)}>Huỷ</Button><Button loading={busy} disabled={spec && !spec.allowed} onClick={connect}>Kết nối</Button></>}>
        <Field label="Loại kênh"><Select value={kind} onChange={(e) => { setKind(e.target.value); setCreds({}); }}>
          {kinds.map((k) => <option key={k.kind} value={k.kind} disabled={!k.allowed}>{k.label}{k.allowed ? "" : " — không có trong gói"}</option>)}
        </Select></Field>
        {spec?.note && <p className="text-[12px] text-muted mt-2 font-normal">{spec.note}</p>}
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
    </div>
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
