import React, { useEffect, useState } from "react";
import { api } from "./api";
import { Badge, Button, Card, CardTitle, Empty, Field, Input, Kpi, Msg, Select, Spinner, Table, Td, Textarea } from "./ui";
import { RefreshCw, Upload, Plug, Send } from "lucide-react";

// ---------------------------------------------------------------- Products
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
        <CardTitle sub="AI dùng dữ liệu này để trả lời về giá, tồn kho, biến thể.">Thêm sản phẩm</CardTitle>
        <div className="grid md:grid-cols-2 gap-3">
          <Field label="Tên"><Input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="Áo thun cotton" /></Field>
          <Field label="Giá (VND)"><Input type="number" value={f.price} onChange={(e) => setF({ ...f, price: e.target.value })} /></Field>
        </div>
        <div className="mt-3"><Field label="Mô tả"><Textarea value={f.description} onChange={(e) => setF({ ...f, description: e.target.value })} /></Field></div>
        <div className="mt-3"><Field label="Biến thể / tồn kho" hint="ví dụ: Size M:10, Size L:3"><Input value={f.variants} onChange={(e) => setF({ ...f, variants: e.target.value })} placeholder="Size M:10, Size L:3" /></Field></div>
        <div className="mt-4"><Button onClick={add}>Lưu sản phẩm</Button></div>
        <Msg type="err">{err}</Msg>
      </Card>
      <Card>
        <CardTitle>Danh sách sản phẩm</CardTitle>
        {!items ? <Spinner /> : items.length === 0 ? <Empty>Chưa có sản phẩm.</Empty> :
          <Table head={["Tên", "Giá", "Biến thể / tồn kho"]}>
            {items.map((p) => (
              <tr key={p.id}>
                <Td><div className="font-semibold">{p.name}</div><div className="text-xs text-muted">{(p.description || "").slice(0, 60)}</div></Td>
                <Td>{p.price != null ? `${p.price.toLocaleString()} ${p.currency}` : "—"}</Td>
                <Td><div className="flex flex-wrap gap-1">{(p.variants || []).map((v: any, i: number) => <span key={i} className="text-xs bg-card2 border border-line rounded px-2 py-0.5">{v.name} · {v.stock}</span>)}</div></Td>
              </tr>
            ))}
          </Table>}
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------- Knowledge
export function Knowledge({ shopId }: { shopId: string }) {
  const [docs, setDocs] = useState<any[] | null>(null);
  const [title, setTitle] = useState(""); const [text, setText] = useState("");
  const [msg, setMsg] = useState(""); const [err, setErr] = useState("");
  const fileRef = React.useRef<HTMLInputElement>(null);
  const load = () => api.get(`/api/knowledge/documents?shop_id=${shopId}`).then((d) => { setDocs(d); if (d.some((x: any) => x.status !== "ready" && x.status !== "error")) setTimeout(load, 2000); }).catch((e) => setErr(e.message));
  useEffect(() => { setDocs(null); load(); }, [shopId]);
  const addText = async () => { setErr(""); setMsg(""); try { await api.post("/api/knowledge/documents", { shop_id: shopId, title, text }); setTitle(""); setText(""); setMsg("Đã thêm, đang tạo embedding…"); load(); } catch (e: any) { setErr(e.message); } };
  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return; setErr(""); setMsg(`Đang xử lý ${file.name}…`);
    const fd = new FormData(); fd.append("shop_id", shopId); fd.append("file", file);
    try { const r = await api.upload("/api/knowledge/upload", fd); setMsg(`Đã tải ${file.name} (${r.extracted_chars} ký tự, ${r.chunks} đoạn).`); load(); }
    catch (ex: any) { setErr(ex.message); setMsg(""); } finally { if (fileRef.current) fileRef.current.value = ""; }
  };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="Nhập text hoặc tải file. Nội dung được cắt đoạn và tạo embedding (chạy nền).">Tải tài liệu</CardTitle>
        <div onClick={() => fileRef.current?.click()} className="border border-dashed border-line rounded-xl p-6 text-center text-muted cursor-pointer hover:border-accent hover:text-fg transition flex flex-col items-center gap-2">
          <Upload className="w-6 h-6" />
          <div className="text-sm">Bấm để tải file — PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, ảnh…</div>
          <div className="text-xs">Ảnh & PDF scan sẽ chạy OCR tự động</div>
        </div>
        <input ref={fileRef} type="file" className="hidden" onChange={onFile} />
        <div className="mt-4 grid gap-3">
          <Field label="Hoặc nhập tiêu đề"><Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Chính sách đổi trả" /></Field>
          <Field label="Nội dung"><Textarea value={text} onChange={(e) => setText(e.target.value)} className="min-h-[110px]" /></Field>
        </div>
        <div className="mt-3"><Button variant="sec" onClick={addText}>Thêm text</Button></div>
        <Msg type="ok">{msg}</Msg><Msg type="err">{err}</Msg>
      </Card>
      <Card>
        <CardTitle>Tài liệu</CardTitle>
        {!docs ? <Spinner /> : docs.length === 0 ? <Empty>Chưa có tài liệu.</Empty> :
          <Table head={["Tiêu đề", "Nguồn", "Trạng thái", "Đoạn"]}>
            {docs.map((d) => <tr key={d.id}><Td>{d.title}</Td><Td className="text-muted">{d.source || "text"}</Td><Td><Badge kind={d.status}>{d.status}</Badge></Td><Td>{d.chunks}</Td></tr>)}
          </Table>}
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------- Channels
export function Channels({ shopId }: { shopId: string }) {
  const [items, setItems] = useState<any[] | null>(null); const [err, setErr] = useState("");
  const load = () => api.get(`/api/channels?shop_id=${shopId}`).then(setItems).catch((e) => setErr(e.message));
  useEffect(() => { setItems(null); load(); }, [shopId]);
  const create = async () => { setErr(""); try { await api.post("/api/channels", { shop_id: shopId, kind: "website", name: "Website widget" }); load(); } catch (e: any) { setErr(e.message); } };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub="MVP hỗ trợ Website widget (chạy ngay). Facebook/Instagram/TikTok/Shopee thêm sau khi qua app-review.">Kênh bán hàng</CardTitle>
        <Button onClick={create}><Plug className="w-4 h-4" /> Tạo Website widget</Button>
        <Msg type="err">{err}</Msg>
      </Card>
      {!items ? <Spinner /> : items.map((ch) => {
        const url = `${location.origin}/widget.html?key=${ch.public_key}`;
        return (
          <Card key={ch.id}>
            <div className="flex items-center gap-2 mb-2"><span className="font-semibold">{ch.name}</span><Badge kind={ch.status}>{ch.status}</Badge></div>
            <div className="text-xs text-muted mb-1">Public key</div>
            <code className="text-xs bg-bg px-2 py-1 rounded border border-line">{ch.public_key}</code>
            <div className="mt-3"><a className="text-accent text-sm" href={url} target="_blank">➜ Mở thử widget</a></div>
            <div className="mt-2"><Field label="Mã nhúng"><pre className="bg-bg border border-line rounded-lg p-3 text-xs overflow-x-auto">{`<iframe src="${url}" style="border:0;width:380px;height:560px"></iframe>`}</pre></Field></div>
          </Card>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- Inbox
export function Inbox({ shopId }: { shopId: string }) {
  const [convs, setConvs] = useState<any[] | null>(null);
  const [active, setActive] = useState<any>(null); const [msgs, setMsgs] = useState<any[]>([]); const [reply, setReply] = useState("");
  const loadList = () => api.get(`/api/conversations?shop_id=${shopId}`).then(setConvs);
  useEffect(() => { setConvs(null); setActive(null); loadList(); }, [shopId]);
  const open = async (c: any) => { setActive(c); setMsgs(await api.get(`/api/conversations/${c.id}/messages`)); };
  const send = async () => { if (!reply.trim()) return; await api.post(`/api/conversations/${active.id}/reply`, { text: reply }); setReply(""); open(active); };
  return (
    <Card>
      <CardTitle>Hộp thư hợp nhất</CardTitle>
      <div className="flex gap-4 items-start flex-col md:flex-row">
        <div className="flex-1 min-w-[220px] w-full space-y-2">
          {!convs ? <Spinner /> : convs.length === 0 ? <Empty>Chưa có hội thoại. Thử nhắn qua widget!</Empty> :
            convs.map((c) => (
              <div key={c.id} onClick={() => open(c)} className={"cursor-pointer rounded-xl border p-3 transition " + (active?.id === c.id ? "border-accent bg-card2" : "border-line hover:bg-card2")}>
                <div className="flex items-center gap-2"><span className="font-semibold text-sm">{c.customer_ref}</span><Badge kind={c.status}>{c.status}</Badge></div>
                <div className="text-xs text-muted mt-1 line-clamp-1">{(c.last_message || "").slice(0, 70)}</div>
              </div>
            ))}
        </div>
        <div className="flex-[1.4] min-w-[260px] w-full">
          {!active ? <Empty>Chọn một hội thoại.</Empty> : (
            <div>
              <div className="flex items-center gap-2 mb-3"><span className="font-semibold text-sm">Khách: {active.customer_ref}</span><Badge kind={active.status}>{active.status}</Badge></div>
              <div className="max-h-[52vh] overflow-auto pr-1 space-y-2">
                {msgs.map((m, i) => (
                  <div key={i} className={"px-3 py-2 rounded-xl text-sm max-w-[85%] whitespace-pre-wrap " +
                    (m.role === "customer" ? "bg-bg border border-line ml-auto" : m.role === "ai" ? "bg-indigo-900/40" : m.role === "agent" ? "bg-emerald-900/40" : "bg-amber-900/30 text-xs")}>{m.content}</div>
                ))}
              </div>
              <div className="flex gap-2 mt-3">
                <Input value={reply} onChange={(e) => setReply(e.target.value)} placeholder="Trả lời với tư cách nhân viên…" onKeyDown={(e) => e.key === "Enter" && send()} />
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

// ---------------------------------------------------------------- Usage
export function Usage() {
  const [s, setS] = useState<any>(null); const [sub, setSub] = useState<any>(null); const [err, setErr] = useState("");
  useEffect(() => { Promise.all([api.get("/api/usage/summary"), api.get("/api/subscription")]).then(([a, b]) => { setS(a); setSub(b); }).catch((e) => setErr(e.message)); }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!s || !sub) return <Spinner />;
  return (
    <Card>
      <CardTitle sub="Chi phí AI được đo cho từng tổ chức (COGS) để tính biên lợi nhuận.">Sử dụng tháng này</CardTitle>
      <div className="grid gap-3 grid-cols-2 md:grid-cols-3">
        <Kpi n={`${s.ai_messages} / ${sub.quota.limit}`} l="Tin nhắn AI" />
        <Kpi n={s.input_tokens.toLocaleString()} l="Input tokens" />
        <Kpi n={s.output_tokens.toLocaleString()} l="Output tokens" />
        <Kpi n={s.embedding_tokens.toLocaleString()} l="Embedding tokens" />
        <Kpi n={`$${s.total_cost.toFixed(4)}`} l="Chi phí ước tính (COGS)" />
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------- Plan
export function Plan({ onChange }: { onChange: () => void }) {
  const [plans, setPlans] = useState<any[] | null>(null); const [sub, setSub] = useState<any>(null); const [err, setErr] = useState("");
  const load = () => Promise.all([api.get("/api/plans"), api.get("/api/subscription")]).then(([p, s]) => { setPlans(p); setSub(s); }).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!plans || !sub) return <Spinner />;
  const cur = sub.entitlements._plan;
  return (
    <div className="space-y-4">
      <Card><CardTitle sub={`Quota AI/tháng: ${sub.quota.used} / ${sub.quota.limit}`}>Gói hiện tại: {sub.entitlements._plan_name}</CardTitle></Card>
      <div className="grid md:grid-cols-3 gap-4">
        {plans.map((p) => (
          <Card key={p.code} className={cur === p.code ? "border-accent shadow-glow" : ""}>
            <div className="flex items-center justify-between"><span className="font-bold">{p.name}</span><span className="text-muted text-sm">${p.price_month}/th</span></div>
            <div className="text-[13px] text-muted mt-2 space-y-1">
              <div>AI/tháng: <b className="text-fg">{p.entitlements.ai_messages_month.toLocaleString()}</b></div>
              <div>Shops: {p.entitlements.shops} · Kênh: {(p.entitlements.channels_allowed || []).join(", ")}</div>
            </div>
            <div className="mt-4">{cur === p.code ? <Badge kind="active">đang dùng</Badge> :
              <Button variant="sec" onClick={async () => { await api.post("/api/subscription", { plan_code: p.code }); await load(); onChange(); }}>Chọn gói</Button>}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- LLM form (reused)
export function LlmForm({ initial, providers, endpoints, showModels = true }: {
  initial: any; providers: { id: string; label: string; base_url?: string }[];
  endpoints: { save: string; test?: string; models?: string; del?: string };
  showModels?: boolean;
}) {
  const findIdx = () => {
    if (!initial) return 0;
    const i = providers.findIndex((p) => p.id === initial.provider && (!p.base_url || p.base_url === initial.base_url));
    return i >= 0 ? i : Math.max(0, providers.findIndex((p) => p.id === initial.provider));
  };
  const [idx, setIdx] = useState(findIdx());
  const [model, setModel] = useState(initial?.model || "");
  const [baseUrl, setBaseUrl] = useState(initial?.base_url || "");
  const [apiKey, setApiKey] = useState("");
  const [maxTokens, setMaxTokens] = useState(initial?.extra?.max_tokens ? String(initial.extra.max_tokens) : "");
  const [models, setModels] = useState<string[]>([]);
  const [busy, setBusy] = useState(""); const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  const body = () => ({ provider: providers[idx].id, model, base_url: baseUrl, api_key: apiKey || null, max_tokens: maxTokens ? parseInt(maxTokens) : null });
  const onProvider = (v: number) => { setIdx(v); const p = providers[v]; if (p.base_url && !baseUrl) setBaseUrl(p.base_url); setModels([]); };
  const loadModels = async () => { if (!endpoints.models) return; setBusy("models"); setErr(""); try { const r = await api.post(endpoints.models, body()); setModels(r.models || []); if (!r.ok && r.error) setErr("Model list: " + r.error); } catch (e: any) { setErr(e.message); } finally { setBusy(""); } };
  const test = async () => { if (!endpoints.test) return; setBusy("test"); setOk(""); setErr(""); try { const r = await api.post(endpoints.test, body()); if (r.ok) setOk("✅ OK " + (r.model ? `(${r.model})` : r.dim ? `(dim ${r.dim})` : "") + (r.reply ? ": " + r.reply : "")); else setErr("❌ " + r.error); } catch (e: any) { setErr(e.message); } finally { setBusy(""); } };
  const save = async () => { setBusy("save"); setOk(""); setErr(""); try { await api.put(endpoints.save, body()); setOk("Đã lưu."); setApiKey(""); } catch (e: any) { setErr(e.message); } finally { setBusy(""); } };
  const del = async () => { if (!endpoints.del) return; try { await api.del(endpoints.del); setOk("Đã xoá cấu hình riêng, dùng mặc định."); } catch (e: any) { setErr(e.message); } };
  return (
    <div>
      <Field label="Nhà cung cấp"><Select value={idx} onChange={(e) => onProvider(parseInt(e.target.value))}>{providers.map((p, i) => <option key={i} value={i}>{p.label}</option>)}</Select></Field>
      <div className="grid md:grid-cols-2 gap-3 mt-3">
        <Field label="Model">
          <div className="flex gap-2">
            <Input value={model} onChange={(e) => setModel(e.target.value)} placeholder="chọn hoặc nhập model" list="modellist" />
            {showModels && endpoints.models && <Button variant="sec" size="sm" loading={busy === "models"} onClick={loadModels}><RefreshCw className="w-3.5 h-3.5" /></Button>}
          </div>
          {models.length > 0 && <datalist id="modellist">{models.map((m) => <option key={m} value={m} />)}</datalist>}
          {models.length > 0 && <select className="mt-2 w-full bg-bg border border-line rounded-lg px-3 py-2 text-sm" onChange={(e) => setModel(e.target.value)} value=""><option value="">— chọn từ {models.length} model —</option>{models.map((m) => <option key={m} value={m}>{m}</option>)}</select>}
        </Field>
        <Field label="Base URL" hint="để trống nếu dùng mặc định"><Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} /></Field>
      </div>
      <div className="grid md:grid-cols-2 gap-3 mt-3">
        <Field label="API key" hint="để trống = giữ nguyên"><Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="••••••••" /></Field>
        <Field label="Max tokens"><Input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} placeholder="1024" /></Field>
      </div>
      <div className="flex gap-2 mt-4 flex-wrap">
        {endpoints.test && <Button loading={busy === "test"} onClick={test}><Plug className="w-4 h-4" /> Test</Button>}
        <Button variant="sec" loading={busy === "save"} onClick={save}>Lưu</Button>
        {endpoints.del && <Button variant="danger" onClick={del}>Xoá override</Button>}
      </div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </div>
  );
}

// ---------------------------------------------------------------- Settings (tenant)
export function Settings() {
  const [llm, setLlm] = useState<any>(null); const [ocr, setOcr] = useState<any>(null); const [err, setErr] = useState("");
  useEffect(() => { api.get("/api/settings/llm").then(setLlm).catch((e) => setErr(e.message)); api.get("/api/settings/ocr").then(setOcr).catch(() => {}); }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!llm) return <Spinner />;
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle sub={llm.can_edit ? `Đang dùng: ${llm.effective.provider} · ${llm.effective.model || ""}` : "Quản trị nền tảng đã khoá — đang dùng mặc định nền tảng."}>Nhà cung cấp LLM</CardTitle>
        {llm.can_edit
          ? <LlmForm initial={llm.org_config} providers={llm.providers}
              endpoints={{ save: "/api/settings/llm", test: "/api/settings/llm/test", models: "/api/settings/llm/models", del: llm.org_config ? "/api/settings/llm" : undefined }} />
          : <Empty>Bạn không có quyền chỉnh LLM. Liên hệ quản trị nền tảng.</Empty>}
      </Card>
      {ocr && <OcrCard ocr={ocr} />}
    </div>
  );
}
function OcrCard({ ocr }: { ocr: any }) {
  const [idx, setIdx] = useState(Math.max(0, ocr.providers.findIndex((p: any) => p.id === (ocr.org_config?.provider || ocr.effective.provider))));
  const [model, setModel] = useState(ocr.org_config?.model || ""); const [lang, setLang] = useState("");
  const [ok, setOk] = useState(""); const [err, setErr] = useState("");
  const save = async () => { setOk(""); setErr(""); try { await api.put("/api/settings/ocr", { provider: ocr.providers[idx].id, model, lang }); setOk("Đã lưu."); } catch (e: any) { setErr(e.message); } };
  return (
    <Card>
      <CardTitle sub={`Đang dùng: ${ocr.effective.provider}. Đọc ảnh & PDF scan; thay Tesseract ↔ VLM.`}>OCR</CardTitle>
      <Field label="Backend OCR"><Select value={idx} onChange={(e) => setIdx(parseInt(e.target.value))}>{ocr.providers.map((p: any, i: number) => <option key={i} value={i}>{p.label}</option>)}</Select></Field>
      <div className="grid md:grid-cols-2 gap-3 mt-3">
        <Field label="Model (VLM)" hint="để trống = dùng LLM hiện tại"><Input value={model} onChange={(e) => setModel(e.target.value)} /></Field>
        <Field label="Ngôn ngữ (Tesseract)" hint="vd: vie+eng"><Input value={lang} onChange={(e) => setLang(e.target.value)} /></Field>
      </div>
      <div className="mt-4"><Button variant="sec" onClick={save}>Lưu OCR</Button></div>
      <Msg type="ok">{ok}</Msg><Msg type="err">{err}</Msg>
    </Card>
  );
}

// ---------------------------------------------------------------- Admin (platform)
export function Admin() {
  const [ov, setOv] = useState<any>(null); const [s, setS] = useState<any>(null); const [tenants, setTenants] = useState<any[]>([]);
  const [pol, setPol] = useState<any>({}); const [polMsg, setPolMsg] = useState(""); const [err, setErr] = useState("");
  useEffect(() => {
    api.get("/api/admin/overview").then(setOv).catch((e) => setErr(e.message));
    api.get("/api/admin/settings").then((d) => { setS(d); setPol(d.policy); }).catch((e) => setErr(e.message));
    api.get("/api/admin/tenants").then(setTenants).catch(() => {});
  }, []);
  if (err) return <Msg type="err">{err}</Msg>;
  if (!ov || !s) return <Spinner />;
  const embProviders = [{ id: "local", label: "Local (không cần key)" }, { id: "openai_compatible", label: "OpenAI-compatible", base_url: "https://api.openai.com/v1" }, { id: "gemini", label: "Gemini" }];
  const savePolicy = async () => { await api.put("/api/admin/settings/policy", pol); setPolMsg("Đã lưu chính sách."); };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>Tổng quan nền tảng</CardTitle>
        <div className="grid gap-3 grid-cols-2 md:grid-cols-5">
          <Kpi n={ov.tenants} l="Tenants" /><Kpi n={ov.shops} l="Shops" /><Kpi n={ov.conversations} l="Hội thoại" />
          <Kpi n={ov.ai_messages_month} l="Tin AI/tháng" /><Kpi n={`$${ov.cost_month.toFixed(4)}`} l="Chi phí/tháng" />
        </div>
      </Card>
      <Card>
        <CardTitle sub="Cho phép tenant tự chọn provider hay không.">Chính sách</CardTitle>
        <label className="flex items-center gap-2 text-sm mb-2"><input type="checkbox" checked={!!pol.allow_tenant_llm} onChange={(e) => setPol({ ...pol, allow_tenant_llm: e.target.checked })} /> Cho phép tenant tự cấu hình LLM</label>
        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={!!pol.allow_tenant_ocr} onChange={(e) => setPol({ ...pol, allow_tenant_ocr: e.target.checked })} /> Cho phép tenant tự cấu hình OCR</label>
        <div className="mt-3"><Button variant="sec" onClick={savePolicy}>Lưu chính sách</Button></div><Msg type="ok">{polMsg}</Msg>
      </Card>
      <Card>
        <CardTitle sub="Áp dụng khi tenant không cấu hình riêng.">LLM mặc định nền tảng</CardTitle>
        <LlmForm initial={s.llm} providers={s.llm_providers} endpoints={{ save: "/api/admin/settings/llm", test: "/api/admin/settings/llm/test", models: "/api/admin/settings/llm/models" }} />
      </Card>
      <Card>
        <CardTitle sub="Dùng chung toàn nền tảng (đổi model cần re-embed). Dim cố định 384.">Embedding nền tảng</CardTitle>
        <LlmForm initial={s.embedding} providers={embProviders} endpoints={{ save: "/api/admin/settings/embedding", test: "/api/admin/settings/embedding/test", models: "/api/admin/settings/embedding/models" }} />
      </Card>
      <Card>
        <CardTitle>Tenants</CardTitle>
        {tenants.length === 0 ? <Empty>Chưa có tenant.</Empty> :
          <Table head={["Tổ chức", "Gói", "Shops", "Tin AI/tháng", "Chi phí"]}>
            {tenants.map((t) => <tr key={t.id}><Td>{t.name}</Td><Td><Badge kind="active">{t.plan}</Badge></Td><Td>{t.shops}</Td><Td>{t.ai_messages}</Td><Td>${t.cost_month.toFixed(4)}</Td></tr>)}
          </Table>}
      </Card>
    </div>
  );
}
