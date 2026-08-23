import React, { useEffect, useState } from "react";
import { api, clearAuth, loadAuth, saveAuth } from "./api";
import { Button, Card, Field, Input, Msg, Select } from "./ui";
import { Admin, Channels, Inbox, Knowledge, Plan, Products, Settings, Usage } from "./pages";
import {
  Bot, Package, BookOpen, Plug, MessagesSquare, BarChart3, CreditCard,
  Settings as Cog, Shield, LogOut, Store, Sparkles,
} from "lucide-react";

type User = { id: string; email: string; is_platform_admin: boolean };
type Org = { id: string; name: string; role: string };

const NAV: [string, any, string][] = [
  ["products", Package, "Sản phẩm"],
  ["knowledge", BookOpen, "Kiến thức"],
  ["channels", Plug, "Kênh & Widget"],
  ["inbox", MessagesSquare, "Hộp thư"],
  ["usage", BarChart3, "Sử dụng"],
  ["plan", CreditCard, "Gói"],
  ["settings", Cog, "Cài đặt AI"],
];

export default function App() {
  const [ready, setReady] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [orgs, setOrgs] = useState<Org[]>([]);
  const [org, setOrg] = useState<Org | null>(null);
  const [shops, setShops] = useState<any[]>([]);
  const [shop, setShop] = useState<any>(null);
  const [tab, setTab] = useState<string>(localStorage.getItem("omnishop.tab") || "products");

  const setActiveOrg = async (o: Org | null) => {
    setOrg(o); saveAuth({ token: loadAuth().token, orgId: o?.id });
    if (o) { try { const s = await api.get("/api/shops"); setShops(s); setShop(s[0] || null); } catch { setShops([]); setShop(null); } }
  };

  const boot = async () => {
    const a = loadAuth();
    if (!a.token) { setReady(true); return; }
    try {
      const me = await api.get("/api/auth/me");
      setUser(me.user); setOrgs(me.orgs);
      const chosen = me.orgs.find((o: Org) => o.id === a.orgId) || me.orgs[0] || null;
      await setActiveOrg(chosen);
    } catch { clearAuth(); }
    setReady(true);
  };
  useEffect(() => { boot(); }, []);
  useEffect(() => { localStorage.setItem("omnishop.tab", tab); }, [tab]);

  if (!ready) return <div className="h-screen flex items-center justify-center text-muted">…</div>;
  if (!user) return <Login onAuthed={boot} />;

  const isAdmin = user.is_platform_admin;
  const nav = isAdmin ? [...NAV, ["admin", Shield, "Quản trị nền tảng"] as [string, any, string]] : NAV;
  const needShop = !shop && !["plan", "settings", "admin"].includes(tab);

  return (
    <div className="min-h-screen grid grid-cols-1 md:grid-cols-[240px_1fr]">
      {/* sidebar */}
      <aside className="border-r border-line bg-panel/70 backdrop-blur px-3 py-4 flex md:flex-col gap-1 md:min-h-screen sticky top-0 z-20 overflow-x-auto">
        <div className="flex items-center gap-2 px-2 pb-4 font-extrabold text-lg shrink-0">
          <span className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent2 flex items-center justify-center"><Bot className="w-5 h-5 text-white" /></span>
          <span>OmniShop<span className="text-muted text-xs font-semibold"> AI</span></span>
        </div>
        {nav.map(([id, Icon, label]) => (
          <button key={id} onClick={() => setTab(id)}
            className={"flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-semibold transition shrink-0 " +
              (tab === id ? "bg-gradient-to-r from-accent/25 to-transparent text-fg border border-line" : "text-muted hover:bg-card hover:text-fg")}>
            <Icon className="w-[18px] h-[18px]" /> <span className="hidden md:inline">{label}</span>
          </button>
        ))}
      </aside>

      {/* main */}
      <div className="flex flex-col min-w-0">
        <header className="flex items-center gap-3 px-5 py-3 border-b border-line bg-panel/60 backdrop-blur sticky top-0 z-10">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted hidden sm:inline">Org</span>
            <Select className="w-auto min-w-[130px]" value={org?.id} onChange={(e) => setActiveOrg(orgs.find((o) => o.id === e.target.value) || null)}>
              {orgs.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
            </Select>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Store className="w-4 h-4 text-muted" />
            <Select className="w-auto min-w-[130px]" value={shop?.id || ""} onChange={(e) => setShop(shops.find((s) => s.id === e.target.value))}>
              {shops.length ? shops.map((s) => <option key={s.id} value={s.id}>{s.name}</option>) : <option>(chưa có shop)</option>}
            </Select>
          </div>
          <div className="flex-1" />
          <span className="text-muted text-[13px] hidden sm:inline">{user.email}</span>
          <Button variant="ghost" size="sm" onClick={() => { clearAuth(); location.reload(); }}><LogOut className="w-4 h-4" /></Button>
        </header>

        <main className="p-5 md:p-6 max-w-[980px] w-full">
          {needShop ? <ShopSetup onCreated={async () => { const s = await api.get("/api/shops"); setShops(s); setShop(s[0]); }} />
            : tab === "products" ? <Products shopId={shop.id} />
            : tab === "knowledge" ? <Knowledge shopId={shop.id} />
            : tab === "channels" ? <Channels shopId={shop.id} />
            : tab === "inbox" ? <Inbox shopId={shop.id} />
            : tab === "usage" ? <Usage />
            : tab === "plan" ? <Plan onChange={() => {}} />
            : tab === "settings" ? <Settings />
            : tab === "admin" ? <Admin />
            : null}
        </main>
      </div>
    </div>
  );
}

function ShopSetup({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState(""); const [err, setErr] = useState("");
  return (
    <Card className="max-w-md">
      <div className="text-[15px] font-bold mb-1">Tạo cửa hàng đầu tiên</div>
      <p className="text-[13px] text-muted mb-3">Bạn cần một cửa hàng để thêm sản phẩm, kiến thức và bật chatbot.</p>
      <Field label="Tên cửa hàng"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Boutique của tôi" /></Field>
      <div className="mt-3"><Button onClick={async () => { try { await api.post("/api/shops", { name: name || "Cửa hàng của tôi" }); onCreated(); } catch (e: any) { setErr(e.message); } }}>Tạo cửa hàng</Button></div>
      <Msg type="err">{err}</Msg>
    </Card>
  );
}

function Login({ onAuthed }: { onAuthed: () => void }) {
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [orgName, setOrgName] = useState("");
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  const go = async (path: string, body: any) => {
    setBusy(true); setErr("");
    try { const d = await api.post(path, body); saveAuth({ token: d.token, orgId: d.orgs?.[0]?.id }); onAuthed(); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      <div className="hidden lg:flex flex-col justify-between p-12 bg-gradient-to-br from-panel to-bg border-r border-line">
        <div className="flex items-center gap-2 font-extrabold text-xl"><span className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-accent2 flex items-center justify-center"><Bot className="w-5 h-5 text-white" /></span> OmniShop AI</div>
        <div>
          <h1 className="text-4xl font-extrabold leading-tight">Chatbot AI đa kênh<br />cho cửa hàng của bạn.</h1>
          <p className="text-muted mt-4 max-w-md">Kết nối kênh bán, nhập sản phẩm & kiến thức, AI trả lời khách tự động — có chuyển nhân viên khi cần. <span className="text-fg">Pay → Connect → AI works.</span></p>
          <div className="flex gap-4 mt-8 text-sm text-muted">
            <span className="flex items-center gap-2"><Sparkles className="w-4 h-4 text-accent" /> RAG product-aware</span>
            <span className="flex items-center gap-2"><Plug className="w-4 h-4 text-accent" /> Đa nhà cung cấp LLM</span>
          </div>
        </div>
        <div className="text-xs text-muted">© OmniShop AI</div>
      </div>
      <div className="flex items-center justify-center p-6">
        <Card className="w-full max-w-sm">
          <div className="text-lg font-extrabold mb-1 lg:hidden flex items-center gap-2"><Bot className="w-5 h-5 text-accent" /> OmniShop AI</div>
          <div className="text-[15px] font-bold mb-1">Chào mừng trở lại</div>
          <p className="text-[13px] text-muted mb-4">Đăng nhập, hoặc đăng ký workspace mới.</p>
          <Field label="Email"><Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" /></Field>
          <div className="mt-3"><Field label="Mật khẩu"><Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="≥ 8 ký tự" /></Field></div>
          <div className="mt-3"><Field label="Workspace (khi đăng ký)"><Input value={orgName} onChange={(e) => setOrgName(e.target.value)} placeholder="Cửa hàng của tôi" /></Field></div>
          <div className="flex gap-2 mt-4">
            <Button loading={busy} onClick={() => go("/api/auth/login", { email, password })}>Đăng nhập</Button>
            <Button variant="sec" onClick={() => go("/api/auth/signup", { email, password, org_name: orgName })}>Đăng ký</Button>
          </div>
          <Msg type="err">{err}</Msg>
          <p className="text-xs text-muted mt-3">Demo: demo@omnishop.local / demo12345</p>
        </Card>
      </div>
    </div>
  );
}
