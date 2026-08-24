import React, { useEffect, useState } from "react";
import { api, clearAuth, loadAuth, saveAuth } from "./api";
import { Button, Card, Field, Input, Msg, Select } from "./ui";
import { Admin, Billing, Bots, Channels, Help, Inbox, Knowledge, Members, Overview, Products, Settings } from "./pages";
import {
  Bot, LayoutDashboard, MessagesSquare, BookOpen, Plug, Package, Users,
  CreditCard, Settings as Cog, Shield, LifeBuoy, LogOut, Store, Sparkles,
} from "lucide-react";

type User = { id: string; email: string; is_platform_admin: boolean };
type Org = { id: string; name: string; role: string };

const GROUPS: { section: string | null; items: [string, any, string][] }[] = [
  { section: null, items: [["overview", LayoutDashboard, "Tổng quan"]] },
  { section: "Vận hành", items: [["inbox", MessagesSquare, "Hộp thư"]] },
  { section: "Trợ lý AI", items: [["bots", Bot, "Trợ lý"], ["knowledge", BookOpen, "Kiến thức"], ["channels", Plug, "Kênh kết nối"]] },
  { section: "Cửa hàng", items: [["products", Package, "Sản phẩm"]] },
  { section: "Quản lý", items: [["members", Users, "Thành viên"], ["billing", CreditCard, "Thanh toán"], ["settings", Cog, "Cài đặt"]] },
  { section: "Hỗ trợ", items: [["help", LifeBuoy, "Hướng dẫn"]] },
];
const NEEDS_SHOP = ["overview", "inbox", "bots", "products", "knowledge", "channels"];

export default function App() {
  const [ready, setReady] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [orgs, setOrgs] = useState<Org[]>([]);
  const [org, setOrg] = useState<Org | null>(null);
  const [shops, setShops] = useState<any[]>([]);
  const [shop, setShop] = useState<any>(null);
  const [tab, setTab] = useState<string>(localStorage.getItem("omnishop.tab") || "overview");

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
      await setActiveOrg(me.orgs.find((o: Org) => o.id === a.orgId) || me.orgs[0] || null);
    } catch { clearAuth(); }
    setReady(true);
  };
  useEffect(() => { boot(); }, []);
  useEffect(() => { localStorage.setItem("omnishop.tab", tab); }, [tab]);

  if (!ready) return <div className="h-screen flex items-center justify-center text-muted">…</div>;
  if (!user) return <Login onAuthed={boot} />;

  const groups = user.is_platform_admin
    ? [...GROUPS, { section: "Quản trị", items: [["admin", Shield, "Quản trị hệ thống"]] as [string, any, string][] }]
    : GROUPS;
  const needShop = !shop && NEEDS_SHOP.includes(tab);
  const role = org?.role || "viewer";

  return (
    <div className="min-h-screen grid grid-cols-1 md:grid-cols-[248px_1fr]">
      <aside className="border-r border-line bg-panel/70 backdrop-blur px-3 py-4 md:min-h-screen sticky top-0 z-20 overflow-y-auto hidden md:block">
        <div className="flex items-center gap-2 px-2 pb-4 font-extrabold text-lg">
          <span className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent2 flex items-center justify-center"><Bot className="w-5 h-5 text-white" /></span>
          <span>OmniShop<span className="text-muted text-xs font-semibold"> AI</span></span>
        </div>
        {groups.map((g, gi) => (
          <div key={gi} className="mb-1.5">
            {g.section && <div className="px-3 pt-3 pb-1 text-[10px] font-bold uppercase tracking-wider text-muted/70">{g.section}</div>}
            {g.items.map(([id, Icon, label]) => (
              <button key={id} onClick={() => setTab(id)}
                className={"w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-semibold transition " +
                  (tab === id ? "bg-gradient-to-r from-accent/25 to-transparent text-fg border border-line" : "text-muted hover:bg-card hover:text-fg")}>
                <Icon className="w-[18px] h-[18px]" /> {label}
              </button>
            ))}
          </div>
        ))}
      </aside>

      <div className="flex flex-col min-w-0">
        <header className="flex items-center gap-3 px-4 md:px-5 py-3 border-b border-line bg-panel/60 backdrop-blur sticky top-0 z-10">
          <span className="md:hidden font-extrabold flex items-center gap-1"><Bot className="w-5 h-5 text-accent" /></span>
          <Select className="w-auto min-w-[120px]" value={org?.id} onChange={(e) => setActiveOrg(orgs.find((o) => o.id === e.target.value) || null)}>
            {orgs.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
          </Select>
          <div className="flex items-center gap-2 text-sm">
            <Store className="w-4 h-4 text-muted hidden sm:block" />
            <Select className="w-auto min-w-[120px]" value={shop?.id || ""} onChange={(e) => setShop(shops.find((s) => s.id === e.target.value))}>
              {shops.length ? shops.map((s) => <option key={s.id} value={s.id}>{s.name}</option>) : <option>Chưa có cửa hàng</option>}
            </Select>
          </div>
          {/* mobile tab select */}
          <Select className="w-auto md:hidden" value={tab} onChange={(e) => setTab(e.target.value)}>
            {groups.flatMap((g) => g.items).map(([id, , label]) => <option key={id} value={id}>{label}</option>)}
          </Select>
          <div className="flex-1" />
          <span className="text-muted text-[13px] hidden lg:inline">{user.email}</span>
          <Button variant="ghost" size="sm" onClick={() => { clearAuth(); location.reload(); }}><LogOut className="w-4 h-4" /></Button>
        </header>

        <main className="p-4 md:p-6 w-full max-w-[1440px]">
          {needShop ? <ShopSetup onCreated={async () => { const s = await api.get("/api/shops"); setShops(s); setShop(s[0]); }} />
            : tab === "overview" ? <Overview shopId={shop.id} />
            : tab === "inbox" ? <Inbox shopId={shop.id} />
            : tab === "bots" ? <Bots shopId={shop.id} role={role} />
            : tab === "knowledge" ? <Knowledge shopId={shop.id} />
            : tab === "channels" ? <Channels shopId={shop.id} />
            : tab === "products" ? <Products shopId={shop.id} />
            : tab === "members" ? <Members role={role} />
            : tab === "billing" ? <Billing role={role} />
            : tab === "settings" ? <Settings />
            : tab === "help" ? <Help />
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
      <p className="text-[13px] text-muted mb-3 font-normal">Bạn cần một cửa hàng để tạo trợ lý AI, thêm sản phẩm và kết nối kênh.</p>
      <Field label="Tên cửa hàng"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Cửa hàng của tôi" /></Field>
      <div className="mt-3"><Button onClick={async () => { try { await api.post("/api/shops", { name: name || "Cửa hàng của tôi" }); onCreated(); } catch (e: any) { setErr(e.message); } }}>Tạo cửa hàng</Button></div>
      <Msg type="err">{err}</Msg>
    </Card>
  );
}

function Login({ onAuthed }: { onAuthed: () => void }) {
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [orgName, setOrgName] = useState("");
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false); const [mode, setMode] = useState<"login" | "signup">("login");
  const go = async () => {
    setBusy(true); setErr("");
    try {
      const path = mode === "login" ? "/api/auth/login" : "/api/auth/signup";
      const body = mode === "login" ? { email, password } : { email, password, org_name: orgName };
      const d = await api.post(path, body); saveAuth({ token: d.token, orgId: d.orgs?.[0]?.id }); onAuthed();
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      <div className="hidden lg:flex flex-col justify-between p-12 bg-gradient-to-br from-panel to-bg border-r border-line">
        <div className="flex items-center gap-2 font-extrabold text-xl"><span className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-accent2 flex items-center justify-center"><Bot className="w-5 h-5 text-white" /></span> OmniShop AI</div>
        <div>
          <h1 className="text-4xl font-extrabold leading-tight tracking-tight">Nhiều trợ lý AI,<br />mọi kênh bán hàng</h1>
          <p className="text-muted mt-4 max-w-md font-normal">Tạo trợ lý riêng cho từng cửa hàng và từng trang, kết nối vào Website, Facebook, Instagram và hơn thế. Trả lời khách tự động, chuyển nhân viên khi cần.</p>
          <div className="flex gap-4 mt-8 text-sm text-muted font-normal">
            <span className="flex items-center gap-2"><Sparkles className="w-4 h-4 text-accent" /> Prompt tuỳ chỉnh cho mỗi trợ lý</span>
            <span className="flex items-center gap-2"><Plug className="w-4 h-4 text-accent" /> Đa kênh, đa nhà cung cấp AI</span>
          </div>
        </div>
        <div className="text-xs text-muted font-normal">© OmniShop AI</div>
      </div>
      <div className="flex items-center justify-center p-6">
        <Card className="w-full max-w-sm">
          <div className="text-lg font-extrabold mb-1 lg:hidden flex items-center gap-2"><Bot className="w-5 h-5 text-accent" /> OmniShop AI</div>
          <div className="text-[15px] font-bold mb-1">{mode === "login" ? "Đăng nhập" : "Tạo tài khoản"}</div>
          <p className="text-[13px] text-muted mb-4 font-normal">{mode === "login" ? "Chào mừng trở lại." : "Bắt đầu với một workspace mới."}</p>
          <Field label="Email"><Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="ban@congty.com" onKeyDown={(e) => e.key === "Enter" && go()} /></Field>
          <div className="mt-3"><Field label="Mật khẩu"><Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Tối thiểu 8 ký tự" onKeyDown={(e) => e.key === "Enter" && go()} /></Field></div>
          {mode === "signup" && <div className="mt-3"><Field label="Tên workspace"><Input value={orgName} onChange={(e) => setOrgName(e.target.value)} placeholder="Cửa hàng của tôi" /></Field></div>}
          <div className="mt-4"><Button loading={busy} onClick={go} className="w-full">{mode === "login" ? "Đăng nhập" : "Đăng ký"}</Button></div>
          <Msg type="err">{err}</Msg>
          <button className="text-[13px] text-accent mt-3 font-semibold" onClick={() => { setMode(mode === "login" ? "signup" : "login"); setErr(""); }}>
            {mode === "login" ? "Chưa có tài khoản? Đăng ký" : "Đã có tài khoản? Đăng nhập"}
          </button>
          {mode === "login" && <p className="text-xs text-muted mt-3 font-normal">Tài khoản dùng thử: demo@omnishop.local / demo12345</p>}
        </Card>
      </div>
    </div>
  );
}
