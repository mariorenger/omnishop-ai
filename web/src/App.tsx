import React, { useEffect, useState } from "react";
import { api, clearAuth, loadAuth, saveAuth } from "./api";
import { Button, Card, ConfirmHost, Field, Input, Msg, Select, Toaster } from "./ui";
import { Admin, Billing, Bots, Channels, Help, Inbox, Knowledge, Members, Overview, Products, Settings } from "./pages";
import {
  Bot, LayoutDashboard, MessagesSquare, BookOpen, Plug, Package, Users,
  CreditCard, Settings as Cog, Shield, LifeBuoy, LogOut, Store, Sparkles, AlertTriangle,
} from "lucide-react";

type User = { id: string; email: string; is_platform_admin: boolean; platform_role?: string | null };
type Org = { id: string; name: string; role: string };

const GROUPS: { section: string | null; items: [string, any, string][] }[] = [
  { section: null, items: [["overview", LayoutDashboard, "Tổng quan"]] },
  { section: "Trợ lý AI", items: [["bots", Bot, "Trợ lý"], ["knowledge", BookOpen, "Kiến thức"], ["products", Package, "Sản phẩm"], ["channels", Plug, "Kênh kết nối"]] },
  { section: "Vận hành", items: [["inbox", MessagesSquare, "Hộp thư"]] },
  { section: "Quản lý", items: [["members", Users, "Thành viên"], ["billing", CreditCard, "Gói & Thanh toán"], ["settings", Cog, "Cài đặt"]] },
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
  const [renewal, setRenewal] = useState<any>(null);
  const [brand, setBrand] = useState<{ app_name: string; logo_url: string | null }>({ app_name: "OmniShop AI", logo_url: null });
  useEffect(() => { api.get("/api/branding").then(setBrand).catch(() => {}); }, []);
  useEffect(() => {
    document.title = brand.app_name;
    // apply the admin-uploaded logo as the browser-tab favicon (was the default globe)
    if (brand.logo_url) {
      let link = document.getElementById("favicon") as HTMLLinkElement | null;
      if (!link) { link = document.createElement("link"); link.id = "favicon"; link.rel = "icon"; document.head.appendChild(link); }
      link.href = brand.logo_url;
    }
  }, [brand]);

  useEffect(() => { if (org) api.get("/api/subscription").then((s) => setRenewal(s.renewal)).catch(() => setRenewal(null)); else setRenewal(null); }, [org]);

  const setActiveOrg = async (o: Org | null) => {
    setOrg(o); saveAuth({ token: loadAuth().token, orgId: o?.id });
    if (o) { try { const s = await api.get("/api/shops"); setShops(s); setShop(s[0] || null); } catch { setShops([]); setShop(null); } }
  };
  const boot = async () => {
    // capture a token handed back by an OAuth redirect (?token=...), then clean the URL
    const params = new URLSearchParams(location.search);
    const oauthToken = params.get("token");
    if (oauthToken) { saveAuth({ token: oauthToken }); history.replaceState({}, "", location.pathname); }
    const a = loadAuth();
    if (!a.token) { setReady(true); return; }
    try {
      const me = await api.get("/api/auth/me");
      setUser(me.user); setOrgs(me.orgs);
      await setActiveOrg(me.orgs.find((o: Org) => o.id === a.orgId) || me.orgs[0] || null);
      // a platform admin/manager with no workspace lands on the control panel, not a broken shop screen
      if (!me.orgs.length && (me.user.platform_role === "admin" || me.user.platform_role === "manager")) setTab("admin");
    } catch { clearAuth(); }
    setReady(true);
  };
  useEffect(() => { boot(); }, []);
  useEffect(() => { localStorage.setItem("omnishop.tab", tab); }, [tab]);

  if (!ready) return <div className="h-screen flex items-center justify-center text-muted">…</div>;
  if (!user) return <Login onAuthed={boot} brand={brand} />;

  const isStaff = user.platform_role === "admin" || user.platform_role === "manager";
  const groups = isStaff
    ? [...GROUPS, { section: "Quản trị", items: [["admin", Shield, user.platform_role === "manager" ? "Báo cáo hệ thống" : "Quản trị hệ thống"]] as [string, any, string][] }]
    : GROUPS;
  const TENANT_TABS = [...NEEDS_SHOP, "members", "billing", "settings"];
  const needOrg = !org && TENANT_TABS.includes(tab);   // no workspace yet
  const needShop = !!org && !shop && NEEDS_SHOP.includes(tab);
  const role = org?.role || "viewer";

  return (
    <div className="min-h-screen grid grid-cols-1 md:grid-cols-[248px_1fr]">
      <Toaster /><ConfirmHost />
      <aside className="border-r border-line bg-panel/70 backdrop-blur px-3 py-4 md:min-h-screen sticky top-0 z-20 overflow-y-auto hidden md:block">
        <div className="flex items-center gap-2 px-2 pb-4 font-extrabold text-lg">
          <span className="w-8 h-8 rounded-lg bg-pastel flex items-center justify-center shadow-[0_4px_16px_rgba(129,140,248,.4)] overflow-hidden">
            {brand.logo_url ? <img src={brand.logo_url} className="w-8 h-8 object-contain" /> : <Bot className="w-5 h-5 text-[#0b0e1a]" />}</span>
          <span>{brand.app_name}</span>
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

        <main className="p-4 md:p-6 w-full max-w-[1680px] mx-auto">
          {renewal?.expires && (renewal.expiring || renewal.expired) && tab !== "billing" && (
            <div className={"mb-4 flex items-center gap-3 rounded-xl border px-4 py-3 text-[13px] font-normal " + (renewal.expired ? "border-bad/50 bg-bad/10 text-bad" : "border-warn/50 bg-warn/10 text-warn")}>
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span className="flex-1">{renewal.expired
                ? "Gói dịch vụ của bạn đã hết hạn. Vui lòng gia hạn để tiếp tục sử dụng đầy đủ tính năng."
                : `Gói dịch vụ sắp hết hạn (còn ${renewal.days_left} ngày). Hãy gia hạn để không bị gián đoạn.`}</span>
              <Button size="sm" onClick={() => setTab("billing")}>Gia hạn ngay</Button>
            </div>
          )}
          {needOrg ? <WorkspaceSetup onCreated={boot} isAdmin={isStaff} />
            : needShop ? <ShopSetup onCreated={async () => { const s = await api.get("/api/shops"); setShops(s); setShop(s[0]); }} />
            : tab === "overview" ? <Overview shopId={shop.id} onGoInbox={() => setTab("inbox")} />
            : tab === "inbox" ? <Inbox shopId={shop.id} me={user} role={role} />
            : tab === "bots" ? <Bots shopId={shop.id} role={role} />
            : tab === "knowledge" ? <Knowledge shopId={shop.id} role={role} />
            : tab === "channels" ? <Channels shopId={shop.id} />
            : tab === "products" ? <Products shopId={shop.id} role={role} />
            : tab === "members" ? <Members role={role} />
            : tab === "billing" ? <Billing role={role} />
            : tab === "settings" ? <Settings />
            : tab === "help" ? <Help />
            : tab === "admin" ? <Admin role={user.platform_role || ""} />
            : null}
        </main>
      </div>
    </div>
  );
}

function WorkspaceSetup({ onCreated, isAdmin }: { onCreated: () => void; isAdmin: boolean }) {
  const [name, setName] = useState(""); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  const create = async () => {
    setBusy(true); setErr("");
    try {
      const o = await api.post("/api/orgs", { name: name || "Cửa hàng của tôi" });
      saveAuth({ token: loadAuth().token, orgId: o.id });
      onCreated();
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <Card className="max-w-md">
      <div className="text-[15px] font-bold mb-1">Tạo workspace</div>
      <p className="text-[13px] text-muted mb-3 font-normal">
        {isAdmin
          ? "Tài khoản quản trị hệ thống chưa có workspace bán hàng. Tạo một workspace nếu bạn cũng muốn dùng như người bán, hoặc vào “Quản trị hệ thống” để quản lý nền tảng."
          : "Bạn cần một workspace để tạo cửa hàng, trợ lý AI và kết nối kênh."}
      </p>
      <Field label="Tên workspace"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Cửa hàng của tôi" onKeyDown={(e) => e.key === "Enter" && create()} /></Field>
      <div className="mt-3"><Button loading={busy} onClick={create}>Tạo workspace</Button></div>
      <Msg type="err">{err}</Msg>
    </Card>
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

function Login({ onAuthed, brand }: { onAuthed: () => void; brand?: { app_name: string; logo_url: string | null } }) {
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [confirm, setConfirm] = useState(""); const [orgName, setOrgName] = useState("");
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false); const [mode, setMode] = useState<"login" | "signup">("login");
  const [google, setGoogle] = useState(false);
  const appName = brand?.app_name || "OmniShop AI";
  const Mark = ({ cls }: { cls: string }) => brand?.logo_url ? <img src={brand.logo_url} className={cls + " object-contain"} /> : <Bot className="w-5 h-5 text-[#0b0e1a]" />;
  useEffect(() => { api.get("/api/auth/google/config").then((d) => setGoogle(!!d.enabled)).catch(() => {}); }, []);
  const googleLogin = async () => { try { const d = await api.get("/api/auth/google/start"); if (d.url) location.href = d.url; else setErr(d.error || "Google chưa cấu hình"); } catch (e: any) { setErr(e.message); } };
  const go = async () => {
    setErr("");
    if (mode === "signup") {
      if (password.length < 8) { setErr("Mật khẩu tối thiểu 8 ký tự."); return; }
      if (password !== confirm) { setErr("Mật khẩu nhập lại không khớp."); return; }
    }
    setBusy(true);
    try {
      const path = mode === "login" ? "/api/auth/login" : "/api/auth/signup";
      const body = mode === "login" ? { email, password } : { email, password, org_name: orgName };
      const d = await api.post(path, body); saveAuth({ token: d.token, orgId: d.orgs?.[0]?.id }); onAuthed();
    } catch (e: any) {
      const m = String(e.message || "");
      setErr(mode === "signup" && /already registered|exist/i.test(m)
        ? "Email này đã có tài khoản. Hãy đăng nhập, hoặc dùng “Tiếp tục với Google” nếu bạn đăng ký bằng Gmail đó."
        : m);
    } finally { setBusy(false); }
  };
  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      <div className="hidden lg:flex flex-col justify-between p-12 border-r border-line relative overflow-hidden">
        <div className="absolute -top-24 -left-16 w-96 h-96 rounded-full bg-pastel-soft blur-3xl" />
        <div className="absolute bottom-0 right-0 w-80 h-80 rounded-full bg-pastel-soft blur-3xl opacity-70" />
        <div className="relative flex items-center gap-2 font-extrabold text-xl"><span className="w-9 h-9 rounded-xl bg-pastel flex items-center justify-center shadow-[0_6px_20px_rgba(129,140,248,.45)] overflow-hidden"><Mark cls="w-9 h-9" /></span> {appName}</div>
        <div className="relative">
          <h1 className="text-[2.7rem] font-extrabold leading-[1.08] tracking-tight">Nhiều trợ lý AI,<br /><span className="text-gradient">mọi kênh bán hàng</span></h1>
          <p className="text-muted mt-4 max-w-md font-normal">Tạo trợ lý riêng cho từng cửa hàng và từng trang, kết nối vào Website, Facebook, Instagram và hơn thế. Trả lời khách tự động, chuyển nhân viên khi cần.</p>
          <div className="flex gap-4 mt-8 text-sm text-muted font-normal">
            <span className="flex items-center gap-2"><Sparkles className="w-4 h-4 text-accent" /> Prompt tuỳ chỉnh cho mỗi trợ lý</span>
            <span className="flex items-center gap-2"><Plug className="w-4 h-4 text-accent" /> Đa kênh, đa nhà cung cấp AI</span>
          </div>
        </div>
        <div className="text-xs text-muted font-normal">© {appName}</div>
      </div>
      <div className="flex items-center justify-center p-6">
        <Card className="w-full max-w-sm">
          <div className="text-lg font-extrabold mb-1 lg:hidden flex items-center gap-2"><span className="w-6 h-6 rounded-md bg-pastel flex items-center justify-center overflow-hidden"><Mark cls="w-6 h-6" /></span> {appName}</div>
          <div className="text-[15px] font-bold mb-1">{mode === "login" ? "Đăng nhập" : "Tạo tài khoản"}</div>
          <p className="text-[13px] text-muted mb-4 font-normal">{mode === "login" ? "Chào mừng trở lại." : "Bắt đầu với một workspace mới."}</p>
          <Field label="Email"><Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="ban@congty.com" onKeyDown={(e) => e.key === "Enter" && go()} /></Field>
          <div className="mt-3"><Field label="Mật khẩu"><Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Tối thiểu 8 ký tự" onKeyDown={(e) => e.key === "Enter" && go()} /></Field></div>
          {mode === "signup" && <>
            <div className="mt-3"><Field label="Nhập lại mật khẩu"><Input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="Gõ lại mật khẩu" onKeyDown={(e) => e.key === "Enter" && go()} />
              {confirm && password !== confirm && <span className="text-[12px] text-bad font-normal">Mật khẩu nhập lại chưa khớp.</span>}</Field></div>
            <div className="mt-3"><Field label="Tên workspace"><Input value={orgName} onChange={(e) => setOrgName(e.target.value)} placeholder="Cửa hàng của tôi" /></Field></div>
          </>}
          <div className="mt-4"><Button loading={busy} onClick={go} className="w-full">{mode === "login" ? "Đăng nhập" : "Đăng ký"}</Button></div>
          {google && (
            <>
              <div className="flex items-center gap-3 my-3 text-[12px] text-muted"><span className="h-px bg-line flex-1" />hoặc<span className="h-px bg-line flex-1" /></div>
              <button onClick={googleLogin} className="w-full inline-flex items-center justify-center gap-2 text-sm font-semibold rounded-lg py-2 bg-white text-[#1f2328] hover:brightness-95 transition">
                <svg width="16" height="16" viewBox="0 0 48 48"><path fill="#4285F4" d="M45 24c0-1.6-.1-2.8-.4-4H24v7.6h12c-.2 1.9-1.5 4.7-4.3 6.6l-.1.3 6.2 4.8.4.1C42.5 35.9 45 30.5 45 24z"/><path fill="#34A853" d="M24 46c5.7 0 10.5-1.9 14-5.1l-6.7-5.2c-1.8 1.2-4.2 2.1-7.3 2.1-5.6 0-10.3-3.7-12-8.8l-.3.1-6.4 5-.1.3C8.6 41.1 15.7 46 24 46z"/><path fill="#FBBC05" d="M12 29c-.4-1.3-.7-2.6-.7-4s.3-2.7.6-4l-.1-.3-6.5-5-.2.1C3.9 14.7 3 19.2 3 24s.9 9.3 2.6 13.2L12 29z"/><path fill="#EA4335" d="M24 10.8c3.9 0 6.6 1.7 8.1 3.1l5.9-5.8C34.5 4.7 29.7 2.7 24 2.7 15.7 2.7 8.6 7.6 5.1 14.7l6.9 5.3c1.7-5.1 6.4-9.2 12-9.2z"/></svg>
                Tiếp tục với Google
              </button>
            </>
          )}
          <Msg type="err">{err}</Msg>
          <button className="text-[13px] text-accent mt-3 font-semibold" onClick={() => { setMode(mode === "login" ? "signup" : "login"); setErr(""); }}>
            {mode === "login" ? "Chưa có tài khoản? Đăng ký" : "Đã có tài khoản? Đăng nhập"}
          </button>
        </Card>
      </div>
    </div>
  );
}
