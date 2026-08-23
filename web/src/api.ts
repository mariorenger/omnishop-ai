// Minimal API client. Same-origin /api (vite proxy in dev, nginx in prod).
const LS = "omnishop.auth";

export type Auth = { token: string; orgId?: string };
export function loadAuth(): Auth { try { return JSON.parse(localStorage.getItem(LS) || "{}"); } catch { return {} as Auth; } }
export function saveAuth(a: Auth) { localStorage.setItem(LS, JSON.stringify(a)); }
export function clearAuth() { localStorage.removeItem(LS); }

function headers(json = true): Record<string, string> {
  const a = loadAuth();
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (a.token) h["Authorization"] = `Bearer ${a.token}`;
  if (a.orgId) h["X-Org-Id"] = a.orgId;
  return h;
}

async function handle(res: Response) {
  if (!res.ok) {
    let detail = res.statusText;
    try { const d = await res.json(); detail = d.detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  get: (p: string) => fetch(p, { headers: headers() }).then(handle),
  post: (p: string, body?: any) => fetch(p, { method: "POST", headers: headers(), body: body ? JSON.stringify(body) : undefined }).then(handle),
  put: (p: string, body?: any) => fetch(p, { method: "PUT", headers: headers(), body: body ? JSON.stringify(body) : undefined }).then(handle),
  del: (p: string) => fetch(p, { method: "DELETE", headers: headers() }).then(handle),
  upload: (p: string, form: FormData) => fetch(p, { method: "POST", headers: headers(false), body: form }).then(handle),
};
