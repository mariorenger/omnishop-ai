"""Transactional email. Provider + credentials are resolved from the admin
config store (registry.resolve_email_config) so the platform admin sets them in
the UI; env vars are only the first-boot fallback. Default 'console' logs to
stdout so the app works with no setup."""
from __future__ import annotations


def _send(cfg: dict, to: str, subject: str, html: str) -> None:
    provider = cfg.get("provider") or "console"
    sender = cfg.get("from") or "OmniShop AI <no-reply@omnishop.local>"
    if provider == "resend" and cfg.get("secret"):
        import httpx
        httpx.post("https://api.resend.com/emails",
                   headers={"Authorization": f"Bearer {cfg['secret']}"},
                   json={"from": sender, "to": [to], "subject": subject, "html": html}, timeout=20)
    elif provider == "smtp" and cfg.get("smtp_host"):
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(html, "html", "utf-8")
        msg["Subject"] = subject; msg["From"] = sender; msg["To"] = to
        with smtplib.SMTP(cfg["smtp_host"], int(cfg.get("smtp_port") or 587), timeout=20) as s:
            s.starttls()
            if cfg.get("smtp_user"):
                s.login(cfg["smtp_user"], cfg.get("secret", ""))
            s.sendmail(sender, [to], msg.as_string())
    else:
        print(f"[email:console] to={to} subject={subject}\n{html}\n", flush=True)


def send_safe(to: str, subject: str, html: str) -> None:
    """Best-effort: never break a request because email failed."""
    try:
        from .registry import resolve_email_config
        _send(resolve_email_config(), to, subject, html)
    except Exception as e:  # noqa: BLE001
        print(f"[email] send failed to={to}: {e}", flush=True)


def send_test(cfg: dict, to: str) -> dict:
    """Send a test message with an explicit config (admin 'send test' button)."""
    try:
        _send(cfg, to, "[OmniShop AI] Email thử nghiệm",
              "<p>Đây là email kiểm tra cấu hình gửi thư của OmniShop AI. "
              "Nếu bạn nhận được thư này, cấu hình đã hoạt động.</p>")
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
