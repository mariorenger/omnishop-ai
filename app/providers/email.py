"""EmailProvider (transactional email). Default 'console' logs to stdout so the
app works with no setup; 'smtp' and 'resend' send for real via env config."""
from __future__ import annotations
from typing import Protocol

from ..config import config


class EmailProvider(Protocol):
    def send(self, to: str, subject: str, html: str) -> None: ...


class ConsoleEmail:
    def send(self, to, subject, html):
        print(f"[email:console] to={to} subject={subject}\n{html}\n", flush=True)


class ResendEmail:
    def send(self, to, subject, html):
        import httpx
        httpx.post("https://api.resend.com/emails",
                   headers={"Authorization": f"Bearer {config.RESEND_API_KEY}"},
                   json={"from": config.EMAIL_FROM, "to": [to], "subject": subject, "html": html}, timeout=20)


class SmtpEmail:
    def send(self, to, subject, html):
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(html, "html", "utf-8")
        msg["Subject"] = subject; msg["From"] = config.EMAIL_FROM; msg["To"] = to
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as s:
            s.starttls()
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASS)
            s.sendmail(config.EMAIL_FROM, [to], msg.as_string())


_provider: EmailProvider | None = None


def get_email() -> EmailProvider:
    global _provider
    if _provider is None:
        p = config.EMAIL_PROVIDER
        _provider = ResendEmail() if (p == "resend" and config.RESEND_API_KEY) else \
                    SmtpEmail() if (p == "smtp" and config.SMTP_HOST) else ConsoleEmail()
    return _provider


def send_safe(to: str, subject: str, html: str) -> None:
    """Best-effort: never break a request because email failed."""
    try:
        get_email().send(to, subject, html)
    except Exception as e:  # noqa: BLE001
        print(f"[email] send failed to={to}: {e}", flush=True)
