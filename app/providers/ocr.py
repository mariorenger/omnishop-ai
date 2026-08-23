"""OCRProvider — flexible/swappable OCR for images and scanned PDFs.

Backends:
  - tesseract : local Tesseract (pytesseract). Fast, offline; needs the binary.
  - vlm       : use a vision-capable LLM (the org's configured provider) to
                transcribe the image — swap the OCR "model" just by changing the
                LLM config. Great for handwriting/complex layouts.
  - disabled  : no OCR (returns empty text).

All backends degrade gracefully (return "" with a note) if unavailable, so
ingestion never hard-fails on a scanned page.
"""
from __future__ import annotations
import base64
from typing import Optional, Protocol


class OCRProvider(Protocol):
    name: str
    def extract_image(self, image_bytes: bytes, media_type: str = "image/png") -> str: ...


class DisabledOCR:
    name = "disabled"
    def extract_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        return ""


class TesseractOCR:
    name = "tesseract"
    def __init__(self, lang: str = "vie+eng"):
        self.lang = lang

    def extract_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        try:
            import io
            import pytesseract
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            try:
                return pytesseract.image_to_string(img, lang=self.lang).strip()
            except Exception:
                return pytesseract.image_to_string(img).strip()  # fallback to default lang
        except Exception as e:  # noqa: BLE001 — tesseract/PIL missing
            return f"[OCR unavailable: {e}]"


class VLMOCR:
    """Transcribe via a vision-capable LLM (the org's configured provider)."""
    name = "vlm"

    def __init__(self, org_id: Optional[str], model_override: str = ""):
        from .registry import resolve_llm_config
        self.cfg = resolve_llm_config(org_id)
        self.model_override = model_override

    _PROMPT = ("Trích xuất toàn bộ văn bản xuất hiện trong ảnh này, giữ nguyên thứ tự đọc. "
               "Chỉ trả về phần văn bản, không thêm giải thích.")

    def extract_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        provider = (self.cfg.get("provider") or "stub").lower()
        b64 = base64.b64encode(image_bytes).decode()
        try:
            if provider == "anthropic":
                return self._anthropic(b64, media_type)
            if provider in ("openai_compatible", "gemini"):
                return self._openai(b64, media_type, gemini=(provider == "gemini"))
            return "[OCR: cấu hình LLM hiện tại không hỗ trợ đọc ảnh]"
        except Exception as e:  # noqa: BLE001
            return f"[OCR error: {e}]"

    def _anthropic(self, b64: str, media_type: str) -> str:
        from anthropic import Anthropic
        from ..config import config
        client = Anthropic(api_key=self.cfg.get("api_key") or config.ANTHROPIC_API_KEY)
        model = self.model_override or self.cfg.get("model") or config.LLM_MODEL
        resp = client.messages.create(
            model=model, max_tokens=2048,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": self._PROMPT},
            ]}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()

    def _openai(self, b64: str, media_type: str, gemini: bool) -> str:
        import httpx
        from ..config import config
        from .llm import GEMINI_OPENAI_BASE
        base = (self.cfg.get("base_url") or (GEMINI_OPENAI_BASE if gemini else config.OPENAI_BASE_URL)).rstrip("/")
        model = self.model_override or self.cfg.get("model") or ("gemini-1.5-flash" if gemini else "gpt-4o-mini")
        headers = {"Content-Type": "application/json"}
        if self.cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {self.cfg['api_key']}"
        r = httpx.post(f"{base}/chat/completions", headers=headers, timeout=90, json={
            "model": model, "max_tokens": 2048,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self._PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            ]}],
        })
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()


def build_ocr(cfg: dict, org_id: Optional[str] = None) -> OCRProvider:
    p = (cfg.get("provider") or "disabled").lower()
    extra = cfg.get("extra") or {}
    if p == "tesseract":
        return TesseractOCR(lang=extra.get("lang", "vie+eng"))
    if p == "vlm":
        return VLMOCR(org_id, model_override=cfg.get("model") or "")
    return DisabledOCR()
