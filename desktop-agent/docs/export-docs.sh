#!/usr/bin/env bash
# Export the Markdown guides to HTML, Word (.docx) and PDF.   Usage:  docs/export-docs.sh [outdir]
# Needs: python3 + `pip install markdown`  and  LibreOffice (soffice).   (pandoc is used instead if installed.)
set -euo pipefail
cd "$(dirname "$0")"
out="${1:-../dist/docs}"; mkdir -p "$out"
css='body{font-family:"Segoe UI",Arial,sans-serif;line-height:1.55;max-width:900px;margin:24px auto;padding:0 16px}
code,pre{font-family:Consolas,monospace;background:#f3f3f3}pre{padding:10px;overflow:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;vertical-align:top}h1,h2,h3{margin-top:1.4em}'
for f in HUONG_DAN HUONG_DAN_TAO_BOT NHAT_KY_PHAT_TRIEN; do
  [ -f "$f.md" ] || continue
  if command -v pandoc >/dev/null; then
    pandoc "$f.md" -s --metadata title="$f" -c /dev/null -o "$out/$f.html" || true
  else
    python3 - "$f.md" "$out/$f.html" "$css" <<'PY'
import sys, markdown
src, dst, css = sys.argv[1:4]
body = markdown.markdown(open(src, encoding='utf-8').read(), extensions=['tables', 'fenced_code', 'sane_lists'])
open(dst, 'w', encoding='utf-8').write(f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{body}</body></html>')
PY
  fi
  if command -v soffice >/dev/null; then
    (cd "$out" && soffice --headless --convert-to docx:"MS Word 2007 XML" "$f.html" >/dev/null 2>&1 || true
                  soffice --headless --convert-to pdf "$f.html" >/dev/null 2>&1 || true)
  fi
done
ls -la "$out"
