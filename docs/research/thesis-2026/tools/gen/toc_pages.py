"""Second pass of the report build: read the page on which every heading lands in the rendered PDF.

usage: python toc_pages.py report.pdf report.toc.json pages.json
The TOC itself lists every heading, so the search starts after the page where the LAST entry first appears
(the end of the TOC) and moves forward monotonically through the document.
"""
import json
import re
import sys

import pymupdf


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


pdf, toc_json, out = sys.argv[1:4]
doc = pymupdf.open(pdf)
pages = [norm(p.get_text()) for p in doc]
entries = json.load(open(toc_json, encoding="utf-8"))
key = lambda t: norm(t)[:38]

last = key(entries[-1]["text"])
toc_end = next(i for i, t in enumerate(pages) if last in t)
res, cur, missing = {}, toc_end + 1, []
for e in entries:
    k = key(e["text"])
    hit = next((i for i in range(cur, len(pages)) if k in pages[i]), None)
    if hit is None:
        missing.append(e["text"])
        continue
    res[e["id"]] = hit + 1
    cur = hit
json.dump(res, open(out, "w"), indent=1)
print(f"pages {len(pages)}, toc ends on page {toc_end + 1}, found {len(res)}/{len(entries)}", "missing:", missing)
