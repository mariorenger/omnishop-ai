// Builds the Word report from report_content.js (blocks) + refs.js (verified references) + out_facts.json (measured numbers).
// Usage: node build_report.js <out.docx> [pages.json]
//   pages.json (optional) maps heading bookmark ids to page numbers, produced by toc_pages.py from a first render,
//   so the static table of contents carries real page numbers (two-pass build).
const fs = require("fs");
const path = require("path");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType, BorderStyle,
  ShadingType, ImageRun, Footer, PageNumber, TabStopType, LevelFormat, InternalHyperlink, Bookmark, VerticalAlign, PageBreak,
} = D;

const OUT = process.argv[2] || "out/report.docx";
const PAGES = process.argv[3] && fs.existsSync(process.argv[3]) ? JSON.parse(fs.readFileSync(process.argv[3], "utf8")) : {};
const HERE = __dirname;
const F = JSON.parse(fs.readFileSync(path.join(HERE, "..", "out_facts.json"), "utf8"));
const REFS = require("./refs.js");
const FIG = path.join(HERE, "..", "out", "fig");

const NAVY = "003366", RED = "C00000", BLUE = "1F4E8C", GRAYTXT = "555555";
const FONT = "Times New Roman";
const TEXT_W = 9071;              // A4 11906 - left 1701 - right 1134 (twips)
const BODY = 26;                  // 13 pt

// ------------------------------------------------------------------ citations & numbering
const refOrder = [];
function citeNum(key) {
  if (!REFS[key]) throw new Error("unknown reference: " + key);
  let i = refOrder.indexOf(key);
  if (i < 0) { refOrder.push(key); i = refOrder.length - 1; }
  return i + 1;
}
function expandCites(s) {
  return s.replace(/\[@([a-z0-9_,@ ]+)\]/g, (_m, body) => {
    const nums = body.split(",").map((k) => citeNum(k.trim().replace(/^@/, "")));
    return "[" + nums.join(", ") + "]";
  });
}
let nTable = 0, nFig = 0, nEq = 0;

// ------------------------------------------------------------------ inline markup: **bold**, *italic*, `mono`
const XREF = { T: {}, H: {}, E: {} };
function xref(s) {
  return s.replace(/\{([THE]):([a-z0-9_]+)\}/g, (_m, k, id) => {
    if (XREF[k][id] == null) throw new Error("unknown cross-reference " + k + ":" + id);
    return String(XREF[k][id]);
  });
}
function numberBlocks(groups) {               // same order as rendering: front, body, appendix
  let t = 0, f = 0, e = 0;
  for (const g of groups) for (const b of g) {
    if (b.t === "table") { t++; if (b.id) XREF.T[b.id] = t; }
    if (b.t === "fig") { f++; if (b.id) XREF.H[b.id] = f; }
    if (b.t === "eq") { e++; if (b.id) XREF.E[b.id] = e; }
  }
}
function runs(text, base = {}) {
  text = xref(expandCites(text));
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Courier New", ...base, size: (base.size || BODY) - 3 }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}),
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: opts.line ?? 288, lineRule: "auto" },
  indent: opts.indent,
  keepNext: opts.keepNext,
});

// ------------------------------------------------------------------ headings with bookmarks (for the static TOC)
const tocEntries = [];
let hCount = 0;
function heading(level, text) {
  const id = "h" + (++hCount);
  tocEntries.push({ level, text, id });
  const lvl = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][level - 1];
  return new Paragraph({ heading: lvl, children: [new Bookmark({ id, children: [new TextRun({ text, font: FONT })] })], keepNext: true });
}

// ------------------------------------------------------------------ tables
const thin = { style: BorderStyle.SINGLE, size: 4, color: "B7C0D0" };
const borders = { top: thin, bottom: thin, left: thin, right: thin, insideHorizontal: thin, insideVertical: thin };
function cell(text, w, o = {}) {
  const keep = !!o.keepNext;
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: o.align || AlignmentType.LEFT, keepNext: keep, spacing: { after: 0, line: 252, lineRule: "auto" },
      children: runs(String(text ?? ""), { size: o.size || 20, bold: o.bold, color: o.color }) })],
  });
}
function table(b) {
  const widths = b.widths.map((f) => Math.round(f * TEXT_W));
  const diff = TEXT_W - widths.reduce((a, c) => a + c, 0); widths[0] += diff;
  const al = (j) => ((b.align || [])[j] === "r" ? AlignmentType.RIGHT : (b.align || [])[j] === "c" ? AlignmentType.CENTER : AlignmentType.LEFT);
  const rows = [new TableRow({ tableHeader: true, children: b.header.map((h, j) => cell(h, widths[j], { fill: NAVY, bold: true, color: "FFFFFF", align: al(j), size: b.size || 20 })) })];
  const small = b.rows.length <= 12;
  b.rows.forEach((r, i) => {
    const hl = (b.highlight || []).includes(i);
    const fill = hl ? "FDECEC" : (i % 2 ? "F3F5FA" : undefined);
    const keepNext = small && i < b.rows.length - 1;
    rows.push(new TableRow({ cantSplit: true, children: r.map((v, j) => cell(v, widths[j], { fill, align: al(j), bold: hl && j === 0, size: b.size || 20, keepNext })) }));
  });
  const n = ++nTable;
  const out = [];
  out.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 80 },
    children: runs(`**Bảng ${n}.** ${b.caption}`, { size: 22, color: NAVY }) }));
  out.push(new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, borders, rows }));
  if (b.note) out.push(new Paragraph({ spacing: { before: 60, after: 160, line: 252, lineRule: "auto" }, alignment: AlignmentType.JUSTIFIED, children: runs("*Ghi chú:* " + b.note, { size: 20, color: GRAYTXT }) }));
  else out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

// ------------------------------------------------------------------ figures
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function figure(b) {
  const f = pngSize(path.join(b.dir || FIG, b.file));
  const wIn = b.width || 6.2;
  const wPx = Math.round(wIn * 96), hPx = Math.round(wPx * f.h / f.w);
  const n = ++nFig;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60, line: 240, lineRule: "auto" },
      children: [new ImageRun({ type: "png", data: f.data, transformation: { width: wPx, height: hPx },
        altText: { title: `Hình ${n}`, description: b.caption, name: b.file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200, line: 252, lineRule: "auto" }, children: runs(`**Hình ${n}.** ${b.caption}`, { size: 21, color: NAVY }) }),
  ];
}

// ------------------------------------------------------------------ equations (native Word equations, centred, numbered on the right)
const { MathRun: MR, MathFraction: MFrac, MathSum: MSum, MathSubScript: MSub, MathRadical: MRad,
  MathRoundBrackets: MRB, MathSquareBrackets: MSB, MathAngledBrackets: MAB } = D;
const r = (t) => new MR(t);
const sb = (b, s) => new MSub({ children: [r(b)], subScript: [r(s)] });
const frac = (n, d) => new MFrac({ numerator: n, denominator: d });
const vhi = () => new MSub({ children: [r("v")], subScript: [sb("h", "i")] });
const EQ = {
  ci: () => [r("Δ̄ = "), frac([r("1")], [r("N")]), new MSum({ children: [new MSB({ children: [sb("m", "i"), r("(A) − "), sb("m", "i"), r("(B)")] })], subScript: [r("i=1")], superScript: [r("N")] }),
    r(",    "), sb("CI", "95"), r(" = Δ̄ ± 1,96 · "), sb("SE", "cụm"), r("(Δ̄)")],
  mean: () => [sb("s", "mean"), r("(u, c) = "), new MAB({ children: [frac([r("1")], [r("L")]), new MSum({ children: [sb("v", "h")], subScript: [r("h∈H")] }), r(" , "), sb("v", "c")] })],
  lse: () => [sb("s", "LSE"), r("(u, c) = τ · "), new MSB({ children: [r("log "), new MSum({ children: [r("exp"), new MRB({ children: [frac([new MAB({ children: [sb("v", "h"), r(", "), sb("v", "c")] })], [r("τ")])] })], subScript: [r("h∈H")] }), r(" − log L")] })],
  km: () => [sb("s", "KM"), r("(u, c) = τ · log "), new MSum({ children: [sb("π", "k"), r(" exp"), new MRB({ children: [frac([new MAB({ children: [sb("μ", "k"), r(", "), sb("v", "c")] })], [r("τ")])] })], subScript: [r("k=1")], superScript: [r("K")] })],
  reader: () => [sb("α", "c,i"), r(" = "), sb("softmax", "i"), new MRB({ children: [frac([new MAB({ children: [r("W g("), sb("v", "c"), r("), g("), vhi(), r(")")] })], [new MRad({ children: [r("m")] })])] }),
    r(",    s(u, c) = "), new MAB({ children: [new MSum({ children: [sb("α", "c,i"), r(" g("), vhi(), r(")")], subScript: [r("i")] }), r(" , g("), sb("v", "c"), r(")")] })],
  zfuse: () => [sb("z", "k"), r("(u, c) = "), frac([sb("s", "k"), r("(u, c) − "), sb("μ", "k")], [sb("σ", "k")]), r(",    "), sb("s", "super"), r("(u, c) = "), sb("z", "nền"), r(" + "),
    new MSum({ children: [sb("w", "k"), r(" "), sb("z", "k")], subScript: [r("k")] })],
};
function equation(b) {
  const n = ++nEq;
  const body = b.math && EQ[b.math] ? [new D.Math({ children: EQ[b.math]() })] : [new TextRun({ text: b.text, font: "Cambria Math", size: BODY })];
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 100, after: 140, line: 240, lineRule: "auto" }, keepLines: true,
    children: [new TextRun({ text: "\t", font: FONT }), ...body, new TextRun({ text: `\t(${n})`, font: FONT, size: BODY })],
  });
}

// ------------------------------------------------------------------ call-out box (one-cell table)
function callout(b) {
  const box = { style: BorderStyle.SINGLE, size: 6, color: "9FB3D9" };
  const kids = [new Paragraph({ spacing: { after: 60 }, children: runs(`**${b.title}**`, { color: NAVY, size: 24 }) })];
  for (const t of [].concat(b.text)) kids.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276, lineRule: "auto" }, children: runs(t, { size: 24 }) }));
  return [new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W],
    borders: { top: box, bottom: box, left: box, right: box, insideHorizontal: box, insideVertical: box },
    rows: [new TableRow({ children: [new TableCell({ width: { size: TEXT_W, type: WidthType.DXA }, shading: { fill: "EEF2FA", type: ShadingType.CLEAR, color: "auto" },
      margins: { top: 110, bottom: 110, left: 170, right: 170 }, children: kids })] })] }),
    new Paragraph({ spacing: { after: 120 }, children: [] })];
}

// ------------------------------------------------------------------ lists
function list(items, ref) {
  const out = [];
  for (const it of items) {
    const t = typeof it === "string" ? it : it.text;
    out.push(new Paragraph({ numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 70, line: 288, lineRule: "auto" }, children: runs(t) }));
    if (typeof it === "object" && Array.isArray(it.sub)) for (const s of it.sub) out.push(new Paragraph({ numbering: { reference: "bul", level: 1 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 50, line: 276, lineRule: "auto" }, children: runs(s) }));
  }
  return out;
}
let numCount = 0;
const numberingConfigs = [
  { reference: "bul", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1021, hanging: 283 } } } }] },
];
function newNumbering() {
  const ref = "num" + (++numCount);
  numberingConfigs.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] });
  return ref;
}

// ------------------------------------------------------------------ render blocks
function render(blocks) {
  const out = [];
  for (const b of blocks) {
    switch (b.t) {
      case "h1": out.push(heading(1, b.text)); break;
      case "h2": out.push(heading(2, b.text)); break;
      case "h3": out.push(heading(3, b.text)); break;
      case "p": out.push(P(b.text, b)); break;
      case "bullets": out.push(...list(b.items, "bul")); break;
      case "numbers": out.push(...list(b.items, newNumbering())); break;
      case "eq": out.push(equation(b)); break;
      case "table": out.push(...table(b)); break;
      case "fig": out.push(...figure(b)); break;
      case "box": out.push(...callout(b)); break;
      case "pagebreak": out.push(new Paragraph({ children: [new PageBreak()] })); break;
      case "raw": out.push(...b.items); break;
      default: throw new Error("unknown block " + b.t);
    }
  }
  return out;
}

// ------------------------------------------------------------------ title page, abstract, TOC
const C = require("./report_content.js")(F);
numberBlocks([C.front, C.body, C.appendix]);
const abbrev = render(C.front);
const body = render(C.body);                  // fills tocEntries and the citation order
tocEntries.push({ level: 1, text: "TÀI LIỆU THAM KHẢO", id: "h_refs" });
const appendix = render(C.appendix);
const refsBlock = [new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
  children: [new Bookmark({ id: "h_refs", children: [new TextRun({ text: "TÀI LIỆU THAM KHẢO", font: FONT })] })] })];
refOrder.forEach((k, i) => refsBlock.push(new Paragraph({ spacing: { after: 70, line: 264, lineRule: "auto" }, indent: { left: 567, hanging: 567 }, alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 22 }), new TextRun({ text: REFS[k], font: FONT, size: 22 })],
  tabStops: [{ type: TabStopType.LEFT, position: 567 }] })));
const unused = Object.keys(REFS).filter((k) => !refOrder.includes(k));
if (unused.length) console.error("references never cited (left out of the list):", unused.join(", "));

function center(text, o = {}) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 120, before: o.before ?? 0, line: o.line ?? 276, lineRule: "auto" },
    children: runs(text, { size: o.size || BODY, bold: o.bold, color: o.color, italics: o.italics }) });
}
const logo = fs.existsSync(path.join(HERE, "..", "out", "hust_logo.png")) ? pngSize(path.join(HERE, "..", "out", "hust_logo.png")) : null;
const titlePage = [
  center("ĐẠI HỌC BÁCH KHOA HÀ NỘI", { size: 28, bold: true, color: NAVY, after: 40 }),
  center("TRƯỜNG CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG", { size: 26, bold: true, color: NAVY, after: 360 }),
  ...(logo ? [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 480, line: 240, lineRule: "auto" }, children: [new ImageRun({ type: "png", data: logo.data,
    transformation: { width: 92, height: Math.round(92 * logo.h / logo.w) }, altText: { title: "Logo", description: "Logo Đại học Bách khoa Hà Nội", name: "logo" } })] })] : []),
  center("BÁO CÁO NGHIÊN CỨU", { size: 30, bold: true, color: RED, after: 240 }),
  center(C.title, { size: 32, bold: true, color: NAVY, after: 200, line: 312, lineRule: "auto" }),
  center(C.titleEn, { size: 24, italics: true, color: BLUE, after: 720 }),
  center("Học viên: **Nguyễn Nam Hán** — MSHV: 20252095M", { size: 26, after: 60 }),
  center("Chuyên ngành: Khoa học Máy tính", { size: 26, after: 60 }),
  center("Cán bộ hướng dẫn: [Họ và tên CBHD]", { size: 26, after: 960 }),
  center("Hà Nội, tháng 10/2026", { size: 26, italics: true, after: 0 }),
  new Paragraph({ children: [new PageBreak()] }),
];

const abstractPage = [
  center("**TÓM TẮT**", { size: 28, color: NAVY, after: 160 }),
  ...C.abstractVi.map((t) => P(t, { after: 100 })),
  P("**Từ khoá:** " + C.keywordsVi, { after: 280 }),
  new Paragraph({ children: [new PageBreak()] }),
  center("**ABSTRACT**", { size: 28, color: NAVY, after: 160 }),
  ...C.abstractEn.map((t) => P(t, { after: 100, run: { italics: false } })),
  P("**Keywords:** " + C.keywordsEn, { after: 120 }),
  new Paragraph({ children: [new PageBreak()] }),
];

function tocBlock() {
  const out = [center("**MỤC LỤC**", { size: 28, color: NAVY, after: 200 })];
  for (const e of tocEntries) {
    if (e.level > 2) continue;
    const pg = PAGES[e.id] != null ? String(PAGES[e.id]) : "…";
    out.push(new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W, leader: "dot" }],
      indent: { left: e.level === 1 ? 0 : 400 }, spacing: { after: e.level === 1 ? 60 : 30, before: e.level === 1 ? 80 : 0, line: 264, lineRule: "auto" },
      children: [new InternalHyperlink({ anchor: e.id, children: [new TextRun({ text: e.text, font: FONT, size: e.level === 1 ? 24 : 23, bold: e.level === 1 })] }),
        new TextRun({ text: "\t" + pg, font: FONT, size: e.level === 1 ? 24 : 23, bold: e.level === 1 })],
    }));
  }
  out.push(new Paragraph({ children: [new PageBreak()] }));
  return out;
}

const doc = new Document({
  creator: "Nguyễn Nam Hán", title: C.title, description: "Báo cáo nghiên cứu — LLM trong gợi ý tin tức (MIND)",
  styles: {
    default: { document: { run: { font: FONT, size: BODY }, paragraph: { spacing: { line: 288, lineRule: "auto" } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: NAVY, font: FONT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: BLUE, font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, italics: true, color: "333333", font: FONT }, paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: numberingConfigs },
  features: { updateFields: false },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1418, bottom: 1418, left: 1701, right: 1134, footer: 567 } }, titlePage: true },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22, color: GRAYTXT })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children: [...titlePage, ...abstractPage, ...tocBlock(), ...abbrev, ...body, ...refsBlock, ...appendix],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  fs.writeFileSync(OUT.replace(/\.docx$/, ".toc.json"), JSON.stringify(tocEntries, null, 1));
  console.log("wrote", OUT, "tables", nTable, "figures", nFig, "equations", nEq, "refs", refOrder.length, "headings", tocEntries.length);
});
