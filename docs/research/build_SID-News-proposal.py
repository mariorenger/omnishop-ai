# -*- coding: utf-8 -*-
"""Build the LLM-RecSys thesis-proposal Word report."""
import docx
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

NAVY = RGBColor(0x0B, 0x2E, 0x59)
RED = RGBColor(0xB3, 0x1B, 0x1B)
GREY = RGBColor(0x44, 0x44, 0x44)
HDR_BG = "0B2E59"
ALT_BG = "EEF2F8"

doc = Document()

# base style
base = doc.styles["Normal"]
base.font.name = "Calibri"
base.font.size = Pt(10.5)
base.paragraph_format.space_after = Pt(4)

for lvl, sz, col in [(1, 15, NAVY), (2, 12.5, NAVY), (3, 11, RED)]:
    st = doc.styles[f"Heading {lvl}"]
    st.font.name = "Calibri"
    st.font.size = Pt(sz)
    st.font.color.rgb = col
    st.font.bold = True


def set_cell_bg(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def para(text="", bold=False, italic=False, size=None, color=None, align=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if align:
        p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    return p


def bullets(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        p.paragraph_format.space_after = Pt(2)
        if isinstance(it, tuple):
            r = p.add_run(it[0] + ": ")
            r.bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        set_cell_bg(hdr[i], HDR_BG)
        p = hdr[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(9.5)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            if ri % 2 == 1:
                set_cell_bg(cells[ci], ALT_BG)
            p = cells[ci].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(str(val))
            r.font.size = Pt(9)
    if widths:
        for ci, w in enumerate(widths):
            for row in t.rows:
                row.cells[ci].width = Inches(w)
    return t


def hrule():
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "B31B1B")
    pbdr.append(bottom)
    pPr.append(pbdr)


def add_toc():
    p = doc.add_paragraph()
    run = p.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-2" \h \z \u')
    r2 = OxmlElement("w:r")
    t2 = OxmlElement("w:t")
    t2.text = "Nhấp chuột phải → Update Field để hiện mục lục."
    r2.append(t2)
    fld.append(r2)
    p._p.append(fld)


def footer_pagenum():
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fldb = OxmlElement("w:fldChar"); fldb.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    flde = OxmlElement("w:fldChar"); flde.set(qn("w:fldCharType"), "end")
    run._r.append(fldb); run._r.append(instr); run._r.append(flde)


# ============================== TITLE ==============================
para("BÁO CÁO NGHIÊN CỨU ĐỊNH HƯỚNG", bold=True, size=11, color=RED,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("Hệ gợi ý dựa trên LLM (2024–2026): Bản đồ SOTA, Xu hướng 2026 "
     "và Đề xuất luận văn khả thi với GPU hạn chế",
     bold=True, size=18, color=NAVY, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para("Trọng tâm: Gợi ý tin tức (News Recommendation) — Semantic-ID Generative "
     "kết hợp Hợp nhất tín hiệu cộng tác (semantic ⊕ collaborative)",
     italic=True, size=11, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
para("Người thực hiện: ……………………………          Giảng viên hướng dẫn: ……………………………",
     size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("Ngày lập: 19/09/2026", size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
hrule()

doc.add_heading("Mục lục", level=1)
add_toc()

# ============================== TÓM TẮT ==============================
doc.add_heading("Tóm tắt", level=1)
para("Gợi ý tin tức truyền thống (NRMS, NAML, Fastformer) đã bão hoà quanh AUC 0.67–0.70 trên MIND. "
     "Giai đoạn 2024–2026, hai làn sóng thống trị: (i) dùng LLM/PLM làm bộ mã hoá tin (MANNeR, embedding "
     "BGE-M3/E5/LLM2Vec) cho gain chắc chắn; (ii) gợi ý sinh (generative recommendation) dựa trên "
     "Semantic ID (TIGER và hậu duệ), là hướng NÓNG nhất tại WWW/SIGIR/KDD 2026. Tuy vậy, Semantic-ID "
     "generative gần như CHƯA được áp dụng cho tin tức — nơi bài toán cold-start khắc nghiệt lại rất hợp "
     "với biểu diễn ngữ nghĩa. Báo cáo này khảo sát hiện trạng SOTA, hệ thống hoá kỹ thuật và xu hướng "
     "2026, rồi đề xuất SID-News: gợi ý tin tức sinh trên Semantic ID có khôi phục tín hiệu cộng tác, "
     "kèm khử nhiễu lịch sử bằng differential attention. Toàn bộ được thiết kế CHẠY ĐƯỢC trên một GPU "
     "miễn phí (Kaggle P100/T4) nhờ backbone T5 nhỏ (4–6 lớp) + RQ-VAE gọn.", space_after=6)

# ============================== 1. PHÂN LOẠI ==============================
doc.add_heading("1. Phân loại bài toán recommendation, dataset và metric", level=1)
table(
    ["Loại bài toán", "Dataset chuẩn", "Metric", "Đặc thù"],
    [
        ["Sequential / next-item", "Amazon Reviews, MovieLens, Steam, Yelp", "Recall@K, NDCG@K", "Lịch sử → item kế tiếp"],
        ["CTR / click", "Criteo, Avazu, Amazon-M2, Taobao", "AUC, LogLoss", "Nhấn / không nhấn"],
        ["News recommendation", "MIND (small/large), EB-NeRD (RecSys Challenge 2024)", "AUC, MRR, nDCG@5/10", "Cold-start mạnh, tin sống ngắn"],
        ["Session-based", "Amazon-M2 (KDD Cup'23), Diginetica", "Recall@K, MRR", "Không có user-id ổn định"],
        ["Conversational (CRS)", "ReDial, INSPIRED, OpenDialKG", "Recall, success", "Hội thoại nhiều lượt"],
        ["Embedding/retrieval", "MTEB, BEIR", "nDCG, Recall", "Đo chất lượng vector (vai trò encoder)"],
        ["Framework benchmark", "BARS, RecBole, OpenP5, SRBench (2026)", "chuẩn hoá split", "So sánh công bằng, chống 'SOTA giả'"],
    ],
    widths=[1.6, 2.3, 1.3, 2.0],
)
para("Ghi chú: RecSys KHÔNG có một leaderboard hợp nhất như MMLU của NLP; mỗi paper báo cáo trên cùng "
     "split chuẩn. Vì vậy 'model cao điểm nhất' luôn kèm dataset và cấu hình cụ thể.", italic=True, size=9.5)

# ============================== 2. SOTA ==============================
doc.add_heading("2. Hiện trạng SOTA — model cao điểm nhất theo từng nhánh", level=1)
table(
    ["Nhánh / Dataset", "Model cao điểm nhất", "Hội nghị", "Kỹ thuật cốt lõi", "Điểm (reported)"],
    [
        ["News / MIND-large", "Fastformer + PLM-NR (ensemble)", "SIGIR'21", "additive attn + PLM, ensemble", "AUC 72.68 (top leaderboard, đóng băng ~2021)"],
        ["News / MIND", "MANNeR", "EMNLP-F'24", "PLM encoder + metric/aspect learning", "AUC tốt nhất (~+13% so TANR)"],
        ["News / MIND-large", "GLORY", "RecSys'23", "global click + entity graph", "AUC 69.54"],
        ["Sequential / Amazon", "TIGER → LETTER → LSIG", "NeurIPS'23 / SIGIR'25 / WWW'26", "Semantic ID (RQ-VAE) + sinh tự hồi quy", "SOTA Recall/NDCG (gây tranh cãi vs SASRec)"],
        ["Seq (tune LLM)", "TALLRec, LLaRA", "RecSys'23, SIGIR'24", "instruction-tune (LoRA)", "mạnh ở few-shot / cold"],
        ["CTR", "CoLLM, CTRL", "TKDE'25", "nhét CF-embedding vào LLM", "AUC > baseline CTR"],
        ["Embedding (MTEB)", "LLM2Vec, E5-Mistral, NV-Embed, BGE-M3", "COLM'24 …", "decoder-only → embedder + contrastive", "top MTEB"],
    ],
    widths=[1.5, 1.9, 1.5, 1.9, 1.6],
)
para("Cảnh báo đánh giá: nhiều 'SOTA LLM-rec' CHƯA chắc vượt baseline cổ điển tune kỹ (SASRec/NRMS+PLM); "
     "so sánh tokenizer semantic-ID cũng thiếu tin cậy (SIDInspector, arXiv 2606.10375). Luôn reproduce baseline mạnh.",
     italic=True, size=9.5)

# ============================== 3. TAXONOMY ==============================
doc.add_heading("3. Taxonomy kỹ thuật LLM cho recommendation (8 họ)", level=1)
table(
    ["#", "Họ kỹ thuật", "Method tiêu biểu (hội nghị)", "Chiêu enabling"],
    [
        ["1", "LLM-as-embedder (two-tower)", "E5, BGE-M3, LLM2Vec (COLM'24)", "contrastive align text ↔ CF"],
        ["2", "Generative + Semantic IDs ★", "TIGER (NeurIPS'23), LETTER (SIGIR'25), LSIG (WWW'26)", "RQ-VAE tokenize item → sinh mã (beam-search có ràng buộc)"],
        ["3", "Instruction-tuning LLM", "TALLRec (RecSys'23), LLaRA (SIGIR'24)", "LoRA/QLoRA (tune rẻ)"],
        ["4", "Nhét CF vào LLM", "CoLLM (TKDE'25), E4SRec", "bind ID/CF-embedding thành token"],
        ["5", "LLM-as-reranker", "LLM4Rerank (WWW'25), LlamaRec", "prompt listwise/pairwise"],
        ["6", "RAG cho rec", "K-RagRec / KG-RAG (2025), RAG-Rec", "retrieve user tương tự / KG-subgraph → nạp LLM"],
        ["7", "Agentic / hội thoại", "RecMind, InteRecAgent, AgentCF", "multi-agent + tool + planning"],
        ["8", "Reasoning / CoT", "Reasoning over Semantic IDs (KDD'26)", "scaffolding suy luận trước khi gợi ý"],
    ],
    widths=[0.3, 1.9, 2.7, 2.5],
)

# ============================== 4. XU HƯỚNG 2026 ==============================
doc.add_heading("4. XU HƯỚNG 2026 (chi tiết, kèm hội nghị)", level=1)
para("Xếp theo độ nóng × độ khả thi cho luận văn GPU hạn chế:", bold=True)

doc.add_heading("4.1. Generative Recommendation + Semantic IDs — dòng nóng nhất", level=2)
bullets([
    ("TIGER (NeurIPS 2023)", "công thức chuẩn: RQ-VAE lượng tử hoá item thành ID ngữ nghĩa, sinh tự hồi quy."),
    ("LETTER (SIGIR 2025)", "học tokenizer đầu-cuối, tránh RQ-VAE tĩnh."),
    ("LSIG — Long Semantic IDs (WWW 2026)", "ID dài để tăng biểu đạt; RPG (2025) sinh ID dài SONG SONG, bỏ beam-search đắt → hợp GPU yếu."),
    ("Differentiable Semantic ID (SIGIR 2026)", "tokenizer khả vi, tối ưu chung với recommender."),
    ("Trie-Aware Transformers (2026), GEMs multi-stream decoder (2026)", "giải nút thắt chuỗi dài & decode."),
    ("SIDInspector (2026)", "công cụ chẩn đoán tokenizer — nhắc rằng so sánh SID còn thiếu tin cậy."),
])

doc.add_heading("4.2. Hợp nhất Semantic ⊕ Collaborative — khe hở lý thuyết trung tâm", level=2)
bullets([
    ("DiscRec (2025)", "tách rời (disentangle) tín hiệu ngữ nghĩa và cộng tác vì chúng xung đột mục tiêu."),
    ("Do LLMs Understand Collaborative Signals? Diagnosis and Repair (2026)", "chẩn đoán & vá việc LLM 'mù' tín hiệu cộng tác."),
    ("LC-Rec (ICDE 2024), CoLLM (TKDE 2025)", "căn chỉnh không gian ngữ nghĩa của LLM với tín hiệu cộng tác."),
])

doc.add_heading("4.3. Reasoning Recommendation (CoT) — mới nổi 2025–2026", level=2)
bullets([
    ("Reasoning over Semantic IDs Enhances Generative Recommendation (KDD 2026)", "chèn bước suy luận trên chuỗi Semantic ID."),
    ("Where Reasoning Matters: Rethinking Latent Reasoning in SID-based GenRec (2026)", "phân tích khi nào reasoning thực sự giúp."),
])

doc.add_heading("4.4. Agentic + KG-RAG", level=2)
bullets([
    ("K-RagRec / KG-RAG for LLM-based Recommendation (2025)", "retrieve KG-subgraph nạp LLM."),
    ("Survey on LLM-powered Agents for Recommender Systems (EMNLP 2025)", "multi-agent + planning + tool."),
])

doc.add_heading("4.5. LLM-encoder + Distillation (âm thầm thắng, dễ triển khai)", level=2)
bullets([
    ("LLM2Vec (COLM 2024), E5-Mistral, BGE-M3, NV-Embed", "encoder mạnh, chạy CPU/GPU nhẹ."),
    ("LLMs Need Encoders for Semantic IDs Too (2026)", "encoder tốt là nền cho semantic ID tốt."),
    ("Distillation LLM → model nhỏ", "giữ chất lượng, giảm độ trễ serving."),
])

# ============================== 5. GAP ==============================
doc.add_heading("5. Khoảng trống & cơ hội (vì sao chọn news)", level=1)
bullets([
    "Semantic-ID generative BÙNG NỔ trên e-commerce/phim (Amazon/MovieLens) nhưng RẤT ÍT cho tin tức.",
    "Tin tức = cold-start khắc nghiệt (tin mới liên tục) — đúng thế mạnh của Semantic ID (tin mới vẫn có mã ngữ nghĩa).",
    "Xung đột semantic ⊕ collaborative đặc biệt gắt với news (tin sống ngắn → tín hiệu cộng tác thưa) — chưa ai giải cho news.",
    "→ Cơ hội: đưa Semantic-ID generative + khôi phục collaborative sang MIND/EB-NeRD, một đề tài 2026 rõ ràng và khả thi.",
])

# ============================== 6. ĐỀ XUẤT CHÍNH ==============================
doc.add_heading("6. Đề xuất chính — SID-News (hướng đóng góp nhất)", level=1)
para("SID-News = Collaborative-aware Semantic-ID Generative News Recommendation. Bốn khối:", bold=True)
bullets([
    ("(A) News tokenizer", "mã hoá tiêu đề+tóm tắt bằng Sentence-T5/BGE-M3 → RQ-VAE lượng tử hoá thành Semantic ID (vd 3–4 mã, codebook 64)."),
    ("(B) Generative recommender", "T5 nhỏ (4–6 lớp) sinh tự hồi quy Semantic ID của tin kế tiếp từ lịch sử đọc; decode có ràng buộc theo trie."),
    ("(C) Khôi phục tín hiệu cộng tác", "thêm nhánh/token cộng tác (ý tưởng DiscRec/CoLLM) để ID không chỉ ngữ nghĩa mà còn phản ánh co-click → điểm mới cho news."),
    ("(D) Khử nhiễu lịch sử", "differential attention (đóng góp riêng từ benchmark của nhóm) làm sạch click clickbait trước khi sinh."),
])
para("Vì sao mới: (1) Semantic-ID generative cho NEWS gần như chưa có; (2) giải trực tiếp xung đột "
     "semantic ⊕ collaborative — vấn đề trung tâm 2026 — trong bối cảnh cold-start của news; (3) ghép "
     "differential-attention (denoise) vào bộ sinh là cầu nối kỹ thuật hai mảng. Bám sát xu hướng 4.1 + 4.2.",
     space_after=6)

# ============================== 7. KẾ HOẠCH KAGGLE ==============================
doc.add_heading("7. Kế hoạch khả thi với GPU miễn phí (Kaggle P100/T4)", level=1)
para("Kaggle free: GPU P100 16GB hoặc T4×2, phiên ≤ 12h, ~30h/tuần. SID-News được cấu hình NHỎ để vừa:", space_after=3)
table(
    ["Thành phần", "Cấu hình gọn (khả thi 1 GPU)", "Ghi chú GPU"],
    [
        ["Text encoder", "BGE-small / Sentence-T5 (~33–110M), ĐÓNG BĂNG", "Precompute embedding OFFLINE 1 lần → lưu .npz"],
        ["Tokenizer", "RQ-VAE 3–4 lớp, 4 mã, codebook 64", "Rất nhẹ, train vài phút"],
        ["Recommender", "T5 4–6 lớp, hidden 128–256", "Vừa 16GB, batch 64–128"],
        ["Dataset", "MIND-small trước; MIND-large khi ổn", "MIND-small chạy trong 1 phiên 12h"],
        ["Kỹ thuật tiết kiệm", "gradient checkpointing, fp16, batch nhỏ, RPG (sinh song song, bỏ beam-search)", "Giảm VRAM & thời gian"],
    ],
    widths=[1.5, 3.2, 2.3],
)
para("Bằng chứng khả thi: các paper 2025–2026 dùng đúng cấu hình nhỏ này — T5 4–6 lớp hidden 128, "
     "RQ-VAE 4 mã codebook 64, Sentence-T5 làm encoder (RPG, Differentiable SID). Đây là quy mô single-GPU.",
     italic=True, size=9.5)

# ============================== 8. BASELINES & THÍ NGHIỆM ==============================
doc.add_heading("8. Baselines, thiết kế thí nghiệm và ablation", level=1)
para("Baselines cần so (đều có code công khai):", bold=True)
table(
    ["Nhóm", "Baseline", "Hội nghị"],
    [
        ["Content", "NRMS, NAML, LSTUR, Fastformer", "EMNLP/IJCAI/ACL'19, arXiv'21"],
        ["Content SOTA", "CAUM, MANNeR", "SIGIR'22, EMNLP-F'24"],
        ["Graph", "GLORY", "RecSys'23"],
        ["LLM-encoder", "two-tower + BGE-M3 (đã có trong newsrec_bench)", "—"],
        ["Generative", "TIGER (text-only) áp cho news", "NeurIPS'23"],
    ],
    widths=[1.4, 3.6, 2.0],
)
para("Ablation: (a) tokenizer RQ-VAE vs LETTER; (b) CÓ/KHÔNG khôi phục collaborative; (c) CÓ/KHÔNG "
     "differential-attention denoise; (d) độ dài Semantic ID. Metric: AUC/MRR/nDCG@5/10 + TÁCH cold-start "
     "(tin mới) + chi phí/độ trễ. Đo beyond-accuracy (diversity) để tạo khác biệt học thuật.", space_after=6)

# ============================== 9. DỰ PHÒNG ==============================
doc.add_heading("9. Phương án dự phòng (nếu thiếu thời gian/GPU)", level=1)
bullets([
    ("B1 — DistillNews", "teacher = BGE-M3/E5 (embedding tin) → student = two-tower Fastformer nhanh; dễ ra số, rủi ro thấp."),
    ("B2 — CF-aware LLM reranker", "two-tower lấy top-K → LLM nhỏ rerank có nhét CF-embedding (CoLLM+LLM4Rerank) + giải thích."),
])

# ============================== 10. RỦI RO ==============================
doc.add_heading("10. Rủi ro & lưu ý đánh giá", level=1)
bullets([
    "SOTA gây tranh cãi → BẮT BUỘC reproduce baseline mạnh (NRMS+PLM, SASRec).",
    "Decode sinh dễ tạo ID không hợp lệ → cần constrained/trie decoding.",
    "Xung đột semantic ⊕ collaborative → cần cơ chế hợp nhất (chính là đóng góp).",
    "Chi phí/độ trễ LLM → luôn báo cáo song song accuracy; cân nhắc distillation.",
    "Cold-start & diversity là nơi dễ ghi điểm hơn đua AUC thuần.",
])

# ============================== 11. LỘ TRÌNH ==============================
doc.add_heading("11. Lộ trình đề xuất (8–10 tuần)", level=1)
table(
    ["Tuần", "Công việc", "Đầu ra"],
    [
        ["1–2", "Đọc TIGER/LETTER/LSIG/DiscRec; reproduce baseline NRMS+BGE trên MIND-small", "Bảng baseline"],
        ["3–4", "Xây tokenizer RQ-VAE + Semantic ID cho tin (BGE-small)", "Semantic ID + phân tích"],
        ["5–6", "Bộ sinh T5 nhỏ + constrained decode; SID-News v1", "Số AUC/nDCG đầu tiên"],
        ["7", "Khôi phục collaborative + differential-attention denoise", "Ablation"],
        ["8", "Cold-start split + chi phí/độ trễ + diversity", "Bảng đánh giá đầy đủ"],
        ["9–10", "Viết báo cáo/luận văn + slide", "Bản thảo + slide HUST"],
    ],
    widths=[0.8, 4.2, 2.0],
)

# ============================== 12. TÀI LIỆU ==============================
doc.add_heading("12. Tài liệu tham khảo", level=1)
refs = [
    "Rajput et al. Recommender Systems with Generative Retrieval (TIGER). NeurIPS 2023.",
    "Wang et al. Learnable Item Tokenization for Generative Recommendation (LETTER). SIGIR 2025.",
    "LSIG: Long Semantic IDs for Generative Recommendation. WWW (ACM Web Conference) 2026.",
    "Differentiable Semantic ID for Generative Recommendation. SIGIR 2026. arXiv:2601.19711.",
    "Reasoning over Semantic IDs Enhances Generative Recommendation. KDD 2026. arXiv:2603.23183.",
    "DiscRec: Disentangled Semantic-Collaborative Modeling for Generative Recommendation. arXiv:2506.15576 (2025).",
    "Generating Long Semantic IDs in Parallel for Recommendation (RPG). arXiv:2506.05781 (2025).",
    "Do LLMs Understand Collaborative Signals? Diagnosis and Repair. arXiv:2505.20730 (2025/26).",
    "LLMs Need Encoders for Semantic IDs Too. arXiv:2606.00324 (2026).",
    "Generative Recommendation with Semantic IDs: A Practitioner's Handbook. CIKM 2025. arXiv:2507.22224.",
    "Zheng et al. LC-Rec: Adapting LLMs by Integrating Collaborative Semantics. ICDE 2024.",
    "Bao et al. TALLRec: Aligning LLMs with Recommendation. RecSys 2023.",
    "Zhang et al. CoLLM: Integrating Collaborative Embeddings into LLMs. TKDE 2025.",
    "Liao et al. LLaRA: Large Language-and-Recommendation Assistant. SIGIR 2024.",
    "BehnamGhader et al. LLM2Vec: LLMs Are Secretly Powerful Text Encoders. COLM 2024.",
    "K-RagRec: Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation. arXiv:2501.02226 (2025).",
    "Iana et al. MANNeR: Modular Multi-Aspect Neural News Recommendation. EMNLP Findings 2024.",
    "Yang et al. GLORY: Going Beyond Local — Global Graph-Enhanced News Recommendation. RecSys 2023.",
    "Wu et al. Empowering News Recommendation with Pre-trained Language Models (PLM-NR). SIGIR 2021.",
    "Qi et al. News Recommendation with Candidate-aware User Modeling (CAUM). SIGIR 2022.",
    "Wu et al. MIND: A Large-scale Dataset for News Recommendation. ACL 2020.",
    "Kruse et al. EB-NeRD: Ekstra Bladet News Recommendation Dataset. RecSys Challenge 2024.",
    "A Survey on LLM-based News Recommender Systems. arXiv:2502.09797 (2025).",
    "A Survey on Generative Recommendation: Data, Model, and Tasks. arXiv:2510.27157 (2025).",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.add_run(f"[{i}] ").bold = True
    run = p.add_run(r)
    run.font.size = Pt(9)

footer_pagenum()

out = "/home/user/omnishop-ai/docs/research/SID-News-Research-Proposal.docx"
doc.save(out)
print("saved", out)
