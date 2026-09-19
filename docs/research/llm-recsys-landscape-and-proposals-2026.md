# Recommendation × LLM — Bản đồ 2024–2026, SOTA, và đề xuất ghép kỹ thuật

> Báo cáo định hướng luận văn: phân loại bài toán → dataset → kỹ thuật → model cao điểm nhất →
> **xu hướng 2026** → **đề xuất ghép kỹ thuật (cross-pollination)** để tạo điểm mới.
> Gắn với benchmark đã dựng: [`experiments/newsrec_bench/`](../../experiments/newsrec_bench/).
> Ngày: 2026-09-19. *Số SOTA là reported per-paper/per-split — không phải leaderboard hợp nhất
> (RecSys không có leaderboard "chính chủ" duy nhất như MMLU của NLP).*

---

## 1. Các loại bài toán recommendation + dataset + metric

| Loại bài toán | Dataset chuẩn | Metric | Đặc thù |
|---|---|---|---|
| **Sequential / next-item** | Amazon Reviews (Beauty/Sports/Toys…), MovieLens (1M/20M), Steam, Yelp, LastFM | Recall@K, NDCG@K, HR@K | Lịch sử → item kế tiếp |
| **CTR / click prediction** | Criteo, Avazu, Amazon-M2, Taobao, MovieLens-CTR | AUC, LogLoss | Nhấn/không nhấn |
| **News recommendation** | **MIND** (small/large), **EB-NeRD** (RecSys Challenge 2024) | AUC, MRR, nDCG@5/10 | Cold-start cực mạnh, tin sống ngắn |
| **Session-based** | Amazon-M2 (KDD Cup'23), Diginetica, YooChoose | Recall@K, MRR | Không có user id ổn định |
| **Conversational / CRS** | ReDial, INSPIRED, OpenDialKG | Recall, success rate | Hội thoại nhiều lượt |
| **Embedding/retrieval** (vai trò encoder) | **MTEB**, BEIR | nDCG, Recall | Đo chất lượng vector item/text |
| **Benchmark framework** (so công bằng) | BARS/OpenBenchmark, RecBole, OpenP5, RecBench, SRBench (2026) | chuẩn hoá split | Chống "SOTA giả" |

---

## 2. Model cao điểm nhất theo từng nhánh (2024–2026)

| Nhánh | Dataset | Model top / SOTA | Kỹ thuật cốt lõi | Điểm (reported) |
|---|---|---|---|---|
| **News** | MIND-large | **Fastformer + PLM-NR (ensemble)** | additive attn + PLM, ensemble | **AUC 72.68**, nDCG@10 41.51 (leaderboard chính thức, đóng băng ~2021) |
| News (research SOTA) | MIND | **MANNeR** (2024) | PLM encoder + metric/aspect learning | best AUC (~+13% so TANR) |
| News (graph) | MIND-large | **GLORY** (2023) | global click + **entity graph** | AUC 69.54 |
| News (PLM boost) | MIND | PLM-NRMS, +LLM category desc | RoBERTa; LLM sinh mô tả | +5.8% AUC so GloVe |
| **Sequential** | Amazon/ML | **TIGER, LC-Rec, LETTER** | **Semantic ID (RQ-VAE) + sinh tự hồi quy** | SOTA Recall/NDCG *(gây tranh cãi vs SASRec)* |
| Sequential (tune) | Amazon/ML | **TALLRec, LLaRA** | instruction-tune LLM (LoRA) | mạnh ở few-shot/cold |
| **CTR** | Amazon/ML-CTR | **CoLLM, CTRL** | nhét CF-embedding vào LLM | AUC > baseline CTR |
| **Embedding** | MTEB | **LLM2Vec, E5-Mistral, NV-Embed, BGE-M3** | decoder-only → embedder + contrastive | top MTEB |
| **Reasoning rec** (2025–26) | Amazon/ML | "Reasoning over Semantic IDs", OneRec-Think | CoT/suy luận trước khi gợi ý | hướng mới nổi |

> ⚠️ **Cảnh báo đánh giá:** nhiều "SOTA LLM-rec" **chưa chắc vượt baseline cổ điển tune kỹ**
> (SASRec/GRU4Rec); so sánh **tokenizer semantic-ID cũng thiếu tin cậy**. Luôn reproduce baseline mạnh.

---

## 3. Taxonomy kỹ thuật LLM-rec (8 họ) + chiêu enabling

| # | Họ | Method tiêu biểu | Chiêu enabling |
|---|---|---|---|
| 1 | **LLM-as-embedder** (two-tower) | E5/BGE/**LLM2Vec** | contrastive align text↔CF |
| 2 | **Generative + Semantic IDs** ⭐ | **TIGER, LC-Rec, LETTER** | **RQ-VAE** tokenize item → sinh mã bằng **beam-search có ràng buộc**; tốt cold/long-tail |
| 3 | **Instruction-tuning LLM** | **TALLRec, LLaRA, BIGRec** | **LoRA/QLoRA** (tune rẻ) |
| 4 | **Nhét CF vào LLM** | **CoLLM, E4SRec, CTRL** | bind ID/CF-embedding thành token |
| 5 | **LLM-as-reranker** | **LLM4Rerank, LlamaRec** | prompt listwise/pairwise |
| 6 | **RAG cho rec** | **RAG-Rec, K-RagRec (KG-RAG)** | retrieve user tương tự / KG-subgraph → nạp LLM |
| 7 | **Agentic / hội thoại** | RecMind, InteRecAgent, AgentCF | multi-agent + tool + planning |
| 8 | **Reasoning / CoT** | OneRec-Think, "Reasoning over SID" | scaffolding suy luận |

Xuyên suốt: **LLM sinh feature/profile** (GENRE, KAR), **alignment DPO**, **distillation** LLM→student nhỏ.
**Căng thẳng cốt lõi mọi paper đang giải:** *semantic ⚔ collaborative* xung đột; decode sinh dễ lỗi;
độ trễ serving; đánh giá thiếu tin cậy.

---

## 4. Xu hướng 2026 — nên theo cái nào?

Xếp theo **độ nóng × độ khả thi cho luận văn** (cao → thấp):

1. **⭐ Generative recommendation + Semantic IDs** — *dòng nóng nhất 2024–2026.* Sub-trend 2026:
   tokenizer tốt hơn (LETTER, universal tokenization), **khôi phục tín hiệu collaborative trong semantic ID**,
   **semantic ID cho JOINT search+rec**. Khả thi vừa (cần RQ-VAE + sinh có ràng buộc).
2. **Reasoning recommendation (CoT)** — mới, ít người làm cho news → dễ tạo điểm mới, nhưng cần LLM lớn.
3. **Hợp nhất semantic ⊕ collaborative** — vấn đề trung tâm; LC-Rec/DiscRec/"restoring collaborative signals".
   Đây là **khe hở lý thuyết tốt** để đóng góp.
4. **Agentic + KG-RAG** — retrieve KG/similar-user → LLM reason/rerank; mạnh cho cold-start + explainability.
5. **LLM-encoder + distillation (deployable)** — "âm thầm thắng": embedder mạnh + distill cho serving rẻ.
   An toàn, dễ ra số, hợp nếu muốn chắc ăn.

**Khuyến nghị định hướng:** với nền tảng news + benchmark sẵn có của bạn, **(1) đưa Semantic-ID generative
sang news** hoặc **(3)/(4) hợp nhất collaborative vào LLM + RAG cho news** là hai lựa chọn "đón trend mà vẫn khả thi".

---

## 5. Đề xuất ghép kỹ thuật (cross-pollination) — 5 ý tưởng cụ thể

Ý tưởng = **lấy kỹ thuật đang mạnh ở mảng A áp sang mảng B (news)**, hoặc **ghép hai kỹ thuật**.

### 💡 P1 — "GenNews": Semantic-ID Generative News Rec (TIGER → news) ⭐ *khuyến nghị*
- **Ghép:** #2 (TIGER/RQ-VAE, từ sequential) → **news**.
- **Cách:** dùng **BGE-M3/Jina** encode tin → **RQ-VAE** lượng tử hoá thành *semantic ID*; huấn luyện LLM/decoder
  **sinh tự hồi quy ID của tin kế tiếp** từ lịch sử đọc. Ràng buộc decode theo cây mã.
- **Vì sao mới:** semantic-ID generative **gần như chưa được áp cho news** — mà news là **cold-start cực mạnh**,
  đúng thế mạnh của semantic ID (tin mới vẫn có mã ngữ nghĩa). 
- **Eval:** MIND / EB-NeRD, so với TIGER-lite vs two-tower (`llmenc`) của bạn.
- **Độ khó:** trung bình (RQ-VAE + constrained decode). **Novelty:** cao.

### 💡 P2 — "CF-Rerank": LLM reranker có tín hiệu cộng tác cho news (CoLLM + LLM4Rerank → news)
- **Ghép:** #4 (CoLLM nhét CF) + #5 (LLM reranker) → **news**.
- **Cách:** tầng 1 two-tower (của bạn) lấy top-K; tầng 2 **LLM reranker** nhận *cả text tin* **lẫn CF-embedding
  nhét vào prompt* → rerank + **giải thích**.
- **Vì sao mới:** LLM reranker hiện **không thấy tín hiệu cộng tác**; CoLLM chưa làm cho news.
- **Eval:** MIND (rerank top-K của NRMS/`supermodel`), đo AUC/nDCG + chi phí + chất lượng giải thích.
- **Độ khó:** trung bình (cần LLM + prompt). **Novelty:** trung bình-cao.

### 💡 P3 — "KG-RAG-News": RAG + suy luận trên đồ thị tri thức tin tức (K-RagRec + #8 → news)
- **Ghép:** #6 (KG-RAG) + #8 (reasoning) → **news**.
- **Cách:** dựng **news KG (entity)**; lúc suy luận **retrieve KG-subgraph + user tương tự** → nạp LLM **reason (CoT)**
  vì sao nên gợi ý. Bổ trợ nhánh graph GLORY.
- **Vì sao mới:** RAG/KG-RAG cho **news** còn hiếm; mạnh cho cold-start + explainability.
- **Độ khó:** cao (KG + RAG + LLM). **Novelty:** cao.

### 💡 P4 — "DenoiseGen": đưa differential-attention (của bạn) vào bộ sinh Semantic-ID
- **Ghép:** **kỹ thuật của bạn** (diff-attn denoise, `supermodel`) → **bộ decoder generative (#2)**.
- **Cách:** dùng **differential attention** làm sạch lịch sử click **trước khi** sinh semantic ID → giảm nhiễu
  clickbait làm hỏng chuỗi sinh.
- **Vì sao mới:** mang đóng góp riêng của bạn (denoise) sang dòng generative đang hot → "cầu nối" hai mảng.
- **Độ khó:** trung bình. **Novelty:** trung bình-cao (ghép chính cái bạn đã có).

### 💡 P5 — "DistillNews": distill LLM-encoder → two-tower nhanh (deployable)
- **Ghép:** distillation + #1 → **news serving**.
- **Cách:** teacher = **E5-Mistral/LLM2Vec** (embedding tin + có thể + reasoning); student = **Fastformer two-tower**
  của bạn học khớp → giữ ~chất lượng, **serving rẻ**.
- **Vì sao đáng:** đón trend "làm LLM-rec rẻ"; dễ ra số, rủi ro thấp.
- **Độ khó:** thấp-trung bình. **Novelty:** trung bình.

**Bảng chọn nhanh**

| Ý tưởng | Đón trend 2026 | Novelty | Độ khó | Rủi ro |
|---|---|---|---|---|
| **P1 GenNews** ⭐ | #1 (nóng nhất) | Cao | TB | TB |
| P2 CF-Rerank | #4+#5 | TB-Cao | TB | TB |
| P3 KG-RAG-News | #6+#8 | Cao | Cao | Cao |
| P4 DenoiseGen | #1 + của bạn | TB-Cao | TB | Thấp |
| P5 DistillNews | encoder+distill | TB | Thấp | Thấp |

---

## 6. Đề xuất kỹ thuật chính cho luận văn (khuyến nghị)

**Hướng chính: P1 "GenNews" + lồng P4 (denoise) + so với two-tower `llmenc`.**

- **Đóng góp tuyên bố:** "Semantic-ID generative recommendation cho **news** với tokenizer từ embedding đa ngữ
  (BGE-M3), **khôi phục tín hiệu collaborative** trong semantic ID, và **denoise lịch sử bằng differential
  attention** — nhắm cold-start tốt hơn two-tower/PLM."
- **Baseline phải vượt/so:** NRMS+PLM, CAUM, GLORY, MANNeR (discriminative) + `llmenc`/`supermodel` (của bạn).
- **Kế hoạch eval:** MIND-small (dev nhanh) → MIND-large + EB-NeRD; metric AUC/MRR/nDCG **+ cold-start tách
  riêng** (tin mới) **+ chi phí/độ trễ** (điểm mạnh để so LLM).
- **Ablation:** (a) tokenizer (RQ-VAE vs LETTER); (b) có/không khôi phục collaborative; (c) có/không denoise.
- **Đòn bẩy quan trọng nhất:** chất lượng **encoder** (BGE-M3/E5) — benchmark của bạn đã chứng minh
  `llmenc` một mình rất mạnh; semantic ID xây trên embedding tốt sẽ lợi thế.

**Phương án an toàn (nếu ít GPU/thời gian):** P5 "DistillNews" hoặc P2 "CF-Rerank" — dễ ra số, vẫn có câu chuyện.

---

## 7. Rủi ro & lưu ý đánh giá (đọc trước khi viết kết quả)
- **SOTA gây tranh cãi:** reproduce baseline mạnh (SASRec/NRMS+PLM), đừng chỉ tin số trong paper.
- **Decode generative dễ lỗi** (sinh ra ID không hợp lệ) → cần constrained decoding + xử lý.
- **semantic ⚔ collaborative** xung đột → cần cơ chế hợp nhất (đó cũng là đóng góp).
- **Chi phí/độ trễ** LLM lớn → luôn báo cáo song song với accuracy; cân nhắc distillation.
- **Đánh giá cold-start & beyond-accuracy** (diversity) là nơi dễ ghi điểm học thuật hơn đua AUC.

---

## Nguồn
- Semantic-ID / generative rec: TIGER; LC-Rec; LETTER; universal tokenization
  ([2504.04405](https://arxiv.org/pdf/2504.04405)); scaling view ([2509.25522](https://arxiv.org/pdf/2509.25522));
  "LLMs need encoders for semantic IDs too" ([2606.00324](https://arxiv.org/pdf/2606.00324));
  reliability of tokenizer comparisons ([2605.25330](https://arxiv.org/pdf/2605.25330)).
- Reasoning rec: "Reasoning over Semantic IDs" ([2603.23183](https://arxiv.org/pdf/2603.23183));
  Generative Reasoning Recommendation ([2510.20815](https://arxiv.org/pdf/2510.20815)).
- Tune/CF-inject: TALLRec (RecSys'23); CoLLM (TKDE'25); restoring collaborative in SID
  ([2607.27682](https://arxiv.org/html/2607.27682)).
- RAG/agent: K-RagRec / KG-RAG ([2501.02226](https://arxiv.org/pdf/2501.02226)); RAG-Rec;
  survey LLM-agents for RecSys ([EMNLP'25](https://aclanthology.org/2025.findings-emnlp.620.pdf)).
- News: MIND ([msnews.github.io](https://msnews.github.io/)); GLORY ([2307.06576](https://arxiv.org/pdf/2307.06576));
  survey LLM news-rec ([2502.09797](https://www.arxiv.org/pdf/2502.09797v1)).
- Embedding: LLM2Vec; benchmarking LLMs as semantic encoders ([2403.03952](https://arxiv.org/pdf/2403.03952)).
- Benchmarks: SRBench ([2604.09553](https://arxiv.org/html/2604.09553)); compendium
  ([LLMSearchRecommender](https://github.com/alopatenko/LLMSearchRecommender));
  item-ID gen survey ([Awesome-Item-ID-Gen-RecSys](https://github.com/HKBU-LAGAS/Awesome-Item-ID-Gen-RecSys)).
