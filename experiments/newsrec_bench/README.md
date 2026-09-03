# newsrec_bench — Base transformers vs cải tiến attention 2024–2026 trên một khung

Bộ benchmark **một khung thống nhất**: mọi model xuất `score(batch) -> logits[B, C]`,
cùng loss (softmax cross-entropy trên ứng viên) và cùng metric — nên khác biệt trong
bảng **chỉ đến từ kiến trúc và tốc độ**. So trên **2 trục**:

- **Chất lượng**: AUC, MRR, nDCG@5, nDCG@10 (định nghĩa như MIND leaderboard).
- **Tốc độ / chi phí**: `params`, `train_s`, `s/epoch`, `infer_impr/s`.

## Các model

> **Chốt lại (không dàn trải):** mặc định chạy **bộ CORE 6 model** — vài base top +
> LLM-encoder + **1 `supermodel` gộp mọi cải tiến**. Các variant lẻ (diff/mla/ssm/rope/
> multi/graph/contrastive) vẫn có trong `REGISTRY` như **menu ablation** (`--models improved`
> hoặc gọi tên), để soi từng cải tiến khi cần chứ không làm loãng bảng chính.

### CORE — bộ mặc định (`--models core`)
| Tên | Họ | Ghi chú |
|---|---|---|
| `nrms` | Transformer (multi-head self-attn) | Baseline mạnh, O(L²) theo độ dài tiêu đề |
| `fastformer` | Additive/linear attention O(L) | Base "Fastformer" — nhanh nhất nhóm content |
| `caum` | Candidate-aware user modeling | **Top MIND** trong nhóm content reproducible |
| `lightgcn` | **Thuần graph** (không đọc chữ) | Tham chiếu cộng tác / cold-start |
| `llmenc` | **LLM-as-encoder** — embedding **BGE-M3 / Jina** (nhẹ, mạnh) đóng băng + head nhẹ | Tham chiếu hướng LLM; online rất rẻ |
| **`supermodel`** | **Gộp HẾT** cải tiến (xem dưới) | Model "siêu cải tiến" để **đọ với `llmenc`** |

*(`naml` cũng có sẵn — thêm bằng `--models core naml` hoặc `--models base`.)*

### Menu ablation — từng cải tiến theo hướng attention LLM 2024–2026 (`--models improved`)
Tất cả dùng **chung news encoder NRMS**, chỉ thay **cơ chế user encoder** → cô lập đúng phần cải tiến.

| Tên | Cải tiến (nguồn) | Giải quyết gì | Loại |
|---|---|---|---|
| `nrms_diff` | **Differential Attention** (DIFF Transformer, ICLR 2025) | Trừ 2 softmax-map → **triệt nhiễu click** trong lịch sử | Chất lượng |
| `nrms_mla` | **Multi-head Latent Attention** (DeepSeek-V2/V3) | Nén KV low-rank → rẻ bộ nhớ/serving, chất lượng ≈/hơn MHA | Tốc độ |
| `nrms_ssm` | **Selective SSM** (Mamba4Rec, KDD-RelKD'24) | Hồi quy tuyến tính **O(H)** cho lịch sử dài | Tốc độ |
| `nrms_rope` | **RoPE + recency bias** (RoPE/ALiBi) | Thêm **vị trí** cho token tiêu đề + ưu tiên tin đọc gần đây | Chất lượng |
| `nrms_multi` | **Multi-interest poly-attention** (MINS/MINER) | Nhiều vector sở thích; chấm điểm theo interest khớp nhất | Chất lượng |
| `graphrec` | **Graph fusion** — content + **LightGCN GNN** message passing | Gộp nội dung + tín hiệu cộng tác bậc cao (GNN thật, khác `hybridopt` chỉ dùng id) | Chất lượng |
| `nrms_cl` | **Contrastive learning** (InfoNCE/CL4SRec-style aux) | 2 view dropout của user kéo lại gần, user khác đẩy xa → biểu diễn user chắc hơn | Chất lượng |
| `hybridopt` | Fastformer + candidate-aware + **fusion cộng tác (id)** | Gộp content (rẻ) + graph nhẹ; **không sập cold-start** | Cả hai |
| **`supermodel`** | **Gộp HẾT**: diff-attn denoise + candidate-aware + **graph GNN** + **contrastive** (+ tùy chọn **news embedding BGE/Jina** qua `--news pretrained`) | "Siêu cải tiến" để **đọ với `llmenc`** | Cả hai |

> `nrms_ssm` là SSM chọn lọc **nhẹ bằng PyTorch thuần** (không phải Mamba CUDA kernel) — bắt
> đúng ý tưởng "tuyến tính, long-range", chạy được trên CPU.

## Cài đặt

```bash
pip install -r requirements.txt        # torch>=2.0, numpy
# CPU-only gọn hơn:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Chạy

```bash
# MẶC ĐỊNH: bộ CORE gọn (nrms, fastformer, caum, lightgcn, llmenc, supermodel):
python bench.py --epochs 4

# menu ablation từng cải tiến / toàn bộ 15 model:
python bench.py --models improved --epochs 4
python bench.py --models all --epochs 4

# "siêu cải tiến" đọ với LLM-base, và bản dùng news embedding BGE/Jina:
python bench.py --models supermodel llmenc caum --epochs 5
python bench.py --models supermodel --news pretrained --epochs 5   # supermodel + BGE/Jina news emb

# vài model cụ thể:
python bench.py --models nrms fastformer nrms_diff nrms_ssm graphrec nrms_cl --epochs 5

# MIND thật (tải MINDsmall ở https://msnews.github.io):
python bench.py --source mind \
  --mind-train /data/MINDsmall_train --mind-dev /data/MINDsmall_dev --epochs 4
```

Cờ hữu ích: `--dim 64 --heads 2 --batch-size 64 --lr 1e-3 --dropout 0.2`,
`--gcn-layers 2` (lightgcn/graphrec/supermodel), `--latent 16` (nrms_mla), `--n-interest 4` (nrms_multi),
`--cl-tau 0.1 --cl-weight 0.1` (nrms_cl/supermodel contrastive),
`--syn-users/--syn-news/--syn-train/--syn-dev` (kích thước synthetic), `--device cuda`.
Kết quả in ra + ghi `results/<out>/results.md` và `results.json`.

## ⏱️ Ước tính thời gian

Đo trên **CPU 4 nhân** (không GPU), bộ code này:

| Cấu hình | Lệnh | Tổng | s/epoch mỗi model |
|---|---|---|---|
| Smoke (400u/600n/1.2k, dim32, 1 ep) | config nhỏ | ~30 s (15 model) | ~2 s |
| **CORE demo** (1.2k/2k/4.5k, dim48, 5 ep) | `--models core` | **~4–5 phút (6 model)** | llmenc/lightgcn <1 · fastformer ~7 · nrms/caum ~10 · supermodel ~18 |
| All 15 model (1.2k/2k/4.5k, dim48, 5 ep) | `--models all` | ~12–18 phút | như trên + các variant ~8–12 |

**MIND thật (CPU):** MINDsmall có ~156k impression → sinh vài trăm nghìn mẫu train;
trên CPU **chậm (nhiều giờ/epoch cho model content)** → **khuyến nghị GPU**.
Trên 1 GPU tầm trung: model content ~**vài phút/epoch** trên MINDsmall; `lightgcn`/`llmenc`
nhanh hơn nhiều (không encode text online). MINDlarge nên chạy GPU + tăng `--batch-size`.

Mẹo tăng tốc CPU: giảm `--dim`, `--syn-*`, `--epochs`; chạy `--models improved` riêng;
đặt `--threads 4`. Lưu ý: `caum` mở rộng tensor `[B,C,H,D]` khi eval — với impression rất
lớn của MIND, giảm `--eval-batch` (hoặc bỏ `caum`) để tránh tốn RAM; `supermodel` không bị.

## Kết quả ví dụ (synthetic demo, **đã chạy thật trên CPU** — `results/synthetic_demo/`)

Lệnh: `python bench.py --models core --epochs 5 --dim 48 --syn-users 1200 --syn-news 2000
--syn-train 4500 --syn-dev 1200` (CPU 4 nhân, ~4–5 phút). Models **học rõ và phân biệt** (AUC >> 0.5):

| model | auc | mrr | ndcg@5 | ndcg@10 | params | s/epoch | infer impr/s |
|---|---|---|---|---|---|---|---|
| `llmenc` (BGE/Jina emb) | **0.776** | 0.234 | 0.778 | 0.692 | 15,472 | **0.65** | **16,130** |
| `supermodel` (gộp hết) | 0.741 | 0.203 | 0.647 | 0.576 | 238,065 | 17.96 | 686 |
| `caum` (top MIND) | 0.731 | 0.193 | 0.605 | 0.545 | 71,969 | 9.61 | 454 |
| `nrms` | 0.718 | 0.187 | 0.581 | 0.523 | 77,264 | 9.82 | 674 |
| `fastformer` | 0.713 | 0.183 | 0.576 | 0.515 | 77,364 | 7.18 | 765 |
| `lightgcn` (thuần graph) | 0.622 | 0.125 | 0.342 | 0.328 | 153,600 | 0.98 | 10,872 |

**Đọc bảng (câu chuyện đúng thực tế):**
- **`llmenc` với embedding pretrained (BGE-M3/Jina) dẫn đầu chất lượng VÀ rẻ/nhanh nhất** (15k params,
  0.65 s/epoch, 16k impr/s) → *một embedding tốt rất khó bị vượt và cực rẻ khi serving*.
- **`supermodel` (gộp diff-attn + candidate-aware + graph + contrastive) hạng 2**, **vượt mọi content
  base** (caum/nrms/fastformer) → cộng dồn cải tiến CÓ tác dụng; nhưng **vẫn thua `llmenc`** dù nặng
  hơn ~15× params và ~28× chậm hơn/epoch → đánh đổi không đáng nếu đã có embedding tốt.
- **`fastformer` nhanh nhất nhóm content** (7.18 vs 9.82 s/epoch của nrms), chất lượng suýt soát.
- **`lightgcn` thuần graph thấp nhất** (không đọc chữ trên task topic-driven) nhưng rất nhanh —
  minh hoạ: graph một mình không đủ cho news, cần content.

> ⚠️ Đây là **synthetic có kiểm soát** — kết luận "embedding pretrained thắng" ở đây khớp với xu hướng
> thực tế (MANNeR/ONCE), nhưng số tuyệt đối **không so được với MIND thật**; chạy `--source mind` để chốt.

Cách đọc: cột chất lượng (auc/mrr/ndcg) cao hơn = tốt hơn; `params`/`train_s` thấp hơn = rẻ hơn;
`infer_impr/s` cao hơn = phục vụ nhanh hơn (graph/LLM-enc rất nhanh vì không encode text online).

> ⚠️ **Số trên synthetic KHÔNG so được với leaderboard MIND thật** — nó minh họa *xu hướng
> tương đối + đánh đổi tốc độ/cold-start*. Kết luận khoa học: chạy `--source mind`.

## Dữ liệu synthetic được thiết kế để mọi họ đều có cái để học

`MindData.synthetic` tạo dữ liệu **đúng định dạng MIND** với quá trình sinh **biết trước**:
- Mỗi tin có **1 topic chính** → chi phối **từ trong tiêu đề**, **category**, và **embedding LLM** ⇒ content học được rõ.
- User có **sở thích peaked** trên vài topic; **click chủ yếu do topic**: tin đúng gu ~85% click, lệch gu ~6% ⇒ tín hiệu xếp hạng học được mạnh (không phải tung đồng xu).
- Thêm **bump cộng tác nhỏ** từ nhân tố tiềm ẩn user–tin ⇒ chừa phần cho **graph**.
- **Embedding "LLM" đóng băng** = ảnh tuyến tính có nhiễu của topic ⇒ mô phỏng BGE/Jina.

⇒ content bắt topic; graph bắt cộng tác + popularity; LLM-enc (embedding topic gần chuẩn) mạnh & rẻ.
Kiểm chứng: oracle topic đạt AUC ~0.79, `llmenc`≈0.79, `nrms`≈0.73, `lightgcn`≈0.73 (rõ ràng >> 0.5).

## Hai nhóm hướng cải thiện (đã hiện thực / gợi ý tiếp)

**Nhóm A — sửa attention theo LLM 2024–2026** (đã code: `nrms_diff`, `nrms_mla`, `nrms_ssm`, `nrms_rope`):
denoise (differential), nén KV (MLA), tuyến tính long-range (SSM/Mamba), vị trí (RoPE/recency).
Gợi ý thêm: **sliding-window/global** cho lịch sử rất dài, **gated attention / attention-sink**.

**Nhóm B — cải thiện model cũ KHÔNG đụng attention** (thường ăn tiền hơn):
- **Graph** (đã code `graphrec`, `supermodel`): fuse **LightGCN GNN propagation** vào content two-tower → tín hiệu cộng tác bậc cao + đỡ cold-start. Mở rộng: DIGAT/GLoCIM, KG+GNN.
- **Contrastive / objective** (đã code `nrms_cl`, `supermodel`): **InfoNCE self-supervised** (2 view dropout). Mở rộng: in-batch negatives thay softmax-candidate, hard-negative mining, listwise; **debiasing** (IPS, popularity).
- **Biểu diễn tin**: GloVe → **PLM (BERT/RoBERTa)** hoặc **frozen LLM embeddings** (`llmenc`); feature do LLM sinh (GENRE); entity/KG (DKN).
- **User modeling**: **multi-interest** (`nrms_multi`), **candidate-aware** (`caum`, `hybridopt`), long/short-term (LSTUR).
- **Popularity & cold-start**: PP-Rec (popularity-aware), time-aware. · **Distillation**: LLM/PLM teacher → student nhỏ.

**`supermodel` = gộp hết** (diff-attn denoise + candidate-aware + graph GNN + contrastive) — dựng để trả lời trực tiếp câu "kết hợp hết lại có mạnh hơn `llmenc` không". Chạy `--models supermodel llmenc caum` để so.

## Ghi chú "paper 2026"

Chưa có mốc son riêng cho news-rec 2026 lật đổ dòng LLM-encoder 2024–2025. Động lực 2026 ở RecSys
tổng quát rồi lan sang news: **generative recommendation + Semantic IDs**, **làm generative-rec rẻ**
(MLP distilled), **LLM+Graph augmentation** (LLMRec), **agentic/RAG**. → Khoảng trống cho luận văn:
đưa **Semantic-ID/generative-retrieval** hoặc **DIFF-attn/Mamba** lên **MIND** rồi đo.

## Cấu trúc & mở rộng

```
newsrec_bench/
├── bench.py     # train/eval/timing + CLI (all/base/improved) + xuất bảng
├── data.py      # MindData: loader MIND thật + generator synthetic + đồ thị + batching
├── models.py    # 15 model sau REGISTRY (BASE_MODELS / IMPROVED_MODELS), chung score()
├── metrics.py   # AUC / MRR / nDCG (thuần numpy)
├── llm_embed.py # (tùy chọn) tính embedding LLM/ST cho tin của MIND
└── requirements.txt
```

Thêm model: hiện thực `score(batch)->[B,C]` (tùy chọn `_encode` của `_ContentTwoTower`),
đăng ký vào `REGISTRY` trong `models.py`. Thêm metric beyond-accuracy (diversity/coverage) trong `metrics.py`.

## Nguồn
- Differential Transformer — <https://arxiv.org/abs/2410.05258>
- Mamba4Rec — <https://github.com/chengkai-liu/Mamba4Rec> · SSD4Rec <https://arxiv.org/html/2409.01192v2>
- DeepSeek-V3 / MLA — <https://arxiv.org/abs/2412.19437>
- CAUM (Candidate-aware User Modeling) — Qi et al., IJCAI 2022
- NRMS / NAML — Wu et al., 2019 · Fastformer — Wu et al., 2021 · LightGCN — He et al., 2020
- MIND dataset — <https://msnews.github.io>
