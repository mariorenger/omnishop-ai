# newsrec_bench — So sánh NRMS / NAML / Fastformer vs Graph vs LLM trên một khung

Bộ benchmark **tối giản, một khung thống nhất** để so sánh các họ mô hình gợi ý
tin tức trên **cùng dữ liệu, cùng loss, cùng metric** — nên khác biệt trong bảng
kết quả **chỉ đến từ kiến trúc mô hình và tốc độ**, không phải do khác biệt pipeline.

So sánh trên **2 trục**:
- **Chất lượng**: AUC, MRR, nDCG@5, nDCG@10 (đúng định nghĩa MIND leaderboard).
- **Tốc độ / chi phí**: số tham số, thời gian train, thời gian/epoch, throughput suy luận (impressions/giây).

## 6 mô hình được so

| Tên | Họ | Ý tưởng | Kỳ vọng |
|---|---|---|---|
| `nrms` | Content, Transformer | Multi-head self-attention (Wu 2019) | Baseline mạnh, **O(L²)** theo độ dài tiêu đề |
| `naml` | Content, CNN | CNN + multi-view attention (title+category) | Nhanh hơn NRMS, chất lượng tương đương |
| `fastformer` | Content, Linear-attn | Additive attention **O(L)** (Wu 2021) | **Nhanh nhất** nhóm content, chất lượng ~ NRMS |
| `lightgcn` | **Thuần graph** | User/news ID embeddings trên đồ thị click, **không đọc chữ** (He 2020) | Mạnh khi có tín hiệu cộng tác, **sập ở cold-start / tin mới** |
| `llmenc` | **LLM-as-encoder** | Embedding tin **tính sẵn (LLM/sentence-transformer) đóng băng** + head nhẹ (kiểu ONCE/DIRE) | Mạnh khi ít dữ liệu; **online rất rẻ**; offline nặng |
| `hybridopt` | **Tối ưu của repo này** | Fastformer news + **candidate-aware user** (CAUM) + **fusion tín hiệu cộng tác** | Nhắm **vượt NRMS về chất lượng, rẻ hơn, không sập cold-start** |

## Cài đặt

```bash
pip install -r requirements.txt      # torch>=2.0, numpy (chạy được CPU)
```

## Chạy

### 1) Demo có kiểm soát (synthetic — mặc định, chạy CPU vài phút)
```bash
python bench.py --source synthetic --epochs 3
# chọn model:
python bench.py --models nrms fastformer lightgcn hybridopt --epochs 5
```

### 2) MIND thật
Tải **MINDsmall** (hoặc MINDlarge) tại <https://msnews.github.io>, giải nén thành
`MINDsmall_train/` và `MINDsmall_dev/` (mỗi thư mục có `news.tsv`, `behaviors.tsv`):
```bash
python bench.py --source mind \
  --mind-train /data/MINDsmall_train --mind-dev /data/MINDsmall_dev --epochs 4
```

### 3) LLM-encoder "thật" trên MIND (tùy chọn)
Mặc định `llmenc` trên MIND dùng embedding placeholder. Để chạy đúng tinh thần
LLM-as-encoder, tính embedding tiêu đề bằng sentence-transformer/LLM:
```bash
pip install sentence-transformers
python llm_embed.py --news /data/MINDsmall_train/news.tsv --out news_emb.npz
```
rồi nạp `llm_embeddings=` vào `MindData.from_mind(...)` (xem docstring `llm_embed.py`).

Kết quả in ra terminal và ghi vào `results/results.md` + `results/results.json`.

## Cấu trúc

```
newsrec_bench/
├── bench.py       # vòng train/eval/timing chung + CLI + xuất bảng
├── data.py        # MindData: loader MIND thật + generator synthetic + đồ thị + batching
├── models.py      # 6 mô hình sau REGISTRY, chung interface score(batch)->[B,C]
├── metrics.py     # AUC / MRR / nDCG (thuần numpy)
├── llm_embed.py   # (tùy chọn) tính embedding LLM/ST cho tin của MIND
└── requirements.txt
```

## Vì sao so sánh này công bằng

- **Cùng interface**: mọi model xuất `score(batch) -> logits[B, C]`; loss là
  softmax cross-entropy trên các ứng viên (1 dương ở slot 0 + K âm), eval dùng
  cùng metric. Không model nào được "ưu ái" ở phần huấn luyện.
- **Cùng ngân sách**: cùng `--dim`, cùng optimizer/lr, cùng số epoch, cùng seed init.
- **Tốc độ đo tường minh**: `train_s`, `s/epoch`, `infer_impr/s` để thấy đánh đổi
  chất lượng ↔ chi phí — đúng câu hỏi "graph/LLM có đáng so với Fastformer không".

## Dữ liệu synthetic được thiết kế để mọi họ đều có cái để học

Generator (`MindData.synthetic`) tạo dữ liệu **đúng định dạng MIND** với quá trình
sinh **biết trước**, để phép so sánh có ý nghĩa như một *demo có kiểm soát*:

- Mỗi tin có **phân bố chủ đề (topic)** → chi phối các **từ trong tiêu đề** ⇒ các
  model content (NRMS/NAML/Fastformer/LLM) có thể học được "hợp gu chủ đề".
- Có thêm **nhân tố tiềm ẩn user–tin hạng thấp (collaborative)** *không* suy ra được
  từ chữ ⇒ model **thuần graph (LightGCN)** khai thác được.
- **Embedding "LLM" đóng băng** là ảnh tuyến tính có nhiễu của vector chủ đề ⇒ mô
  phỏng embedding câu của sentence-transformer/LLM.

Nhờ vậy: content bắt chủ đề; graph bắt tín hiệu cộng tác; LLM-encoder mạnh khi ít
dữ liệu; **hybrid có cả hai** ⇒ thường đứng đầu, và **không sập cold-start** như graph.

> ⚠️ **Quan trọng**: số tuyệt đối trên synthetic **không so được** với leaderboard
> MIND thật. Nó minh họa *xu hướng tương đối và đánh đổi tốc độ*. Muốn kết luận
> khoa học, chạy `--source mind` trên MINDsmall/large (cùng bộ code).

## `hybridopt` — phần "tự sửa để tối ưu"

Ý tưởng gộp 3 điểm mạnh, nhắm đúng điểm yếu của từng nhóm:
1. **News encoder Fastformer** (additive, **O(L)**) — rẻ hơn self-attention của NRMS.
2. **Candidate-aware user attention** (kiểu **CAUM**): lịch sử đọc được **tái trọng
   số theo từng ứng viên**, thứ mà additive/self-attention thường không làm được.
3. **Fusion tín hiệu cộng tác**: cộng thêm tích vô hướng của embedding ID user–tin
   với **trọng số học được** `exp(log_lambda)` ⇒ có phần "graph", **nhưng vì luôn có
   nhánh nội dung nên tin mới (cold) không bị sập** như LightGCN thuần.

Đây là hiện thực trực tiếp của khuyến nghị trong
`docs/research/news-recommendation-survey-2024-2026.md`: *xương sống content (rẻ) +
một lớp tín hiệu cộng tác/LLM*. Dễ mở rộng: đổi `_news()` sang `llmenc` head để
thành **Hybrid + LLM-encoder**.

## Ghi chú về "paper 2026"

Tính đến 2026, **chưa có mốc son mới cho riêng news recommendation** lật đổ dòng
LLM-as-encoder 2024–2025 (MANNeR, ONCE/DIRE). Động lực 2026 nằm ở **RecSys tổng quát**
rồi lan sang news:
- **Generative recommendation + Semantic IDs**: tokenize item để LLM *sinh thẳng*
  item (generative retrieval); nhiều survey 2025–2026.
- **Làm generative-rec rẻ đi**: ví dụ *MLP distilled generative recommenders* (2026).
- **LLM + Graph augmentation**: *LLMRec* (LLM làm giàu đồ thị tương tác) là hướng lai
  đáng chú ý; **agentic / multi-agent + RAG** cho gợi ý.
- Với news cụ thể: các biến thể **LLM prompting phân cấp** và **mô tả category bằng
  LLM** (2025) tiếp tục cải thiện trên MIND, nhưng vẫn trong khung 2.2–2.3 của survey.

→ Nếu làm luận văn "đón đầu 2026": thử **Semantic ID / generative retrieval cho news**
hoặc **LLM+graph augmentation** — hai khoảng trống còn ít người khai thác trên MIND.

## Ý nghĩa các cột kết quả

| Cột | Ý nghĩa |
|---|---|
| `auc/mrr/ndcg@k` | Chất lượng xếp hạng (cao hơn = tốt hơn) |
| `params` | Số tham số học được (chi phí bộ nhớ/độ phức tạp) |
| `train_s`, `s/epoch` | Thời gian huấn luyện (thấp hơn = rẻ hơn) |
| `infer_impr/s` | Throughput suy luận trên dev (cao hơn = phục vụ nhanh hơn) |

## Mở rộng gợi ý
- Thêm model: hiện thực `score(batch)->[B,C]`, đăng ký vào `REGISTRY` trong `models.py`.
- Thêm metric beyond-accuracy (diversity/coverage) trong `metrics.py`.
- Đổi negative sampling / số âm `n_neg`, độ dài lịch sử `max_hist`, `--dim` để khảo sát.
