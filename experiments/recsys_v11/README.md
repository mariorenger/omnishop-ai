# V11 — bộ so sánh đầy đủ các hướng LLM cho news recommendation (MIND-small)

**Một lần Run All trên Kaggle (T4/P100) ≈ 6.5–8.5 giờ, ngân sách đồng hồ cứng 9.5 giờ** cho ~13 họ kỹ thuật / ~100 biến thể,
tất cả đo trên đúng giao thức V10, có khoảng tin cậy ghép cặp, lát cắt tin mới/tin cũ và hiệu chỉnh Bonferroni.
Mục tiêu không phải tối đa hoá điểm, mà là **biết hướng LLM nào đáng làm luận văn, bằng bằng chứng có CI**.

> ⚠️ **Trạng thái trung thực.** Toàn bộ code đã được kiểm thử đầu-cuối **trên CPU** với dữ liệu đúng định dạng MIND và các mô hình
> ngẫu nhiên tí hon (BERT, Qwen3). **Chưa có con số nào trên MIND thật, chưa đo thời gian trên GPU, chưa chạy với weight thật
> của BGE / Qwen3** (sandbox phát triển không có GPU, không tải được weight, không có MIND). Mọi thời gian ở dưới là **ước lượng từ FLOPs**.
> Số liệu quyết định hướng đi do **lần chạy Kaggle của bạn** tạo ra.

## 1. Giả thuyết ↔ kỹ thuật ↔ paper ↔ mức độ cài đặt

| | Kỹ thuật đã cài | Dòng paper (venue đã tra cứu) | **Chưa cài** (nói rõ để không nhận vơ) | Stage |
|---|---|---|---|---|
| **H1** | History-as-text: ghép 10 tiêu đề gần nhất thành 1 query *có instruction*, embed bằng cùng encoder (zero-shot), ± gộp với mean-pool | Qwen3-Embedding (arXiv 2506.05176, instruction-aware); language-based user profile | — | `histquery` |
| **H2** | Co-click contrastive adaptation: residual head (lin/mlp), **LoRA / DoRA** trong vòng lặp encoder, **Matryoshka (MRL)**, **hard negative theo impression**, chọn epoch theo validation (epoch 0 = frozen cũng được chọn) | LLM2Rec (KDD'25); MRL (NeurIPS'22), fMRLRec (EMNLP-Findings'24), SMEC (EMNLP'25); DoRA (ICML'24) | LLM2Rec đầy đủ (pretrain nhiều giai đoạn) | `head`, `encoder` |
| **H3** | RAG ở mức *biểu diễn*: bổ sung bài bằng context của **thực thể Wikidata** hoặc **k bài cũ gần nhất** (bộ nhớ nhân quả) | RAG; K-RagRec (ACL'25) | LLM đọc ngữ cảnh truy xuất (xem H4b/H6) | `graph` |
| **H4** | GraphRAG-lite: cộng đồng **Louvain** trên đồ thị kNN bài báo + truy xuất *người dùng tương tự* | Microsoft GraphRAG (arXiv 2404.16130); *"GraphRAG thường kém RAG thường"* (arXiv 2506.05690) ⇒ phải đo | Knowledge graph đầy đủ | `graph` |
| **H4b** | **GraphRAG có LLM**: Qwen3-1.7B viết tóm tắt 1 câu cho từng community (từ 8 tiêu đề gần tâm nhất); vector tóm tắt thay centroid / làm context bài báo | Microsoft GraphRAG (community summaries) | Global-search trả lời truy vấn | `graphrag_llm` |
| **H5** | **LLM làm user encoder, tinh chỉnh LoRA**: query = lịch sử dạng text, vector = hidden state token cuối, InfoNCE với bài được click tiếp theo (document tower đóng băng, in-batch + hard negative theo impression). Ba mask: **causal**, **bidirectional** (mask 4D tự dựng, kiểu LLM2Vec), **soft** (cộng `log λ` lên vị trí tương lai, λ: 0→1 trong nửa đầu huấn luyện). Có kiểm tra *runtime* xem model có tôn trọng mask 4D không; BGE chạy `ut_native` làm đối chứng encoder 2 chiều | nghiên cứu mask của arXiv 2602.10622 (Ant Group, 2/2026); LLM2Vec (COLM'24) | ⚠️ **`soft` chỉ là phần *scheduler tuyến tính*** của bài 2602.10622 (bài gọi đó là baseline "scheduler-only"). **Gradient-Guided Soft Masking** (pre-warmup bằng gradient từ "left tower" đóng băng) và mask *hybrid* **chưa cài** | `llm_user` |
| **H6** | **KAR-style**: LLM mở rộng từng bài (chủ đề + độc giả quan tâm) và viết **hồ sơ độc giả** từ 10 tiêu đề; embed rồi trộn (`E0 + α·E_gen`) / dùng làm query | KAR (RecSys'24); LLM-UP (2026) | Factorization prompting + hybrid-expert adaptor huấn luyện cùng CTR model (ở đây dùng zero-shot) | `augment` |
| **H7** | **Semantic IDs**: RQ-KMeans 3 tầng × 256 (fit trên nội dung bài đã quan sát, bài mới gán bằng centroid gần nhất). (a) `sid_profile`: độ trùng tiền tố mã giữa lịch sử và ứng viên (tầng thô khái quát hoá cho tin mới). (b) **SID-GPT** (transformer nhân quả 4 tầng, next-code-token trên chuỗi đọc) dùng làm **slate ranker**: `log P(mã ứng viên | lịch sử)`, có biến thể PMI trừ `log P(mã ứng viên)` | TIGER (NeurIPS'23); OneRec Technical Report (arXiv 2506.13695, RQ-Kmeans) | Tokenizer *học được* (LETTER, CIKM'24); generative **retrieval** (MIND là xếp hạng slate cho sẵn) | `sid` |
| **H8** | Pooling: trọng số **recency** `exp(-λ·tuổi)`, **MaxSim** (late interaction), top-k, log-sum-exp | ColBERT (SIGIR'20) | — | `pooling` |
| **H9** | **LLM judge zero-shot** xếp lại top-10 của mean-pool: log-odds `Yes−No` token đầu tiên của câu trả lời (Qwen3-1.7B, tắt thinking), prompt = 8 tiêu đề gần nhất + tiêu đề/chuyên mục ứng viên; ± gộp với điểm gốc | LLMRank "LLMs are Zero-Shot Rankers" (ECIR'24); LLM4Rerank (WWW'25) | LLM4Rerank dạng CoT/đồ thị nhiều mục tiêu; listwise | `rerank` |
| **H10** | User model **học được** = bản sao V10 `llmenc_ca` (đã kiểm: điểm số giống hệt mã V10) trên từng biểu diễn (frozen / head tốt nhất / LoRA tốt nhất / entity_rag / kNN-RAG) — cải thiện biểu diễn có sống sót khi đã có user model không? | V10 | — | `usermodel` |
| — | Quét encoder: TF-IDF+LSA (sàn từ vựng), BGE small/base/large, Qwen3-Embedding-0.6B | MTEB; Qwen3-Embedding | Qwen3-Embedding-4B/8B **chưa hỗ trợ** (cần nạp fp16; với mã hiện tại nạp fp32 sẽ hết bộ nhớ T4 và stage `sweep` chỉ ghi log rồi bỏ qua) | `sweep` |

Các venue/tên bài được **tra cứu lại bằng web trong phiên này**: arXiv 2602.10622, LLM2Vec (COLM'24), LLM4Rerank (WWW'25), TIGER (NeurIPS'23), LETTER (**CIKM'24**, không phải SIGIR'25 như ghi nhầm trước đây),
OneRec (RQ-Kmeans), LLMRank (ECIR'24), KAR (RecSys'24), arXiv 2506.05690. Các mục còn lại lấy từ khảo sát trước trong `docs/research/`.

## 2. Giao thức (chống tự lừa mình)

* **Y hệt V10**: `data.py`/`metrics.py` vendor nguyên văn (`PROVENANCE.md`); validation tách theo ngày, `MINDsmall_dev` là test, chỉ chạm **một lần** cho cấu hình đã chọn trên validation. Evaluator vector hoá khớp `metrics.py` tới ~1e-8 (slate có điểm bằng nhau được giao lại cho `metrics.py`).
* **Bộ nhớ nhân quả** cho RAG/GraphRAG: validation chỉ dùng nội dung bài đã quan sát trong `train_core`; test thêm validation. Chỉ dùng nội dung (embedding, thực thể), không dùng nhãn click. *Ngoại lệ có chủ đích:* codebook RQ-KMeans (H7) fit trên nội dung bài đã quan sát ở train+validation cho cả hai split (chỉ ảnh hưởng việc chọn cấu hình trên validation; bài chỉ xuất hiện ở test được gán mã bằng centroid gần nhất, đúng như hệ thống thật xử lý tin mới).
* **Mọi so sánh kèm CI 95% ghép cặp theo impression** (positive cùng impression gộp cụm). Dòng chạy trên **tập con ngẫu nhiên lồng nhau** (H1 ở run B, H5, H6, H9) luôn được so với control **trên đúng các impression đó** (cột `ref nDCG@10`), không so con số tuyệt đối với dòng chạy toàn bộ.
* **Chọn đại diện từng họ theo validation** (`val_delta` = chênh lệch ghép cặp trên validation, nên hợp lệ cả với tập con), không theo test; kết luận có **Bonferroni** theo số họ. **HEAD-TO-HEAD** giữa top-5 họ kỹ thuật (xếp theo validation), Bonferroni theo số cặp.
* **Lát cắt tin mới/tin cũ**: *cold* = bài được click mà các dòng train không hề quan sát — nơi embedding LLM lẽ ra phải thắng.
* **Kiểm tra tái lập**: `frozen` phải khớp V10 `bge_zeroshot` (AUC 0.6241, nDCG@10 0.3909) và `um_frozen` phải gần V10 `llmenc_ca` (0.6494 / 0.3997, dung sai 0.01 vì V10 chỉ 1 seed). (V10 embed bằng Sentence-Transformers với `max_seq_length` mặc định 512 của BGE; ở đây cắt 256 token — chênh lệch không đáng kể vì gần như mọi bài MIND < 256 token; nếu `MISMATCH` thì thử `--max-len 512`.) **Guard pooling**: `HFEncoder` được so với Sentence-Transformers trên 64 bài trước khi chạy (bắt lỗi pooling/EOS/prefix, quan trọng với Qwen3 dùng last-token pooling).
* **Leaderboard chung** (một control toàn cục = BGE-small frozen) + `forest.png`; hàng H10 bị loại khỏi leaderboard vì có control riêng (`um_frozen`).
* **Ngân sách đồng hồ chung** (`--budget-hours`): vòng lặp dài nhận `deadline` và dừng êm, stage nào không còn thời gian bị bỏ qua, kết quả được ghi sau *từng* stage; gọi lại sẽ **resume** (`state.pkl`, chỉ khi dữ liệu *và mọi tham số ảnh hưởng kết quả* giống hệt — QUICK không bao giờ lẫn vào bản đầy đủ). Dòng bị cắt giữa chừng được gắn cờ (`truncated`, `queries_missing`) để không so sánh nhầm.

## 3. Chạy

**Kaggle:** upload `v11_full_suite_v4.ipynb` → Settings: GPU + Internet On → *Add Input* dataset MIND có **cả** `MINDsmall_train` và `MINDsmall_dev` (vd `thinhhuynh3108/mindsmall`) → Run All.
Lần đầu nên `QUICK=1` (biến môi trường `V11_QUICK=1` hoặc sửa ô cấu hình; phần BGE đã đo thật ≈ 13 phút trên Kaggle, phần Qwen3/LLM chưa đo (ước ~30–60 phút vì phải encode 65k bài bằng Qwen3, chạy Louvain và tải model); mục đích bắt lỗi môi trường: tải model, bộ nhớ GPU, phiên bản `transformers`/`peft`, mask 4D, chat template). Chỉ thử BGE: `V11_RUN_B=0`.

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `V11_QUICK` | 0 | bản rút gọn |
| `V11_BUDGET_H` | 9.5 | ngân sách đồng hồ chung (giờ); đặt 6 nếu muốn ngắn hơn |
| `V11_ENCODER` / `V11_ENCODER_B` | `BAAI/bge-small-en-v1.5` / `Qwen/Qwen3-Embedding-0.6B` | encoder run A / run B |
| `V11_GEN_MODEL` | `Qwen/Qwen3-1.7B` | LLM sinh văn bản / giám khảo (có thể thử `Qwen/Qwen3-4B-Instruct-2507`, chậm ~2.5×) |
| `V11_RUN_B` | 1 | 0 = bỏ phần LLM (chỉ còn run A, ~3 giờ) |
| `V11_REFERENCE` | 1 | 0 = bỏ bước so HFEncoder với Sentence-Transformers |
| `V11_SWEEP` | bge-base, bge-large, Qwen3-Emb-0.6B | danh sách encoder quét |

**Local:**
```bash
pip install torch transformers peft scipy networkx scikit-learn sentence-transformers matplotlib
python selftest.py            # ~2 phút CPU, không cần dữ liệu thật
python suite.py --mind-train /data/MINDsmall_train --mind-dev /data/MINDsmall_dev --work out --stages controls,pooling,graph,head,histquery,sid
python make_notebook.py       # sinh lại notebook (đổi NOTEBOOK trong file để tránh cache của Kaggle)
```

## 4. Thời gian ước tính (T4; chỉ có một số đo thật: QUICK phần BGE ≈ 13 phút — nhanh hơn ước lượng, nên các con số dưới đây có thể dư; mỗi run in `STAGE TIMINGS` và lưu `timings.json` để hiệu chỉnh)

| Phần | Nội dung | Ước tính |
|---|---|---|
| **A-core** (BGE-small) | E0 3' · controls 1' · pooling 3' · graph (Louvain ×2, RAG, user-RAG) 20–30' · head (6×3 seed) 20' · histquery 12' · sid (RQ-KMeans, SID-GPT, chấm cả slate) 20–25' | 1.1–1.5 h |
| **B** (Qwen3-Embedding-0.6B + Qwen3-1.7B) | E0 15' · histquery (tập con) 11' · **llm_user 3 mode × 32–40'** · rerank ≤35' · augment ≤35' · graphrag_llm 15' · pooling+graph+head 35' · usermodel 22' | 3.8–4.8 h |
| **A-phần còn lại** | encoder LoRA/DoRA ×3 ≈ 60' · usermodel (5 view × 2 seed) 35' · sweep (bge-base 3', bge-large 10', Qwen3 đã cache) 18' | 1.6–2.2 h |
| **Tổng** | | **≈ 6.5–8.5 h** (chặn ở 9.5 h) |

Nếu GPU chậm hơn dự kiến, thứ tự ưu tiên là: A-core → B → A-phần còn lại, nên thứ bị bỏ trước là LoRA/DoRA, sweep, user model của run A. Muốn chắc chắn dưới 6 giờ: `V11_BUDGET_H=6`.

## 5. Cách đọc kết quả → hướng luận văn

| Kết quả (CI loại trừ 0 trên test, sau Bonferroni) | Hướng nên theo |
|---|---|
| **H5** thắng (nhất là `ut_soft`/`ut_bidir` > `ut_causal`) | "Tinh chỉnh LLM làm user encoder + lịch trình attention mask" — đúng dòng 2026 |
| **H7** thắng, nhất là trên *cold* | "Semantic ID cho cold-start tin tức; generative model dùng làm slate ranker" |
| **H3/H4/H4b** thắng trên *cold* | "Retrieval-/Graph-augmented representation cho tin mới", mở rộng bằng LLM đọc ngữ cảnh |
| **H6/H9** thắng | "LLM-as-augmenter / LLM-as-reranker" — nhớ nêu chi phí suy luận |
| **H2** thắng | "Collaborative-aware adaptation của embedding LLM" |
| **H10** không còn thắng | cải thiện biểu diễn bị user model học được "nuốt" — kết luận âm quan trọng |
| Không hướng nào | **Kết quả âm có giá trị**: mean-pool của embedding đóng băng đã gần trần trên MIND-small ⇒ chuyển sang MIND-large / EB-NeRD |

## 6. Đã kiểm chứng ở đây / chưa kiểm chứng

✅ `selftest.py` (CPU, dữ liệu định dạng MIND, BERT + Qwen3 ngẫu nhiên tí hon): evaluator == `metrics.py`; head là identity ở bước 0; adapter không thấp hơn frozen trên validation; cờ `hardneg`/MRL không phải no-op; trainer LoRA học thật;
**mask**: causal không nhìn tương lai, bidirectional có nhìn, soft nội suy, batch left-pad == từng chuỗi riêng lẻ (cả 3 mode), mask 4D causal == đường mặc định của HF; tower học (loss giảm) ở cả 3 mode;
**Semantic ID**: k-means, chấm điểm theo batch == tham chiếu từng chuỗi (kể cả lịch sử rỗng), PMI(lịch sử rỗng)=0, học được quy luật "bài kế tiếp" tất định (AUC 1.0);
**generator**: greedy không phụ thuộc batch, cache JSONL, deadline; Yes/No log-odds batch == đơn lẻ; **so sánh trên tập con** == cắt thủ công; `ScoreCache`/z-fusion bảo toàn thứ hạng; clone `LLMEncCA` cho điểm **giống hệt** mã V10;
**resume** (gọi lần 2 bỏ qua stage đã xong, dùng lại `kv`/records, không trùng dòng) và **ngân sách** (ngân sách ~0 → bỏ qua mọi stage tuỳ chọn nhưng vẫn ghi kết quả);
mô phỏng "Run All" của notebook (QUICK) từ thư mục trống trên fixture: 3 lần gọi `suite.main`, không stage nào lỗi, có `leaderboard.md` + `forest.png`.

❌ **Chưa kiểm chứng:** mọi hiệu năng trên MIND thật; thời gian trên T4/P100; fp16/AMP + LoRA/GradScaler trên GPU thật; weight thật của BGE/Qwen3 (guard pooling chạy lúc runtime); hành vi mask 4D với phiên bản `transformers` trên Kaggle (có probe runtime, variant nào fail sẽ bị bỏ qua và ghi log);
tốc độ sinh văn bản của Qwen3-1.7B; thời gian Louvain trên ~50k bài; bộ nhớ GPU khi chạy tower (có tự giảm batch khi OOM).

## 6b. Sự cố môi trường đã gặp khi chạy thật

* **`ImportError: Found an incompatible version of torchao`** (peft từ chối torchao 0.10 có sẵn trong ảnh Kaggle ngay khi gắn LoRA; lộ ra ở stage `encoder`, và sẽ chặn cả `llm_user`): đã vá bằng `common.fix_peft_torchao()` (coi torchao là không có — ta không dùng nó), có test tái hiện bằng torchao giả trong `selftest.py`.
* Dòng `WORSE` với Δ = −0.0000 [−0.0000, −0.0000] ở các head không học được gì (adapter giữ nguyên epoch 0 = frozen): do làm tròn float32/float64 giữa dòng mới và dòng tham chiếu; đã sửa, giờ in `identical`.

## 7. Hạn chế cần nói trong luận văn

Một dataset, một tập dev. Tower/LoRA/DoRA/SID-GPT mặc định **1 seed** (chênh lệch nhỏ hơn ~0.005 nDCG@10 giữa các mask có thể là nhiễu huấn luyện). Dòng chạy tập con (H5/H6/H9) có CI rộng hơn: chỉ phát hiện được hiệu ứng lớn.
`soft` ≠ Gradient-Guided Soft Masking. Giám khảo LLM 1.7B pointwise zero-shot khá yếu so với các hệ rerank mạnh. Entity graph dựa trên chú thích Wikidata của MIND. Popularity/CTR online **cố ý không** dùng.
Đo ở mức biểu diễn zero-shot cho phần lớn họ; H10 là cầu nối sang user model học được nhưng vẫn chỉ là một kiến trúc (V10 `llmenc_ca`).

## 8. Bản đồ file

`data.py`/`metrics.py` (V10, nguyên văn) · `common.py` (đồng hồ ngân sách, fusion, `ScoreCache`, `HistQueries`) · `rep_eval.py` (evaluator vector hoá, CI ghép cặp, so sánh qua tập con) ·
`semgraph.py` (RAG/GraphRAG-lite) · `adapt.py` (head, LoRA/DoRA, MRL) · `pooling.py` (H8) · `sid.py` (H7) · `llm_user.py` (H5) · `llm_gen.py` (sinh văn bản, giám khảo) · `usermodel.py` (H10) ·
`stages_ext.py` (các stage mới) · `suite.py` (registry, resume, ngân sách, tổng hợp, leaderboard) · `selftest.py` · `make_notebook.py` · `v11_full_suite_v4.ipynb`.

## 9. Tài liệu

TIGER "Recommender Systems with Generative Retrieval" (NeurIPS'23) · LETTER "Learnable Item Tokenization for Generative Recommendation" (CIKM'24) · OneRec Technical Report (arXiv 2506.13695) ·
LLM2Vec (COLM'24) · "How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning" (arXiv 2602.10622) · LLM2Rec (KDD'25) · Qwen3-Embedding (arXiv 2506.05176) ·
Matryoshka Representation Learning (NeurIPS'22) · fMRLRec (EMNLP-Findings'24) · SMEC (EMNLP'25) · DoRA (ICML'24) · ColBERT (SIGIR'20) ·
K-RagRec (ACL'25) · Microsoft GraphRAG (arXiv 2404.16130) · "When to use Graphs in RAG" (arXiv 2506.05690) ·
KAR "Towards Open-World Recommendation with Knowledge Augmentation from LLMs" (RecSys'24) · "LLMs are Zero-Shot Rankers for Recommender Systems" (ECIR'24) · LLM4Rerank (WWW'25) ·
"LLM-Driven News Recommendation via Lightweight Task-Adaptive Modules" (Applied Sciences 2026).
