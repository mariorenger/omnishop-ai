# V11 — bộ so sánh đầy đủ các hướng LLM cho news recommendation (MIND-small)

**Một lần Run All trên Kaggle (T4) — đã đo 7,52 giờ cho bản v7 đầy đủ (3,2 giờ cho bản v4 cũ); ngân sách đồng hồ cứng 9.5 giờ** cho ~14 họ kỹ thuật / ~110 biến thể
**và các baseline NRMS / NAML / Fastformer huấn luyện đúng công thức đã công bố**, tất cả đo trên đúng giao thức V10, có khoảng tin cậy ghép cặp, lát cắt tin mới/tin cũ và hiệu chỉnh Bonferroni.
Mục tiêu không phải tối đa hoá điểm, mà là **biết hướng LLM nào đáng làm luận văn, bằng bằng chứng có CI — và so với baseline huấn luyện đàng hoàng, không phải với baseline yếu**.

**Vì sao v7 có baseline mới.** Ở V10, NRMS / NAML / Fastformer đạt AUC 0,60–0,64 và nDCG@10 0,36–0,40, thấp hơn khoảng 0,03–0,05 ở mọi chỉ số so với các số công bố cho MIND-small
(NRMS ≈ 0,66 / 0,41 theo arXiv 2304.03112 và NewsReX; xem §10, lấy từ kết quả tìm kiếm, **chưa mở PDF**). Chênh lệch này lớn hơn nhiều độ lệch chuẩn giữa các seed (±0,002–0,003) nên không phải may rủi.
Bộ đo của V11 thì đúng (khớp `metrics.py`, khớp `bge_zeroshot` của V10, vector ngẫu nhiên cho AUC 0,50). **Đã đo ở lần chạy v7 (xem `RESULTS_v7.md`):** nguyên nhân chính là *vector từ huấn luyện sẵn (GloVe)*, không phải tokenizer / mask / negative: tái hiện công thức V10 trong harness này cho 0,3694 (V10: 0,3685); sửa dữ liệu (title-only, UNK, mask, negative bốc lại mỗi epoch) không giúp (−0,003); cỡ chuẩn + lr 1e-4 với vector ngẫu nhiên còn tệ hơn (chưa hội tụ trong 10 epoch); thêm GloVe +0,049. Huấn luyện thêm cả các ngày validation (`nrms_ref_refit`) cho thêm +0,014 và gần mức các bài báo công bố. **Nhiễu seed lớn hơn nhiều so với số trích trong văn liệu: 0,007 (nDCG@10) giữa hai seed của NRMS, 0,010–0,012 giữa hai seed của tower Qwen3**. Vì vậy câu "hơn Fastformer của V10" không đủ làm luận điểm; so với Fastformer chuẩn (0,3926) thì frozen BGE ngang (−0,002, không khác biệt), `pool_lse` hơn +0,011.

> ⚠️ **Trạng thái trung thực.** Code được kiểm thử trên CPU với dữ liệu đúng định dạng MIND và các mô hình ngẫu nhiên tí hon; sandbox phát triển không có GPU/MIND/weight.
> Các con số trên MIND thật (và thời gian) đến từ **các lần chạy Kaggle của bạn** (v4 đầy đủ trên Tesla T4: 3,2 giờ); README không chép lại kết quả — xem `results.md`, `leaderboard*.md`, `direct_comparisons.md` của lần chạy.

## 1. Giả thuyết ↔ kỹ thuật ↔ paper ↔ mức độ cài đặt

| | Kỹ thuật đã cài | Dòng paper (venue đã tra cứu) | **Chưa cài** (nói rõ để không nhận vơ) | Stage |
|---|---|---|---|---|
| **R** | **Baseline tham chiếu** — NRMS, NAML, Fastformer huấn luyện đúng công thức đã công bố: title-only regex (30 token; NAML thêm abstract 50 token + category/subcategory), từ vựng có UNK (không đổi từ hiếm thành PAD), **GloVe-300**, 16 head × 16, additive attention 200, dropout 0,2, **negative lấy lại mỗi epoch**, Adam lr 1e-4, **mask padding**; chọn epoch theo validation nDCG@10 như V10. Hàng `nrms_ref`, `naml_ref`, `fastformer_ref` (+ `+pop`), `nrms_ref@1` (cùng công thức, seed khác = sàn nhiễu huấn luyện), `nrms_ref_refit` (huấn luyện lại trên train_core **+ validation** với số epoch `nrms_ref` đã chọn — số so được với bài báo vốn dùng toàn bộ train; cột validation của hàng này lặp lại lượt chọn epoch, chỉ cột test có ý nghĩa). **Thang ablation** NRMS: `ladder0_v10recipe` (tái hiện công thức V10 trong harness này; kiến trúc cho điểm **giống hệt** NRMS của V10 khi chép trọng số, lệch ≤ 3e-8 — đã kiểm) → `ladder1_data` (+ title-only regex, UNK, mask, negative mỗi epoch) → `ladder2_recipe` (+ cỡ/ lr chuẩn, vector ngẫu nhiên N(0,0.1)) → `nrms_ref` (+ GloVe), bảng `ladder.md` | NRMS (EMNLP-IJCNLP'19), NAML (IJCAI'19), Fastformer (arXiv 2108.09084), GloVe (EMNLP'14), MIND (ACL'20) | Fastformer là **cài lại** theo cấu trúc mã chính thức (additive attention theo head, query đóng vai value, transform + residual, FFN, embedding vị trí; 1 lớp, 256-d) — *không* phải mã gốc; NAML dùng đúng 4 view nhưng chưa đối chiếu từng siêu tham số với mã gốc. Không tinh chỉnh siêu tham số | `ref_nrms`, `ref_naml`, `ref_ff`, `ref_nrms_refit`, `ref_nrms_s1`, `ref_l0`, `ref_l1`, `ref_l2` |
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
| **H10** | User model **học được** = bản sao V10 `llmenc_ca` (đã kiểm: điểm số giống hệt mã V10) trên từng biểu diễn (frozen / head tốt nhất / LoRA tốt nhất / entity_rag / kNN-RAG) — cải thiện biểu diễn có sống sót khi đã có user model không? **Thêm (v7):** cùng bảng vector đóng băng và cùng head 2 lớp nhưng reader *không nhìn ứng viên* là **Fastformer** (`um_ff_*`), **self-attention kiểu NRMS** (`um_nrms_*`) hoặc additive attention (`um_add_*`) ⇒ cô lập xem *module attention* nào đáng giá khi đã có vector LLM | V10; Fastformer; NRMS | — | `usermodel` (`--um-extra`) |
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
* **Hai khung so sánh**: *content only* (control `frozen`) và *with popularity* (mọi biến thể zero-shot có bản sinh đôi `+pop`, control `frozen+pop` = V10 `bge_zs_pop`; kèm `leaderboard_pop.md`, `forest_pop.png`, `head_to_head_pop.json`). Lý do: trong bảng V10, `popularity` thuần (nDCG@10 0.4026) đã vượt mọi model nơ-ron huấn luyện (NAML 0.3953, llmenc_ca 0.3997, NRMS 0.3685, Fastformer 0.3642) và `bge_zs_pop` (0.4330) là model tốt nhất — nên một kỹ thuật LLM chỉ thật sự có giá trị nếu còn đóng góp *khi đã có popularity* (hoặc nếu nhắm cold-start, nơi popularity chưa có). `controls` kiểm tra `popularity` và `frozen+pop` khớp V10 (`--repro-pop`).
* **Baseline tham chiếu cùng split, cùng evaluator.** Các dòng huấn luyện được dựng lại từ `behaviors.tsv` đúng như `data.py` (cùng cách tách ngày, cùng lọc) và **bị từ chối nếu lệch**: số impression/ngày validation phải bằng `MindData`, mỗi mẫu huấn luyện phải có cùng bài được click + cùng lịch sử, và mọi negative mà V10 đã bốc phải nằm trong pool của impression (bắt lỗi ánh xạ id bài báo khác nhau mà số lượng vẫn khớp; test đảo thứ tự id để chắc chắn lỗi này bị bắt). Baseline được chấm bằng đúng `Suite.run` (cùng CI ghép cặp, cùng lát cắt cold/warm theo `seen_tr` của V10). Chúng nằm ở họ `ref-baseline` / `ref-ladder`: **có trong leaderboard** nhưng **không** tham gia DECISION SUMMARY / HEAD-TO-HEAD như một "hướng ứng viên".
* **`pop` là phản hồi online.** `_online_log_ctr` (V10) đếm click của các impression *trước đó trong chính tập test* (ngày 15/11). Đó là thiết lập streaming hợp lệ nhưng khác giao thức offline của các bài báo: các hàng `+pop` (nDCG@10 ≈ 0,43) **không so được** với số công bố; chỉ bảng *content only* so được. Không mô hình độ trễ phản hồi.
* **Sàn nhiễu huấn luyện.** CI ghép cặp chỉ tính nhiễu *mẫu test*, không tính nhiễu *seed huấn luyện*. `nrms_ref@1` (cùng công thức, seed 1) và `ut_causal@1` cho biết độ lớn đó; chênh lệch nhỏ hơn nó giữa hai mô hình *đã huấn luyện* không phải phát hiện.
* **Ngân sách đồng hồ chung** (`--budget-hours`): vòng lặp dài nhận `deadline` và dừng êm, stage nào không còn thời gian bị bỏ qua, kết quả được ghi sau *từng* stage; gọi lại sẽ **resume** (`state.pkl`, chỉ khi dữ liệu *và mọi tham số ảnh hưởng kết quả* giống hệt — QUICK không bao giờ lẫn vào bản đầy đủ). Dòng bị cắt giữa chừng được gắn cờ (`truncated`, `queries_missing`) để không so sánh nhầm.

## 3. Chạy

**Kaggle:** upload `v11_full_suite_v7.ipynb` → Settings: GPU + Internet On → *Add Input* dataset MIND có **cả** `MINDsmall_train` và `MINDsmall_dev` (vd `thinhhuynh3108/mindsmall`) → Run All.
Lần đầu nên `QUICK=1` (biến môi trường `V11_QUICK=1` hoặc sửa ô cấu hình; phần BGE đã đo thật ≈ 13 phút trên Kaggle, phần Qwen3/LLM chưa đo (ước ~30–60 phút vì phải encode 65k bài bằng Qwen3, chạy Louvain và tải model); mục đích bắt lỗi môi trường: tải model, bộ nhớ GPU, phiên bản `transformers`/`peft`, mask 4D, chat template). Chỉ thử BGE: `V11_RUN_B=0`; bỏ baseline: `V11_REF=0`. Thứ tự chạy: **A-core → baseline (R) → B (Qwen3) → A-phần còn lại**; cuối cùng ghi `SUMMARY_v11.md` (gộp mọi bảng) — gửi file đó để đọc kết quả.

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `V11_QUICK` | 0 | bản rút gọn |
| `V11_BUDGET_H` | 9.5 | ngân sách đồng hồ chung (giờ); đặt 6 nếu muốn ngắn hơn |
| `V11_ENCODER` / `V11_ENCODER_B` | `BAAI/bge-small-en-v1.5` / `Qwen/Qwen3-Embedding-0.6B` | encoder run A / run B |
| `V11_GEN_MODEL` | `Qwen/Qwen3-1.7B` | LLM sinh văn bản / giám khảo (có thể thử `Qwen/Qwen3-4B-Instruct-2507`, chậm ~2.5×) |
| `V11_RUN_B` | 1 | 0 = bỏ phần LLM (chỉ còn run A, ~3 giờ) |
| `V11_REFERENCE` | 1 | 0 = bỏ bước so HFEncoder với Sentence-Transformers |
| `V11_REF` | 1 | 0 = bỏ 8 mô hình baseline (ước ~1,7–2,2 giờ, chưa đo) |
| `V11_GLOVE_PATH` | (trống) | file GloVe-300 `txt`/`zip` nếu bạn đã Add Input; trống ⇒ tự dò `/kaggle/input/**/glove*300d*`, rồi tải bản `sentence-transformers/average_word_embeddings_glove.6B.300d` (~480 MB), rồi `gensim`. Hỏng hết ⇒ vector ngẫu nhiên, hàng gắn nhãn `_noglove` |
| `V11_SWEEP` | bge-base, bge-large, Qwen3-Emb-0.6B | danh sách encoder quét |

**Local:**
```bash
pip install torch transformers peft scipy networkx scikit-learn sentence-transformers matplotlib
python selftest.py            # ~2 phút CPU, không cần dữ liệu thật
python suite.py --mind-train /data/MINDsmall_train --mind-dev /data/MINDsmall_dev --work out --stages controls,pooling,graph,head,histquery,sid
python make_notebook.py       # sinh lại notebook (đổi NOTEBOOK trong file để tránh cache của Kaggle)
```

## 4. Thời gian

**v7 đầy đủ, đã đo trên Tesla T4 (Kaggle, dùng 1 trong 2 GPU): 27 092 giây = 7,52 giờ**, không stage nào lỗi. Mỗi run in `STAGE TIMINGS` và lưu `timings.json`.

| Phần | Số đo thật (giây) | Ghi chú |
|---|---|---|
| **A-core** (BGE-small) | controls 10 · pooling 29 · graph 96 · head 226 · histquery 155 · sid 596 | **18,5 phút** |
| **R — baseline** | `ref_nrms` 952 (91 s/epoch, 10 epoch) · `ref_naml` 1063 (173 s/epoch, 6 epoch) · `ref_ff` 1741 (170 s/epoch, 10 epoch) · `ref_nrms_refit` 912 (113 s/epoch, 8 epoch) · `ref_l0` 244 · `ref_l1` 265 · `ref_l2` 925 · `ref_nrms_s1` 741 | **1,90 giờ** (ước trước 1,7–2,2 giờ); GloVe + dựng dữ liệu 23 s sau khi tải |
| **B** (Qwen3-Embedding-0.6B + Qwen3-1.7B) | E0 550 · histquery 524 · **llm_user 8998 (4 mode ≈ 37 phút/mode gồm đánh giá)** · rerank 978 · augment 1183 · graphrag_llm 94 · pooling 33 · graph 93 · head 147 · usermodel 751 | **3,56 giờ** (+ E0) |
| **A-phần còn lại** | encoder LoRA/DoRA ×3 3712 · llm_user (tower BGE) 137 · usermodel 1293 · sweep 339 | **1,52 giờ** |
| **Tổng v7** | | **7,52 giờ** |

Phần "V11 cũ" của v6/v7 (không tính baseline) ≈ 5,6 giờ, đúng dải ước 5,5–6 giờ đã nêu trước đó. Nếu cần ngắn hơn: `V11_BUDGET_H=6`, `V11_REF=0` (−1,9 giờ), hoặc bớt `causal@1` khỏi `--llm-user-modes` (−0,6 giờ; nhưng đó là hàng đo nhiễu seed của tower, nên giữ nếu muốn kết luận về mask).

## 5. Cách đọc kết quả → hướng luận văn

| Kết quả (CI loại trừ 0 trên test, sau Bonferroni) | Hướng nên theo |
|---|---|
| **Biến thể LLM/zero-shot thắng cả `nrms_ref` / `fastformer_ref`** (bảng content-only trong `direct_comparisons.md`) | luận điểm mạnh: "vượt baseline đã huấn luyện chuẩn", không chỉ vượt baseline yếu của V10 |
| Chỉ thắng baseline V10, **thua** `nrms_ref` | luận điểm cũ không đứng vững; báo cáo trung thực |
| `ladder.md`: rung nào tăng nhiều nhất | nguồn chính của khoảng cách V10 ↔ bài báo (Δ phụ thuộc thứ tự rung; có tương tác) |
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
mô phỏng "Run All" của notebook (QUICK) từ thư mục trống trên fixture: 4 lần gọi `suite.main` (A-core, baseline, B, A-còn-lại), không stage nào lỗi, có `leaderboard.md` + `forest.png` + `ladder.md` + `SUMMARY_v11.md`.

**Baseline mới (v7):** token/từ vựng/UNK; nạp GloVe từ `txt`, `zip`, thư mục sentence-transformers và chuỗi lỗi → ngẫu nhiên có nhãn; ma trận embedding mang đúng hàng GloVe, PAD = 0; **mẫu huấn luyện dựng lại == `train_core` của V10** (cùng bài click, cùng lịch sử, mọi negative của V10 nằm trong pool; đảo thứ tự id bị từ chối); negative thật sự được bốc lại mỗi epoch, không lặp khi pool ≥ 4, có lặp khi pool nhỏ hơn (như V10);
**rung 0 của thang ablation == NRMS của V10** (chép trọng số: lệch điểm ≤ 3e-8 trên 50 impression); attention Fastformer == bản vòng lặp tham chiếu; encoder có mask bất biến theo phần PAD thêm vào (bố cục V10 thì *không*); scorer (mã hoá cả catalogue một lần) == forward huấn luyện (mã hoá theo batch); lịch sử rỗng ⇒ mọi ứng viên bằng điểm; cả 3 họ đều học được cấu trúc chủ đề giả (loss giảm, val tăng);
reader Fastformer/NRMS/additive trên vector đóng băng: add/nrms bất biến theo thứ tự lịch sử, Fastformer thì không; stage baseline trong suite: hàng + twin `+pop`, bảng ablation, harness check, không lọt vào DECISION SUMMARY, resume theo từng mô hình, đổi cấu hình baseline chỉ chạy lại các stage baseline, fallback không GloVe (`_noglove`), chạy con (QUICK).

✅ **Đã kiểm trên Kaggle thật (lần chạy v7, Tesla T4, Python 3.13):** `frozen`, `popularity`, `frozen+pop`, `um_frozen` đều MATCH V10; guard pooling của Qwen3 (cos nhỏ nhất 0,9998); tải GloVe qua `huggingface:sentence-transformers/average_word_embeddings_glove.6B` chạy được (52 722 / 60 992 từ của kho có vector, 0,12% token title là UNK);
mẫu huấn luyện dựng lại == `train_core` của V10 (kiểm từng mẫu); rung 0 của thang ablation khớp NRMS của V10 (0,6156 / 0,3694 vs 0,6181 / 0,3685); mask `bidir` và `soft` chạy được trên Qwen3 thật (probe OK); fp16/AMP + LoRA/DoRA + GradScaler; sinh văn bản Qwen3-1.7B (≈ 44 ms/prompt khi giám khảo); Louvain trên ~65k bài (26–36 s); không OOM ở tower và baseline; thời gian đo được ở §4.

❌ **Chưa kiểm chứng:** Fastformer cài lại có khớp mã chính thức về siêu tham số hay không; số bài báo ở §10 (lấy từ kết quả tìm kiếm, chưa mở PDF); kết luận có chuyển sang dataset khác (MIND-large, EB-NeRD) hay không; hầu hết hàng chỉ có **1 seed huấn luyện** (nhiễu seed đo được: 0,007 nDCG@10 giữa hai seed của NRMS, 0,010–0,012 giữa hai seed của tower — xem `RESULTS_v7.md`), nên chênh lệch nhỏ hơn mức đó giữa hai mô hình *đã huấn luyện* chưa phải kết luận.

## 6b. Sự cố môi trường đã gặp khi chạy thật

* **Probe mask loại nhầm `bidir` và `soft`** (lần chạy đầy đủ v4): tiêu chí `soft(λ=1e-6) ≈ causal` quá chặt với LLM thật (đo được 0,099 trên Qwen3 vì logit rất lớn ở vài token đặc biệt; trong khi `causal-4D == native` và `soft(1) == bidir` đều chính xác 0,0). Đã đổi: số đó chỉ để tham khảo, `probe_ok()` chỉ kiểm tra 3 điều kiện còn lại; lịch trình `soft` bắt đầu đúng bằng causal (λ=0). Test tái hiện báo cáo thật trong `selftest.py`.

* **`ImportError: Found an incompatible version of torchao`** (peft từ chối torchao 0.10 có sẵn trong ảnh Kaggle ngay khi gắn LoRA; lộ ra ở stage `encoder`, và sẽ chặn cả `llm_user`): đã vá bằng `common.fix_peft_torchao()` (coi torchao là không có — ta không dùng nó), có test tái hiện bằng torchao giả trong `selftest.py`.
* Dòng `WORSE` với Δ = −0.0000 [−0.0000, −0.0000] ở các head không học được gì (adapter giữ nguyên epoch 0 = frozen): do làm tròn float32/float64 giữa dòng mới và dòng tham chiếu; đã sửa, giờ in `identical`.

## 7. Hạn chế cần nói trong luận văn

Một dataset, một tập dev. Tower/LoRA/DoRA/SID-GPT mặc định **1 seed** (chênh lệch nhỏ hơn ~0.005 nDCG@10 giữa các mask có thể là nhiễu huấn luyện). Dòng chạy tập con (H5/H6/H9) có CI rộng hơn: chỉ phát hiện được hiệu ứng lớn.
`soft` ≠ Gradient-Guided Soft Masking. Giám khảo LLM 1.7B pointwise zero-shot khá yếu so với các hệ rerank mạnh. Entity graph dựa trên chú thích Wikidata của MIND. Popularity (CTR online của V10) **không** nằm trong các hàng gốc; nó chỉ xuất hiện ở các hàng sinh đôi `+pop` (cùng cấu hình, gộp z-score 1:1 như `bge_zs_pop`, không tinh chỉnh trọng số; hàng `rerank_*` không có bản `+pop`).
**Baseline:** (i) các hàng baseline thường huấn luyện trên `train_core` (≈ 80,7% số impression của MINDsmall_train; 10% ngày cuối là validation) trong khi các bài báo dùng toàn bộ train ⇒ hơi bất lợi cho baseline; riêng `nrms_ref_refit` huấn luyện lại trên cả validation (nhưng chỉ NRMS, và số epoch lấy từ lượt chọn trên validation); (ii) GloVe-6B (chữ thường) thay vì 840B; (iii) mặc định 1 seed (nhiễu seed trong các nguồn khác ±0,002–0,003); (iv) chỉ số bài báo trích từ kết quả tìm kiếm, cần đối chiếu PDF; (v) thang ablation là tích luỹ nên Δ của mỗi rung phụ thuộc thứ tự; (vi) không tinh chỉnh siêu tham số cho bất kỳ mô hình nào (baseline lẫn các hướng LLM) — so sánh công bằng ở mức "cấu hình mặc định", không phải "mỗi bên tối ưu hết cỡ".
Đo ở mức biểu diễn zero-shot cho phần lớn họ; H10 là cầu nối sang user model học được nhưng vẫn chỉ là một kiến trúc (V10 `llmenc_ca`).

## 8. Bản đồ file

`data.py`/`metrics.py` (V10, nguyên văn) · `common.py` (đồng hồ ngân sách, fusion, `ScoreCache`, `HistQueries`) · `rep_eval.py` (evaluator vector hoá, CI ghép cặp, so sánh qua tập con) ·
`semgraph.py` (RAG/GraphRAG-lite) · `adapt.py` (head, LoRA/DoRA, MRL) · `pooling.py` (H8) · `sid.py` (H7) · `llm_user.py` (H5) · `llm_gen.py` (sinh văn bản, giám khảo) · `usermodel.py` (H10, reader Fastformer/NRMS/additive) ·
`refdata.py` (token, GloVe, dựng lại mẫu huấn luyện V10 + negative bốc lại mỗi epoch) · `refmodels.py` (NRMS/NAML/Fastformer, trainer) · `stages_ref.py` (stage baseline + bảng ablation) ·
`stages_ext.py` (các stage mới) · `suite.py` (registry, resume, ngân sách, tổng hợp, leaderboard) · `selftest.py` · `make_notebook.py` · `v11_full_suite_v7.ipynb` · `RESULTS_v7.md` (kết quả đo thật của lần chạy đầy đủ v7: bảng, nhiễu seed, nhận định, giới hạn).

## 9. Tài liệu

TIGER "Recommender Systems with Generative Retrieval" (NeurIPS'23) · LETTER "Learnable Item Tokenization for Generative Recommendation" (CIKM'24) · OneRec Technical Report (arXiv 2506.13695) ·
LLM2Vec (COLM'24) · "How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning" (arXiv 2602.10622) · LLM2Rec (KDD'25) · Qwen3-Embedding (arXiv 2506.05176) ·
Matryoshka Representation Learning (NeurIPS'22) · fMRLRec (EMNLP-Findings'24) · SMEC (EMNLP'25) · DoRA (ICML'24) · ColBERT (SIGIR'20) ·
K-RagRec (ACL'25) · Microsoft GraphRAG (arXiv 2404.16130) · "When to use Graphs in RAG" (arXiv 2506.05690) ·
KAR "Towards Open-World Recommendation with Knowledge Augmentation from LLMs" (RecSys'24) · "LLMs are Zero-Shot Rankers for Recommender Systems" (ECIR'24) · LLM4Rerank (WWW'25) ·
"LLM-Driven News Recommendation via Lightweight Task-Adaptive Modules" (Applied Sciences 2026) ·
NRMS "Neural News Recommendation with Multi-Head Self-Attention" (EMNLP-IJCNLP'19) · NAML "Neural News Recommendation with Attentive Multi-View Learning" (IJCAI'19) · Fastformer "Additive Attention Can Be All You Need" (arXiv 2108.09084) ·
GloVe (EMNLP'14) · MIND (ACL'20).

## 10. Số công bố cho MIND-small (chỉ để kiểm tra độ lớn — **chưa mở PDF**, lấy từ kết quả tìm kiếm)

| Mô hình | AUC | MRR | nDCG@5 | nDCG@10 | Nguồn |
|---|---|---|---|---|---|
| NRMS | 66,58 ± 0,17 | 31,44 ± 0,15 | 34,99 ± 0,19 | 41,21 ± 0,16 | arXiv 2304.03112 |
| NAML | 67,14 ± 0,20 | 31,58 ± 0,28 | 35,20 ± 0,29 | 41,52 ± 0,28 | arXiv 2304.03112 |
| NRMS (JAX) | 66,14 – 66,39 | 31,30 – 31,35 | 34,56 – 34,69 | 40,96 – 40,97 | NewsReX (model card) |
| NAML (JAX) | 66,61 | 31,49 | 34,78 | 41,19 | NewsReX (model card) |

Quy luật cấu trúc để phát hiện bảng sai: ở mọi nguồn trên và trong cả 163 dòng kết quả của lần chạy đầy đủ v4 (kể cả các dòng gần ngẫu nhiên), **nDCG@5 luôn lớn hơn MRR**; ở 141 dòng có AUC > 0,60 mức chênh nằm trong 0,022–0,033. Một bảng có MRR > nDCG@5 (như bảng không trích dẫn tìm được trên Google) không khớp quy luật đó — đừng dùng làm tài liệu tham chiếu.
