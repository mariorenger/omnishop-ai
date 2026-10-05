# V11 — bộ thử nghiệm quyết định hướng nghiên cứu (LLM embedding cho news recommendation)

Mục tiêu: **một lần Run All trên Kaggle cho biết hướng LLM nào đáng làm luận văn**, bằng bằng chứng có
khoảng tin cậy, chứ không phải bằng một con số AUC đơn lẻ.

> ⚠️ **Trạng thái trung thực:** toàn bộ code đã được kiểm thử đầu-cuối trên CPU (dữ liệu đúng định dạng MIND +
> encoder BERT tí hon ngẫu nhiên). **Chưa có con số nào trên MIND thật**, vì sandbox phát triển không có GPU,
> không tải được weight (huggingface.co bị chặn) và không có MIND. Số liệu quyết định hướng đi do **lần chạy
> Kaggle của bạn** tạo ra.

## Vì sao đo ở mức *biểu diễn* (zero-shot)

User = trung bình embedding các bài trong lịch sử; điểm = tích vô hướng với bài ứng viên (đúng scorer của
`bge_zeroshot` trong V10). Không có user-model học thêm ⇒ chênh lệch giữa các dòng **chỉ đến từ chất lượng
embedding / cách truy xuất**, mỗi biến thể chạy vài chục giây, và có thể chạy nhiều seed + khoảng tin cậy.
Embedding tốt nhất xuất ra `.npz` để cắm vào `bench.py --news-emb` của V10 và train `llmenc_ca` phía trên.

## Giả thuyết ↔ kỹ thuật ↔ paper

| | Kỹ thuật (đã cài) | Dòng paper (venue đã kiểm trong session) | Stage |
|---|---|---|---|
| **H1** | **History-as-text**: ghép các tiêu đề đã đọc thành 1 query *có instruction* (LLM làm user encoder) | Qwen3-Embedding (2025, instruction-aware); language-based user profiles (SIGIR'26 LLM-UP workshop) | `histquery` |
| **H2** | **Co-click contrastive adaptation**: residual head (TAM-style), **LoRA / DoRA** trong vòng lặp encoder, **Matryoshka (MRL)**, **hard negative theo impression** (bài *đã hiển thị nhưng bị bỏ qua*) | LLM2Rec (KDD'25); MRL (NeurIPS'22), fMRLRec (EMNLP-Findings'24), SMEC (EMNLP'25); DoRA (ICML'24); TAM cho news (Applied Sciences 2026) | `head`, `encoder` |
| **H3** | **Retrieval-augmented article representation**: bổ sung bài mới bằng context của *thực thể Wikidata* và *k bài cũ gần nhất* | RAG; K-RagRec (ACL'25) | `graph` |
| **H4** | **GraphRAG-lite**: cộng đồng Louvain trên đồ thị kNN bài báo + truy xuất *người dùng tương tự* | Microsoft GraphRAG (local/global); cảnh báo "GraphRAG thường kém RAG thường" (arXiv 2506.05690) ⇒ phải *đo* | `graph` |

"RAG" ở đây là **retrieval-augmented *representation*** (chưa có LLM sinh văn bản). Bước kế tiếp tự nhiên là
để LLM đọc ngữ cảnh truy xuất — xem *Stage-2* bên dưới.

## Giao thức (chống tự lừa mình)

* **Y hệt V10**: `data.py`/`metrics.py` được vendor nguyên văn (xem `PROVENANCE.md`); validation tách theo ngày,
  `MINDsmall_dev` là test, chỉ chạm **một lần** cho cấu hình đã chọn trên validation.
* **Evaluator vector hoá khớp `metrics.py` tới ~1e-8** (kể cả slate có điểm bằng nhau, ví dụ người dùng lịch sử rỗng
  — các hàng này được giao lại cho đúng hàm của `metrics.py` để tie-break giống hệt).
* **Bộ nhớ nhân quả** cho RAG: validation chỉ truy xuất từ bài đã quan sát trong `train_core`; test thêm validation.
  Chỉ dùng *nội dung* (embedding + thực thể), không dùng nhãn click.
* **Mọi so sánh với control kèm CI 95% ghép cặp** theo impression (positive cùng impression gộp cụm). Head adapter chạy
  nhiều seed (`--seeds`), báo độ lệch chuẩn giữa các seed.
* **Chọn đại diện từng hướng theo *validation*, không theo test**; kết luận cuối cùng có **hiệu chỉnh Bonferroni** theo số hướng
  (chọn 1-trong-4 là bài toán so sánh bội). Với ~20 dòng, nếu mọi hiệu ứng đều bằng 0 thì vẫn kỳ vọng ~1 dòng "BETTER" do ngẫu nhiên.
* **So sánh trực tiếp giữa các hướng** (bảng HEAD-TO-HEAD, ghép cặp, Bonferroni theo số cặp) — ngoài so với control. `test_arrays.npz` lưu metric
  theo từng impression của mọi biến thể để phân tích thêm mà không phải chạy lại.
* **Lát cắt tin mới/tin cũ**: *cold* = bài được click mà các dòng train không hề quan sát. Đây là nơi embedding LLM
  *lẽ ra* phải thắng; một hướng không thắng ở đây thì khó nói là "tận dụng LLM".
* **Epoch 0 (= embedding gốc) luôn tham gia chọn epoch** ⇒ adapter không bao giờ tệ hơn frozen trên validation.
* **Kiểm tra tái lập**: dòng `frozen` phải khớp `bge_zeroshot` của V10 (AUC 0.6241, nDCG@10 0.3909, dung sai 0.003).
  Nếu in `MISMATCH`, dừng lại — pipeline sai, mọi dòng còn lại không đáng tin.
* **Guard pooling**: `HFEncoder` được so với Sentence-Transformers trên 64 bài trước khi chạy (bắt lỗi pooling/EOS/prefix,
  đặc biệt với Qwen3 dùng last-token pooling).

## Chạy

**Kaggle (khuyến nghị):** upload `v11_research_suite_v2.ipynb` → Settings: GPU + Internet On → *Add Input* dataset MIND có
**cả** `MINDsmall_train` và `MINDsmall_dev` (vd `thinhhuynh3108/mindsmall`) → Run All. Lần đầu đặt `QUICK = True`
(~10 phút, kiểm tra môi trường) rồi mới chạy đầy đủ. Đặt `RUN_QWEN3 = True` để chạy lại các stage rẻ với Qwen3-Embedding-0.6B.

**Local:**
```bash
pip install torch transformers peft scipy networkx scikit-learn sentence-transformers
python selftest.py            # ~1-2 phút CPU, không cần dữ liệu thật
python suite.py --mind-train /data/MINDsmall_train --mind-dev /data/MINDsmall_dev --work out
```
Kết quả: `results.md`, `results.json`, `emb_<variant>.npz`.

**Thời gian ước tính (chưa đo trên T4):** E0 + load ~5 phút; `graph` ~15 phút; `head` ~20 phút; `encoder`
(3 lần LoRA/DoRA × `--enc-steps 1500`) ~50 phút; `histquery` ~15 phút ⇒ **~2 giờ** với bge-small. Qwen3-0.6B tốn thêm ~20–30 phút cho các stage rẻ.

## Cách đọc kết quả → hướng luận văn

| Kết quả (CI loại trừ 0 trên test) | Hướng nên theo |
|---|---|
| H2 **BETTER**, nhất là trên *cold* | "Collaborative-aware adaptation của LLM embedding chuyển giao sang tin mới" (+ MRL, + hard negative theo impression) |
| H3/H4 **BETTER** trên cold, H2 không | "Retrieval-augmented representation cho tin mới" (entity memory / GraphRAG), mở rộng bằng LLM đọc ngữ cảnh |
| H1 **BETTER** | "Language-based user representation với embedder biết instruction" |
| Không giả thuyết nào | **Kết quả âm có giá trị**: mean-pool của embedding đóng băng đã gần trần trên MIND-small ⇒ chuyển sang user model học được và/hoặc MIND-large, EB-NeRD |

## Đã kiểm chứng ở đây / chưa kiểm chứng

✅ `selftest.py` (CPU, dữ liệu MIND-format, BERT ngẫu nhiên): evaluator == `metrics.py`; head là identity ở bước 0; adapter
không thấp hơn frozen trên validation; cờ `hardneg` không phải no-op; MRL có tác dụng khi d lớn; trainer LoRA thật sự học;
đủ mọi stage; mô phỏng "Run All" của notebook từ thư mục trống chạy xong, không stage nào lỗi.
Qua kiểm thử đã tìm và sửa 2 lỗi thật: (i) tie-break của MRR/nDCG lệch `metrics.py` khi điểm bằng nhau; (ii) `train_head` luôn chọn
epoch 0 khi không có `val_fn`.

❌ **Chưa kiểm chứng:** mọi hiệu năng trên MIND thật; fp16/AMP + LoRA trên GPU thật; thời gian Louvain trên ~50k bài; hành vi với
weight BGE/Qwen3 thật (đã có guard pooling lúc chạy); thời gian ước tính ở trên.

## Hạn chế cần nói trong luận văn

Zero-shot mean-pool **không** kiểm chứng user-model học được (phần này nằm ở `bench.py` V10). Một dataset, một tập dev.
LoRA/DoRA mặc định 1 seed (đắt). Entity graph dựa trên chú thích Wikidata của MIND. Popularity/CTR online **cố ý không** đưa vào.

## Stage-2 (chưa cài, đáng làm nếu stage-1 cho tín hiệu)

* **Decoder-only LLM làm user encoder** với lịch trình attention mask causal→bidirectional (arXiv 2602.10622, 2/2026) áp lên lịch sử đọc.
* **GraphRAG thật**: LLM viết tóm tắt cho từng community rồi nhúng làm ngữ cảnh của ứng viên (hiện chỉ dùng centroid).
* **Reasoning/profile do LLM sinh** (cache offline, kiểu KAR RecSys'24) cho tin mới.
* **Semantic ID** làm tokenizer cho cold-start (không dùng generative retrieval — MIND là bài toán xếp hạng slate cho sẵn).

## Tài liệu

LLM2Rec (KDD'25) · Matryoshka Representation Learning (NeurIPS'22) · fMRLRec "Train Once, Deploy Anywhere" (EMNLP-Findings'24) ·
SMEC (EMNLP'25) · DoRA (ICML'24) · K-RagRec (ACL'25) · Microsoft GraphRAG · "When to use Graphs in RAG" (arXiv 2506.05690) ·
Qwen3-Embedding (arXiv 2506.05176) · "How Do Decoder-Only LLMs Perceive Users?" (arXiv 2602.10622) · KAR (RecSys'24) ·
LLM-Driven News Recommendation via Lightweight Task-Adaptive Modules (Applied Sciences 2026).
