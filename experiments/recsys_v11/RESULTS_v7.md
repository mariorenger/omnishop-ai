# Kết quả đo thật — lần chạy đầy đủ v7 (MIND-small, Kaggle Tesla T4, 7,52 giờ)

Nguồn: log của lần chạy `v11_full_suite_v7.ipynb` (QUICK=0, ngân sách 9,5 giờ, 1 trong 2 T4). Mọi số là trên **MINDsmall_dev (73 152 impression)** trừ các hàng tập con (n ghi rõ).
CI 95% là **ghép cặp theo impression**: chỉ tính nhiễu *mẫu test*, **không** tính nhiễu *seed huấn luyện* (xem §3, nhiễu này lớn). Số bài báo ở §1 lấy từ kết quả tìm kiếm, chưa mở PDF.

## 0. Chạy có ổn không?

* Không stage nào lỗi (log không có `!!`), tổng 7,52 giờ (§4 của README có bảng thời gian).
* Kiểm tra tái lập đều **MATCH**: `frozen` = V10 `bge_zeroshot` (0,6242 / 0,3909); `popularity` và `frozen+pop` = V10 (`bge_zs_pop` 0,6769 / 0,4330); `um_frozen` = V10 `llmenc_ca` trong dung sai 0,01 (0,6449 / 0,3957 vs 0,6494 / 0,3997); guard pooling Qwen3 cos nhỏ nhất 0,9998.
* **Harness check đạt:** rung 0 của thang ablation (công thức V10 cài lại) cho **0,6156 / 0,3694**, V10 báo **0,6181 / 0,3685** ⇒ code huấn luyện mới tái hiện đúng NRMS của V10, và NRMS của V10 không hỏng vì lỗi cài đặt.
* GloVe tải được từ Hugging Face: 52 722 / 60 992 từ của kho có vector, chỉ 0,12% token title là UNK.

## 1. Baseline huấn luyện chuẩn so với V10 và với zero-shot (nội dung, không popularity)

| Hàng | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC |
|---|---|---|---|---|---|---|
| V10 NRMS / NAML / Fastformer (số V10 báo) | 0,6181 / 0,6387 / 0,6022 | 0,2809 / 0,3036 / 0,2798 | 0,3020 / 0,3327 / 0,2998 | 0,3685 / 0,3953 / 0,3642 | | |
| `ladder0_v10recipe` (công thức V10, harness mới) | 0,6156 | 0,2822 | 0,3034 | 0,3694 | 0,6288 | 0,5920 |
| `nrms_ref` (GloVe, công thức chuẩn, seed 0) | 0,6392 | 0,3037 | 0,3293 | 0,3937 | 0,6471 | 0,6400 |
| `nrms_ref@1` (cùng công thức, seed 1) | 0,6501 | 0,3091 | 0,3372 | 0,4005 | 0,6630 | 0,6152 |
| `naml_ref` | 0,6471 | 0,3079 | 0,3364 | 0,4006 | 0,6698 | 0,5640 |
| `fastformer_ref` | 0,6455 | 0,2993 | 0,3288 | 0,3926 | 0,6674 | 0,5785 |
| `nrms_ref_refit` (train_core + validation, 8 epoch) | 0,6555 | 0,3140 | 0,3443 | 0,4078 | 0,6700 | 0,6284 |
| BGE-small `frozen` (zero-shot mean-pool) | 0,6242 | 0,3002 | 0,3299 | 0,3909 | 0,6617 | 0,4416 |
| BGE-small `pool_lse` (zero-shot, late-interaction) | 0,6386 | 0,3118 | 0,3427 | 0,4037 | 0,6675 | 0,5036 |
| `um_knn_rag` (user model học được trên vector BGE + kNN-RAG) | 0,6531 | 0,3090 | 0,3419 | 0,4033 | 0,6824 | 0,5365 |
| Bài báo, NRMS (arXiv 2304.03112; NewsReX) | 0,661 – 0,666 | 0,313 – 0,314 | 0,346 – 0,350 | 0,410 – 0,412 | | |
| Bài báo, NAML | 0,666 – 0,671 | 0,315 – 0,316 | 0,348 – 0,352 | 0,412 – 0,415 | | |

Đọc bảng:

* Baseline huấn luyện đúng cách tốt hơn V10 rõ: NRMS +0,025 (refit +0,039), Fastformer +0,028, NAML +0,005 nDCG@10. **NRMS refit** (0,6555 / 0,4078) nằm sát khoảng bài báo (kém 0,006–0,011 AUC, 0,002–0,004 nDCG@10), trong nhiễu seed đo ở §3.
* **Fastformer chuẩn ≈ NRMS chuẩn** (−0,0011, CI [−0,0026, +0,0004]): nó nhanh hơn về lý thuyết, không chính xác hơn; "Fastformer kém" ở V10 là hệ quả của công thức huấn luyện.
* **Zero-shot BGE-small `frozen` ≈ baseline huấn luyện một seed** (so với `nrms_ref` −0,0028, với `fastformer_ref` −0,0017 không khác biệt), thấp hơn `naml_ref`, `nrms_ref@1` (−0,010), thấp hơn NRMS refit (−0,017).
* `pool_lse` (không tham số) hơn `nrms_ref` / `fastformer_ref` +0,010 / +0,011 nhưng chỉ hơn trung bình hai seed NRMS (0,3971) khoảng +0,007 và **thua NRMS refit −0,0041 [−0,0061, −0,0021]**.
* Trên tin **cold** (85,9% số click test), các mô hình dựa vector LLM (`um_knn_rag` 0,6824, `pool_lse` 0,6675) xấp xỉ hoặc nhỉnh hơn baseline (0,647–0,670); trên tin **warm** (đã thấy ở huấn luyện) baseline huấn luyện vượt xa (0,56–0,64 so với 0,44–0,54): chúng học thuộc nội dung bài đã thấy.

## 2. Thang ablation NRMS: khoảng cách V10 ↔ bài báo đến từ đâu

| Rung | Thay đổi | AUC | nDCG@10 | Δ nDCG@10 so với rung trước [95% CI] |
|---|---|---|---|---|
| `ladder0_v10recipe` | công thức V10 | 0,6156 | 0,3694 | — |
| `ladder1_data` | + title-only regex 30 từ, UNK thay PAD, mask padding, negative bốc lại mỗi epoch | 0,6102 | 0,3661 | −0,0033 [−0,0051, −0,0015] |
| `ladder2_recipe` | + cỡ chuẩn (300-d, 16 head × 16), lr 1e-4, vector ngẫu nhiên | 0,5818 | 0,3447 | −0,0213 [−0,0229, −0,0197] |
| `nrms_ref` | + GloVe-300 | 0,6392 | 0,3937 | **+0,0490** [+0,0471, +0,0509] |
| `nrms_ref_refit` | huấn luyện lại thêm các ngày validation | 0,6555 | 0,4078 | +0,0141 [+0,0126, +0,0156] (so với `nrms_ref`) |

* **Yếu tố chính là vector từ huấn luyện sẵn (GloVe)**, không phải tokenizer, mask hay cách bốc negative (giả thuyết trước đó của tôi về các yếu tố này **sai**: sửa chúng không giúp, −0,003).
* **Lưu ý rung 2:** mô hình cỡ chuẩn với vector ngẫu nhiên **chưa hội tụ** — epoch tốt nhất là epoch 10 (đúng trần) và validation còn tăng (0,3531). Nên rung 2 → GloVe (+0,049) có phần thổi phồng bởi chưa train đủ; kết luận an toàn là "GloVe là yếu tố lớn nhất", không phải "đúng +0,049".
* Dữ liệu huấn luyện thêm các ngày sát ngày test (refit) cho +0,014: tin tức trôi theo thời gian, các bài báo huấn luyện trên toàn bộ train nên có lợi thế này.

## 3. Nhiễu seed (đo được, lớn hơn số trong văn liệu)

| Cặp cùng công thức, khác seed | nDCG@10 | AUC |
|---|---|---|
| `nrms_ref` seed 0 vs 1 (baseline chuẩn) | 0,3937 vs 0,4005 (chênh 0,0068) | 0,6392 vs 0,6501 (chênh 0,0109) |
| tower Qwen3 `ut_causal` seed 0 vs 1 | 0,3944 vs 0,3820 (chênh 0,0124) | 0,6367 vs 0,6245 |
| tower Qwen3 `ut_causal+meanpool` seed 0 vs 1 | 0,4026 vs 0,3923 (chênh 0,0103) | 0,6471 vs 0,6361 |

CI ghép cặp của hai hàng seed khác nhau vẫn hẹp (±0,002) vì chỉ phản ánh nhiễu mẫu test; **nhiễu huấn luyện là 0,007–0,012**. Hệ quả: chênh lệch nhỏ hơn mức đó giữa hai mô hình *đã huấn luyện* (một seed) không phải kết luận. Các hàng zero-shot (không huấn luyện) không có nhiễu này.

## 4. Mask của tower Qwen3 (`causal` / `bidir` / `soft`): chưa kết luận được

n = 30 000 impression, mọi hàng chung các impression đó; Δ so với `frozen` Qwen3 trên cùng impression.

| Mode | nDCG@10 | + meanpool | AUC (+ meanpool) |
|---|---|---|---|
| `causal` (seed 0) | 0,3944 (+0,0173) | 0,4026 (+0,0255) | 0,6367 (0,6471) |
| `bidir` | 0,3854 (+0,0083) | 0,3965 (+0,0194) | 0,6275 (0,6411) |
| `soft` | 0,3964 (+0,0193) | 0,4043 (+0,0273) | 0,6381 (0,6474) |
| `causal@1` (seed 1) | 0,3820 (+0,0049) | 0,3923 (+0,0152) | 0,6245 (0,6361) |
| tower BGE `ut_native` (mã hoá hai chiều gốc) | 0,3849 | 0,3973 | 0,6179 (0,6298) |

Chênh `soft − causal` = +0,0020 [+0,0003, +0,0036]; `bidir − causal` = −0,0090; nhưng **cùng công thức khác seed lệch tới −0,0124** (`causal@1 − causal`). Mọi chênh lệch giữa các mode nằm trong hoặc ngang mức nhiễu seed ⇒ **dữ liệu này không chứng minh được rằng mask có ảnh hưởng**.
Chắc chắn hơn: tower Qwen3 hơn tower BGE cùng điều kiện (+0,0095, `ut_causal` vs `ut_native`), và tower Qwen3 + meanpool ≈ `pool_lse` của BGE (−0,0007, CI [−0,0035, +0,0022]) nhưng **thua NRMS refit** (−0,0035 [−0,0068, −0,0002]).
Để kết luận về mask cần ≥ 3 seed mỗi mode (mỗi lần huấn luyện + đánh giá ≈ 37 phút trên T4).

## 5. Khi có popularity online (các hàng `+pop`)

`pop` đếm click của các impression *trước đó trong chính ngày test*: thiết lập streaming, **không so được với số offline của bài báo**.

| Hàng | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| `popularity` (V10) | 0,6533 | 0,3114 | 0,3402 | 0,4026 |
| `frozen+pop` (V10 `bge_zs_pop`) | 0,6769 | 0,3350 | 0,3727 | 0,4330 |
| `nrms_ref+pop` / `naml_ref+pop` / `fastformer_ref+pop` | 0,6946 / 0,6896 / 0,6913 | 0,3401 / 0,3385 / 0,3402 | 0,3759 / 0,3731 / 0,3767 | 0,4383 / 0,4366 / 0,4384 |
| `pool_lse+pop` / `pool_topk+pop` | 0,6867 / 0,6871 | 0,3413 / 0,3414 | 0,3808 / 0,3807 | 0,4409 / 0,4410 |
| Tower Qwen3 `ut_causal+meanpool+pop` (n = 30 000) | 0,6935 | — | — | 0,4415 |

* `bge_zs_pop` (0,4330) **không còn là mô hình tốt nhất** của V10: baseline huấn luyện chuẩn + pop đạt 0,4366–0,4384 (+0,0036 … +0,0055 so với `frozen+pop`).
* `pool_lse+pop` hơn `nrms_ref+pop` +0,0026 [+0,0009, +0,0043]; tower Qwen3 + pop hơn `nrms_ref+pop` +0,0038 [+0,0010, +0,0065] (n = 30 000). Chênh nhỏ và baseline chỉ một seed ⇒ coi là xấp xỉ.

## 6. Các hướng LLM khác (so với `frozen`, nội dung)

| Hướng | Kết quả | Ghi chú |
|---|---|---|
| **H8 pooling** (`pool_lse`, `pool_topk`) | **+0,0128 / +0,0107** nDCG@10, cold AUC +0,0058 / +0,0032 | tái lập trên Qwen3 (`pool_lse` +0,0127 so với `frozen` Qwen3); không tham số, không huấn luyện |
| H7 `sid_profile+meanpool` | +0,0035 | nhỏ; `sid_gpt` (xếp hạng bằng mô hình sinh) ngang ngẫu nhiên (0,2898) |
| H4 `community_graphrag` | +0,0043 nhưng cold AUC −0,0031 | |
| H2 LoRA/DoRA | +0,0016 – +0,0029; cold AUC giảm | head MLP −0,005 |
| H10 `um_knn_rag` / `um_encoder` | +0,0077 / +0,0049 so với `um_frozen` | ≈ `pool_lse` (−0,0004, không khác biệt) |
| H3 RAG (entity / kNN / user) | −0,0080 … −0,0003 | không giúp |
| H1 history-as-text | −0,0211 … −0,0014 | |
| H6 KAR-style (n = 2 500) | `kar_item` +0,0079 [+0,0021, +0,0137] nhưng hết khác biệt khi có pop (+0,0036); `kar_user` −0,0308 | CI rộng vì n nhỏ |
| H9 LLM judge (n = 2 000) | `rerank_llm` −0,0112 [−0,0216, −0,0008], `rerank_fused` −0,0051 (không khác biệt) | giám khảo 1,7B zero-shot yếu |
| Quét encoder | bge-base −0,0185, bge-large +0,0015 (n.s.), Qwen3-Embedding-0.6B −0,0125, TF-IDF/LSA −0,0443 | encoder lớn hơn / LLM **không** giúp mean-pool zero-shot |

**Reader Fastformer / NRMS trên vector BGE đóng băng** (đối chứng: `um_frozen` = V10 `llmenc_ca`): `um_ff_frozen` 0,3949 (−0,0008, không khác biệt), `um_nrms_frozen` 0,3862 (−0,0095), `um_ff_knn_rag` 0,3984 (kém `um_knn_rag` 0,4033 là −0,0049). Module attention **không phải yếu tố quyết định**; và `pool_lse` không tham số (0,4037) hơn `um_frozen` +0,0080 [+0,0061, +0,0099]: user encoder học được trên vector BGE đóng băng **không** hơn một phép pooling cố định.

## 7. Hệ quả cho luận văn (nhận định, không phải kết quả đo)

1. **Không thể khẳng định "LLM embedding hơn NRMS/NAML/Fastformer"** trên giao thức này. Điều có bằng chứng: embedding đóng băng nhỏ + pooling đơn giản **ngang** baseline huấn luyện đúng cách (không huấn luyện, không cần ID), xấp xỉ hoặc nhỉnh hơn trên tin cold, thua baseline khi baseline được huấn luyện trên mọi ngày.
2. Kết luận có bằng chứng nhất và sạch nhất (xác định, CI ghép cặp, tái lập trên hai encoder): **late-interaction pooling của lịch sử** (+0,013) và việc **baseline V10 yếu vì thiếu vector huấn luyện sẵn**.
3. Hướng "mask của LLM user tower" **chưa có đủ bằng chứng** (nhiễu seed ≥ chênh lệch giữa các mode). Hướng RAG/GraphRAG/Semantic ID/LLM judge/augment cho hiệu ứng ≤ 0,005 hoặc âm.
4. Trong luận văn cần **báo cáo nhiễu seed** và dùng ≥ 3 seed cho mọi so sánh giữa mô hình đã huấn luyện; dùng bảng *content only* để so với bài báo, bảng `+pop` chỉ cho thiết lập streaming.
5. Hai lối đi khả dĩ: (a) luận văn dạng *benchmark nghiêm ngặt* (baseline chuẩn, nhiễu seed, giao thức chronological, cold/warm) — bằng chứng hiện có đã đủ; (b) tìm hiệu ứng dương ở dữ liệu lớn hơn (MIND-large, EB-NeRD), nơi LLM có thể hơn.

## 8. Giới hạn

Một dataset, một tập dev; hầu hết hàng một seed; baseline huấn luyện trên `train_core` (80,7% số impression, trừ `nrms_ref_refit`); GloVe-6B; Fastformer cài lại; không tinh chỉnh siêu tham số cho bất kỳ mô hình nào; số bài báo chưa đối chiếu PDF; rung 2 của thang ablation chưa hội tụ.
