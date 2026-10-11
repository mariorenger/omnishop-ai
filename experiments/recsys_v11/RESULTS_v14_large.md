# Kết quả đo thật — v14 trên MIND-large (Kaggle, 2 × Tesla T4 nhưng mã chỉ dùng 1 GPU, 2,90 giờ)

Nguồn: log Kaggle của lần chạy `v14_mindlarge_hf_supermodel.ipynb` (QUICK=0, ngân sách 10 giờ) do người dùng gửi lại. Mọi số là trên **`MINDlarge_dev` (376 471 impression)**
trừ các hàng tập con (n ghi rõ). CI 95% là ghép cặp theo impression: chỉ tính nhiễu *mẫu test*, **không** tính nhiễu huấn luyện. Số công bố ở §4 lấy từ kết quả tìm kiếm (bài báo / khảo sát), **chưa mở PDF**.

## 0. Chạy có ổn không?

* Không stage nào lỗi (`Stages: 7 done`). Bố cục repo Hugging Face `Recommenders/MIND` đúng như bộ tải giả định: 16 file ở cấp cao nhất (`MINDlarge_train.zip` 531 MB, `MINDlarge_dev.zip` 104 MB tải trong 10 s và 2 s). Train 2 232 748 impression, 101 527 bài; dev 376 471 impression, 72 023 bài; 104 151 bài sau khi gộp.
* Tổng 10 500 s ≈ **2,90 giờ** (ước trước: 3–5 giờ). Giây của stage: mã hoá E0 1 046, mã hoá văn bản `cta` 1 196 (nằm trong `moredata` 1 314), `controls` 39, `pooling` 165, `graph` 201, **`usermodel` 4 574**, `llm_user` 2 614, `supermodel` 154.
* Bộ nhớ: RAM tiến trình đỉnh **12,6 GB** (máy 34 GB), GPU đỉnh **6,0 GB** (16 GB). Guard mã hoá Qwen3: cos nhỏ nhất 0,9998.
* Chia dữ liệu: 1 801 231 train_core / 431 517 validation (lấy ngẫu nhiên 60 000) / 376 471 test; 2 712 612 mẫu huấn luyện (dùng 600 000). **82,2% click của test là bài lạnh** (chưa thấy trong train).
* Không có kiểm tra tái lập V10 (V10 không có số cho MIND-large).

## 1. Kết quả chính (nội dung thuần, không popularity)

| Hàng | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC |
|---|---|---|---|---|---|---|
| `frozen` (mean-pool Qwen3) | 0,6215 | 0,2922 | 0,3168 | 0,3787 | 0,6583 | 0,4784 |
| `popularity` (CTR online V10, không nội dung) | 0,6540 | 0,3126 | 0,3415 | 0,4040 | 0,6986 | 0,5213 |
| `pool_lse` | 0,6352 | 0,3046 | 0,3319 | 0,3933 | 0,6662 | 0,5179 |
| `txt_cta_lse` (text có category, late-interaction) | 0,6554 | 0,3206 | 0,3509 | 0,4102 | 0,6620 | 0,6333 |
| `um_knn_rag` (reader tốt nhất, 2 seed) | 0,6599 | 0,3117 | 0,3463 | 0,4089 | 0,6966 | 0,5507 |
| `ut_causal+meanpool` (tower LoRA, n = 30 000) | 0,6384 | 0,2986 | 0,3277 | 0,3904 | 0,6663 | 0,5454 |
| `super_base` (= `um_knn_rag` ghép 2 seed) | 0,6624 | 0,3135 | 0,3483 | 0,4111 | 0,6995 | 0,5519 |
| **`super_zs`** (+ `txt_cta_lse`, tốt nhất nội dung thuần) | **0,6762** | 0,3295 | 0,3663 | **0,4277** | 0,7081 | 0,5828 |
| `super_llm` (+ 3 reader, `hist100_lse`) | 0,6756 | 0,3283 | 0,3646 | 0,4263 | 0,7031 | 0,6028 |

Có popularity online (khung streaming: đếm click của các impression *trước đó trong chính ngày test*; **không so được** với leaderboard offline):
`frozen+pop` 0,6757 / 0,4312; `txt_cta_lse+pop` 0,6914 / 0,4441; **`super_llm+pop` 0,7048 / 0,4542**; `super_apex+pop` (n = 30 000) 0,6980 / 0,4444.

## 2. Phát hiện

**2.1 Thang supermodel (ghép cặp).** `super_base → super_zs` +0,0166 [+0,0161, +0,0171] (thêm `txt_cta_lse` trọng số 0,5); `super_zs → super_llm` **−0,0014** [−0,0017, −0,0011] (thêm 3 reader + `hist100_lse` làm test *giảm* dù validation tăng 0,4152 → 0,4177);
`super_llm → super_apex` −0,0002 [−0,0009, +0,0006] (tower không giúp); `super_llm → super_llm+pop` +0,0279 [+0,0273, +0,0284]. Bỏ-một-nhóm từ `super_llm+pop`: bỏ zero-shot −0,0077, bỏ popularity −0,0288, **bỏ reader +0,0015** (tăng).
Lợi của ghép nằm ở việc kết hợp *một* reader với *một* zero-shot (khác loại); các thành phần thêm không giúp.

**2.2 Validation quá "ấm" so với test (lỗi thiết kế của supermodel v14).** Lợi so với `frozen` trên validation và trên test (nDCG@10):

| Hàng | val − frozen | test − frozen |
|---|---|---|
| `random_vec` | −0,0699 | −0,0906 |
| `pool_lse` | +0,0232 | +0,0146 |
| `txt_cta_lse` | +0,0370 | +0,0315 |
| `knn_rag` | **+0,0130** | **−0,0037** |
| `community_graphrag` | **+0,0199** | **−0,0066** |
| `um_frozen` | **+0,0596** | +0,0192 |
| `um_knn_rag` | **+0,0653** | +0,0302 |

Hàng zero-shot giữ gần nguyên (giảm 0,005–0,009); reader mất khoảng 0,035–0,040; RAG và GraphRAG đổi dấu. Reader có warm AUC rất cao (0,55–0,62 so với 0,478 của `frozen`) và cold AUC khiêm tốn: lợi của chúng nằm ở bài đã thấy.
Test có 82% click là bài lạnh; validation là các ngày sát train nên (giả thuyết, log không in tỉ lệ lạnh của validation) ấm hơn ⇒ trọng số khớp trên validation ưu ái reader / RAG quá mức. Ngoài ra mọi hàng đều "dễ hơn" trên test (ngay cả `random_vec` +0,020): mức nDCG tuyệt đối không so được giữa hai tập, chỉ so được *chênh so với đối chứng*.

**2.3 Lỗi khởi tạo `super_apex+pop`.** Bản `+pop` của bậc tower khởi tạo từ trọng số *không-pop* của chính nó thay vì từ `super_llm+pop`, nên thua nó −0,0065 [−0,0079, −0,0052] trên cùng 30 000 impression. Mô hình tốt nhất thật sự là `super_llm+pop`, không phải `super_apex+pop`.

**2.4 Từng kỹ thuật (đối chứng của chính nó).** Tốt: category trong text +0,0169 [+0,0163, +0,0175] so với `pool_lse` (mean-pool +0,0186 so với `frozen`); late-interaction +0,0146; top-k +0,0143; k-means interest +0,0104; reader `knn_rag` +0,0110 so với `um_frozen`.
Không giúp: RAG entity −0,0013, kNN −0,0037, GraphRAG cộng đồng −0,0066, 100 click ≈ 0 (+0,0004 / −0,0004), multi-interest học được (`mi4` ≈ `mi1`, cosine giữa các interest 0,94–0,96: các interest gần như trùng nhau), tower LoRA = `pool_lse` (−0,0000 [−0,0027, +0,0026], n = 30 000).
Tower: 800 bước × 16 = 12 800 mẫu (0,5% của 2,7 triệu); validation còn tăng ở bước 800 (0,3513 → 0,3622 → 0,3631).

**2.5 Nhiễu seed của reader (std mẫu, n−1):** `um_frozen` 0,0018, `um_text` 0,0099 (0,3993 / 0,3934 / 0,4127), `um_knn_rag` 0,0025, `um_mi4_text` 0,0056.

## 3. Đã đo ở đây so với dự đoán ghi trước (README §0a)

Đúng: RAM 10–16 GB (12,6 GB), Hub chạy ngay lần đầu, nền của supermodel là một reader (`um_knn_rag`). Sai: thời gian (2,9 giờ, thấp hơn 3–5 giờ); `super_llm` hơn `super_base` +0,0152 nhưng *toàn bộ* mức tăng đến từ nhóm zero-shot, các reader thêm vào làm giảm.

## 4. So với số công bố (từ kết quả tìm kiếm, chưa mở PDF)

NRMS (GloVe) trên MIND-large: AUC 0,670 – 0,691 và nDCG@10 0,418 – 0,434 tuỳ bài báo; PLM-NR 0,7064 / 0,4438; UNBERT 0,7068 / 0,4478; đứng đầu leaderboard: ensemble Fastformer + PLM-NR AUC 0,7268.
`super_zs` (0,6762 / 0,4277) nằm trong dải NRMS và thấp hơn các mô hình PLM đơn khoảng 0,03 AUC, thấp hơn top khoảng 0,05. Dev khác test của leaderboard nên lệch ±0,01 là bình thường.
Encoder lớn hơn *không* phải nguyên nhân chính: ở MIND-small (RESULTS_v7 §6) so với bge-small mean-pool, bge-base −0,0185, bge-large +0,0015 (không khác biệt), Qwen3-0,6B −0,0125.

## 5. Việc tiếp theo suy ra từ số đo

1. Điều kiện hoá vào mức "lạnh/ấm" thay vì khớp một bộ trọng số tuyến tính trên validation ấm (xếp hạng học được trên validation với cờ lạnh làm đặc trưng).
2. Thêm đặc trưng nhãn-tự-do về độ mới / mức hiển thị của bài (popularity online đóng góp +0,028; phần không cần nhãn của test chưa được khai thác).
3. Reader huấn luyện trên toàn bộ mẫu thay vì 600 nghìn; bỏ các view không giúp (entity, mi4).
4. Một encoder thứ hai khác họ (BGE) để ghép; encoder rất lớn chỉ thử sau.
