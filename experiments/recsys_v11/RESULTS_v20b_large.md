# Kết quả đo thật — v20b trên MIND-large (Kaggle, 2 × Tesla T4, 3,2 giờ, mọi stage xong)

Nguồn: log và `results_v20b/` của `v20b_mindlarge_sid_histq.ipynb` (kernel `ngpnguyen/v20b-mindlarge-sid-histq`, phiên bản 1). Mọi số là trên **`MINDlarge_dev` (376 471 impression)**, trừ đối chứng lịch sử xáo trộn (20 000 impression). Encoder Qwen3-Embedding-0,6B; CI 95% ghép cặp theo impression.

## 0. Chạy có ổn không?
* Thời gian: controls 58 s, pooling 300 s, `sid2` 3 944 s, `histq_mlp` 6 349 s (mã hoá 314 732 lịch sử validation / test trong 3 123 s và 300 000 lịch sử train trong 2 963 s trên 2 GPU). Tổng 3,2 giờ.
* RAM đỉnh 21,5 GB. Đối chứng `frozen` 0,3787 và `pool_lse` 0,3916 trùng các run trước.

## 1. SID v2 (H7)
Mã RQ-KMeans 3 tầng × 256 từ vector Qwen3 đọc `cta` (tầng 1: 256 cụm; tầng 2: 23 416 tiền tố; tầng 3: 79 962). SID-GPT 3,6 triệu tham số, 4 epoch (18 392 bước).

| Hàng | test nDCG@10 | AUC | cold AUC |
|---|---|---|---|
| `sid_gpt_sumlevels` (cách chấm của v7: cộng cả 3 tầng) | 0,2990 | 0,5334 | 0,4735 |
| `sid_gpt` (chọn trên validation: **chỉ tầng 1**, không PMI) | 0,3473 | 0,5717 | 0,5505 |
| `sid_profile` | 0,3492 | 0,5640 | 0,5609 |
| `frozen` (mean pooling) | 0,3787 | 0,6215 | 0,6583 |

* **Giả thuyết của v2 đúng:** chỉ chấm tầng chủ đề hơn cách cộng cả 3 tầng **+0,0483 [+0,0474, +0,0493]**; tầng mịn của bài mới làm hỏng điểm.
* **Mô hình có dùng lịch sử:** đổi lịch sử sang của người khác làm giảm −0,0626 [−0,0662, −0,0589] (20 000 impression).
* **Nhưng vẫn kém mean pooling:** `sid_gpt` −0,0314 [−0,0324, −0,0303], `sid_profile` −0,0294; cold AUC kém xa (−0,10). Mã rời rạc làm mất chi tiết của từng bài mà vector liên tục giữ được.

## 2. H1 + lớp học được (`histq_mlp`)
300 000 lịch sử train (754 956 mẫu), 2 seed (0,3771 / 0,3739), bước tốt nhất epoch 2.

| So sánh (A − B) | Δ nDCG@10 [CI 95%] | Δ cold AUC |
|---|---|---|
| `histq_mlp` (0,3755) − `histq_instr` (H1 zero-shot, 0,3676) | **+0,0080 [+0,0073, +0,0086]** | +0,0123 |
| `histq_mlp` − `frozen` | −0,0032 [−0,0041, −0,0023] | −0,0098 |
| `histq_mlp` − `pool_lse` | −0,0161 [−0,0170, −0,0152] | −0,0157 |
| `histq_instr` − `frozen` | −0,0111 [−0,0119, −0,0104] | −0,0221 |

* Lớp học được thu hẹp khoảng cách của H1 (+0,0080) nhưng vẫn dưới mean pooling. So với tháp H5 tinh chỉnh LLM (0,4002, v19b), phần do tinh chỉnh chính LLM lớn hơn nhiều: khoảng +0,025 trên cùng đầu vào (hai run khác nhau, không ghép cặp ở đây; v20c đo ghép cặp).

## 3. Kết luận
* SID: lỗi "ngang ngẫu nhiên" của v7 là do cách chấm (cộng tầng mịn), không phải lỗi code; sửa cách chấm +0,048, nhưng SID vẫn là biểu diễn kém hơn vector liên tục.
* Đọc lịch sử dạng văn bản bằng LLM đóng băng: thêm lớp học được giúp (+0,008) nhưng chưa bằng mean pooling; chỉ tinh chỉnh LLM (H5) mới vượt.
* Các thành phần này vẫn được đưa vào v20c để bộ ghép tự quyết có dùng hay không.
