# Kết quả đo thật — v19b trên MIND-large (Kaggle, 2 × Tesla T4, 10,09 giờ, mọi stage xong)

Nguồn: log và thư mục kết quả của lần chạy `v19b_mindlarge_h5tower.ipynb` (kernel `nguyenpn/v19b-mindlarge-h5tower`, phiên bản 1, đẩy lên 2026-10-19 04:58 UTC, chạy song song với v19a trên cùng tài khoản, QUICK=0, `BUDGET_H` 11,3). Mọi số là trên **`MINDlarge_dev` (376 471 impression)**, trừ H1 (tập con 60 000 impression test). CI 95% ghép cặp theo impression chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện.

## 0. Chạy có ổn không?

* **Thời gian** (đo thật):
  * Run 1 BGE-large: E0 233 s; controls 59, pooling 299, moredata 456, graph 274, usermodel 1 758 s.
  * Run 2 vector nối: E0 515 s; controls 64, graph 420, usermodel 1 785 s.
  * Run 3 Qwen3-0,6B (**đối chứng**): controls 52, pooling 283, moredata 762, graph 252, usermodel 3 001, supermodel 255, stack 3 320 s. Đối chứng xong ở 4,12 giờ.
  * **Nhánh thí nghiệm** trong run 3:
    * H1 (`histquery`): 728 s, 70 017 lịch sử khác nhau, mã hoá trên 2 GPU trong 706 s.
    * H5 (`llm_user`): **16 582 s** (4,6 giờ). Huấn luyện 200 phút (1,84 s mỗi bước, 5 872 bước × batch 16 = 93 952 lịch sử, khoảng 3,5% của 2 712 612 mẫu train). Chấm toàn bộ split: 122 559 lịch sử validation trong 1 531 s và 246 591 lịch sử test trong 3 008 s, trên 2 GPU.
    * Supermodel 258 s, stack 3 392 s.
  * Tổng **10,09 giờ**. 42 file điểm thành phần được ghi ra output, gồm điểm của tháp H5 (`ut_causal`).
* **Bộ nhớ:** RAM đỉnh 25,9 GB ở stacker của nhánh thí nghiệm (cảnh báo 70–80% của 32 GB); GPU đỉnh 6,3 GB (tháp H5).
* **Tái lập:** đối chứng trùng v16b đến 4 chữ số ở cả bảy hàng ghép (lần thứ tư: v18a, v18b, v19a, v19b).

## 1. Tháp người dùng LLM H5 (Qwen3-Embedding-0,6B + LoRA r=16, causal, đọc 10 tiêu đề gần nhất dạng văn bản)

**Đường học.** Validation đo giữa chừng trên 20 000 impression:

| bước | 1 938 | 3 934 | 5 872 |
|---|---|---|---|
| validation nDCG@10 | 0,3813 | 0,3899 | 0,3906 |

* Loss: 5,17 (đầu) → 4,07 (bước 1 950) → 3,92 (bước 3 950) → 3,96 (bước 5 850).
* Bước tốt nhất là bước cuối; lịch cosine đưa learning rate về 0 ở bước 5 872.
* Huấn luyện không bị ngân sách cắt ngang.

**Trên toàn bộ test** (376 471 impression; cùng impression nên mọi CI là ghép cặp):

| So sánh (A − B) | Δ nDCG@10 [CI 95%] | Δ cold AUC |
|---|---|---|
| `ut_causal` (0,4002, AUC 0,6470) − `frozen` (mean pooling) | **+0,0215 [+0,0206, +0,0224]** | +0,0071 |
| `ut_causal` − `pool_lse` (zero-shot tốt nhất cùng encoder, văn bản ta) | **+0,0085 [+0,0077, +0,0094]** | +0,0012 |
| `ut_causal` − `txt_cta_lse` (zero-shot, văn bản cta) | −0,0100 [−0,0109, −0,0092] | +0,0034 |
| `ut_causal` − `qwen3:um_frozen` (reader trên vector đóng băng) | −0,0048 [−0,0055, −0,0040] | −0,0041 |
| `ut_causal` − `qwen3:um_knn_rag` | −0,0093 [−0,0101, −0,0086] | −0,0171 |
| `ut_causal` − `cat:um_knn_rag` (reader tốt nhất của v16b) | −0,0224 [−0,0232, −0,0216] | −0,0411 |
| `ut_causal+meanpool` (0,4082) − `pool_lse` | +0,0165 [+0,0158, +0,0173] | +0,0165 |

Với popularity: `ut_causal+pop` 0,4456, `ut_causal+meanpool+pop` 0,4502.

**H1 — cùng encoder, zero-shot, lịch sử thành một câu truy vấn có chỉ dẫn** (tập con 20 000 validation / 60 000 test):
* `histq_instr` 0,3664: −0,0101 [−0,0120, −0,0083] so với `frozen` (WORSE).
* `histq_instr+meanpool` 0,3780: +0,0014 [+0,0002, +0,0027] so với `frozen`; −0,0113 [−0,0130, −0,0096] so với `pool_lse`.
* Phần do huấn luyện mang lại, `ut_causal` − `histq_instr` trên cùng 60 000 impression: **+0,0319 [+0,0299, +0,0340]**.

## 2. Bảng chính: đối chứng (pipeline v16b) với thí nghiệm (+ H5), cùng impression

| Hàng | đối chứng | thí nghiệm | Δ nDCG@10 [CI 95%] | Δ AUC [CI 95%] |
|---|---|---|---|---|
| `super_llm` | 0,4373 | 0,4373 | 0 (H5 không được chọn) | 0 |
| `stack_llm` | 0,4369 | 0,4370 | +0,0001 [−0,0002, +0,0003] | −0,0003 [−0,0005, −0,0001] |
| `stack_meta` | 0,4516 | 0,4515 | −0,0001 [−0,0004, +0,0002] | −0,0009 [−0,0012, −0,0006] |
| `stack_meta_v15cfg` | 0,4543 | 0,4533 | −0,0011 [−0,0014, −0,0008] | −0,0013 [−0,0016, −0,0010] |
| `super_llm+pop` | 0,4638 | 0,4638 | 0 | 0 |
| `stack_pop` | 0,4731 | 0,4727 | −0,0004 [−0,0007, −0,0001] | −0,0015 [−0,0018, −0,0013] |

**Cổng viết trước khi chạy** (`gate_v19b.md`):

| Cổng | Δ [CI 95%] | Kết luận |
|---|---|---|
| G1 nội dung (`super_llm`) | 0 | **NO GAIN** |
| G2 offline (`stack_meta`) | −0,0001 [−0,0004, +0,0002] | **NO GAIN** |
| G3 online (`stack_pop`) | −0,0004 [−0,0007, −0,0001] | **WORSE** |

Trong supermodel, H5 xếp thứ 9 trên validation có trọng số bài lạnh (0,4276; các reader 0,4335–0,4573), nên không vào nhóm 3 reader được ghép. Trong stacker, nó là thêm một cột nhưng các hàng không đổi (±0,001).

## 3. Kết luận

* **H5 lần đầu được huấn luyện đủ để hơn zero-shot trên MIND-large.** Sau 5 872 bước (3,5% dữ liệu train, 200 phút trên T4), tháp đạt 0,4002: hơn `pool_lse` +0,0085, hơn cùng LLM zero-shot (H1) +0,0319. v14 sau 800 bước chỉ bằng `pool_lse`.
* **Nhưng vẫn dưới reader trên vector đóng băng:** −0,0048 so với reader Qwen3, −0,0224 so với reader tốt nhất của v16b. Reader đó chỉ huấn luyện khoảng 5–6 phút mỗi seed trên 1,2 triệu mẫu (vector đóng băng, mạng nhỏ).
* **Đường học gần phẳng ở cuối** (0,3899 → 0,3906 trong 1 938 bước cuối), một phần do learning rate giảm về 0. Muốn biết H5 còn lên được bao nhiêu thì cần huấn luyện dài hơn nhiều lần, vượt một phiên Kaggle.
* **Ở mức hệ thống H5 không thêm gì** (G1 / G2 NO GAIN, G3 WORSE −0,0004).
* **H1 (đọc lịch sử như một đoạn văn, zero-shot) kém hơn mean pooling.** Đây là lần thứ hai, sau MIND-small.
* **So với kết quả tốt nhất trước đó:** không vượt hàng nào. Chỉ nội dung vẫn là 0,4380 (v19a), offline 0,4570 (v18a), online 0,4731 (v16b).
* **So với PLM-NR (chỉ nội dung + lịch sử):** `stack_llm` của nhánh thí nghiệm có AUC 0,6850, MRR 0,3373, nDCG@10 0,4370, vẫn dưới PLM-NR (0,7064 / 0,3539 / 0,4438, tập test khác).
