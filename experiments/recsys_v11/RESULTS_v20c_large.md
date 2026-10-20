# Kết quả đo thật — v20c trên MIND-large: ghép các thành phần LLM-text mới vào hệ tốt nhất (Kaggle, 2 × T4, 3,72 giờ)

Nguồn: log và `results_v20c/` của `v20c_mindlarge_fusion.ipynb` (kernel `nguyenpn/v20c-mindlarge-fusion`, phiên bản 1). Đầu vào: output của v19a, v19b (`kernel_sources`) và dataset riêng tư `nguyenpn/v20-components` (13 file điểm thành phần của v20a / v20b). Mọi số trên **`MINDlarge_dev` (376 471 impression)**, CI 95% ghép cặp theo impression (không tính nhiễu huấn luyện).

* **Đối chứng** = thí nghiệm của v19c (hệ tốt nhất trước đó: pipeline v16b + reader huấn luyện kỹ hơn + tháp H5).
* **Thí nghiệm** = đối chứng + H6 toàn bộ (`kar_item`, `kar_item_lse`, `kar_exp`, reader `catkarplus`, `qwen3kar`) + SID v2 (`sid_gpt`, `sid_profile`) + `histq_instr`, `histq_mlp`.

## 0. Chạy có ổn không?
* **Tái lập:** đối chứng trùng thí nghiệm của v19c đến 4 chữ số ở cả 8 hàng (`control_vs_v19c.md`).
* Bốn nhánh xong: `ctrl_s0`, `treat_s0`, `ctrl_cw0`, `treat_cw0`. Hai nhánh seed 1 **bị bỏ** vì hết giờ (còn 0,88 giờ < 1,6 giờ cần) → không có số đo nhiễu seed riêng cho v20c; dùng số đo ở v19c: ±0,0008.
* Bảng so sánh từng thành phần (`components_v20c.md`) **không được tạo**: khi không có nhánh seed 1, dữ liệu của nhánh cuối đã được giải phóng trước bước đó (lỗi của notebook, không ảnh hưởng các hàng ghép).
* Stacker của thí nghiệm: 69 thành phần, 131 đặc trưng.

## 1. Bảng chính (stacker seed 0; cổng viết trước)

| Hàng | đối chứng (hệ tốt nhất cũ) | thí nghiệm | Δ nDCG@10 [CI 95%] | Δ AUC [CI 95%] |
|---|---|---|---|---|
| `super_zs` | 0,4373 | 0,4376 | +0,0003 [+0,0001, +0,0005] | +0,0002 |
| `super_llm` | 0,4380 | 0,4383 | +0,0003 [+0,0000, +0,0006] | +0,0009 |
| `stack_llm` | 0,4358 | 0,4390 | **+0,0032 [+0,0029, +0,0035]** | +0,0040 |
| `stack_meta` | 0,4531 | 0,4534 | +0,0004 [+0,0000, +0,0007] | +0,0002 |
| `stack_meta_v15cfg` | 0,4562 | 0,4566 | +0,0004 [+0,0000, +0,0008] | +0,0004 |
| `super_llm+pop` | 0,4634 | 0,4635 | +0,0001 [−0,0001, +0,0004] | −0,0001 |
| `stack_pop` | 0,4731 | **0,4749** | **+0,0018 [+0,0015, +0,0022]** | +0,0026 |

| Cổng | Δ [CI 95%] | Kết luận |
|---|---|---|
| G1 nội dung (`super_llm`) | +0,0003 [+0,0000, +0,0006] | PASS (sát 0) |
| G2 offline (`stack_meta`) | +0,0004 [+0,0000, +0,0007] | PASS (sát 0; nhỏ hơn nhiễu seed ±0,0008) |
| G3 online (`stack_pop`) | +0,0018 [+0,0015, +0,0022] | **PASS**, lớn hơn nhiễu seed |

## 2. Quy tắc phụ (đăng ký trước ở v20c): supermodel chọn theo validation không trọng số

| Hàng | đối chứng | thí nghiệm | Δ thí nghiệm − đối chứng [CI 95%] |
|---|---|---|---|
| `super_base` | 0,4270 | 0,4325 | +0,0054 [+0,0050, +0,0059] |
| `super_zs` | 0,4379 | 0,4416 | +0,0037 [+0,0034, +0,0041] |
| `super_llm` | 0,4398 | **0,4407** | +0,0009 [+0,0006, +0,0012] |
| `super_llm+pop` | 0,4609 | 0,4643 | +0,0034 [+0,0030, +0,0037] |

Theo quy tắc này nền được chọn là reader trên vector **có đoạn LLM viết** (`catkarplus:um_knn_rag`). Cold AUC thấp hơn quy tắc có trọng số (0,7097 so với 0,7181 ở `super_llm`).

## 3. Thành phần mới được dùng thế nào
* **Supermodel (quy tắc chính):** nền `cat:um_knn_rag`; thêm `kar:kar_item_lse` (trọng số 0,25) và `qwen3kar:um_knn_rag` (0,25) — cả hai là thành phần có đoạn LLM viết. SID và `histq_*` không được chọn.
* **Stacker:** đặc trưng có gain lớn nhất trong các hàng `stack_*` là `catkarplus:um_frozen` (0,08–0,11) và `catkarplus:um_knn_rag` (0,06–0,07) — reader trên vector có đoạn LLM viết thay chỗ `catplus` ở đầu bảng.
* Phần lợi của v20c đến từ H6 (LLM viết thêm kiến thức cho bài) dùng như **thành phần bổ sung** cho stacker, dù riêng lẻ nó không hơn reader cũ (v20a). Đây là suy luận từ trọng số / gain, không phải phép thử bỏ-từng-thành-phần.

## 4. Mô hình mới nhất (để đưa vào báo cáo)

**Thành phần** (mọi encoder giữ đóng băng; không dùng nhãn test để chọn gì):
1. Vector bài: Qwen3-Embedding-0,6B và BGE-large (tiêu đề + tóm tắt; thêm biến thể có chuyên mục), vector nối hai encoder.
2. **LLM-text:** Qwen3-1,7B viết 1–2 câu mở rộng cho từng bài (104 151 bài); vector đoạn mở rộng được cộng vào vector bài (α = 1).
3. Reader học trên vector đóng băng (theo ứng viên, có / không kNN-RAG): trên vector nối, trên BGE-large, trên vector nối có đoạn LLM viết (`catkarplus`), trên Qwen3 có đoạn LLM viết (`qwen3kar`); tháp người dùng LLM H5 (LoRA) là một thành phần.
4. Zero-shot: late interaction (LSE) trên lịch sử 50/100 click, văn bản có chuyên mục, vector có đoạn LLM viết.
5. Ghép: supermodel tuyến tính (trọng số chọn trên validation) và stacker XGBoost trên 69 thành phần + đặc trưng ngữ cảnh / phơi nhiễm (offline) / popularity (online).

**Số đo (test `MINDlarge_dev`):**

| Chế độ | Hàng | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| Chỉ nội dung + lịch sử (quy tắc chính) | `super_llm` | 0,6880 | 0,3392 | 0,3772 | 0,4383 |
| Chỉ nội dung + lịch sử (quy tắc phụ, đăng ký trước) | `super_llm` (cw0) | 0,6939 | 0,3418 | 0,3795 | **0,4407** |
| Offline (+ phơi nhiễm, không dùng click ngày test) | `stack_meta_v15cfg` | 0,7054 | 0,3529 | 0,3947 | 0,4566 |
| Offline, hàng chính đăng ký trước | `stack_meta` | 0,7038 | 0,3504 | 0,3914 | 0,4534 |
| Online (+ popularity từ click trước đó trong ngày test) | `stack_pop` | **0,7216** | **0,3676** | **0,4142** | **0,4749** |

**So với hệ tốt nhất trước (v19c, cùng impression):** online +0,0018 (vượt nhiễu seed), offline +0,0004, nội dung +0,0003 (quy tắc chính) / +0,0009 (quy tắc phụ).
**So với PLM-NR (chỉ nội dung + lịch sử):** 0,6939 / 0,3418 / 0,4407 so với 0,7064 / 0,3539 / 0,4438 — vẫn thấp hơn; tập test khác (bài PLM-NR không ghi rõ tập).

## 5. Giới hạn
* Một seed stacker cho v20c (nhánh seed 1 hết giờ); nhiễu seed ±0,0008 lấy từ v19c.
* Một ngày test; CI chỉ tính nhiễu mẫu test.
* Bảng so sánh từng thành phần chưa có (xem §0).
