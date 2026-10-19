# Kết quả đo thật — v18a trên MIND-large (Kaggle, 2 × Tesla T4, 8,15 giờ, mọi stage xong)

Nguồn: log và thư mục kết quả của lần chạy `v18a_mindlarge_q4b.ipynb` (kernel `mario123475/v18a-mindlarge-q4b`, phiên bản 1, đẩy lên 2026-10-18 19:28 UTC, QUICK=0, `BUDGET_H` 11,2). Mọi số là trên **`MINDlarge_dev` (376 471 impression)**. CI 95% ghép cặp theo impression chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện. Đối chứng và thí nghiệm nằm trong **cùng** notebook và đọc cùng validation / test.

## 0. Chạy có ổn không?

* **Thời gian** (đo thật):
  * Run 1 BGE-large: E0 234 s (2 GPU; v16b 1 GPU: 405 s), controls 54, pooling 288, moredata 438, graph 264, usermodel 1 713 s.
  * Run 2 vector nối Qwen3-0,6B ⊕ BGE-large: E0 509 s, controls 66, graph 424, usermodel 1 827 s.
  * Run 3 Qwen3-0,6B (**đối chứng**): pooling 291, moredata 758, graph 250, usermodel 3 056, supermodel 266, stack 3 331 s. Đối chứng xong ở 4,13 giờ (v16b: 4,49 giờ trên 1 GPU).
  * Run 4 `q4b`: **mã hoá 104 151 bài bằng Qwen3-Embedding-4B mất 2 923 s (0,81 giờ)**, trọng số linear fp16 + embedding / norm fp32, chia cho 2 GPU, không có hàng không hữu hạn; controls 73, pooling 467, graph 506, usermodel 1 908 s.
  * Run 5 `cat4` (vector 4B ⊕ BGE-large, 3 584 chiều, lấy từ cache): controls 86, graph 670, usermodel 2 101 s.
  * **Nhánh thí nghiệm** (supermodel + stacker của run 3 chạy lại với 60 thành phần từ 5 run, 116 cột): supermodel 346 s, stack 4 303 s.
  * Tổng **8,15 giờ**.
* **Bộ nhớ:** RAM đỉnh **28,5 GB** ở stacker của nhánh thí nghiệm, trên giới hạn container 32 GB. Có cảnh báo ≥ 70% ở `graph` của run 4 và 5 và ở cả hai stacker, nhưng không hết RAM. Thêm thành phần nữa ở quy mô này sẽ chạm giới hạn. GPU đỉnh 12,3 GB (4B fp16).
* **Đối chứng tái lập v16b chính xác:** `super_base` 0,4260, `super_llm` 0,4373, `stack_llm` 0,4369, `stack_meta` 0,4516, `stack_meta_v15cfg` 0,4543, `super_llm+pop` 0,4638, `stack_pop` 0,4731. Cả bảy hàng trùng v16b đến 4 chữ số, kể cả stacker XGBoost trên GPU. v18b cũng tái lập các reader của v16b. Vậy trên phần cứng này pipeline tất định giữa các lần chạy, và mọi chênh lệch dưới đây là do thêm các thành phần 4B (cùng với việc chọn trọng số và dựng cây lại khi có chúng).

## 1. Bảng chính: đối chứng (pipeline v16b) với thí nghiệm (+ thành phần 4B), cùng impression

| Hàng | đối chứng | thí nghiệm | Δ nDCG@10 [CI 95%] | Δ AUC [CI 95%] |
|---|---|---|---|---|
| `super_base` | 0,4260 | 0,4260 | 0 (cùng nền `cat:um_knn_rag`) | 0 |
| `super_zs` | 0,4373 | 0,4373 | 0 | 0 |
| `super_llm` (chỉ nội dung + lịch sử) | 0,4373 | 0,4365 | **−0,0009 [−0,0011, −0,0006]** | −0,0005 [−0,0007, −0,0003] |
| `stack_llm` | 0,4369 | **0,4420** | **+0,0051 [+0,0047, +0,0055]** | +0,0064 [+0,0060, +0,0068] |
| `stack_meta` (hàng chính, đăng ký trước) | 0,4516 | 0,4524 | **+0,0008 [+0,0004, +0,0012]** | +0,0002 [−0,0001, +0,0005] |
| `stack_meta_v15cfg` | 0,4543 | **0,4570** | **+0,0026 [+0,0023, +0,0030]** | +0,0027 [+0,0024, +0,0031] |
| `super_llm+pop` (streaming) | 0,4638 | 0,4632 | −0,0006 [−0,0008, −0,0004] | −0,0002 [−0,0003, −0,0001] |
| `stack_pop` (streaming) | 0,4731 | 0,4712 | **−0,0019 [−0,0022, −0,0015]** | +0,0005 [+0,0002, +0,0009] |

Cold AUC (thí nghiệm − đối chứng): `stack_llm` +0,0098, `stack_meta` +0,0042, `stack_meta_v15cfg` +0,0039, `stack_pop` −0,0006, `super_llm` −0,0006.

**Cổng viết trước khi chạy** (`gate_v18a.md`):

| Cổng | Δ [CI 95%] | Kết luận | Số của nhánh thí nghiệm | Tốt nhất trước đó |
|---|---|---|---|---|
| G1 nội dung (`super_llm`) | −0,0009 [−0,0011, −0,0006] | **WORSE** | 0,4365 | 0,4373 |
| G2 offline (`stack_meta`) | +0,0008 [+0,0004, +0,0012] | **PASS** | 0,4570 (`stack_meta_v15cfg`) | 0,4543 |
| G3 online (`stack_pop`) | −0,0019 [−0,0022, −0,0015] | **WORSE** | 0,4712 | 0,4731 |

## 2. Encoder 4B làm đầu vào (test nDCG@10, 2 seed mỗi reader)

| | Qwen3-4B (cta) | Qwen3-0,6B | BGE-large | 0,6B ⊕ BGE (ta) | 4B ⊕ BGE (cta) |
|---|---|---|---|---|---|
| mean pooling `frozen` | 0,4036 | 0,3787 (văn bản ta) | 0,3933 (ta) | — | 0,4052 |
| zero-shot `pool_lse` | 0,4092 | 0,4102 (`txt_cta_lse`) | — | — | — |
| reader `um_frozen` | 0,4098 | 0,4092 (`um_text`, cùng văn bản cta) | 0,4152 | **0,4223** | 0,4158 |
| reader `um_knn_rag` | 0,4097 | 0,4095 | 0,4189 | **0,4226** | 0,4171 |

Cặp ghép:
* `q4b:um_frozen` − `qwen3:um_text` (cùng văn bản, 4B so với 0,6B): **+0,0006 [+0,0001, +0,0012]**, cold AUC +0,0100.
* `q4b:pool_lse` − `qwen3:txt_cta_lse`: −0,0010 [−0,0016, −0,0004].
* `q4b:um_frozen` − `bge:um_frozen`: −0,0053; `q4b:um_knn_rag` − `bge:um_knn_rag`: −0,0091.
* `cat4` − `cat`: −0,0065 (`um_frozen`), −0,0054 (`um_knn_rag`). So sánh này **bị lẫn**: nửa BGE của `cat4` đọc văn bản cta, còn nửa BGE của `cat` đọc ta. Ở BGE, cta kém ta 0,0030 khi zero-shot (v16b).

**Validation lệch so với test:** trên validation có trọng số bài lạnh, reader 4B đứng thứ 2 và 3 (`cat4:um_knn_rag` 0,4549, `q4b:um_frozen` 0,4532, so với `cat:um_frozen` 0,4467). Trên test thì chúng thấp hơn reader của `cat` và BGE. Vì vậy supermodel cho chúng trọng số (`cat4:um_knn_rag` 0,1, `q4b:um_frozen` 0,25; validation tăng 0,4275 → 0,4290) và mất 0,0009 trên test.

## 3. Kết luận

* **Encoder LLM lớn hơn không cho reader tốt hơn.** Qwen3-Embedding-4B chỉ ngang Qwen3-0,6B trên cùng văn bản (+0,0006 với reader, −0,0010 với zero-shot), và kém BGE-large 335M (−0,005 đến −0,009). Đây là câu trả lời đo được cho "có cần embedding lớn hơn không": không. Điều này khớp với v16b: BGE-large hơn Qwen3-0,6B, và độ đa dạng giữa hai không gian vector quan trọng hơn kích thước. Mã hoá bằng 4B tốn 0,81 giờ trên 2 T4.
* **Thêm thành phần 4B làm đặc trưng cho stacker giúp nhỏ, chỉ ở các hàng offline.**
  * `stack_llm` (không đặc trưng phơi nhiễm): **+0,0051**, rõ.
  * `stack_meta_v15cfg`: **+0,0026**, thành **hàng offline tốt nhất từ trước đến nay: 0,4570** (trước đó 0,4543).
  * `stack_meta` (hàng chính đăng ký trước): +0,0008, CI nằm trên 0 nên cổng G2 **PASS**.
  * Ngược lại, `stack_pop` −0,0019 và ghép tuyến tính −0,0009 / −0,0006 đều **kém đi**.
* **So với kết quả tốt nhất trước đó:**
  * Offline: **có vượt**, nhưng nhỏ (0,4570 so với 0,4543, +0,0027; Δ ghép cặp của chính hàng này so với đối chứng trong notebook là +0,0026 [+0,0023, +0,0030]).
  * Online: **không** (0,4712 < 0,4731).
  * Chỉ nội dung: **không** (0,4365 < 0,4373).
* **Lưu ý:** CI không tính nhiễu do dựng lại cây khi tập đặc trưng đổi. Chỉ cần thêm cột là XGBoost đã dựng cây khác. Các hàng stacker dịch theo hai chiều ngược nhau (+0,0026 ở `stack_meta_v15cfg`, −0,0019 ở `stack_pop`), gợi ý nhiễu loại này cỡ 0,002–0,003. Vì vậy mức tăng offline là có thật theo CI nhưng không lớn hơn nhiều so với độ dao động khi đổi đặc trưng.
* **Đặt cạnh v18b:** đòn bẩy lớn nhất ở mức thành phần là huấn luyện reader kỹ hơn. `cat:um_knn_rag` tăng 0,4226 → **0,4316** (+0,0090), lớn hơn nhiều so với mọi thay đổi do encoder 4B. Lần chạy tiếp theo nên ghép các reader của v18b với supermodel + stacker; nhánh này v18b chưa đo được vì hết ngân sách.

## 4. Còn lại sau lần chạy

* Quota GPU (2026-10-19 04:18 UTC): `mario123475` còn 4,98 giờ, `m81extend` còn 2,54 giờ; làm mới 2026-10-24 00:00 UTC.
* Một lần chạy "v16b + reader huấn luyện kỹ hơn trên vector nối (bỏ `sca`) + supermodel + stacker" ước khoảng 5,2 giờ (từ số đo của v18a / v18b). Không vừa quota còn lại của một tài khoản nào.
* Cần sửa trước lần chạy sau: ngưỡng thời gian chỉ áp cho stacker; ghi `scores/*.npz` vào output để có thể chạy lại riêng phần ghép (~1,2 giờ).
