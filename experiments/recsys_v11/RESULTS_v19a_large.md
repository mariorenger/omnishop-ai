# Kết quả đo thật — v19a trên MIND-large (Kaggle, 2 × Tesla T4, 9,54 giờ, mọi stage xong)

Nguồn: log và thư mục kết quả của lần chạy `v19a_mindlarge_plusreaders.ipynb` (kernel `nguyenpn/v19a-mindlarge-plusreaders`, phiên bản 1, đẩy lên 2026-10-19 04:58 UTC, QUICK=0, `BUDGET_H` 11,3). Mọi số là trên **`MINDlarge_dev` (376 471 impression)**, trừ H6 / H9 (tập con, ghi rõ ở mục 4). CI 95% ghép cặp theo impression chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện. Đối chứng và thí nghiệm nằm trong **cùng** notebook và đọc cùng validation / test.

## 0. Chạy có ổn không?

* **Thời gian** (đo thật):
  * Run 1 BGE-large: E0 224 s; controls 54, pooling 286, moredata 436, graph 269, usermodel 1 724 s.
  * Run 2 vector nối Qwen3-0,6B ⊕ BGE-large: E0 499 s; controls 66, graph 428, usermodel 1 820 s.
  * Run 3 Qwen3-0,6B (**đối chứng**): controls 55, pooling 287, moredata 750, graph 252, usermodel 3 080, augment (H6) 1 303, rerank (H9) 1 027, supermodel 275, stack 3 303 s. Đối chứng xong ở 4,78 giờ.
  * Run 4 `catplus` (reader kỹ hơn trên vector nối, 2 712 612 mẫu train, 6 epoch): controls 68, graph 424, usermodel 5 090 s (mỗi seed của reader theo ứng viên ~975–1 010 s).
  * Run 5 `bgeplus` (cùng công thức trên BGE-large): controls 57, graph 268, usermodel 5 365 s.
  * **Nhánh thí nghiệm** (supermodel + stacker của run 3 chạy lại với các reader mới): supermodel 336 s, stack 4 313 s.
  * Tổng **9,54 giờ**. 55 file điểm thành phần được ghi ra output (`results_v19a/scores/<run>/`) cho lần ghép riêng v19c.
* **Bộ nhớ:** RAM đỉnh **27,6 GB** ở stacker của nhánh thí nghiệm (cảnh báo 71% → 81% của giới hạn 32 GB, không hết RAM); stacker của đối chứng 23,8 GB. GPU đỉnh 11,1 GB (Qwen3-1,7B của H6 / H9).
* **Tái lập:**
  * Đối chứng trùng v16b đến 4 chữ số ở cả bảy hàng ghép: `super_base` 0,4260, `super_llm` 0,4373, `stack_llm` 0,4369, `stack_meta` 0,4516, `stack_meta_v15cfg` 0,4543, `super_llm+pop` 0,4638, `stack_pop` 0,4731. Đây là lần thứ ba (sau v18a, v18b).
  * Reader của `catplus` trùng v18b từng seed: `um_frozen` 0,4220 / 0,4282, `um_knn_rag` 0,4333 / 0,4300.

## 1. Bảng chính: đối chứng (pipeline v16b) với thí nghiệm (+ reader huấn luyện kỹ hơn), cùng impression

| Hàng | đối chứng | thí nghiệm | Δ nDCG@10 [CI 95%] | Δ AUC [CI 95%] |
|---|---|---|---|---|
| `super_base` | 0,4260 | 0,4260 | 0 (cùng nền `cat:um_knn_rag`) | 0 |
| `super_zs` | 0,4373 | 0,4373 | 0 | 0 |
| `super_llm` (chỉ nội dung + lịch sử) | 0,4373 | **0,4380** | **+0,0006 [+0,0005, +0,0008]** | +0,0008 [+0,0007, +0,0009] |
| `stack_llm` | 0,4369 | 0,4365 | −0,0004 [−0,0008, +0,0001] | +0,0034 [+0,0029, +0,0038] |
| `stack_meta` (hàng chính, đăng ký trước) | 0,4516 | 0,4533 | **+0,0016 [+0,0012, +0,0021]** | +0,0034 [+0,0030, +0,0038] |
| `stack_meta_v15cfg` | 0,4543 | 0,4557 | +0,0014 [+0,0009, +0,0018] | +0,0019 [+0,0015, +0,0023] |
| `super_llm+pop` (streaming) | 0,4638 | 0,4634 | −0,0004 [−0,0006, −0,0002] | −0,0002 [−0,0003, −0,0000] |
| `stack_pop` (streaming) | 0,4731 | 0,4721 | **−0,0010 [−0,0014, −0,0006]** | −0,0021 [−0,0025, −0,0018] |

Cold AUC (thí nghiệm − đối chứng): `stack_llm` −0,0061, `stack_meta` −0,0013, `stack_meta_v15cfg` −0,0017, `stack_pop` −0,0019, `super_llm` −0,0001.

**Cổng viết trước khi chạy** (`gate_v19a.md`):

| Cổng | Δ [CI 95%] | Kết luận | Số của nhánh thí nghiệm | Tốt nhất trước đó |
|---|---|---|---|---|
| G1 nội dung (`super_llm`) | +0,0006 [+0,0005, +0,0008] | **PASS** | 0,4380 | 0,4373 (v16b) |
| G2 offline (`stack_meta`) | +0,0016 [+0,0012, +0,0021] | **PASS** | 0,4557 (`stack_meta_v15cfg`) | 0,4570 (v18a) |
| G3 online (`stack_pop`) | −0,0010 [−0,0014, −0,0006] | **WORSE** | 0,4721 | 0,4731 (v16b) |

## 2. Reader huấn luyện kỹ hơn (test nDCG@10, 2 seed mỗi reader)

| Reader | công thức v16b | công thức kỹ hơn | Δ [CI 95%] | Δ cold AUC | epoch tốt nhất / đã chạy |
|---|---|---|---|---|---|
| BGE-large `um_frozen` | 0,4152 | **0,4251** (0,4236 / 0,4266) | **+0,0099 [+0,0095, +0,0103]** | +0,0005 | 5 / 6, 4 / 6 |
| BGE-large `um_knn_rag` | 0,4189 | 0,4287 (0,4277 / 0,4297) | **+0,0098 [+0,0094, +0,0103]** | −0,0042 | 5 / 6, 4 / 6 |
| vector nối `um_frozen` | 0,4223 | 0,4251 (0,4220 / 0,4282) | +0,0028 [+0,0024, +0,0033] | −0,0053 | 5 / 6, 5 / 6 |
| vector nối `um_knn_rag` | 0,4226 | **0,4316** (0,4333 / 0,4300) | **+0,0091 [+0,0086, +0,0095]** | −0,0096 | 5 / 6, 5 / 6 |

`catplus:um_knn_rag` − `bgeplus:um_knn_rag` +0,0029 [+0,0024, +0,0034]. Công thức kỹ hơn (toàn bộ 2 712 612 mẫu, 6 epoch, learning rate × 0,5 mỗi epoch từ epoch 2, ngưỡng dừng 0,0002) cho BGE-large cùng mức tăng như cho vector nối (~+0,01). Reader tốt nhất vẫn là 0,4316 (vector nối, kNN-RAG). Cold AUC phần lớn giảm.

## 3. Vì sao mức tăng của reader gần như không lên hệ thống

* **Supermodel giữ nền cũ.** Nền và trọng số được chọn theo nDCG@10 trên validation **có trọng số bài lạnh** (đưa phần click bài lạnh về 80%, như test). Trên thước đo đó thứ tự **ngược** với test:
  * Validation có trọng số: `cat:um_knn_rag` (công thức v16b) 0,4573 > `catplus:um_knn_rag` 0,4532 > `catplus:um_frozen` 0,4508.
  * Test: `catplus:um_knn_rag` hơn `cat:um_knn_rag` +0,0091.
  * Validation thường (không trọng số) thì cùng chiều với test: 0,4246 so với 0,4159.
  * Kết quả: nền vẫn là `cat:um_knn_rag`; reader mới chỉ nhận trọng số 0,1 trong `super_llm` (`cat:um_knn_rag`=1, `bge:hist100_lse`=0,1, `bge:pool_lse`=0,1, `qwen3:txt_cta_lse`=0,25, `catplus:um_knn_rag`=0,1). Vì vậy chỉ được +0,0006.
* **Stacker có dùng reader mới nhưng không thêm thông tin.** Cây dùng chúng nhiều nhất: `catplus:um_frozen` có tỷ lệ gain 0,14 ở `stack_llm`, `stack_meta` và `stack_meta_v15cfg`. Dù vậy `stack_llm` không đổi (−0,0004), `stack_meta` chỉ +0,0016, `stack_pop` −0,0010. Thông tin của reader mới phần lớn trùng với reader cũ cộng các đặc trưng ngữ cảnh. Đây là lần thứ hai thấy điều này (v16b: `super_llm` +0,0105 nhưng `stack_meta` chỉ +0,0008).

## 4. LLM sinh văn bản trên MIND-large (Qwen3-1,7B, tập con; lần đầu đo ở quy mô large)

So với `frozen` (mean pooling Qwen3-0,6B) trên đúng các impression đó; số MIND-small từ `RESULTS_v7.md`.

| Kỹ thuật | n test | nDCG@10 | Δ so với `frozen` [CI 95%] | MIND-small | + popularity: Δ so với `frozen+pop` |
|---|---|---|---|---|---|
| H6 `kar_item` (LLM mở rộng bài, α = 1) | 2 500 | 0,3741 (frozen 0,3698) | +0,0043 [−0,0015, +0,0101], không khác | +0,0079 [+0,0021, +0,0137] | −0,0005, không khác |
| H6 `kar_user` (hồ sơ người đọc do LLM viết) | 2 500 | 0,3420 | **−0,0278 [−0,0373, −0,0183], WORSE** | −0,0308 | −0,0130, WORSE |
| H6 `kar_user+meanpool` | 2 500 | 0,3661 | −0,0037 [−0,0086, +0,0012], không khác | — | +0,0010, không khác |
| H6 `kar_item+user` | 2 500 | 0,3677 | −0,0021 [−0,0088, +0,0046], không khác | — | +0,0043, không khác |
| H9 `rerank_llm` (LLM chấm Có / Không top-10) | 2 000 | 0,3638 | −0,0092 [−0,0195, +0,0012], không khác | −0,0112 (WORSE) | −0,0183, WORSE |
| H9 `rerank_fused` (gộp với điểm gốc) | 2 000 | 0,3787 | +0,0057 [−0,0016, +0,0130], không khác | −0,0051 | −0,0171, WORSE |

So với scorer zero-shot mạnh hơn trên cùng impression: `kar_item` − `pool_lse` −0,0033 [−0,0118, +0,0053]; `rerank_fused` − `pool_lse` −0,0028 [−0,0135, +0,0078].

**Chi phí đo được:**
* H6: 0,14 s mỗi mục khi thăm dò; 7 418 bài được mở rộng và 3 124 hồ sơ người đọc được viết; cả stage 1 303 s. Validation 745 / test 2 500 impression.
* H9: 41 ms mỗi câu hỏi; 600 validation / 2 000 test impression (17 923 câu hỏi cho test); stage 1 027 s.

**Ước tính để phủ toàn bộ (từ tốc độ trên, một GPU T4, chưa đo):**
* Mở rộng cả 104 151 bài: khoảng 3–4 giờ. Tốc độ thực trong stage (~0,11 s mỗi mục) nhanh hơn số thăm dò.
* H9 trên toàn bộ test: ~3,4 triệu câu hỏi, khoảng 38 giờ.

**Điều rút ra.** Trên MIND-large, H6 / H9 với LLM 1,7B zero-shot **không cho mức tăng có ý nghĩa** trên tập con.
* Mở rộng bài (`kar_item`) cùng chiều với MIND-small (+0,0043) nhưng CI chứa 0; vẫn thấp hơn `pool_lse`; hết khác biệt khi có popularity.
* Hồ sơ người đọc do LLM viết kém rõ ở cả hai quy mô.
* LLM chấm Có / Không không hơn điểm gốc; gộp với popularity thì kém.

## 5. So với PLM-NR (cùng điều kiện: chỉ nội dung + lịch sử)

| Mô hình | AUC | MRR | nDCG@10 |
|---|---|---|---|
| PLM-NR (công bố, tập test chính thức; nguồn thứ cấp) | 0,7064 | 0,3539 | 0,4438 |
| v19a `super_llm` (thí nghiệm) | 0,6871 | 0,3391 | 0,4380 |
| v19a `stack_llm` (thí nghiệm) | 0,6886 | 0,3365 | 0,4365 |
| v18a `stack_llm` (tốt nhất trước đó ở điều kiện này) | 0,6917 | 0,3416 | 0,4420 |
| v19a `stack_meta_v15cfg` (+ đếm / phơi nhiễm trong log) | 0,7048 | 0,3534 | 0,4557 |

Ở điều kiện chỉ nội dung + lịch sử, hệ vẫn thấp hơn PLM-NR (AUC −0,018 đến −0,019, MRR −0,015, nDCG@10 −0,006). Hai tập test khác nhau (`MINDlarge_dev` so với tập test chính thức), nên chênh nDCG@10 nhỏ cỡ này không đủ để kết luận.

## 6. Kết luận

* **Reader huấn luyện kỹ hơn:** tốt và tái lập được ở mức thành phần (+0,0091 vector nối; +0,0098 / +0,0099 BGE-large). Ở mức hệ thống thì rất nhỏ:
  * `super_llm` +0,0006 (G1 PASS). 0,4380 là hàng chỉ-nội-dung tốt nhất đến nay, nhỉnh hơn 0,4373.
  * `stack_meta` +0,0016 (G2 PASS), nhưng 0,4557 vẫn dưới 0,4570 của v18a.
  * `stack_pop` −0,0010 (G3 WORSE).
* **Nguyên cớ đo được:**
  * Validation có trọng số bài lạnh xếp reader mới **dưới** reader cũ (ngược với test), nên supermodel không đổi nền.
  * Stacker thấy thông tin của reader mới đã có sẵn trong các thành phần khác.
  * Muốn cải thiện thành phần lên được hệ thống thì phải sửa cách chọn, không chỉ thêm thành phần tốt hơn. Ví dụ: chọn nền theo validation thường, hoặc trộn thước đo có / không có trọng số. Thay đổi đó cần cổng mới viết trước khi chạy.
* **H6 / H9 (LLM 1,7B sinh văn bản / chấm điểm):** không có mức tăng có ý nghĩa trên MIND-large; hồ sơ người đọc kém rõ. Mở rộng toàn bộ catalogue khả thi về chi phí (~3–4 giờ T4), nhưng tín hiệu trên tập con chưa đủ để ưu tiên.
* **So với kết quả tốt nhất trước đó:**
  * Chỉ nội dung: 0,4380, nhỉnh hơn 0,4373.
  * Offline: không vượt (0,4557 < 0,4570).
  * Online: không vượt (0,4721 < 0,4731).
