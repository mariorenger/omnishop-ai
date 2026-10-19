# Kết quả đo thật — v18b trên MIND-large (Kaggle, 2 × Tesla T4, 5,13 giờ; supermodel và stacker KHÔNG chạy)

Nguồn: log và thư mục kết quả của lần chạy `v18b_mindlarge_strongreader.ipynb` (kernel `m81extend/v18b-mindlarge-strongreader`, phiên bản 1, đẩy lên 2026-10-18 19:28 UTC, QUICK=0, `BUDGET_H` 6,0). Mọi số là trên **`MINDlarge_dev` (376 471 impression)**; CI 95% ghép cặp theo impression chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện. So sánh với v16b là so giữa hai lần chạy (`RESULTS_v16b_large.md`).

## 0. Chạy có ổn không?

* Máy: **2 × Tesla T4** (log `nvidia-smi -L`). Mã hoá chia cho hai GPU hoạt động: BGE-large mã hoá 104 151 bài trong **249 s** (v16b, 1 GPU: 405 s), Qwen3-Embedding-0,6B trong **545 s** (v16b: 1 016 s). Guard pooling: BGE min cos 1,0000; Qwen3 0,9998.
* Run 1 BGE-large (công thức v16b): controls 57 s, pooling 298, moredata 449, graph 280, usermodel 1 853 s (0,97 giờ). Run 2 `catplus` (vector nối, reader huấn luyện kỹ hơn): E0 545 s, controls 69, graph 443, **usermodel 8 298 s (2,3 giờ)**. Run 3 Qwen3-0,6B: controls 58, pooling 302, moredata 786, graph 262, usermodel 3 289 s.
* **Supermodel và stacker bị bỏ qua** ("wall-clock budget exhausted (5.1h used)"). Hai nguyên nhân, đều là lỗi ước lượng / thiết kế của tôi, không phải lỗi chạy:
  1. Reader của `catplus` mất 2,3 giờ thay vì ~1,3 giờ như ước. MIND-large có **2 712 612** mẫu train (log), gấp 2,26 lần 1,2 triệu chứ không phải 1,5 lần như tôi tính; 6 epoch mỗi seed mất ~1 030 s (công thức v16b: ~400 s); `sca` chậm hơn `ca` 1,44 lần (1 483 s mỗi seed).
  2. Ngưỡng `STACK_MIN_S` = 3 900 s được truyền bằng `--min-stage-s`, nên áp cho **mọi** stage của run cuối, kể cả supermodel (vốn chỉ vài phút). Khi usermodel xong, ngân sách 6,0 giờ còn 0,9 giờ, nên cả supermodel lẫn stacker đều không bắt đầu. Quota GPU của tài khoản lúc đó còn ~2,5 giờ, đủ cho hai stage này nếu ngưỡng chỉ áp cho stacker và ngân sách được đặt cao hơn.
  Điểm của các thành phần nằm ở thư mục làm việc tạm của phiên (không lưu vào output), nên không chạy lại riêng phần ghép được.
* RAM đỉnh 18,6 GB (run 3), GPU đỉnh 5,0 GB; không stage nào lỗi.
* **Tái lập:** mọi hàng dùng công thức v16b trùng v16b đến 4 chữ số, cả từng seed: `bge:um_frozen` 0,4152 (0,4186 / 0,4117), `bge:um_knn_rag` 0,4189 (0,4203 / 0,4175), `qwen3:um_frozen` 0,4050 (0,4091 / 0,4009), `qwen3:um_text` 0,4092, `qwen3:um_knn_rag` 0,4095, `frozen` (BGE) 0,3933. Pipeline tất định giữa hai lần chạy trên cùng loại GPU, nên chênh lệch của các hàng `catplus` so với hàng `cat` của v16b là do công thức huấn luyện, không phải do nhiễu giữa hai lần chạy. Nhiễu giữa các seed vẫn còn (0,001–0,006).

## 1. Reader trên vector nối Qwen3-0,6B ⊕ BGE-large (test nDCG@10, 2 seed, đầu 128 chiều)

| Hàng | v18b (huấn luyện kỹ hơn) | từng seed | epoch tốt nhất / đã chạy | v16b | từng seed v16b | Δ |
|---|---|---|---|---|---|---|
| `um_frozen` | 0,4251 | 0,4220 / 0,4282 | 5 / 6, 5 / 6 | 0,4223 | 0,4229 / 0,4216 | +0,0028 |
| `um_knn_rag` | **0,4316** | 0,4333 / 0,4300 | 5 / 6, 5 / 6 | 0,4226 | 0,4200 / 0,4252 | **+0,0090** |
| `um_sca_knn_rag` (mới) | 0,4311 | 0,4298 / 0,4325 | 6 / 6, 5 / 6 | — | — | — |
| `um_prior_frozen` (prior không cá nhân hoá) | 0,3404 | 0,3370 / 0,3437 | 5 / 6, 3 / 5 | — | — | — |

Công thức "kỹ hơn": toàn bộ 2 712 612 mẫu (thay vì 1 200 000), tối đa 6 epoch (thay vì 5), learning rate × 0,5 sau mỗi epoch từ epoch 2, ngưỡng cải thiện của early stopping 0,0002 (thay vì 0,001). AUC của `um_knn_rag` 0,6840, cold AUC 0,6969.

Cặp ghép trong run (A − B, nDCG@10 [CI 95%]; Δ cold AUC):
* `sca` − `ca` (cùng view kNN-RAG, cùng công thức): **−0,0005 [−0,0009, −0,0001]**; cold AUC +0,0048 [+0,0045, +0,0050].
* `um_knn_rag` − `um_frozen`: **+0,0065 [+0,0062, +0,0069]** (v16b: +0,0003 [−0,0001, +0,0007], không khác).
* `catplus:um_frozen` − `bge:um_frozen` +0,0100 [+0,0094, +0,0106]; `catplus:um_knn_rag` − `bge:um_knn_rag` +0,0127 [+0,0121, +0,0134]; `catplus:um_frozen` − `qwen3:um_frozen` +0,0201 [+0,0197, +0,0206].

## 2. Kết luận

* **Huấn luyện reader kỹ hơn: thành công ở mức thành phần.** Reader tốt nhất tăng từ 0,4226 lên **0,4316** (+0,0090). Cả hai seed mới (0,4333 / 0,4300) đều cao hơn cả hai seed của v16b (0,4200 / 0,4252), và các hàng đối chứng trong cùng lần chạy trùng v16b đến 4 chữ số. Đây là thành phần đơn lẻ tốt nhất đo được đến nay. Có thêm huấn luyện thì view kNN-RAG mới thực sự có ích (+0,0065, trước đó +0,0003). Lần chạy này không tách được phần nào do thêm dữ liệu, phần nào do thêm epoch hay do lịch learning rate. Epoch tốt nhất phần lớn là 5 trên 6, nên reader đã gần hội tụ.
* **Reader `sca` (self-attention kiểu NRMS trên lịch sử, trước attention theo ứng viên): không hơn.** nDCG@10 −0,0005 (CI dưới 0, độ lớn rất nhỏ). Cold AUC +0,0048, nhưng chi phí huấn luyện gấp 1,44 lần. Trên vector LLM đóng băng đã có kNN-RAG, ngữ cảnh giữa các click không thêm thông tin xếp hạng đáng kể.
* **Hệ thống (super_llm, stack_meta, stack_pop): chưa đo** vì supermodel và stacker không chạy. Lần chạy này **không** cho biết v18b có vượt 0,4373 / 0,4543 / 0,4731 hay không. Ở v16b, cải thiện nội dung chuyển lên stacker rất ít (super_llm +0,0105 → stack_meta +0,0008). Vì vậy +0,009 ở reader nền chưa chắc thành mức tăng của hệ thống.

## 3. Việc cần làm để đo mức hệ thống

1. Sửa: ngưỡng thời gian riêng cho stacker (không chặn supermodel); ghi `scores/*.npz` của mọi run vào output để chạy lại riêng phần ghép (~1,2 giờ) mà không phải huấn luyện lại reader.
2. Chạy lại v18b với ngân sách đúng: ~6,5–7 giờ (thực đo: 3 run 5,1 giờ + supermodel và stacker ~1,1 giờ). Quota của tài khoản đã dùng còn 2,54 giờ (làm mới 2026-10-24 00:00 UTC).
