# Kết quả đo thật — v19c trên MIND-large (Kaggle, 2 × Tesla T4, 5,38 giờ, mọi nhánh xong)

Nguồn: log và thư mục kết quả của lần chạy `v19c_mindlarge_fusion.ipynb` (kernel `nguyenpn/v19c-mindlarge-fusion`, phiên bản 1, đẩy lên 2026-10-19 khoảng 15:57 UTC, chờ GPU đến khoảng 19:28 UTC). Notebook **chỉ ghép điểm**: không huấn luyện thành phần nào, mà đọc điểm thành phần đã lưu của v19a và v19b (Kaggle `kernel_sources`). Mọi số là trên **`MINDlarge_dev` (376 471 impression)**. CI 95% ghép cặp theo impression chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện.

* **Đối chứng** = thành phần của pipeline v16b.
* **Thí nghiệm** = đối chứng + reader huấn luyện kỹ hơn của v19a (`catplus:*`, `bgeplus:*`) + tháp H5 của v19b (`ut_causal`).

## 0. Chạy có ổn không?

* **Đầu vào:** đủ điểm của cả v19a và v19b. Các run hai notebook cùng có thì giống hệt từng file: `bge` 16/16, `cat` 7/7, `qwen3` 18/18, chênh lớn nhất 0.
* **Sáu nhánh xong:**
  * `ctrl_s0`, `treat_s0`: supermodel + stacker, seed 0.
  * `ctrl_cw0`, `treat_cw0`: chỉ supermodel, chọn theo validation không trọng số (khám phá, §4).
  * `ctrl_s1`, `treat_s1`: stacker chạy lại với seed 1.
* **Thời gian** (đo thật):
  * Nạp dữ liệu và đầu vào: khoảng 0,3 giờ.
  * Supermodel: 255–310 s mỗi nhánh.
  * Stacker: 3 237 s (`ctrl_s0`), 4 119 s (`treat_s0`), 3 282 s (`ctrl_s1`), 4 140 s (`treat_s1`).
  * Tổng **5,38 giờ**.
* **Bộ nhớ:** RAM đỉnh 23,0 GB (cảnh báo 70% của 32 GB ở stacker của `treat_s1`); GPU đỉnh 5,0 GB.
* **Tái lập:** đối chứng trùng v16b đến 4 chữ số ở cả bảy hàng có số của v16b. Đây là lần thứ năm: v18a, v18b, v19a, v19b, v19c.

## 1. Bảng chính: đối chứng với thí nghiệm, cùng impression, stacker seed 0

| Hàng | đối chứng | thí nghiệm | Δ nDCG@10 [CI 95%] | Δ AUC [CI 95%] | Δ cold AUC [CI 95%] |
|---|---|---|---|---|---|
| `super_base` | 0,4260 | 0,4260 | 0 | 0 | 0 |
| `super_zs` | 0,4373 | 0,4373 | 0 | 0 | 0 |
| `super_llm` | 0,4373 | 0,4380 | +0,0006 [+0,0005, +0,0008] | +0,0008 [+0,0007, +0,0009] | −0,0001 [−0,0001, +0,0000] |
| `stack_llm` | 0,4369 | 0,4358 | −0,0011 [−0,0015, −0,0007] | +0,0019 [+0,0014, +0,0023] | −0,0071 [−0,0074, −0,0068] |
| `stack_meta` | 0,4516 | 0,4531 | **+0,0014 [+0,0010, +0,0019]** | +0,0024 [+0,0020, +0,0027] | −0,0019 [−0,0022, −0,0016] |
| `stack_meta_v15cfg` | 0,4543 | 0,4562 | **+0,0019 [+0,0014, +0,0023]** | +0,0021 [+0,0017, +0,0025] | −0,0002 [−0,0005, +0,0001] |
| `super_llm+pop` | 0,4638 | 0,4634 | −0,0004 [−0,0006, −0,0002] | −0,0002 [−0,0003, −0,0000] | −0,0002 [−0,0003, −0,0001] |
| `stack_pop` | 0,4731 | 0,4731 | −0,0000 [−0,0004, +0,0004] | −0,0006 [−0,0009, −0,0002] | −0,0002 [−0,0005, +0,0000] |

**Cổng viết trước khi chạy** (`gate_v19c.md`):

| Cổng | Δ [CI 95%] | Kết luận |
|---|---|---|
| G1 nội dung (`super_llm`) | +0,0006 [+0,0005, +0,0008] | **PASS** |
| G2 offline (`stack_meta`) | +0,0014 [+0,0010, +0,0019] | **PASS** |
| G3 online (`stack_pop`) | −0,0000 [−0,0004, +0,0004] | **NO GAIN** |

Cột "treatment" của G2 trong `gate_v19c.md` ghi 0,4562: đó là `stack_meta_v15cfg`, vì cổng G2 tính cả hàng này. Hàng chính `stack_meta` là 0,4531.

## 2. Nhiễu seed của stacker và trung bình 2 seed

Stacker lấy mẫu con 70% hàng / 70% cột, nên đổi seed là một nguồn nhiễu mà CI ghép cặp không tính. Bảng dưới đo trực tiếp nhiễu đó.

| Hàng | đối chứng s0 / s1 | nhiễu seed: đối chứng s1 − s0 | thí nghiệm s0 / s1 | Δ TN − ĐC, s0 | Δ TN − ĐC, s1 | Δ TN − ĐC, trung bình 2 seed |
|---|---|---|---|---|---|---|
| `stack_llm` | 0,4369 / 0,4363 | −0,0006 [−0,0008, −0,0004] | 0,4358 / 0,4357 | −0,0011 [−0,0015, −0,0007] | −0,0006 [−0,0011, −0,0002] | −0,0009 [−0,0013, −0,0005] |
| `stack_meta` | 0,4516 / 0,4515 | −0,0001 [−0,0004, +0,0001] | 0,4531 / 0,4521 | +0,0014 [+0,0010, +0,0019] | +0,0006 [+0,0002, +0,0010] | +0,0011 [+0,0007, +0,0015] |
| `stack_meta_v15cfg` | 0,4543 / 0,4551 | +0,0008 [+0,0005, +0,0011] | 0,4562 / 0,4567 | +0,0019 [+0,0014, +0,0023] | +0,0017 [+0,0012, +0,0021] | +0,0019 [+0,0015, +0,0023] |
| `stack_pop` | 0,4731 / 0,4739 | +0,0008 [+0,0005, +0,0010] | 0,4731 / 0,4729 | −0,0000 [−0,0004, +0,0004] | −0,0009 [−0,0013, −0,0005] | −0,0005 [−0,0009, −0,0001] |

Trung bình 2 seed (ghép hai stacker): đối chứng 0,4368 / 0,4519 / 0,4551 / 0,4738; thí nghiệm 0,4359 / 0,4530 / 0,4570 / 0,4733 (cùng thứ tự hàng như bảng).

* **Đổi seed, giữ nguyên thành phần**, làm một hàng đổi tới ±0,0008 ở đối chứng (`stack_meta_v15cfg`, `stack_pop`). Ở nhánh thí nghiệm, `stack_meta` đổi −0,0010 [−0,0012, −0,0007] giữa hai seed.
* **Lợi ích offline lặp lại ở cả hai seed.** `stack_meta_v15cfg` +0,0019 / +0,0017, rõ hơn nhiễu seed. `stack_meta` +0,0014 / +0,0006: cùng dấu ở cả hai seed, nhưng độ lớn cùng cỡ với nhiễu seed.
* **Online (`stack_pop`) không lợi ở seed nào** (0 / −0,0009). Đối chứng ở seed 1 đạt 0,4739 chỉ vì đổi seed. Vì vậy 0,4731 nên được báo kèm nhiễu huấn luyện khoảng ±0,0008.
* **`stack_llm` kém ở cả hai seed** (−0,0011 / −0,0006), cold AUC −0,0071 / −0,0077.

## 3. Thành phần được chọn, và tháp H5

* **Supermodel** (validation có trọng số bài lạnh):
  * Nền vẫn là reader cũ `cat:um_knn_rag`. Reader mới chỉ thêm `catplus:um_knn_rag` với trọng số 0,1.
  * **Tháp H5 không được chọn.** Trên validation không trọng số, H5 đạt 0,3874, thấp nhất trong 12 reader ứng viên (các reader khác 0,4120–0,4264). Ở v19b, theo validation có trọng số, H5 xếp thứ 9.
* **Stacker:** 56 thành phần, 118 đặc trưng.
  * Đóng góp lớn nhất là các reader kỹ hơn: `catplus:um_frozen` 0,12–0,16 gain, `catplus:um_knn_rag` 0,06, `bgeplus:um_frozen` 0,04–0,05.
  * H5 không nằm trong 6 đặc trưng đứng đầu ở hàng nào.
* **So với v19a** (chỉ thêm reader kỹ hơn, không có H5): `stack_meta` +0,0016, `stack_meta_v15cfg` 0,4557, gần bằng v19c. Phần lợi offline vì vậy gần như toàn bộ đến từ reader kỹ hơn, không từ H5. Đây là suy luận từ hai lần chạy, không phải phép thử tách riêng trong v19c.

## 4. Khám phá: supermodel chọn theo validation không trọng số (`--cold-target 0`)

Quy tắc này được chọn **sau khi** xem test của v19a: validation có trọng số xếp reader kỹ hơn dưới reader cũ, còn test thì ngược lại. Vì vậy các hàng dưới đây chỉ là gợi ý cho một lần chạy đăng ký trước, **không phải kết quả**.

| Hàng | ĐC, có trọng số | ĐC, không trọng số | TN, có trọng số | TN, không trọng số | Δ TN − ĐC, không trọng số [CI 95%] |
|---|---|---|---|---|---|
| `super_base` | 0,4260 | 0,4260 | 0,4260 | 0,4270 | +0,0011 [+0,0005, +0,0017] |
| `super_zs` | 0,4373 | 0,4380 | 0,4373 | 0,4379 | −0,0001 [−0,0006, +0,0004] |
| `super_llm` | 0,4373 | 0,4382 | 0,4380 | 0,4398 | +0,0016 [+0,0011, +0,0020] |
| `super_llm+pop` | 0,4638 | 0,4616 | 0,4634 | 0,4609 | −0,0007 [−0,0011, −0,0004] |

* Không trọng số, nhánh thí nghiệm chọn nền `catplus:um_frozen` và các reader `catplus:um_knn_rag`, `bgeplus:um_frozen`, `bgeplus:um_knn_rag`.
* `super_llm` của thí nghiệm lên 0,4398, tức +0,0018 [+0,0013, +0,0023] so với chọn có trọng số. Đổi lại cold AUC −0,0120, và hàng `+pop` giảm −0,0025.

## 5. Kết luận

* **Ghép reader kỹ hơn của v19a và tháp H5 của v19b vào pipeline v16b:**
  * Nội dung +0,0006 (G1 PASS, nhỏ).
  * Offline `stack_meta` +0,0014 và `stack_meta_v15cfg` +0,0019 (G2 PASS), lặp lại ở seed 1.
  * Online không đổi (G3 NO GAIN).
* **H5 không đóng góp:** supermodel không chọn, và nó không có mặt trong các đặc trưng đứng đầu của stacker. v19a (không có H5) đã cho mức lợi offline gần như bằng.
* **Nhiễu seed của stacker (±0,0008) cùng cỡ với các chênh lệch ở hàng online.** Hàng online nên được báo kèm khoảng nhiễu này.
* **Số tốt nhất sau v19c (đăng ký trước):**
  * Chỉ nội dung: 0,4380 (v19a, v19c).
  * Offline: 0,4570 (`stack_meta_v15cfg` của v18a; v19c đạt 0,4562 / 0,4567, trung bình 2 seed 0,4570).
  * Hàng offline chính `stack_meta`: 0,4531 (v19c; +0,0014 so với 0,4516 trong cùng lần chạy).
  * Online: 0,4731 (v16b, v19c; trung bình 2 seed của đối chứng 0,4738).
  * Hàng khám phá 0,4398 không được tính.
* **So với PLM-NR (chỉ nội dung + lịch sử):**
  * `super_llm` của thí nghiệm: AUC 0,6871, MRR 0,3391, nDCG@10 0,4380.
  * Hàng khám phá: 0,6933 / 0,3413 / 0,4398.
  * Cả hai vẫn dưới PLM-NR (0,7064 / 0,3539 / 0,4438). Tập test khác nhau: bài PLM-NR không ghi rõ bảng MIND đo trên tập nào, nhiều khả năng là test chính thức.
