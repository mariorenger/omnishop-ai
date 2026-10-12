# Kết quả đo thật — v16b trên MIND-large (Kaggle, 2 × Tesla T4 nhưng mã chỉ dùng 1 GPU, 4,49 giờ, mọi stage xong)

Nguồn: log của lần chạy `v16b_mindlarge_hf_stacker_honest.ipynb` (QUICK=0, ngân sách 10 giờ) do người dùng gửi lại. Mọi số là trên **`MINDlarge_dev` (376 471 impression)**; CI 95% ghép cặp theo impression chỉ tính nhiễu *mẫu test*, **không** tính nhiễu huấn luyện (hai seed của cùng một reader chênh nhau 0,001 – 0,007 nDCG@10 trên test, bảng §2; XGBoost GPU không tất định đến từng bit). Số công bố (§6) lấy từ kết quả tìm kiếm, chưa mở PDF, và đo trên tập test của benchmark chứ không phải `MINDlarge_dev`. Lần chạy này thay lần chạy v16 chết giữa chừng (`RESULTS_v16_partial.md`).

## 0. Chạy có ổn không?

* Không stage nào lỗi (`Stages: 7 done`). Run 1 (BGE-large) 1,03 giờ: E0 405 s, controls 52, pooling 277, moredata 683, graph 256, usermodel 1 679 s. Run 2 (vector nối): E0 1 016 s (phần BGE lấy từ cache), controls 64, graph 416, usermodel 1 773 s. Run 3 (Qwen3 chính): E0 lấy từ cache, controls 54, pooling 286, moredata 1 333, graph 244, usermodel 2 976, supermodel 251, **stack 3 311 s**. Tổng **4,49 giờ** (ước trước 4–5 giờ). Môi trường được nhận là Kaggle; thư mục làm việc `/content`, `SUMMARY_v16b.md` ghi ở `/kaggle/working/results_large`.
* **Bản sửa RAM hoạt động.** Đỉnh RAM theo stage (lần đầu có số đo thật): run BGE ≤ 11,5 GB; run vector nối ≤ 15,9 GB (stage `graph`, 2048 chiều) và `usermodel` 15,0 GB — đúng chỗ đã chết ở v16; run chính ≤ 16,3 GB ở `usermodel`, 17,7 ở `supermodel`, **21,2 GB ở `stack`** (đỉnh của cả notebook); giới hạn container 32 GB; GPU đỉnh 5,0 GB. Không có cảnh báo `!! RAM`. Hệ quả: MIND-large ở quy mô này cần ≥ 24 GB RAM; **Colab miễn phí (~12,7 GB) chắc chắn không đủ**, kể cả khi tự giảm quy mô chưa ai thử.
* Zero-shot **khớp v15 đến 4 chữ số** ở cả hai encoder (`frozen`, `pool_lse`, `txt_cta_lse`, `hist100_lse` … giống hệt) nên các thay đổi của bộ đọc dữ liệu / stacker không làm lệch số offline.
* Chia dữ liệu như v14/v15: 1 801 231 train_core / 431 517 validation (lấy 150 000) / 376 471 test; 82,2% click của test là bài lạnh; validation chỉ 18,1% (thước đo có trọng số bài lạnh kéo về 80%).

## 1. Kết quả chính (test, offline; `stack_meta` còn đọc độ phơi nhiễm trong log impression, không dùng nhãn)

| Hàng | AUC | nDCG@10 | cold AUC | v15 (AUC / nDCG@10) | Δ so với v15 |
|---|---|---|---|---|---|
| `super_llm` (ghép tuyến tính) | 0,6863 | 0,4373 | 0,7185 | 0,6774 / 0,4268 | **+0,0089 / +0,0105** |
| `stack_llm` | 0,6853 | 0,4369 | 0,7063 | 0,6843 / 0,4332 | +0,0010 / +0,0037 |
| `stack_meta` (hàng chính, đăng ký trước) | 0,7013 | 0,4516 | 0,7248 | 0,6998 / 0,4508 | +0,0015 / +0,0008 |
| `stack_meta_v15cfg` (đối chứng: công thức v15) | 0,7029 | 0,4543 | 0,7285 | — | — |
| `super_llm+pop` (streaming) | 0,7133 | 0,4638 | 0,7539 | 0,7114 / 0,4599 | +0,0019 / +0,0039 |
| `stack_pop` (streaming) | 0,7196 | 0,4731 | 0,7526 | 0,7206 / 0,4730 | −0,0010 / +0,0001 |

Cặp ghép (A − B, nDCG@10): `stack_llm − super_llm` **−0,0004 [−0,0010, +0,0002]** (không khác; ở v15 là +0,0064); `stack_meta − super_llm` +0,0143 [+0,0137, +0,0149]; `stack_meta − stack_llm` +0,0147 [+0,0142, +0,0153]; `stack_pop − stack_meta` +0,0215 [+0,0209, +0,0220]; `stack_pop − super_llm+pop` +0,0093 [+0,0087, +0,0099]; **`stack_meta − stack_meta_v15cfg` −0,0027 [−0,0031, −0,0023]** (AUC −0,0017, cold AUC −0,0037).

**Điều rút ra.** Thành phần nội dung tốt hơn rõ (`super_llm` +0,0105) nhưng gần như **không chuyển lên hàng đầu của stacker**: `stack_meta` chỉ +0,0008 so với v15. Khi có đặc trưng phơi nhiễm / đếm theo bài, GBDT quy về cùng một mức ~0,45 dù điểm nội dung là 0,4268 hay 0,4373; còn khi không có chúng (`stack_llm`) nó không hơn nổi bốn trọng số tuyến tính. Cải thiện thêm thành phần nội dung chỉ đáng giá nếu mang **thông tin cá nhân hoá mới**, không phải thêm một bản ước lượng độ phổ biến của bài.

## 2. Reader học được: encoder nào, độ rộng đầu nào (cùng đầu 128 chiều, 2 seed, test nDCG@10)

| Reader | Qwen3 | BGE-large | Qwen3 ⊕ BGE-large |
|---|---|---|---|
| `um_frozen` | 0,4050 (seed 0,4091 / 0,4009) | 0,4152 (0,4186 / 0,4117) | **0,4223** (0,4229 / 0,4216) |
| `um_knn_rag` | 0,4095 (0,4075 / 0,4116) | 0,4189 (0,4203 / 0,4175) | **0,4226** (0,4200 / 0,4252) |
| `um_text` | 0,4092 (0,4085 / 0,4099) | — | — |

Cặp ghép: `bge:um_frozen − qwen3:um_frozen` **+0,0102 [+0,0095, +0,0108]**; `cat:um_frozen − bge:um_frozen` +0,0071 [+0,0066, +0,0076]; `cat:um_frozen − qwen3:um_frozen` +0,0173; `cat:um_knn_rag − qwen3:um_knn_rag` +0,0130; `bge:um_knn_rag − bge:um_frozen` +0,0037; `cat:um_knn_rag − cat:um_frozen` +0,0003 [−0,0001, +0,0007] (không khác, nhưng cold AUC +0,0088); `bge:um_frozen − bge:pool_lse` (zero-shot) +0,0105.
Thứ tự **cat > BGE > Qwen3** đúng ở cả hai seed của `um_frozen` (các khoảng seed không chồng nhau) và ở `um_knn_rag` giữa BGE và Qwen3; riêng cat so với BGE ở `um_knn_rag` hai khoảng seed chồng nhẹ (0,4200 so với 0,4203). Khoảng cách 0,007 – 0,010 lớn hơn nhiễu seed nhưng chỉ khoảng 1 – 2 lần độ lệch chuẩn của trung bình hai seed, nên coi là gợi ý mạnh, chưa phải kết luận chắc. Đầu 128 so với 64 trên Qwen3: `um_frozen` +0,0044, `um_text` +0,0066, `um_knn_rag` −0,0002 so với v15 (cùng cỡ nhiễu seed). **Câu hỏi "cần embedding lớn hơn?"**: encoder LLM 0,6B (Qwen3-Embedding) *kém* BGE-large 335M khi làm đầu vào cho reader (+0,0102 cho BGE), và ghép hai encoder tốt hơn mỗi cái riêng — độ đa dạng của hai không gian vector có ích hơn kích thước.
Reader `prior` (chỉ nhìn bài, không lịch sử) rất yếu (0,34 – 0,36) và nhiễu seed lớn (độ lệch chuẩn 0,009 – 0,018).

## 3. Supermodel (ghép tuyến tính, chọn nền và trọng số theo nDCG@10 có trọng số bài lạnh)

Nền được chọn: `cat:um_knn_rag` (cold-weighted validation 0,4573); `super_base` 0,6768 / 0,4260 — cao hơn chính hàng `um_knn_rag` của nó (0,4226) vì điểm xuất là **trung bình z-score của hai seed** (hợp hai seed +0,0034). `super_llm` thêm `bge:hist100_lse` 0,1, `bge:pool_lse` 0,1, `qwen3:txt_cta_lse` 0,25: +0,0114 [+0,0109, +0,0118] so với nền. Các reader khác (`cat:um_frozen`, `bge:um_knn_rag`, `qwen3:um_text`) được trọng số 0. Bỏ nhóm zero-shot khỏi `super_llm+pop`: −0,0092 [−0,0096, −0,0087]; bỏ popularity: −0,0265. `qwen3:txt_cta_lse` (category trong văn bản; +0,0186 so với `pool_lse` ở Qwen3 nhưng −0,0030 ở BGE) vẫn là hàng zero-shot tốt nhất và có trọng số cao nhất trong nhóm phụ.
Thước đo có trọng số bài lạnh chọn đúng nền (cũng là thành phần tốt nhất trên test) nhưng **không dự đoán tốt mức test**: test − validation có trọng số là −0,030 (`super_zs`) đến −0,081 (đối chứng), còn test − validation thô chỉ +0,010.

## 4. Stacker: kiểm tra giả thuyết "v15 học thuộc ngày validation"

| Hàng | validation (out-of-fold) | validation trên bài mới | test | test − validation |
|---|---|---|---|---|
| `stack_meta_v15cfg` (fold theo người dùng, mọi cột) | 0,4922 | 0,3738 | 0,4543 | **−0,0379** |
| `stack_llm` | 0,4456 | 0,4109 | 0,4369 | −0,0087 |
| `stack_meta` (fold theo khối thời gian) | 0,4588 | 0,4533 | 0,4516 | **−0,0072** |
| `stack_pop` | 0,4682 | 0,4605 | 0,4731 | +0,0049 |

* **Điểm ước lượng của v15 quả thực lạc quan** (0,4922 so với 0,4543 trên test: lệch 0,038); fold theo khối thời gian + lát cắt "bài mới" cho ước lượng trung thực (0,4588 / 0,4533 so với 0,4516: lệch 0,007). Đây là kết quả đo được, và khớp với ca gài sẵn của selftest.
* **Nhưng mô hình v15 không vì thế mà kém trên test**: công thức mới thua đối chứng **−0,0027 [−0,0031, −0,0023]**. Tức là phần "học thuộc" nằm ở *ước lượng*, không gây hại cho điểm test; test thấp hơn validation ở v15 là do ước lượng bị thổi, không phải vì mô hình hỏng. Kết luận có ba phần: (i) giả thuyết "v15 tự đánh lừa mình" đúng, (ii) giả thuyết "sửa nó thì test tốt hơn" **sai** trong lần đo này, (iii) lợi ích thực của v16 là con số validation đáng tin.
* Holdout tiến chọn cấu hình `v15` (cây sâu 7, mọi cột; nDCG@10 trên bài mới 0,4637; tất cả impression 0,4499) và `clean` (0,4556 / 0,4360); `clean_d5` 0,4535, `clean_d3` 0,4484, `noprior_d5` 0,4537. **Bỏ `ctx_n_cand` và làm thô cột gần-định-danh-bài làm *giảm* điểm trên holdout** (−0,014 trên tất cả impression): cỡ slate có vẻ không chỉ là "hằng số trong impression": cây dường như dùng nó như ngữ cảnh để đổi cách trọng số các đặc trưng khác (suy đoán; không có ablation riêng từng cột). Gain theo nhóm: `stack_llm` điểm 0,66 / ctx 0,34; `stack_meta` điểm 0,53 / ctx 0,27 / meta 0,20; `stack_pop` điểm 0,46 / ctx 0,22 / meta 0,08 / pop 0,24; đặc trưng dùng nhiều nhất: `cat:um_knn_rag` (0,09 – 0,13), `ctx_n_cand` (0,10 – 0,21), `bge:um_frozen`, `ctx_slate_max`, `m_ctr`, `pop`, `popw_ctr_1h`.
* Không có ablation riêng cho CTR theo cửa sổ giờ (đặc trưng mới của `stack_pop`) và đặc trưng trùng entity; `stack_pop` không đổi so với v15 (0,4731 so với 0,4730) nên không có dấu hiệu chúng giúp.
* Vận hành: stage `stack` 55 phút (quét 5 cấu hình 50 – 71 giây mỗi cái; đối chứng 290 s; 6 tập fold 197 – 291 s; dự đoán 14,09 triệu hàng test 1 037 s); bảng validation 6,05 triệu hàng × 99 cột (2,4 GB); XGBoost GPU; đỉnh RAM 21,2 GB.

## 5. Dự đoán ghi trước (README §0c) so với số đo

| Dự đoán | Số đo | Kết quả |
|---|---|---|
| `stack_llm` AUC 0,682 – 0,697; nDCG@10 0,431 – 0,446 | 0,6853; 0,4369 | trong khoảng |
| `stack_meta` AUC 0,698 – 0,712; nDCG@10 0,449 – 0,464 | 0,7013; 0,4516 | trong khoảng |
| `stack_pop` AUC 0,719 – 0,731; nDCG@10 0,471 – 0,484 | 0,7196; 0,4731 | trong khoảng (sát cận dưới) |
| ít nhất một hàng offline tăng > +0,003 nDCG@10 so với v15 (~65%) | `stack_llm` +0,0037 (còn lại +0,0008 / +0,0001) | đúng, sát ngưỡng |
| `stack_llm` tăng ≥ +0,01 (~15 – 20%) | +0,0037 | sai |
| khoảng cách test − validation của `stack_meta` nhỏ hơn đối chứng (~65%) | −0,0072 so với −0,0379 | **đúng** |
| (ngầm) `stack_meta` hơn `stack_meta_v15cfg` | −0,0027 (kém hơn) | **sai** |

## 6. Số công bố (thứ cấp, từ kết quả tìm kiếm) và vị trí của hệ thống này

PLM-NR 0,7064 AUC / 0,4438 nDCG@10, UNBERT 0,7068 / 0,4478, NRMS (GloVe) 0,670 – 0,691 / 0,418 – 0,434, đầu bảng xếp hạng (ensemble Fastformer + PLM-NR) 0,7268 AUC — trên tập test của benchmark, chuẩn bị dữ liệu và tinh chỉnh khác; chênh ~0,01 AUC không phải một phát biểu về mô hình. Hàng *chỉ dùng nội dung và lịch sử* của ta (`super_llm`, `stack_llm`) 0,686 AUC / 0,437: trong dải NRMS, thấp hơn PLM-NR khoảng 0,02 AUC. `stack_meta` (0,7013 / 0,4516) ngang PLM-NR về AUC nhưng đọc độ phơi nhiễm trong log impression nên **không cùng bài toán**. Khoảng trống còn lại so với văn liệu nằm ở phía *mô hình nội dung / cá nhân hoá được huấn luyện end-to-end*, không phải ở stacker.
