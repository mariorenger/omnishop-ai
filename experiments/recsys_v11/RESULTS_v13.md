# Kết quả đo thật — lần chạy đầy đủ v13 (MIND-small, Kaggle Tesla T4, 8,22 giờ) và so sánh với v7

Nguồn: log của lần chạy `v13_suite_moredata.ipynb` (QUICK=0, `V11_FOCUS=0`, `V11_REF_ALL=0`, ngân sách 9,5 giờ) và log của lần chạy v7 đã ghi ở `RESULTS_v7.md`. Mọi số là trên **MINDsmall_dev (73 152 impression)** trừ các hàng tập con (n ghi rõ); số lấy trực tiếp từ các bảng markdown trong hai log.
CI 95% trong ngoặc là **ghép cặp theo impression**, lấy nguyên từ log: chỉ tính nhiễu *mẫu test*, **không** tính nhiễu *seed huấn luyện* (xem §5).
**"CI chặn"** là cách tính của tôi khi log không in CI ghép cặp giữa hai hàng: lấy hiệu của hai Δ so với *cùng một* đối chứng (cả hai đo trên 73 152 impression). Vì Var(X−Y) ≤ (sd_X + sd_Y)², nửa độ rộng CI của hiệu không vượt tổng nửa độ rộng của hai CI đã in. Đây là cận ngoài (rộng hơn CI thật), không phải CI ghép cặp; CI thật cần `test_arrays.npz` của lần chạy.

**Về "v8".** `v11_full_suite_v8.ipynb` (giống hệt từng byte `v12_sota_suite.ipynb`) **chưa có log chạy nào** và không có ô nào có output. Log "(2)" là của **v7**: có các stage NRMS / NAML / thang ablation mà v8 bỏ, không ghi `stages_innov.py`, tổng 7,52 giờ khớp `RESULTS_v7.md`. Vì vậy so sánh dưới đây là **v7 (cũ, đã đo) ↔ v13 (mới, đã đo)**; v8 không có cột số. Hai ý của v8 (multi-interest; thông tin chuyên mục) được đo ở v13 bằng cách cài khác, không phải chạy lại mã v8: k-means và reader `mi<K>` có hệ số nhiệt và phạt đa dạng; category / subcategory đưa thẳng vào văn bản được nhúng (không có mô tả do LLM sinh). Kết quả ở §3.

## 0. Chạy có ổn không?

* Không stage nào lỗi (log không có `Traceback` hay `!!`), tổng 29 599 giây = **8,22 giờ** (v7: 7,52 giờ). Giây của các stage: run A (BGE) 13 344, run B (Qwen3) 15 338; phần còn lại là mã hoá E0 và dựng dữ liệu.
  So với v7: `moredata` +256 s (BGE) và +893 s (Qwen3); `usermodel` +809 s và +884 s (thêm reader `ff` / `mi` / `text` và seed); Fastformer 2 seed + refit 5 409 s thay cho NRMS / NAML / Fastformer / thang ablation 6 843 s.
* Kiểm tra tái lập đều **MATCH**, y hệt v7: `frozen` = V10 `bge_zeroshot` (0,6242 / 0,3909); `popularity` và `frozen+pop` = V10 (`bge_zs_pop` 0,6769 / 0,4330); `um_frozen` = V10 `llmenc_ca` trong dung sai 0,01; guard mã hoá E0 có cos nhỏ nhất 0,9998 (Qwen3).
* Kiểm kê dữ liệu (`data audit` trong log): `news.tsv` có 8 cột, **không có body bài**; 65 238 bài; abstract trung bình 34,9 từ (5,2% rỗng), title 10,7 từ; 18 category, 270 subcategory; 87,5% bài có tên entity (2,4 mỗi bài); người đọc trong test có trung bình 32,3 click, 19,6% có hơn 50 click, nhiều nhất 444.

## 1. Cũ (v7) ↔ mới (v13): cái nào tốt hơn?

**Ngắn gọn.** v13 tốt hơn v7 ở **một điểm**: đưa category / subcategory vào văn bản được nhúng, rõ nhất với Qwen3-Embedding-0.6B (+0,0182 nDCG@10 so với `pool_lse` cùng encoder, validation cùng chiều). Các stage cũ chạy lại cho ra đúng số cũ, trừ phần huấn luyện không ổn định (SID-GPT, tower `soft`; §4), nên phần còn lại là hòa hoặc nằm trong nhiễu.
Điểm cao nhất nhích lên (zero-shot 0,4037 → 0,4094; có popularity 0,4410 → 0,4455), nhưng so với baseline huấn luyện mạnh nhất từng đo (NRMS refit của v7, 0,4078) chỉ hơn +0,0016, chưa phân biệt được với 0, và v13 đã bỏ NRMS / NAML nên bảng v13 không còn hàng đó.

**Bảng 1a. nDCG@10 trên 73 152 impression test, chỉ nội dung (không popularity).** "Chọn theo val" = hàng có validation cao nhất trong nhóm.

| Nhóm | Cũ (v7) | Mới (v13) | Δ mới − cũ | Đọc |
|---|---|---|---|---|
| Mốc `frozen` BGE (V10) | 0,3909 | 0,3909 | 0 | khớp V10 cả hai lần |
| Zero-shot BGE, chọn theo val | `pool_lse` 0,4037 | `hist200_lse` 0,4049 | +0,0012 [+0,0008, +0,0016] | gần như không đổi |
| Zero-shot BGE, cao nhất trên test | `pool_lse` 0,4037 | `txt_ctae_lse` 0,4083 | +0,0046 [+0,0033, +0,0059] | validation ngược dấu (−0,0051); chọn trên test nên lạc quan |
| Zero-shot Qwen3-Emb-0.6B, chọn theo val | `pool_lse` 0,3912 | `txt_cta_lse` 0,4094 | **+0,0182** [+0,0169, +0,0196] | validation cùng chiều (+0,0129) |
| **Zero-shot tốt nhất toàn bộ** | 0,4037 (BGE `pool_lse`) | 0,4094 (Qwen3 `txt_cta_lse`) | +0,0057, CI chặn [+0,0026, +0,0088] | validation xếp hai encoder ngược test (§6) |
| Reader học được tốt nhất | 0,4033 (BGE `um_knn_rag`) | 0,4104 (Qwen3 `um_text`) | +0,0071 | `um_text` hai seed 0,4173 / 0,4036; chưa kết luận |
| Tower LoRA Qwen3 `ut_causal+meanpool` (n = 30 000) | 0,4026 | 0,4024 | −0,0002 | tái lập |
| Tower LoRA Qwen3 `ut_soft+meanpool` (n = 30 000) | 0,4043 | 0,3956 | −0,0087 | **không tái lập** (§4) |
| `fastformer_ref` | 0,3926 (1 seed) | 0,3927 (2 seed) | +0,0001 | tái lập |
| Baseline huấn luyện refit (train_core + validation) | 0,4078 (`nrms_ref_refit`) | 0,4033 (`fastformer_ref_refit`) | −0,0045 | hai mô hình khác nhau; v13 bỏ NRMS |
| Cold AUC cao nhất, không tính `popularity` | 0,6912 (Qwen3 `um_entity_rag`) | 0,6912 (cùng hàng) | 0 | v13 không cải thiện lát cold |

**Bảng 1b. Có popularity (thiết lập streaming, không so được với số bài báo).**

| Nhóm | Cũ (v7) | Mới (v13) | Δ mới − cũ |
|---|---|---|---|
| `frozen+pop` (= V10 `bge_zs_pop`) | 0,4330 | 0,4330 | 0 |
| Zero-shot BGE + pop, chọn theo val | `pool_topk+pop` 0,4410 | `txt_views_lse+pop` 0,4455 | +0,0045, CI chặn [+0,0021, +0,0069] |
| Zero-shot Qwen3 + pop, chọn theo val | `pool_topk+pop` 0,4363 | `txt_cta_lse+pop` 0,4432 | +0,0069 (so với `pool_lse+pop` cùng encoder: +0,0074 [+0,0063, +0,0086]) |
| `fastformer_ref+pop` | 0,4384 | 0,4381 | −0,0003 |
| Tower Qwen3 `ut_causal+meanpool+pop` (n = 30 000) | 0,4415 | 0,4415 | 0 |

**Bảng 1c. Hàng mới tốt nhất (Qwen3 `txt_cta_lse`, 0,4094) so với các baseline huấn luyện.**

| So với | nDCG@10 của baseline | Δ | CI chặn | Ghi chú |
|---|---|---|---|---|
| `fastformer_ref_refit` (v13) | 0,4033 | +0,0061 | [+0,0022, +0,0100] | chưa có nhiễu seed cho refit |
| `nrms_ref_refit` (v7) | 0,4078 | +0,0016 | [−0,0023, +0,0055] (gồm 0) | một seed; hai seed NRMS ở v7 lệch nhau 0,0068 |

Theo từng chỉ số (Bảng 2), Qwen3 `txt_cta_lse` so với `nrms_ref_refit`: AUC ngang (0,6556 vs 0,6555), MRR và nDCG@5 hơn khoảng +0,006, nhưng cold AUC thấp hơn (0,6637 vs 0,6700) và warm AUC thấp hơn (0,6180 vs 0,6284). Không có chiều nào "hơn hẳn".

**Bảng 1d. Năm hàng đầu (73 152 impression, chỉ nội dung, không tính `popularity`).**

| Hạng | Cũ (v7) | nDCG@10 | Mới (v13) | nDCG@10 |
|---|---|---|---|---|
| 1 | `nrms_ref_refit` | 0,4078 | Qwen3 `um_text` (2 seed) | 0,4104 |
| 2 | BGE `pool_lse` | 0,4037 | Qwen3 `txt_cta_lse` | 0,4094 |
| 3 | BGE `um_knn_rag` | 0,4033 | BGE `txt_ctae_lse` | 0,4083 |
| 4 | BGE `pool_topk` | 0,4016 | BGE `txt_cta_lse` | 0,4050 |
| 5 | Qwen3 `um_entity_rag` | 0,4010 | BGE `hist200_lse` | 0,4049 |

Bảng v13 không chạy `nrms_ref_refit`; nếu có, nó (0,4078) sẽ đứng thứ 4 trong cột mới.

## 2. Số đo đầy đủ của các ứng viên chính

Chỉ nội dung (không popularity):

| Hàng | Lần chạy | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC |
|---|---|---|---|---|---|---|---|
| BGE-small `frozen` (mốc V10) | v7 = v13 | 0,6242 | 0,3002 | 0,3299 | 0,3909 | 0,6617 | 0,4416 |
| BGE-small `pool_lse` | v7 = v13 | 0,6386 | 0,3118 | 0,3427 | 0,4037 | 0,6675 | 0,5036 |
| `nrms_ref_refit` (baseline mạnh nhất v7, 1 seed) | v7 | 0,6555 | 0,3140 | 0,3443 | 0,4078 | 0,6700 | 0,6284 |
| `fastformer_ref` (2 seed ở v13) | v13 | 0,6426 | 0,3000 | 0,3289 | 0,3927 | 0,6623 | 0,5869 |
| `fastformer_ref_refit` | v13 | 0,6550 | 0,3097 | 0,3387 | 0,4033 | 0,6767 | 0,5883 |
| Qwen3-Emb `pool_lse` | v7 = v13 | 0,6349 | 0,3025 | 0,3300 | 0,3912 | 0,6632 | 0,4939 |
| Qwen3-Emb `txt_cta_lse` | v13 | 0,6556 | 0,3201 | 0,3509 | 0,4094 | 0,6637 | 0,6180 |
| BGE-small `txt_ctae_lse` | v13 | 0,6516 | 0,3176 | 0,3476 | 0,4083 | 0,6703 | 0,5479 |
| Qwen3-Emb `um_text` (reader học được, 2 seed) | v13 | 0,6535 | 0,3167 | 0,3492 | 0,4104 | 0,6885 | 0,5102 |

Có popularity:

| Hàng | Lần chạy | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC |
|---|---|---|---|---|---|---|---|
| `frozen+pop` (= V10 `bge_zs_pop`) | v7 = v13 | 0,6769 | 0,3350 | 0,3727 | 0,4330 | 0,7249 | 0,4671 |
| BGE `pool_topk+pop` (tốt nhất v7) | v7 = v13 | 0,6871 | 0,3414 | 0,3807 | 0,4410 | 0,7324 | 0,4979 |
| `fastformer_ref+pop` | v13 (v7: 0,4384) | 0,6909 | 0,3400 | 0,3763 | 0,4381 | 0,7265 | 0,5627 |
| BGE `txt_views_lse+pop` (chọn theo val) | v13 | 0,6956 | 0,3441 | 0,3861 | 0,4455 | 0,7369 | 0,5269 |
| BGE `txt_ctae_lse+pop` | v13 | 0,6926 | 0,3432 | 0,3833 | 0,4449 | 0,7336 | 0,5225 |
| Qwen3-Emb `txt_cta_lse+pop` | v13 | 0,6909 | 0,3436 | 0,3835 | 0,4432 | 0,7263 | 0,5551 |

* Mức tăng của Qwen3 `txt_cta_lse` so với `pool_lse` cùng encoder (AUC 0,6349 → 0,6556) nằm ở **lát warm** (AUC 0,4939 → 0,6180); lát cold gần như không đổi (0,6632 → 0,6637, Δ +0,0005 [−0,0007, +0,0017]).
  Warm AUC của các hàng title + abstract không có category là thấp (BGE `frozen` 0,4416, Qwen3 `frozen` 0,4396, tức dưới 0,5). Bỏ abstract (`txt_t`) nâng warm AUC (BGE 0,5963, Qwen3 0,6350) nhưng kéo cold AUC xuống (0,6158, 0,5704); thêm category vào title + abstract nâng warm AUC mà giữ cold AUC. Cơ chế chưa được kiểm tra.
* Với popularity, các hàng BGE có category (`cta`, `ctae`, `views`; dạng `_lse`) đều hơn `pool_lse+pop` (+0,0030, +0,0040, +0,0046; CI không chứa 0) và cold AUC của chúng nhỉnh hơn (+0,0010, +0,0017, +0,0049). Hàng chỉ title thì kém (`txt_t_lse+pop` −0,0097).

## 3. Từng thay đổi của v13

### 3.1 Văn bản nhúng: title / abstract / category / entity

Đối chứng: `frozen` cho hàng mean-pool, `pool_lse` cho hàng `_lse` (cùng encoder). `val` là nDCG@10 trên validation (30 270 impression); Δ val chỉ là hiệu thô, không có CI.

| Encoder | Hàng | val nDCG@10 | test nDCG@10 | Δ val so với đối chứng | Δ test [CI] |
|---|---|---|---|---|---|
| bge | `txt_t` (vs `frozen`) | 0,3579 | 0,3710 | −0,0081 | −0,0199 [−0,0216, −0,0183] |
| bge | `txt_cta` (vs `frozen`) | 0,3571 | 0,3907 | −0,0089 | −0,0002 [−0,0016, +0,0012] |
| bge | `txt_ctae` (vs `frozen`) | 0,3611 | 0,3917 | −0,0049 | +0,0007 [−0,0007, +0,0022] |
| bge | `txt_views` (vs `frozen`) | 0,3700 | 0,3864 | +0,0040 | −0,0045 [−0,0059, −0,0031] |
| bge | `txt_t_lse` (vs `pool_lse`) | 0,3706 | 0,3876 | −0,0109 | −0,0161 [−0,0177, −0,0146] |
| bge | `txt_cta_lse` (vs `pool_lse`) | 0,3719 | 0,4050 | −0,0096 | +0,0013 [+0,0000, +0,0026] |
| bge | `txt_ctae_lse` (vs `pool_lse`) | 0,3764 | 0,4083 | −0,0051 | +0,0046 [+0,0033, +0,0059] |
| bge | `txt_views_lse` (vs `pool_lse`) | 0,3800 | 0,4037 | −0,0015 | +0,0000 [−0,0013, +0,0014] |
| bge | `hist100_lse` (vs `pool_lse`) | 0,3818 | 0,4048 | +0,0003 | +0,0011 [+0,0007, +0,0014] |
| bge | `hist200_lse` (vs `pool_lse`) | 0,3822 | 0,4049 | +0,0007 | +0,0012 [+0,0008, +0,0016] |
| qwen3 | `txt_t` (vs `frozen`) | 0,3228 | 0,3538 | −0,0156 | −0,0247 [−0,0266, −0,0227] |
| qwen3 | `txt_cta` (vs `frozen`) | 0,3628 | 0,3974 | +0,0244 | +0,0189 [+0,0176, +0,0203] |
| qwen3 | `txt_t_lse` (vs `pool_lse`) | 0,3446 | 0,3591 | −0,0187 | −0,0321 [−0,0338, −0,0304] |
| qwen3 | `txt_cta_lse` (vs `pool_lse`) | 0,3762 | 0,4094 | +0,0129 | +0,0182 [+0,0169, +0,0196] |
| qwen3 | `hist100_lse` (vs `pool_lse`) | 0,3644 | 0,3924 | +0,0011 | +0,0013 [+0,0009, +0,0016] |

* **Abstract quan trọng:** bỏ abstract (chỉ title) làm giảm 0,016 – 0,020 ở BGE và 0,025 – 0,032 ở Qwen3, cả mean-pool lẫn `_lse`, validation cùng chiều.
* **Category:** với Qwen3, tăng +0,0189 (mean-pool) và +0,0182 (`_lse`), validation cùng chiều (+0,0244 / +0,0129). Với BGE, test ≈ 0 đến +0,005 nhưng **validation nói ngược** (−0,005 … −0,010) cho `cta` / `ctae`: không có bằng chứng ổn định.
* **Tên entity** (BGE; Qwen3 không chạy): `ctae` hơn `cta` +0,0009 [+0,0001, +0,0018] (mean-pool) và +0,0033 [+0,0025, +0,0041] (`_lse`); validation cùng dấu (+0,0040 / +0,0045). Nhỏ nhưng nhất quán.
* **`views`** (title / abstract / category nhúng riêng rồi ghép): không hơn `cta` (−0,0043 mean-pool; −0,0013 [−0,0027, +0,0001] `_lse`); với popularity thì là hàng tốt nhất của BGE (0,4455, validation cũng cao nhất 0,4166).
* **Lịch sử 100 / 200 click:** +0,0002 … +0,0013 (hàng mean-pool 100 click của Qwen3 không khác 0); CI hẹp nên phần lớn "khác 0" nhưng không có ý nghĩa thực tế; chỉ 19,6% người đọc test có hơn 50 click.
* Trường `abstract` đã có sẵn từ v1; MIND-small không có body bài nên "mở khoá body" không làm được (§0).

### 3.2 Multi-interest

Zero-shot (`pool_km<K>`: gom click thành K nhóm bằng k-means; K = 1 ≡ mean-pool, K ≥ số click ≡ `pool_lse`):

| Encoder | Hàng | val | test nDCG@10 | Δ so với `pool_lse` | Δ so với `frozen` [CI] |
|---|---|---|---|---|---|
| bge | `pool_km2` | 0,3701 | 0,3958 | −0,0079 | +0,0049 [+0,0041, +0,0057] |
| bge | `pool_km3` | 0,3729 | 0,3989 | −0,0048 | +0,0080 [+0,0071, +0,0089] |
| bge | `pool_km5` | 0,3752 | 0,4007 | −0,0030 | +0,0098 [+0,0087, +0,0108] |
| qwen3 | `pool_km2` | 0,3458 | 0,3837 | −0,0075 | +0,0053 [+0,0043, +0,0063] |
| qwen3 | `pool_km3` | 0,3501 | 0,3865 | −0,0047 | +0,0080 [+0,0069, +0,0091] |
| qwen3 | `pool_km5` | 0,3546 | 0,3875 | −0,0037 | +0,0091 [+0,0078, +0,0103] |

Đơn điệu theo K và luôn nằm giữa `frozen` và `pool_lse`: **kết quả âm sạch**, vì `pool_lse` là trường hợp K lớn. CI ghép cặp của BGE so với `pool_lse`: −0,0079 [−0,0089, −0,0068], −0,0048 [−0,0058, −0,0038], −0,0030 [−0,0039, −0,0021].

Reader học được (K query attention; `mi1` là đối chứng một interest, `mi4d` thêm phạt bất đồng; `std` là độ lệch chuẩn **tổng thể**, xem §5):

| Encoder | Hàng | test nDCG@10 (trung bình seed) | số seed | nDCG@10 từng seed | std | cos giữa các interest | TV attention |
|---|---|---|---|---|---|---|---|
| bge | `um_frozen` (V10) | 0,3968 | 3 | 0,3956, 0,3958, 0,3990 | 0,0016 | - | - |
| bge | `um_ff_frozen` | 0,3939 | 3 | 0,3988, 0,3909, 0,3919 | 0,0035 | - | - |
| bge | `um_mi1_frozen` | 0,3906 | 3 | 0,3908, 0,3862, 0,3948 | 0,0035 | - | - |
| bge | `um_mi4_frozen` | 0,3944 | 3 | 0,3956, 0,3930, 0,3945 | 0,0011 | 0,87, 0,85, 0,86 | 0,34, 0,34, 0,33 |
| bge | `um_mi4d_frozen` | 0,3972 | 3 | 0,4043, 0,3907, 0,3966 | 0,0055 | 0,38, 0,39, 0,34 | 0,72, 0,71, 0,72 |
| qwen3 | `um_frozen` (V10) | 0,3996 | 2 | 0,3929, 0,4063 | 0,0067 | - | - |
| qwen3 | `um_ff_frozen` | 0,3944 | 2 | 0,3958, 0,3930 | 0,0014 | - | - |
| qwen3 | `um_mi1_frozen` | 0,4008 | 2 | 0,4025, 0,3990 | 0,0018 | - | - |
| qwen3 | `um_mi4_frozen` | 0,4029 | 2 | 0,4048, 0,4011 | 0,0019 | 0,91, 0,91 | 0,29, 0,30 |
| qwen3 | `um_mi4d_frozen` | 0,3965 | 2 | 0,3933, 0,3997 | 0,0032 | 0,52, 0,38 | 0,63, 0,71 |
| qwen3 | `um_text` | 0,4104 | 2 | 0,4173, 0,4036 | 0,0068 | - | - |

* BGE: `mi4` 0,3944 ≈ `mi1` 0,3906 ≈ `ff` 0,3939 ≈ `um_frozen` 0,3968 (trong ±0,005), và **kém `pool_lse` −0,0093 [−0,0112, −0,0075]**. Qwen3: `mi4` 0,4029 hơn `mi1` 0,4008 và `um_frozen` 0,3996, nhưng chênh nhỏ hơn nhiễu seed; hơn `pool_lse` +0,0118 như `um_frozen` đã hơn ở v7.
* Không có phạt (`mi4`) thì các interest gần như trùng nhau (cos 0,85 – 0,91, nghĩa là thực chất một interest). Có phạt (`mi4d`) thì tách ra (cos 0,34 – 0,52, TV 0,63 – 0,72) nhưng **không có lợi ích nhất quán**: BGE +0,0004 [−0,0004, +0,0012] so với `um_frozen`, Qwen3 −0,0031.
* `um_text` (reader đọc vector của `txt_cta`) +0,0108 [+0,0097, +0,0119] so với `um_frozen` Qwen3 trên mẫu test, nhưng hai seed lệch nhau 0,0137: chưa kết luận.

### 3.3 Dự đoán ghi trước ↔ kết quả đo

| Dự đoán (README §0, ghi trước khi chạy) | Kết quả | Đánh giá |
|---|---|---|
| F: \|Δ nDCG@10\| ≲ 0,01; category là ứng viên có cơ hội nhất | Qwen3: category +0,0182 / +0,0189; BGE ≈ 0 | đúng hướng, **sai độ lớn** ở Qwen3 |
| Tên entity gần như dư thừa | BGE `ctae` hơn `cta` +0,0033 [+0,0025, +0,0041] (`_lse`), validation cùng dấu | sai một phần: nhỏ nhưng có |
| 100 / 200 click ≈ 0 | +0,0011 / +0,0012 (BGE `_lse`) | đúng về ý nghĩa thực tế |
| H11: `pool_km` nằm giữa `frozen` và `pool_lse`, có thể bằng hoặc nhỉnh hơn `pool_lse` | giữa hai hàng đó, **thấp hơn** `pool_lse` (−0,0079 … −0,0030), đơn điệu theo K | đúng nửa đầu, sai nửa sau |
| H11: `um_mi4` ≈ `um_mi1` ≈ `um_ff_frozen` (±0,005) và ≤ `pool_lse` | BGE đúng (0,3944 / 0,3906 / 0,3939; kém `pool_lse` −0,0093). Qwen3: `mi4` hơn `ff` +0,0085 và hơn `pool_lse` +0,0118 | đúng với BGE, sai với Qwen3 (reader học được trên vector Qwen3 hơn pooling cố định, điều đã thấy ở v7: `um_frozen` 0,3996 vs `pool_lse` 0,3912) |

## 4. Độ tái lập giữa hai lần chạy (cùng mã cho các stage cũ)

146 hàng có mặt ở cả hai log (BGE + Qwen3). Chênh |nDCG@10 mới − cũ|:

| Mức lệch | Số hàng | Là những hàng nào |
|---|---|---|
| < 0,00005 (giống hệt) | 115 | mọi hàng không huấn luyện hoặc có cache: controls, pooling, RAG / GraphRAG, `community_llm`, `kar_*`, `rerank_*`, `histq_*`, enc-sweep; 27/30 hàng head / LoRA / DoRA; 9/11 hàng `um_*` cũ; 5 hàng tower |
| ≤ 0,001 | 17 | tower `ut_*` (9), `sid_profile*` (3), head / LoRA (3), `fastformer_ref` và `fastformer_ref+pop` (2) |
| 0,001 – 0,003 | 6 | `um_frozen` (+0,0011) và `um_ff_frozen` (−0,0010) vì v13 dùng 3 seed thay vì 2; tower `ut_bidir*` (+0,0015, +0,0012) và `ut_soft+pop`, `ut_soft+meanpool+pop` (−0,0016, −0,0015) |
| > 0,003 | 8 | 6 hàng SID-GPT (−0,0030 … −0,0246; vốn ngang ngẫu nhiên ở cả hai lần: 0,2898 vs 0,2742, `random_vec` 0,2897) và **2 hàng tower `ut_soft` / `ut_soft+meanpool` (−0,0085, −0,0087)** |

* **`ut_soft` không tái lập.** Cùng mã, cùng seed: `ut_soft+meanpool` 0,4043 → 0,3956 (AUC 0,6474 → 0,6396). Ở v7, `soft − causal` = +0,0020 [+0,0003, +0,0036]; ở v13 là −0,0063 [−0,0080, −0,0046]. Dấu của chênh lệch giữa các mode của tower đảo giữa hai lần chạy, nên mọi kết luận "mode nào của tower tốt hơn" từ một lần chạy là không có cơ sở (khớp §4 của `RESULTS_v7.md`).
  Hạng 2 của `ut_soft+meanpool` trong bảng xếp hạng chung của v7 (+0,0137 so với `frozen`) không ổn định: hai lần chạy cho +0,0137 và +0,0050.
* Hệ quả: so sánh cũ ↔ mới ở các hàng zero-shot không chứa nhiễu huấn luyện; ở các hàng huấn luyện thì có (§5).

## 5. Nhiễu seed

* Bảng trong `noise.md` (và §3.2) dùng `np.std` mặc định, tức **độ lệch chuẩn tổng thể (chia n)**. Độ lệch chuẩn mẫu lớn hơn ×1,22 với 3 seed và ×1,41 với 2 seed (ví dụ Qwen3 `um_frozen`: 0,0067 → 0,0095). Hãy nhân trước khi dùng làm sai số của một lần huấn luyện.
* Nhiễu đã đo ở v7 vẫn đúng: hai seed NRMS lệch nhau 0,0068, hai seed của tower lệch 0,0103 – 0,0124. Trong v13: Qwen3 `um_frozen` lệch 0,0134 giữa hai seed, `um_text` 0,0137, BGE `um_mi4d_frozen` 0,0136 giữa seed lớn nhất và nhỏ nhất.
* Hệ quả: chênh lệch dưới ~0,007 giữa hai hàng *đã huấn luyện* (một lần) không phải kết luận; trung bình 3 seed còn khoảng 0,004. CI ghép cặp trong các bảng không bao gồm nhiễu này.

## 6. Hệ quả cho luận văn (nhận định, không phải kết quả đo)

1. **Kết quả dương mới và sạch nhất:** đưa category / subcategory vào văn bản nhúng giúp embedding Qwen3-0.6B zero-shot +0,018 nDCG@10 (xác định, không có nhiễu huấn luyện, validation cùng chiều, CI hẹp), nhưng không giúp BGE-small. Đây là bằng chứng cho giả thuyết "dùng thêm trường dữ liệu thì encoder LLM phát huy hơn", giới hạn ở một trường (category) và một mẫu chuỗi. **Chưa biết** đó là thông tin chủ đề hay hiệu ứng định dạng: cần đối chứng gán category ngẫu nhiên cùng định dạng và vài mẫu chuỗi khác.
2. **Mức hơn "baseline huấn luyện" chỉ nhỏ.** Zero-shot Qwen3 + category ngang NRMS refit về AUC, hơn về MRR / nDCG@5 (+0,006), thua về cold AUC; hơn Fastformer refit +0,0061. Phát biểu an toàn: "ngang đến nhỉnh hơn, không cần huấn luyện", không phải "vượt".
3. **Xếp hạng giữa hai encoder không ổn định giữa validation và test.** Validation xếp BGE `pool_lse` (0,3815) trên Qwen3 `txt_cta_lse` (0,3762); test xếp ngược (0,4037 vs 0,4094). Khoảng cách `frozen` Qwen3 − BGE là −0,0276 trên validation nhưng −0,0125 trên test. Trong cùng một encoder thì validation và test cùng chiều (Qwen3 +0,0129 / +0,0182), nên "category giúp Qwen3" chắc hơn "Qwen3 hơn BGE". Chỉ một ngày test nên không tách được hai khả năng.
4. **Multi-interest: kết quả âm.** Zero-shot k-means luôn dưới `pool_lse`; reader học được không hơn reader một vector ngoài nhiễu seed. Phạt đa dạng làm các interest tách nhau nhưng không giúp xếp hạng.
5. **Lát cold:** không có cải thiện nào ở cold AUC giữa v7 và v13 (cao nhất 0,6912 cả hai lần); category không làm tăng cold AUC của Qwen3 (+0,0005 [−0,0007, +0,0017]).
6. Các hướng cũ (RAG / GraphRAG / Semantic ID / LLM judge / KAR / LoRA) giữ nguyên kết luận ở `RESULTS_v7.md` §6 (số đo giống hệt).
7. Khi viết: dùng **v13 cho phần dữ liệu / multi-interest** và **v7 cho NRMS / NAML / thang ablation** (v13 không chạy lại); không trộn số hai lần chạy của hàng *đã huấn luyện* như thể cùng một thí nghiệm.

## 7. Giới hạn

* Một dataset (MIND-small), một ngày test; hầu hết hàng huấn luyện 1 – 3 seed; refit chưa có nhiễu seed.
* "Zero-shot tốt nhất" được chọn sau khi xem nhiều hàng trên test (56 hàng mới: BGE 34, Qwen3 22, tính cả `+pop`); chọn theo validation cho kết quả khác ở BGE (§1, Bảng 1a). Trong các hàng mới, chỉ Qwen3 `txt_cta_lse` có validation lẫn test cùng ủng hộ với mức đáng kể so với đối chứng cùng encoder (Δ val +0,0129, Δ test +0,0182).
* Một mẫu chuỗi category (`Category: c. Subcategory: s.`), chưa thử mẫu khác, chưa có đối chứng category ngẫu nhiên, chưa thử category riêng so với subcategory riêng.
* Không có CI ghép cặp giữa hàng của hai encoder khác nhau (cần `test_arrays.npz`); "CI chặn" ở trên là cận ngoài, không phải CI.
* Số liệu v7 và v13 là hai lần chạy khác nhau; các hàng đã huấn luyện chỉ so được trong phạm vi nhiễu seed (§5).
* Thang đo, evaluator và giao thức giữ nguyên V10; số `+pop` là phản hồi streaming, không so được với số bài báo.
