# Kết quả đo thật — v15 trên MIND-large (Kaggle, 2 × Tesla T4 nhưng mã chỉ dùng 1 GPU, 2,73 giờ)

Nguồn: log Kaggle của lần chạy `v15_mindlarge_hf_stacker.ipynb` (QUICK=0, ngân sách 10 giờ) do người dùng gửi lại. Mọi số là trên **`MINDlarge_dev` (376 471 impression)**. CI 95% là ghép cặp theo impression: chỉ tính nhiễu *mẫu test*, **không** tính nhiễu huấn luyện (reader: độ lệch chuẩn giữa 2 seed 0,0008 – 0,0078; XGBoost GPU không tất định đến từng bit).
Số công bố ở §4 lấy từ kết quả tìm kiếm (bài báo / khảo sát), **chưa mở PDF**, và đo trên tập test của benchmark chứ không phải `MINDlarge_dev`.

## 0. Chạy có ổn không?

* Không stage nào lỗi (`Stages: 7 done`). Run BGE-large: mã hoá E0 386 s, rồi controls 52 + pooling 288 + moredata 649 s (moredata gồm 452 s mã hoá text `cta`). Run Qwen3: mã hoá E0 944 s, rồi controls 53, pooling 292, moredata 1 257 (gồm 1 056 s mã hoá text `cta`), graph 250, **usermodel 2 958**, supermodel 256, **stack 1 535** s. Tổng **2,73 giờ** (ước trước: 3–5 giờ).
* **XGBoost chạy trên GPU** (`backend: xgboost (GPU)`); stage `stack` 25,6 phút: bảng validation 6,05 triệu hàng × 72 cột (1,7 GB) dựng trong 20 s, quét 3 mục tiêu 191 + 227 + 66 s, 12 lần huấn luyện fold 205 + 293 + 222 s, dựng đặc trưng + dự đoán 14,09 triệu hàng test 247 s.
* Bộ nhớ: RAM tiến trình đỉnh **16,2 GB** (máy 34 GB; `--stack-free-memory` đã giải phóng mẫu train trước stage `stack`), GPU đỉnh **5,0 GB** (16 GB). Guard mã hoá: Qwen3 cos nhỏ nhất 0,9998; BGE-large 1,0000.
* **Điều không ổn về vận hành:** thư mục làm việc là `/tmp` nên **điểm xuất (`scores/*.npz`) và cache vector bị mất khi phiên kết thúc**; muốn thử lại stacker phải chạy lại 2,7 giờ.
* Chia dữ liệu như v14: 1 801 231 train_core / 431 517 validation (lấy ngẫu nhiên 150 000) / 376 471 test; 2 712 612 mẫu huấn luyện (dùng 1 200 000). **82,2% click của test là bài lạnh.**

## 1. Kết quả chính (test)

Hàng không popularity (offline; `stack_meta` còn đọc độ phơi nhiễm trong log impression, không dùng nhãn):

| Hàng | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC |
|---|---|---|---|---|---|---|
| `frozen` (Qwen3 mean-pool) | 0,6215 | 0,2922 | 0,3168 | 0,3787 | 0,6583 | 0,4784 |
| `frozen` (BGE-large mean-pool) | 0,6297 | 0,3042 | 0,3321 | 0,3933 | 0,6632 | 0,4985 |
| `super_zs` | 0,6702 | 0,3229 | 0,3586 | 0,4205 | 0,6893 | 0,6338 |
| `super_llm` (ghép tuyến tính, tốt nhất theo validation) | 0,6774 | 0,3286 | 0,3654 | 0,4268 | 0,7004 | 0,6207 |
| **`stack_llm`** (GBDT: điểm + lịch sử/slate) | **0,6843** | 0,3355 | 0,3730 | **0,4332** | 0,7003 | 0,6534 |
| **`stack_meta`** (+ đếm theo bài, phơi nhiễm) | **0,6998** | 0,3500 | 0,3901 | **0,4508** | 0,7280 | 0,6187 |

Có popularity online (khung streaming: đếm click của các impression trước đó *trong chính ngày test*; **không so được** với bảng xếp hạng offline và không thể nộp lên leaderboard vì không có nhãn lúc nộp):

| Hàng | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC |
|---|---|---|---|---|---|---|
| `frozen+pop` (Qwen3) | 0,6757 | 0,3324 | 0,3694 | 0,4312 | 0,7259 | 0,5098 |
| `super_llm+pop` | 0,7114 | 0,3574 | 0,3998 | 0,4599 | 0,7446 | 0,6192 |
| **`stack_pop`** | **0,7206** | 0,3680 | 0,4125 | **0,4730** | 0,7524 | 0,6313 |

So với v14 (cùng giao thức, khác cấu hình): `super_llm` 0,6756 / 0,4263 → 0,6774 / 0,4268 (không đổi); `super_zs` 0,6762 / 0,4277 → 0,6702 / 0,4205 (**giảm** −0,0072: nền đổi sang `um_text`, xem §2.5); `super_llm+pop` 0,7048 / 0,4542 → 0,7114 / 0,4599 (+0,0057).

## 2. Phát hiện

**2.1 Stacker hơn supermodel — ít ở phần offline thuần, nhiều khi có đặc trưng log.** Ghép cặp trên cùng 376 471 impression (Δ nDCG@10 [95% CI]):

| A − B | Δ nDCG@10 | Δ AUC | Δ cold AUC |
|---|---|---|---|
| `stack_llm` − `super_llm` | +0,0064 [+0,0059, +0,0069] | +0,0069 | **−0,0001** [−0,0005, +0,0003] |
| `stack_meta` − `super_llm` | +0,0240 [+0,0234, +0,0246] | +0,0224 | +0,0276 |
| `stack_meta` − `stack_llm` | +0,0176 [+0,0170, +0,0182] | +0,0155 | +0,0277 (warm AUC 0,6534 → 0,6187) |
| `stack_pop` − `stack_meta` | +0,0222 [+0,0216, +0,0227] | +0,0208 | +0,0244 |
| `stack_pop` − `super_llm+pop` | +0,0131 [+0,0125, +0,0137] | +0,0093 | +0,0078 |
| `stack_llm` − `txt_cta_lse` (thành phần tốt nhất) | +0,0230 | +0,0289 | +0,0383 |

Giả thuyết "GBDT sửa được điểm yếu bài lạnh" **không được ủng hộ cho `stack_llm`**: cold AUC y hệt `super_llm` (0,7003 vs 0,7004); phần tăng nằm ở bài đã thấy (warm AUC +0,033). Cold AUC chỉ tăng khi có đặc trưng phơi nhiễm (`stack_meta` +0,0277), và đổi lại warm AUC giảm 0,035: mô hình đánh đổi bài cũ lấy bài mới, có lợi vì 82% click là bài lạnh.

**2.2 Validation out-of-fold của stacker lạc quan hơn hẳn** (nDCG@10, validation → test):

| Hàng | validation | test | test − validation |
|---|---|---|---|
| `stack_llm` | 0,4664 | 0,4332 | **−0,0332** |
| `stack_meta` | 0,4909 | 0,4508 | **−0,0401** |
| `stack_pop` | 0,4922 | 0,4730 | −0,0192 |
| `super_llm` | 0,4208 | 0,4268 | +0,0060 |
| `super_llm+pop` | 0,4394 | 0,4599 | +0,0205 |
| `frozen` / `random_vec` | 0,3392 / 0,2685 | 0,3787 / 0,2881 | +0,0395 / +0,0196 |

Hàng không huấn luyện và ghép tuyến tính có test ≥ validation (+0,02 … +0,04: test "dễ" hơn), stacker thì ngược lại: lệch cỡ 0,05 – 0,08 so với mức kỳ vọng. Số cây dùng hết trần (783 / 792 trên 800 ở một số fold) và điểm giữ lại của fold 0 chỉ chênh vài phần nghìn giữa ba mục tiêu (0,4880 / 0,4834 / 0,4895) nên **mục tiêu `binary:logistic` được chọn trên một thước đo không đáng tin**. `ctx_n_cand` (cỡ slate, hằng số trong một impression) chiếm gain lớn nhất (0,19 – 0,25): mục tiêu pointwise tốn sức mô hình hoá tỉ lệ click nền của impression, không giúp xếp hạng trong impression.
**Giả thuyết (chưa kiểm):** các fold chia theo người dùng nhưng **cùng một ngày và cùng các bài**, nên cây học được mức phổ biến *trong ngày đó* của từng bài qua các đặc trưng gần như định danh bài (điểm `prior`, `m_ctr`, `m_log_shown`, tuổi bài …) từ click của người dùng khác; thông tin đó không có ở ngày test. Dấu hiệu ủng hộ: biến thể `stack_pop` (có popularity trong ngày hợp lệ) lệch ít nhất. Cách kiểm: so out-of-fold chia theo người dùng với chia theo khối thời gian, bỏ từng nhóm đặc trưng, trên điểm xuất đã lưu.

**2.3 BGE-large tốt hơn Qwen3 ở mọi hàng zero-shot trên văn bản thường, nhưng mọi reader chỉ được huấn luyện trên vector Qwen3.** (test nDCG@10 / AUC):

| Hàng | BGE-large | Qwen3 | BGE − Qwen3 |
|---|---|---|---|
| `frozen` | 0,3933 / 0,6297 | 0,3787 / 0,6215 | +0,0146 |
| `pool_lse` | 0,4047 / 0,6459 | 0,3916 / 0,6348 | +0,0131 |
| `pool_topk` | 0,4045 / 0,6448 | 0,3930 / 0,6341 | +0,0115 |
| `pool_km5` | 0,4023 / 0,6402 | 0,3879 / 0,6306 | +0,0144 |
| `hist100_lse` | 0,4057 / 0,6471 | 0,3929 / 0,6360 | +0,0128 |
| `txt_cta` (category trong text) so với hàng cùng encoder không có category | −0,0030 | **+0,0186** | |
| `txt_cta_lse` | 0,4022 / 0,6484 | **0,4102 / 0,6554** | −0,0080 |

Category trong văn bản giúp Qwen3-Embedding (+0,0186) nhưng không giúp BGE (−0,0030, −0,0024). Hàng zero-shot tốt nhất là Qwen3 `txt_cta_lse` (0,4102), thứ hai BGE `hist100_lse` (0,4057). Bỏ nhóm "zero-shot extras" (`qwen3:txt_cta_lse` + ba hàng BGE: `hist100_lse`, `pool_lse`, `txt_cta_lse`) khỏi `super_llm+pop` làm test giảm 0,0116; **chưa có ablation riêng cho BGE**, nên chưa tách được phần của encoder thứ hai (BGE được trọng số 0,25 – 0,5 trong các bậc ghép và `bge:pool_topk#raw` nằm trong nhóm đặc trưng nhiều gain của stacker).

**2.4 Reader học được yếu so với kỳ vọng.** `um_frozen` 0,4006 (hai seed 0,3950 / 0,4061), `um_text` 0,4026, `um_knn_rag` 0,4097 (tốt nhất), so với zero-shot `txt_cta_lse` 0,4102. Bỏ cả nhóm reader khỏi `super_llm+pop`: chỉ −0,0008 [−0,0012, −0,0005]. Reader `prior` (chỉ nhìn nội dung bài) rất yếu: 0,3423 / 0,3212 (AUC 0,579 / 0,569), nằm trong top-6 đặc trưng của stacker nhưng với gain nhỏ (0,03).

**2.5 Chọn nền theo validation ưu ái mô hình "ấm".** Nền của supermodel được chọn là `um_text` (validation 0,4090 > `um_knn_rag` 0,4079), nhưng trên test `um_text` 0,4026 < `um_knn_rag` 0,4097; `um_text` mạnh ở bài đã thấy (warm AUC 0,6285) và yếu ở bài lạnh (cold AUC 0,6712 vs 0,6931). Đó là lý do `super_zs` thấp hơn v14.

**2.6 Từng kỹ thuật (đối chứng của chính nó), thêm so với v14.** `pool_topk` +0,0143, `pool_lse` +0,0130, `pool_km5` +0,0092, `txt_cta` (Qwen3) +0,0186, `hist100_lse` +0,0012 so với `pool_lse`, `hist100_mean` +0,0004 so với `frozen`; RAG entity −0,0013, kNN +0,0007 (cold AUC −0,0147), GraphRAG cộng đồng −0,0066 (cold −0,0181) — nên GraphRAG / RAG vẫn **không** đưa vào ghép tuyến tính. `community_*` dùng k-means thay Louvain (`RuntimeError` được ghi trong log: `kmeans(RuntimeError)`), chưa điều tra.

**2.7 Quan sát vận hành.** Mục tiêu quét trên GPU: `rank:ndcg` 191 s, `rank:pairwise` 227 s, `binary:logistic` 66 s (nhanh nhất ~3×). Gain theo nhóm: `stack_llm` scores 0,60 / ctx 0,40; `stack_meta` scores 0,47 / ctx 0,32 / meta 0,21; `stack_pop` scores 0,39 / ctx 0,27 / meta 0,11 / pop 0,22. Đặc trưng nhiều gain nhất ngoài `ctx_n_cand`: `um_text`, `um_knn_rag`, `m_ctr` (0,046), `pop` (0,185).

## 3. Dự đoán ghi trước so với số đo

| Dự đoán (README §0b, ghi trước khi chạy) | Đo được |
|---|---|
| `stack_llm` hơn `super_llm`, CI loại trừ 0; mức tăng nDCG@10 trong +0,003 … +0,02 | **Đúng** (+0,0064) |
| Cold AUC của stacker cao hơn của supermodel | **Sai cho `stack_llm`** (−0,0001); đúng cho `stack_meta` (+0,0276) |
| `stack_meta` > `stack_llm` (xác suất ~60%) | **Đúng** và lớn (+0,0176) |
| `stack_pop` > `stack_meta`, chênh nhỏ hơn +0,028 mà popularity cộng cho supermodel | **Đúng** (+0,0222 so với +0,0331 của `super_llm → super_llm+pop`) |
| Không mong đợi chạm mức PLM end-to-end (~0,706 AUC) | `stack_meta` 0,6998 (chưa chạm, nhưng sát; khác tập test) |
| Thời gian ~3–5 giờ | 2,73 giờ |

Bảng "dự đoán" do người dùng dán vào (một bản kế hoạch khác): AUC `stack_llm` 0,688 – 0,695 (đo 0,6843, **thấp hơn** khoảng đó), `stack_meta` 0,698 – 0,708 (0,6998, trong khoảng), `stack_pop` 0,718 – 0,728 (0,7206, trong khoảng); nDCG@10 `stack_pop` 0,450 – 0,458 (đo **0,4730**, lệch xa — đúng chỗ tôi đã chỉ ra là hai cột không nhất quán). Tuyên bố "`stack_pop` chạm Top 1 (0,7268)" vẫn **không áp dụng được**: 0,7206 < 0,7268, tập đo khác nhau, và `stack_pop` không thể nộp (cần nhãn của chính ngày test).

## 4. So với số công bố (từ kết quả tìm kiếm, chưa mở PDF)

NRMS (GloVe) trên MIND-large: AUC 0,670 – 0,691, nDCG@10 0,418 – 0,434; PLM-NR 0,7064 / 0,4438; UNBERT 0,7068 / 0,4478; ensemble Fastformer + PLM-NR 0,7268 (nguồn thứ cấp; một khảo sát 2021 ghi 73,04 cho một hệ khác, và Papers with Code ghi mục top dùng thêm dữ liệu huấn luyện).
`stack_llm` (chỉ nội dung + lịch sử, không dùng log) 0,6843 / 0,4332 nằm trong dải NRMS và kém PLM-NR 0,022 AUC; `stack_meta` 0,6998 / 0,4508 sát PLM-NR về AUC nhưng đọc độ phơi nhiễm trong log nên **không cùng bài toán** với các mô hình đó.

## 5. Việc tiếp theo suy ra từ số đo

1. **Lưu điểm xuất và cache vector** ra `/kaggle/working`, thêm notebook chỉ chạy stacker: mọi thử nghiệm dưới đây rẻ đi từ 2,7 giờ xuống chừng 20 – 30 phút.
2. **Chẩn đoán độ lạc quan của validation** (§2.2) rồi sửa: out-of-fold theo khối thời gian, bỏ đặc trưng hằng-theo-impression, làm thô đặc trưng gần-định-danh-bài, chọn mục tiêu / số cây bằng giữ-lại theo thời gian, thử `rank:*` và trộn nhiều mục tiêu / XGBoost + LightGBM.
3. **Reader (và RAG) trên BGE-large** và trên vector nối Qwen3 ⊕ BGE; head rộng hơn 64 chiều, khởi tạo giữ hình học zero-shot.
4. **Chọn nền / siêu tham số theo thước đo có trọng số bài lạnh** (§2.5) thay vì nDCG@10 chung trên validation ấm.
5. Đặc trưng khía cạnh rẻ: trùng entity (IDF) và subcategory giữa ứng viên và lịch sử; với biến thể streaming, CTR theo cửa sổ thời gian gần đây.
