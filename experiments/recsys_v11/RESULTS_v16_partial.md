# Kết quả đo thật — lần chạy đầu của v16 trên MIND-large (kernel chết giữa chừng sau 1,6 giờ)

Nguồn: log của lần chạy `v16_mindlarge_hf_stacker_honest.ipynb` (QUICK=0, ngân sách 10 giờ, 2 × Tesla T4 nhưng mã chỉ dùng 1 GPU, RAM 34 GB) do người dùng gửi lại. Mọi số là trên **`MINDlarge_dev` (376 471 impression)**, CI 95% ghép cặp theo impression (chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện). Lần chạy này **không đến được stacker, supermodel hay run Qwen3 chính**: kết quả của v16 (hướng A) chưa có. Số công bố không nằm trong tài liệu này.

## 0. Chuyện gì đã xảy ra

* Run 1 (BGE-large: `controls,pooling,moredata,graph,usermodel`) **xong trong 1,07 giờ**: đọc dữ liệu ≈ 4,5 phút, mã hoá E0 421 s (v15: 386 s), controls 57 s, pooling 301 s, moredata 701 s (gồm 495 s mã hoá text `cta`), graph 277 s, usermodel 1 757 s (6 lần huấn luyện reader).
* Run 2 (vector nối Qwen3 ⊕ BGE-large, 2048 chiều): mã hoá E0 1 027 s (phần BGE lấy từ cache, phần Qwen3 mã hoá lần đầu), controls 68 s, graph 438 s, rồi **kernel chết** ngay đầu stage `usermodel` (≤ 36 giây sau dòng tiêu đề của stage, trước khi in bất kỳ dòng huấn luyện nào). Papermill: `DeadKernelError: Kernel died`; không có traceback Python → tiến trình bị hệ điều hành giết (hết RAM).
* Run 3 (Qwen3 chính + supermodel + stacker) **chưa chạy**.

### Nguyên nhân (tái hiện được)

Stage `usermodel` dựng các "view" cho reader trước khi huấn luyện; view `knn_rag` gọi `semgraph.Memory.knn_view` cho toàn catalogue (104 151 bài, k = 20). Hàm này tính `(E[nbr] * w[:, :, None]).sum(1)`: `E[nbr]` là mảng `[104 151, 20, d]` float32 và phép nhân tạo thêm một mảng nữa.

| d | một mảng `[n, k, d]` | hai mảng | ghi chú |
|---|---|---|---|
| 1024 (BGE-large, Qwen3) | 8,5 GB | 17,1 GB | run BGE-large: 9 GB + 17 GB ≈ 26 GB < 34 GB → **sống sót sát ngưỡng** |
| 2048 (vector nối) | 17,1 GB | **34,1 GB** | 12 GB đang dùng + 34 GB ≫ 34 GB → **bị giết** |

Đo thật trên catalogue cỡ MIND-large (104 151 bài, k = 20, láng giềng ngẫu nhiên) với mã cũ: d = 128 / 256 / 384 → đỉnh RAM tăng **+2,01 / +3,99 / +5,97 GB** = đúng 2 × một mảng (1,07 / 2,13 / 3,20 GB), tuyến tính theo d; ngoại suy d = 1024 → 17 GB, d = 2048 → 34 GB. Sau khi chia khối (`budget` = 1,5·10⁸ phần tử mỗi mảng tạm, ≈ 600 MB): +1,14 / +0,88 / +0,46 GB ở d = 384 / 1024 / 2048 (đỉnh cả tiến trình 3,5 GB ở d = 2048), **kết quả giống hệt** (cùng tổng kiểm ở d = 384; test (48) so với định nghĩa một lần, sai số < 1e-6).

Vì sao không thấy sớm hơn: (1) test (47) chạy cả pipeline vector nối nhưng trên vài nghìn bài, (2) lần "thử quy mô" dùng vector 64 chiều và 30 000 mẫu train (catalogue đủ cỡ nhưng d nhỏ 16–32 lần), (3) con số "RAM đỉnh 16,2 GB" của v15 chỉ là **ảnh chụp tại cuối mỗi stage** (`psutil` RSS lúc stage kết thúc), không phải đỉnh; đỉnh tạm thời bên trong stage (17 GB ở view kNN của run chính v15, 1024 chiều trên Kaggle 29 GB) không bao giờ hiện trong log. Dòng "RAM process 9.6 GB" của run BGE-large vì vậy cũng không nói gì về đỉnh. v16b in `stage peak` / `run peak` (lấy mẫu mỗi 0,25 giây + `ru_maxrss` của nhân), đọc giới hạn cgroup của container, và in cảnh báo kèm dòng mã đang chạy khi RSS vượt 70% giới hạn.

## 1. Run BGE-large (đầy đủ; test, `MINDlarge_dev`)

Zero-shot — **khớp v15 đến 4 chữ số** (cùng encoder, cùng dữ liệu, cùng seed) nên các thay đổi của bộ đọc dữ liệu v16 không làm lệch số offline:

| Hàng | v16 (nDCG@10 / AUC) | v15 | Cold AUC |
|---|---|---|---|
| `frozen` | 0,3933 / 0,6297 | 0,3933 / 0,6297 | 0,6632 |
| `pool_topk` | 0,4045 / 0,6448 | 0,4045 / 0,6448 | 0,6691 |
| `pool_lse` | 0,4047 / 0,6459 | 0,4047 / 0,6459 | 0,6713 |
| `hist100_lse` | 0,4057 / 0,6471 | 0,4057 / 0,6471 | 0,6728 |
| `txt_cta` / `txt_cta_lse` (category trong text) | 0,3903 / 0,6363 · 0,4022 / 0,6484 | −0,0030 · −0,0024 so với hàng không category | 0,6366 · 0,6436 |

Retrieval zero-shot (RAG): `entity_rag` −0,0105 [−0,0110, −0,0100], `knn_rag` −0,0017 [−0,0023, −0,0011], `community_graphrag` −0,0079 [−0,0086, −0,0072] so với `frozen`; hướng giống v15 trên Qwen3 (−0,0013 / +0,0007 / −0,0066) — **RAG zero-shot vẫn không giúp**. Cold AUC của `txt_cta` giảm mạnh (0,6632 → 0,6366): văn bản có category giúp bài đã thấy (warm AUC 0,6231) nhưng hại bài lạnh với BGE.

**Reader học được (mới ở v16; đầu 128 chiều, 2 seed, 5 epoch tối đa):**

| Hàng | val nDCG@10 | test nDCG@10 | test AUC | cold AUC | so với |
|---|---|---|---|---|---|
| `um_frozen` | 0,4113 (seed 0,4140 / 0,4085) | **0,4152** | 0,6674 | 0,6916 | — |
| `um_knn_rag` | 0,4119 (0,4126 / 0,4112) | **0,4189** | 0,6696 | 0,7026 | `um_frozen` +0,0037 [+0,0034, +0,0041], cold AUC +0,0110 |
| `um_prior_frozen` (chỉ nhìn bài ứng viên) | 0,3147 | 0,3447 | 0,5888 | 0,6317 | — (v15 Qwen3: 0,3423 / AUC 0,579) |

* Reader BGE-large hơn zero-shot tốt nhất của chính nó: `um_frozen` 0,4152 so với `hist100_lse` 0,4057 (+0,0095) và `pool_lse` 0,4047 (+0,0105); `um_knn_rag` 0,4189 là **thành phần đơn tốt nhất trong mọi run đến nay** (v15: `txt_cta_lse` của Qwen3 0,4102; +0,0087). (Chênh lệch điểm, chưa có CI ghép cặp giữa các run: bước đó nằm sau stage bị chết.)
* So với reader Qwen3 của v15 (đầu **64** chiều): `um_frozen` 0,4006 → 0,4152 (+0,0146), `um_knn_rag` 0,4097 → 0,4189 (+0,0092). **Không tách được encoder khỏi độ rộng đầu** (128 so với 64); hai seed của `um_frozen` chênh nhau 0,0055 trên validation (v15, Qwen3: 0,0111 trên test). v16b cho cả ba encoder cùng đầu 128 để tách chúng.
* Hướng ngược với v15: ở Qwen3 reader không hơn zero-shot (0,4006 so với 0,4102), ở BGE-large reader hơn zero-shot cỡ +0,01.

## 2. Run vector nối (một phần: `controls` và `graph`)

| Hàng | val nDCG@10 | test nDCG@10 / AUC | cold AUC |
|---|---|---|---|
| `frozen` (Qwen3 ⊕ BGE-large, trung bình hai tích vô hướng) | 0,3560 | 0,3945 / 0,6379 | 0,6772 |
| `frozen` BGE-large một mình | 0,3546 | 0,3933 / 0,6297 | 0,6632 |
| `frozen` Qwen3 một mình (v15) | — | 0,3787 / 0,6215 | — |
| `frozen+pop` | 0,4023 | 0,4365 / 0,6827 | 0,7338 |
| `entity_rag` | 0,3487 | 0,3847 / 0,6238 (−0,0098) | 0,6705 |
| `knn_rag` | 0,3703 | 0,3942 / 0,6401 (−0,0003 [−0,0009, +0,0002], không khác) | 0,6685 |
| `knn_rag+pop` | 0,4129 | 0,4393 / 0,6883 (+0,0028) | 0,7334 |
| `community_graphrag` | 0,3715 | 0,3766 / 0,6172 (−0,0179) | 0,6398 |

Zero-shot trên vector nối chỉ là trung bình hai điểm (tích vô hướng = trung bình hai tích vô hướng) và đạt 0,3945: hơn BGE-large +0,0012 và hơn Qwen3 +0,0158 — tức một "ensemble hai scorer" rẻ. Reader trên vector nối: **chưa có số** (chết ở đây).

## 3. Còn lại sau lần chạy này

Chưa có bất kỳ số nào cho: stacker (`stack_llm`, `stack_meta`, `stack_pop`, đối chứng `stack_meta_v15cfg`), bảng "test − validation", supermodel chọn theo thước đo lạnh-có-trọng-số, reader trên vector nối, reader Qwen3 với đầu 128. Các dự đoán chủ quan ghi trong README §0c vẫn chưa được kiểm.
