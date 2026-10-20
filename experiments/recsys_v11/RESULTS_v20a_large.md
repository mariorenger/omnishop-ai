# Kết quả đo thật — v20a trên MIND-large (Kaggle, 2 × Tesla T4, 5,35 giờ, mọi run xong)

Nguồn: log và `results_v20a/` của `v20a_mindlarge_kar.ipynb` (kernel `ngpnguyen/v20a-mindlarge-kar`, phiên bản 1; lần đẩy đầu ở tài khoản `ccl666lv` dừng ngay vì tài khoản đó không có GPU / Internet). Mọi số trên **`MINDlarge_dev` (376 471 impression)**, CI 95% ghép cặp theo impression.

## 0. Chạy có ổn không?
* **H6 ở quy mô đủ:** Qwen3-1,7B viết mở rộng cho **104 151 / 104 151 bài** (tối đa 56 token mới) trong **96 phút trên 2 GPU**; nhúng lại trong 6,7 phút. Stage `kar_full` 6 328 s.
* Run `catkarplus` (reader trên (Qwen3 + KAR) ⊕ BGE-large, công thức `catplus`): usermodel 5 140 s. Run `qwen3kar`: usermodel 4 893 s. Tổng 5,35 giờ; RAM đỉnh 20,3 GB.
* `frozen` 0,3787 và `pool_lse` 0,3916 trùng các run trước.

## 1. Vector bài có thêm đoạn LLM viết (zero-shot)
α chọn trên validation = 1,0.

| So sánh (A − B) | A | Δ nDCG@10 [CI 95%] | Δ cold AUC |
|---|---|---|---|
| `kar_item` − `frozen` | 0,3792 | +0,0005 [−0,0000, +0,0010] | −0,0144 |
| `kar_item_lse` − `pool_lse` | 0,3930 | **+0,0013 [+0,0009, +0,0018]** | −0,0117 |
| `kar_exp` (chỉ đoạn LLM viết) − `frozen` | 0,3550 | −0,0237 | −0,0576 |

Đoạn LLM viết thêm giúp rất ít ở mức zero-shot (+0,0013 với late interaction) và làm **cold AUC giảm** (−0,012 đến −0,014). Tự nó (không có vector gốc) thì kém hẳn.

## 2. Reader học trên vector có thêm đoạn LLM viết
| Reader | test nDCG@10 | cold AUC |
|---|---|---|
| `qwen3kar:um_frozen` | 0,4181 | 0,6774 |
| `qwen3kar:um_knn_rag` | 0,4237 | 0,6938 |
| `catkarplus:um_frozen` | 0,4295 | 0,6989 |
| `catkarplus:um_knn_rag` | **0,4310** | 0,7019 |

* Tham chiếu từ run khác (cùng công thức, không ghép cặp ở đây): `catplus:um_knn_rag` của v19a = 0,4316 → `catkarplus` **không hơn** (−0,0006, trong nhiễu seed).
* `qwen3kar` cao hơn reader Qwen3 của v19b (≈0,405 / 0,410) nhưng `qwen3kar` dùng công thức huấn luyện kỹ hơn (6 epoch, toàn bộ mẫu), nên phần chênh không chỉ do LLM-text. v20c đo ghép cặp trên cùng impression.

## 3. Kết luận
* Lần đầu H6 chạy trên toàn bộ MIND-large: LLM viết thêm kiến thức cho bài **không** cải thiện rõ — +0,0013 zero-shot, reader mạnh nhất không đổi (0,4310 so với 0,4316), cold AUC giảm.
* Các thành phần vẫn vào v20c để bộ ghép quyết định.
