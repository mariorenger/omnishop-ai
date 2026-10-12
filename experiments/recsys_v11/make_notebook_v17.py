"""Build the two self-contained notebooks of v17 from the module files (they run on Google Colab as well as on Kaggle):

  v17_stacker_ft.ipynb              MIND-small (a Kaggle input dataset when there is one, otherwise Hugging Face); the finetune stage is off by default
  v17_mindlarge_hf_stacker_ft.ipynb the same pipeline on MIND-large downloaded from Hugging Face, with the finetune stage on

v17 = v16b + the ``finetune`` stage (``ftune.py``) in the BGE-large run: the top layers of the encoder unfrozen under the learned reader, against a matched frozen control, with a gate.
The notebooks of v16b (``v16b_*.ipynb``, which produced ``RESULTS_v16b_large.md``) carry the previous modules and cannot be regenerated from this file; see git history.

v16b = v16 + the fix of the crash its first real run met (the kernel died at the start of the learned readers on the 2048-d vectors: ``semgraph.Memory.knn_view`` built
two [104k articles, 20, d] arrays at once, 2 x 17 GB), the true peak RAM of every stage in the log, and the same 128-wide reader head for all three encoders.
The v16 notebooks (``v16_*.ipynb``) were removed: they cannot finish on the vector-concatenation run.

Compared with v15 (``make_notebook_v15.py``): the learned readers and the retrieval (RAG) views also run on BGE-large and on the Qwen3 + BGE-large vectors side by side
(a wider head), the stacker is chosen and stopped on what the test day looks like (time-block folds, impressions whose clicked article is new, a forward holdout over
configurations; the v15 recipe is trained next to it as a control), the supermodel's base and weights are chosen on a validation mean re-balanced to the test split's cold
share, entity-overlap and windowed click-through features, a finished run is released from memory before the next one starts, and a re-run can redo the stacker alone.
Still ONE notebook with "Run All"; no NRMS / NAML / Fastformer, no fine-tuning of the encoder.
Bump the notebook names when the content changes: Kaggle (and Colab, when it is re-uploaded) cache notebooks by file name."""
import json
import os

import make_notebook as V14
from make_notebook import code, md, writefile  # re-used cell helpers

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = {"small": "v17_stacker_ft.ipynb", "large": "v17_mindlarge_hf_stacker_ft.ipynb"}
MODULES = ["data.py", "metrics.py", "common.py", "rep_eval.py", "semgraph.py", "adapt.py", "fields.py", "pooling.py", "sid.py", "llm_user.py", "llm_gen.py",
           "refdata.py", "refmodels.py", "usermodel.py", "stages_ext.py", "stages_ref.py", "supermodel.py", "featscore.py", "stacker.py", "final_report.py", "data_lean.py",
           "hf_mind.py", "ftune.py", "suite.py"]

# ------------------------------------------------------------------------------------------------ text
V17 = """**V17 = v16b + một thí nghiệm có cổng dừng: mở khoá hai lớp cuối của BGE-large dưới reader (công thức PLM-NR).**
*Vì sao (số đo thật của v16b, `RESULTS_v16b_large.md`):* (1) stacker đã bão hoà — `stack_llm` ≈ `super_llm` (−0,0004, không khác) và `stack_meta` chỉ +0,0008 so với v15 dù `super_llm` tăng +0,0105; công thức "trung thực" cho ước lượng validation đúng (lệch test 0,007 thay vì 0,038) nhưng **không** hơn trên test (−0,0027 so với công thức v15); (2) thành phần tốt nhất là reader trên vector đóng băng (vector nối 0,4223 > BGE-large 0,4152 > Qwen3 0,4050 nDCG@10, cùng đầu 128 chiều); (3) khoảng trống còn lại so với văn liệu (chỉ nội dung + lịch sử: 0,686 AUC so với PLM-NR 0,706) nằm ở mô hình nội dung được huấn luyện end-to-end. PLM-NR chỉ fine-tune hai lớp cuối của mô hình ngôn ngữ vì chênh lệch so với fine-tune toàn bộ rất nhỏ (từ kết quả tìm kiếm, chưa mở PDF); và tower LoRA của v14 ngang `pool_lse` sau 0,5% dữ liệu với validation còn tăng.
*Làm thế nào, rẻ:* stage `finetune` (cuối run BGE-large) tính **một lần** trạng thái ẩn của các lớp dưới cho cả catalogue và ghi ra đĩa (float16, khoảng 20 GB; đọc bằng `pread`, không tính vào RAM), rồi huấn luyện chỉ hai lớp trên + reader trên các **ứng viên** của batch; phía lịch sử đọc một bảng vector được làm mới mỗi 600 bước (không gradient). Bắt đầu từ chính reader đã học trên bảng đóng băng.
*Công bằng:* reader khởi động chưa hội tụ (nhiều seed dừng ở trần 5 epoch) nên huấn luyện thêm riêng nó cũng làm validation tăng. Vì vậy có **hai nhánh cùng mẫu, cùng số bước, cùng lịch validation** — nhánh mở khoá (`ft_reader`) và nhánh đối chứng đóng băng (`ft_base`) — và cổng chỉ so hai nhánh với nhau. Checkpoint chọn theo validation có trọng số bài lạnh; test chạm một lần cho mỗi hàng.
*Cổng (`ft_gate.md`, quyết định chỉ bằng validation):* **PASS** nếu nhánh mở khoá hơn nhánh đóng băng ≥ 0,006 nDCG@10 có trọng số bài lạnh (và thô không giảm); **INCONCLUSIVE** nếu ≥ 0,002; còn lại **FAIL**. PASS → bản đầy đủ (nhiều bước và lớp hơn, gradient cả hai phía, dùng `ft_vectors.npy` trong vector nối); FAIL → dừng hướng này, giữ reader đóng băng.
*Bỏ có chủ đích:* scorer NRMS / Fastformer cho stacker (hướng B) — các reader thêm vào được trọng số 0 trong ghép tuyến tính của v16b và stacker bão hoà, nên thêm một scorer nữa có kỳ vọng thấp. Điểm xuất của `ft_reader` vẫn vào supermodel và stacker như mọi reader khác.
*Giới hạn:* một seed; chỉ phía ứng viên có gradient; văn bản cắt ở 96 token (vector đóng băng đọc 256); thời gian stage chưa đo (ước 1,5 – 2 giờ trên T4; vòng huấn luyện có trần `V11_FT_MINUTES` = 110 phút)."""

WHY = """**Vì sao có bản này (số đo thật của v15 trên MIND-large, `RESULTS_v15_large.md`):** `stack_meta` đạt AUC 0,6998 / nDCG@10 0,4508 và `stack_llm` 0,6843 / 0,4332 trên 376 471 impression dev, nhưng
(1) điểm validation out-of-fold của stacker **cao hơn test 0,03–0,04** nDCG@10 (ghép tuyến tính thì test ≥ validation), (2) ba mục tiêu huấn luyện không phân biệt được trên fold giữ lại và `ctx_n_cand` (hằng số trong một impression) chiếm gain lớn nhất,
(3) nền của supermodel được chọn trên validation "ấm" (`um_text`) lại thua `um_knn_rag` trên test, (4) mọi reader chỉ học trên vector Qwen3 trong khi BGE-large zero-shot hơn Qwen3 khoảng 0,013 nDCG@10 ở các hàng văn bản thường,
(5) thư mục làm việc ở `/tmp` nên điểm xuất và cache vector mất khi phiên kết thúc."""

WHAT = """**Bản này thay đổi gì so với v15 (năm hướng, theo thứ tự trong `RESULTS_v15_large.md` §5):**
1. **Lưu và chạy lại.** Thư mục làm việc giữ `scores/*.npz`, cache vector và `state.pkl`; chạy lại notebook thì mọi stage đã xong được bỏ qua, và **đổi một thiết lập `--stack-*` (hoặc đặt `V11_RESTACK=1`) chỉ chạy lại stage `stack`** trên điểm đã lưu. Trên Colab đặt `V11_DRIVE=1` để thư mục làm việc nằm trên Google Drive và sống sót qua mất kết nối.
2. **Reader và RAG trên BGE-large, và trên vector nối Qwen3 ⊕ BGE-large** (đầu đọc rộng 128 chiều thay vì 64 — **ở cả ba encoder, kể cả Qwen3 chính**, để so được với nhau; v15 dùng 64; vector nối là 2 × 1024 chiều, mỗi nửa chuẩn hoá và chia √2 nên tích vô hướng là *trung bình* hai tích vô hướng — zero-shot trên nó chỉ là trung bình hai điểm, chỉ reader và RAG mới được lợi). Cả hai chạy trước encoder chính và được **giải phóng khỏi RAM** khi xong (v15 giữ run đầu sống).
3. **Stacker bớt "học thuộc ngày validation".** Fold là các **khối thời gian** của ngày validation (có đệm hai bên), không chia theo người dùng; dừng sớm và **chọn cấu hình** theo nDCG@10 trên các impression có click vào bài **mới với mô hình** (chưa xuất hiện trong slate huấn luyện) — đúng tình huống của 82% click ở test; cấu hình (cây sâu như v15 / bỏ cột hằng-trong-impression và làm thô cột gần-định-danh-bài / cây nông hơn / bỏ prior) so một lần trên **holdout tiến theo thời gian**, hai cấu hình tốt nhất được huấn luyện ở mọi fold rồi lấy trung bình. Công thức v15 cũng được huấn luyện trên đúng các thành phần đó làm **đối chứng** (`stack_meta_v15cfg`).
4. **Chọn nền và trọng số của supermodel theo nDCG@10 có trọng số bài lạnh** (validation được gán lại trọng số để tỉ lệ click lạnh là 80%, `--cold-target`; test của MIND-large có 82%).
5. **Đặc trưng mới:** trùng entity (IDF) giữa ứng viên và click của người đọc (toàn bộ / ba click gần nhất); với `stack_pop` thêm CTR theo cửa sổ 1 / 6 / 24 giờ vừa qua (đếm click của các giờ *đã hoàn tất* — khung streaming như popularity).
**Không có trong bản này:** fine-tune encoder bằng click (dự kiến là một pilot riêng có cổng dừng, sau khi có kết quả bản này), scorer NRMS / Fastformer cho stacker, tower LoRA."""

HOW = """**Stacker làm thế nào — không nhìn nhãn test để chọn gì:**
1. Mọi stage vẫn đo từng kỹ thuật riêng lẻ (CI ghép cặp so với đối chứng của nó) và giữ **điểm thô** của từng thành phần; có ba run encoder (Qwen3-Embedding chính, BGE-large, vector nối) và reader `prior` (chỉ nhìn nội dung bài).
2. Với mỗi ứng viên, bảng đặc trưng gồm: điểm của mọi thành phần (z-score trong impression, vài scorer thêm giá trị thô) · đặc trưng lịch sử/slate từ vector đóng băng + trùng entity với lịch sử ·
   **meta**: số lần train đã hiển thị / click bài, CTR của category, **độ phơi nhiễm trong log impression** (bao lâu kể từ lần đầu bài xuất hiện, đã hiển thị bao nhiêu lần, trong 1 / 6 / 24 giờ qua — *không dùng nhãn*) · (biến thể cuối) **popularity online** của V10 và CTR theo cửa sổ giờ.
3. Mô hình: **XGBoost** (GPU nếu có; dự phòng LightGBM, rồi scikit-learn), mục tiêu `binary:logistic` (v15: ba mục tiêu cho điểm giữ-lại chênh vài phần nghìn, nên chọn cái nhanh nhất), huấn luyện **chỉ trên impression validation**, fold = **khối thời gian**; điểm validation báo cáo là **out-of-fold**;
   điểm test = trung bình (sau z-score trong impression) của các mô hình fold của các cấu hình được chọn, tính **một lần** ở cuối. Test không bao giờ được dùng để chọn mục tiêu, cấu hình, số cây hay biến thể.
4. Ba biến thể: `stack_llm` (điểm + lịch sử/slate + entity) · `stack_meta` (+ meta, **vẫn offline**) · `stack_pop` (+ popularity online: khung *streaming*, **không** so được với số offline của bài báo), và đối chứng `stack_meta_v15cfg`.
   Lưu ý minh bạch: các đặc trưng phơi nhiễm đọc **danh sách ứng viên của chính impression test** (không đọc nhãn); `stack_llm` không dùng chúng.
5. Các kiểm tra tự động (selftest): bảng đặc trưng == tính lại độc lập; chạy lại **toàn bộ** với nhãn test bị xáo → điểm test của `stack_llm`/`stack_meta` **giống hệt**; điểm out-of-fold của một fold không đổi khi chỉ lật nhãn của chính fold đó;
   CTR theo cửa sổ không bao giờ thấy nhãn của giờ hiện tại hay sau đó; một ca nhân tạo "học thuộc bài trong ngày": fold theo người dùng báo kết quả lạc quan, còn fold theo thời gian và lát cắt "bài mới" của nó thì không."""

COLAB = """**Chạy trên Colab:** Runtime → *Change runtime type* → **T4 GPU** (nếu có thì chọn thêm **High-RAM**) → *Run all*. MIND-large cần RAM. Con số "đỉnh 16,2 GB" của v15 chỉ là ảnh chụp **cuối** mỗi stage nên bỏ sót đỉnh tạm thời bên trong stage (view kNN của reader cần thêm ~17 GB ở 1024 chiều, ~34 GB ở 2048 chiều — chính nó làm chết v16); v16b xử lý phần đó theo khối và in **đỉnh thật** của từng stage (`stage peak`, `run peak`) cùng cảnh báo kèm dòng mã đang chạy khi RAM vượt 70% giới hạn của container. Vẫn **chưa có số đỉnh thật cho cả pipeline** nên **Colab miễn phí (~12,7 GB) rất có thể không đủ**. Notebook đọc RAM (kể cả giới hạn cgroup của container) và tự giảm quy mô: từ 24 GB trở lên giữ như v15 (validation 150 000 impression, 1,2 triệu mẫu train), 16–24 GB: 100 000 / 800 000, dưới 16 GB: 50 000 / 400 000 —
và in cảnh báo; khi đó số **không so thẳng** với v15 và các ngưỡng này là ước tính chưa thử trên máy nhỏ. Muốn sống sót qua mất kết nối: đặt `V11_DRIVE=1` (hỏi quyền Google Drive một lần; thư mục làm việc và cache vector lên Drive, cần khoảng 10 GB trống; dữ liệu MIND vẫn tải về đĩa của máy). Cuối cùng notebook tạo `SUMMARY_v17.md`; trên Colab nó tự tải về trình duyệt (nếu bị chặn, mở ngăn *Files* bên trái).
**Chạy lại sau khi chết giữa chừng:** nếu máy còn giữ thư mục làm việc (`/content/v16_large_out`, hoặc Drive khi `V11_DRIVE=1`), Run all lại sẽ **bỏ qua các stage đã xong** (kể cả toàn bộ run BGE-large) và cache vector; nếu máy đã bị xoá thì phải chạy lại từ đầu."""

SMALL_INTRO = f"""# V17 — v16b + mở khoá hai lớp cuối của BGE-large (tắt mặc định ở bản small: `V11_FT=1` để bật) · MIND-small · Qwen3-Embedding-0.6B + BGE-large (ước ~3 giờ trên T4, chưa đo)

{V17}

Cùng giao thức V10 (validation theo ngày, `MINDsmall_dev` chỉ là test, CI 95% ghép cặp, lát cold/warm) và cùng supermodel tuyến tính của v14, **cộng** bộ xếp hạng ở dưới. Không huấn luyện NRMS / NAML / Fastformer.

{WHY}

{WHAT}

{HOW}

{V14.TECHNIQUES}

**Cách chạy:** (Kaggle) Settings → Accelerator **GPU**, Internet **On**; *Add Input* dataset MIND có **cả** `MINDsmall_train` và `MINDsmall_dev` (ví dụ `thinhhuynh3108/mindsmall`) — hoặc không thêm gì: khi không thấy dataset, notebook tải MIND-small từ Hugging Face. (Colab) xem dưới. Rồi **Run All**.
**Kiểm tra môi trường trước (vài phút, chưa tốn giờ GPU đáng kể):** ở ô *Cấu hình* đặt `QUICK` = `"1"` (dòng `QUICK = ...`), `V11_ENCODER` = `BAAI/bge-small-en-v1.5` và `V11_ENCODER2` = rỗng — mọi stage chạy một lần trên encoder nhỏ. (Chỉ `QUICK=1` mà vẫn dùng Qwen3 + BGE-large thì vẫn mã hoá cả catalogue 65k bài, ~40 phút.) Công tắc khác: `V11_ENCODER2=` (rỗng) bỏ BGE-large và run vector nối; `V11_CONCAT=0` bỏ run vector nối; `V11_BGE_READERS=0` chỉ giữ scorer zero-shot của BGE-large; `V11_RESTACK=1` chạy lại riêng stacker; `V11_STACK_BACKEND=lightgbm|sklearn|xgboost`; `V11_STACK_CV=user` (fold như v15), `V11_STACK_CONFIGS=clean` (một cấu hình, bỏ bước chọn), `V11_STACK_BLEND`, `V11_COLD_TARGET`.
**Kiểm tra giao thức (1–2 phút):** BGE-small `frozen` phải khớp V10 `bge_zeroshot` (AUC 0.6241 / nDCG@10 0.3909); nếu `MISMATCH` thì dừng, không tin số nào.

{COLAB}

**Thời gian:** *ước tính* từ số đo thật v13/v15, chưa đo ở bản này. `BUDGET_H` (mặc định 6) là ngân sách đồng hồ chung; stage nào không còn thời gian thì bị bỏ qua và chạy lại cell sẽ resume.
**Nhận lại:** `SUMMARY_v17.md` (FINAL_REPORT mục 1b là bảng stacker; `ft_gate.md` là cổng fine-tune nếu bật) + log nếu có lỗi."""

LARGE_INTRO = f"""# V17 — v16b + mở khoá hai lớp cuối của BGE-large (có cổng dừng) · MIND-large từ Hugging Face · Kaggle / Colab (ước ~6 – 6,5 giờ trên T4; v16b đo thật 4,49 giờ)

{V17}

**Lịch sử — sửa ở v16b so với v16 (lần chạy thật đầu tiên của v16 chết giữa chừng):** trên máy 34 GB, run BGE-large xong (1,07 giờ) nhưng kernel chết ở đầu stage `usermodel` của run vector nối (2048 chiều). Nguyên nhân: `Memory.knn_view` (view kNN-RAG mà reader dùng) dựng cả mảng [104 151 bài × 20 láng giềng × d] **hai lần** — 2 × 8,5 GB ở 1024 chiều (run BGE-large chạy sát ngưỡng và sống sót), 2 × 17 GB ở 2048 chiều, chồng lên ~12 GB đã dùng. v16b xử lý theo khối (kết quả y hệt, có test), in RAM đỉnh thật của từng stage, và dùng đầu đọc 128 chiều cho cả ba encoder. Số đo của run BGE-large nằm ở `RESULTS_v16_partial.md`.

Tiếp theo lần chạy thật của v15 (2,73 giờ trên Kaggle, mọi stage xong). Cùng giao thức V10 trên `MINDlarge_dev` (376 471 impression, chạm một lần), cùng supermodel tuyến tính của v14, **cộng** bộ xếp hạng. Không huấn luyện NRMS / NAML / Fastformer, không tower LoRA, không fine-tune encoder.

{WHY}

{WHAT}

{HOW}

**Dữ liệu.** `hf_mind.py` tải `MINDlarge_train.zip` và `MINDlarge_dev.zip` từ Hugging Face dataset `Recommenders/MIND` (đã chạy được ở v14/v15: 531 MB + 104 MB trong 12 giây), chỉ giải nén `news.tsv` và `behaviors.tsv`; `MINDlarge_test.zip` (không nhãn) không bao giờ được dùng. Cần Internet.
Khác bản small (đều ghi vào log và `FINAL_REPORT.md`): bộ đọc `data_lean.py` (cùng split / popularity / negative với `MindData.from_mind`; thêm đặc trưng phơi nhiễm, đếm theo bài, CTR theo cửa sổ giờ và thời điểm của impression); validation giữ ngẫu nhiên **150 000** impression (RAM dưới 24 GB: 100 000 hoặc 50 000);
các stage có huấn luyện dùng **1,2 triệu** mẫu train (RAM dưới 24 GB: 800 000 hoặc 400 000), `--um-epochs 5`; `user_rag` bỏ qua; không ghi `test_arrays.npz`.

**Thời gian (v16b đo thật 4,49 giờ trên Kaggle T4):** run BGE-large 1,03 giờ (đọc dữ liệu ≈ 4,5 phút · E0 405 s · controls 52 s · pooling 277 s · moredata 683 s · graph 256 s · usermodel 1 679 s) · run vector nối 0,99 giờ (E0 1 016 s, phần BGE lấy từ cache · controls 64 s · graph 416 s · usermodel 1 773 s) · run Qwen3 chính 2,47 giờ (E0 từ cache · controls 54 s · pooling 286 s · moredata 1 333 s · graph 244 s · usermodel 2 976 s · supermodel 251 s · **stack 3 311 s**). v17 thêm stage `finetune` ≈ **1,5 – 2 giờ (chưa đo)**: trạng thái ẩn ≈ 4 phút, reader khởi động ≈ 6 phút, nhánh mở khoá có trần 110 phút, nhánh đóng băng ≈ 5 phút. Tổng ≈ 6 – 6,5 giờ; `BUDGET_H=10` là trần.

{COLAB}

**Cách chạy:** (Kaggle) Settings → GPU, Internet **On** → **Run All**. (Colab) xem trên. Kiểm tra môi trường trước bằng notebook small (xem ở đó); `QUICK = ...` → `"1"` trên notebook large vẫn tải MIND-large và mã hoá cả catalogue (~40 phút). Hết RAM/thời gian: giảm `V11_TRAIN_SUB`, `V11_VAL_MAX`, hoặc đặt `V11_ENCODER2=` (bỏ BGE-large và vector nối), `V11_CONCAT=0`, `V11_BGE_READERS=0`. Stage fine-tune: `V11_FT=0` tắt; `V11_FT_LAYERS` (2), `V11_FT_STEPS` (18000), `V11_FT_MINUTES` (110), `V11_FT_MAXLEN` (96); cần ≥ 25 GB đĩa trống ở thư mục tải về (nếu không, stage tự bỏ qua).
**Chạy lại chỉ stacker** (sau khi một lần chạy đã xong, cùng thư mục làm việc): đặt `V11_RESTACK=1`, hoặc đổi một trong `V11_STACK_CV` / `V11_STACK_CONFIGS` / `V11_STACK_BLEND` / `V11_STACK_FOLDS` / `V11_STACK_BACKEND` ở ô Cấu hình, rồi Run All — các stage khác được resume, chỉ `stack` chạy lại (vài chục phút cộng thời gian đọc dữ liệu).
**Nhận lại:** `SUMMARY_v17.md` (mục 1b của FINAL_REPORT là bảng stacker, mục 4 là số công bố — thứ cấp; `ft_gate.md` là cổng fine-tune) + log (dòng `-- stage ... finished` có RAM đỉnh thật; dòng `!! RAM` là cảnh báo kèm vị trí mã)."""

CONFIG_HEAD = """import os, glob, sys, gc, time, json, shutil
T_NB = time.time()                                                    # giờ bắt đầu của notebook: ngân sách BUDGET_H tính từ đây cho CẢ ba run
IN_KAGGLE = os.path.isdir("/kaggle/working")
IN_COLAB = (not IN_KAGGLE) and (("google.colab" in sys.modules) or bool(os.environ.get("COLAB_GPU")) or os.path.isdir("/content/sample_data"))
QUICK = bool(int(os.environ.get("V11_QUICK", "0")))                  # True: cỡ đồ chơi để bắt lỗi môi trường
BUDGET_H = float(os.environ.get("V11_BUDGET_H", "@BUDGET@"))          # ngân sách đồng hồ chung (giờ)@BUDGET_NOTE@
ENCODER = os.environ.get("V11_ENCODER", "Qwen/Qwen3-Embedding-0.6B")  # encoder chính: mọi kỹ thuật, reader, RAG
ENCODER2 = os.environ.get("V11_ENCODER2", "BAAI/bge-large-en-v1.5")   # encoder thứ hai; '' = bỏ nó và run vector nối
FT = bool(int(os.environ.get("V11_FT", "@FT@")))                    # stage finetune ở run BGE-large: mở khoá hai lớp cuối dưới reader, có đối chứng đóng băng và cổng
FT_LAYERS = int(os.environ.get("V11_FT_LAYERS", "2"))
FT_STEPS = int(os.environ.get("V11_FT_STEPS", "18000"))
FT_MINUTES = float(os.environ.get("V11_FT_MINUTES", "110"))     # trần của vòng huấn luyện (giờ đồng hồ thực)
FT_MAXLEN = int(os.environ.get("V11_FT_MAXLEN", "96"))          # số token mỗi bài trong cache trạng thái ẩn
CONCAT = bool(int(os.environ.get("V11_CONCAT", "1"))) and bool(ENCODER2)       # run thứ ba: reader + RAG trên vector Qwen3 ⊕ BGE-large nối nhau
BGE_READERS = bool(int(os.environ.get("V11_BGE_READERS", "1")))     # reader + RAG trên BGE-large (0: chỉ scorer zero-shot như v15)
CONTROLS = bool(int(os.environ.get("V11_CONTROLS", "@CONTROLS@")))            # 1: thêm đối chứng "category của bài khác" (txt_ctr*): mã hoá thêm một lần cả catalogue
TEXT_MODES = os.environ.get("V11_TEXT_MODES", "cta,ctr" if CONTROLS else "cta")
PROTOCOL_CHECK = bool(int(os.environ.get("V11_PROTOCOL_CHECK", "@CHECK@")))   # BGE-small frozen phải khớp V10 (1-2 phút)@CHECK_NOTE@
CHECK_ENCODER = os.environ.get("V11_CHECK_ENCODER", "BAAI/bge-small-en-v1.5")
DEVICE = os.environ.get("V11_DEVICE", "auto")
USE_REFERENCE = bool(int(os.environ.get("V11_REFERENCE", "1")))       # so HFEncoder với Sentence-Transformers trên 64 bài trước khi chạy
STACK_FOLDS = int(os.environ.get("V11_STACK_FOLDS", "3" if QUICK else "4"))
STACK_BACKEND = os.environ.get("V11_STACK_BACKEND", "auto")           # auto: xgboost (GPU nếu có) -> lightgbm -> sklearn
SAVE_SCORES = int(os.environ.get("V11_SAVE_SCORES", "1"))            # 1: ghi stack_scores.npz (điểm out-of-fold và test của từng biến thể, ~100-150 MB trên MIND-large)
RESTACK = int(os.environ.get("V11_RESTACK", "0"))                    # 1: chạy lại riêng stage stack trên điểm đã lưu (cùng thư mục làm việc)
STACK_CV = os.environ.get("V11_STACK_CV", "time")                    # fold của stacker: time (khối thời gian của ngày validation) | user (như v15)
STACK_CONFIGS = os.environ.get("V11_STACK_CONFIGS", "v15,clean,clean_d5,clean_d3,noprior_d5")   # các cấu hình so trên holdout tiến (stacker.CONFIGS)
STACK_BLEND = int(os.environ.get("V11_STACK_BLEND", "2"))            # số cấu hình tốt nhất được huấn luyện ở mọi fold và lấy trung bình
COLD_TARGET = float(os.environ.get("V11_COLD_TARGET", "0.8"))        # tỉ lệ click lạnh mà thước đo validation dùng để CHỌN phản ánh (test MIND-large: 82%); 0 = trung bình thô
USE_DRIVE = bool(int(os.environ.get("V11_DRIVE", "0")))             # Colab: thư mục làm việc trên Google Drive (hỏi quyền một lần) để chạy lại / resume sau khi mất kết nối
DRIVE_DIR = None
if USE_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/recsys_v16"
    os.makedirs(DRIVE_DIR, exist_ok=True)
sys.path.insert(0, os.getcwd())
import common                                                        # đã được ghi ra ở bước 1
RAM_GB = common.ram_total_gb()                                       # RAM máy, hoặc giới hạn cgroup của container nếu thấp hơn (psutil báo RAM của máy chủ)
RAM_TIER = "high" if (RAM_GB == 0 or RAM_GB >= 24) else ("mid" if RAM_GB >= 16 else "low")     # 0 = không đọc được RAM: coi như máy lớn
LOW_RAM = RAM_TIER != "high"                                          # Colab miễn phí ~12,7 GB ("low"): MIND-large ở quy mô v15 rất có thể hết RAM
HF_REPO = os.environ.get("V11_HF_REPO", "Recommenders/MIND")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("V11_HF_TOKEN") or None
if HF_TOKEN is None:                                                  # chỉ cần nếu repo trên Hub đòi đăng nhập: Kaggle secret hoặc Colab secret tên HF_TOKEN
    for _get in (lambda: __import__("kaggle_secrets").UserSecretsClient().get_secret("HF_TOKEN"), lambda: __import__("google.colab.userdata", fromlist=["userdata"]).get("HF_TOKEN")):
        try:
            HF_TOKEN = _get() or None
        except Exception:
            HF_TOKEN = None
        if HF_TOKEN:
            break
def pick_base(need_gb=15):
    \"\"\"Thư mục cục bộ: /kaggle/temp, /content, /tmp, /kaggle/working -- cái đầu tiên còn đủ chỗ.\"\"\"
    cands = [d for d in ("/kaggle/temp", "/content", "/tmp", "/kaggle/working") if os.path.isdir(d)]
    for d in cands:
        if shutil.disk_usage(d).free / 1e9 >= need_gb:
            return d
    return cands[0] if cands else "."
LOCAL_BASE = os.environ.get("V11_BASE") or pick_base()                  # file tải về (MIND): tải lại được trong vài phút
BASE = os.environ.get("V11_BASE") or DRIVE_DIR or LOCAL_BASE            # thư mục làm việc + cache vector (mã hoá lại tốn hàng giờ): lên Drive khi bật V11_DRIVE
print(f"môi trường: {'Colab' if IN_COLAB else 'Kaggle' if IN_KAGGLE else 'khác'} | RAM {RAM_GB:.0f} GB ({RAM_TIER}) | thư mục làm việc {BASE}" + (" (Google Drive)" if BASE == DRIVE_DIR else "") + f" | dữ liệu tải về {LOCAL_BASE}")
"""

CONFIG_SMALL = """MIND_TRAIN = os.environ.get("V11_MIND_TRAIN")                      # để None -> tự dò trong /kaggle/input, rồi tải từ Hugging Face
MIND_DEV = os.environ.get("V11_MIND_DEV")
WORK = os.path.join(BASE, "v16_out") + ("_quick" if QUICK else "")    # QUICK không bao giờ lẫn vào bản đầy đủ
CACHE = os.path.join(BASE, "v16_cache")
RESULTS = ("/kaggle/working/results_small" if IN_KAGGLE else "results_small") + ("_quick" if QUICK else "")
"""

LOW_RAM_NOTE = """if LOW_RAM:
    print("!! RAM dưới 24 GB: quy mô MIND-large được giảm tự động (validation / mẫu train nhỏ hơn v15, số không so thẳng được); nếu vẫn hết RAM hãy chọn runtime High-RAM hoặc đặt V11_TRAIN_SUB / V11_VAL_MAX nhỏ hơn.")
"""

CONFIG_LARGE = """VAL_MAX = int(os.environ.get("V11_VAL_MAX", "5000" if QUICK else {"high": "150000", "mid": "100000", "low": "50000"}[RAM_TIER]))     # validation giữ ngẫu nhiên từng này impression (0 = tất cả)
TEST_MAX = int(os.environ.get("V11_TEST_MAX", "20000" if QUICK else "0"))       # 0 = toàn bộ MINDlarge_dev (376k impression)
TRAIN_SUB = int(os.environ.get("V11_TRAIN_SUB", "50000" if QUICK else {"high": "1200000", "mid": "800000", "low": "400000"}[RAM_TIER]))  # mẫu train cho các stage có huấn luyện (0 = tất cả)
UM_EPOCHS = int(os.environ.get("V11_UM_EPOCHS", "2" if QUICK else "5"))
WORK = os.path.join(BASE, "v16_large_out") + ("_quick" if QUICK else "")
RESULTS = ("/kaggle/working/results_large" if IN_KAGGLE else "results_large") + ("_quick" if QUICK else "")
CACHE = os.path.join(BASE, "v16_large_cache")
"""

CONFIG_TAIL = """sys.path.insert(0, os.getcwd())
print(dict(QUICK=QUICK, BUDGET_H=BUDGET_H, ENCODER=ENCODER, ENCODER2=ENCODER2, CONCAT=CONCAT, BGE_READERS=BGE_READERS, FT=FT, TEXT_MODES=TEXT_MODES, PROTOCOL_CHECK=PROTOCOL_CHECK,
           STACK_BACKEND=STACK_BACKEND, STACK_CV=STACK_CV, STACK_CONFIGS=STACK_CONFIGS, STACK_BLEND=STACK_BLEND, COLD_TARGET=COLD_TARGET, RESTACK=RESTACK, WORK=WORK))
"""

DEPS = '''%pip install -q "transformers>=4.51" peft sentence-transformers
import importlib, subprocess, sys
for pkg in ("xgboost", "lightgbm", "psutil"):            # có sẵn trên ảnh Kaggle và Colab; cài nếu thiếu (stacker rơi về scikit-learn nếu cả hai đều không dùng được)
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True).stdout[-300:])
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout or "!! không thấy GPU (Colab: Runtime -> Change runtime type -> T4 GPU)")
'''

DATA_SMALL = '''if not MIND_TRAIN:
    hits = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/news.tsv", recursive=True)})
    trs = [d for d in hits if "train" in d.lower()]
    dvs = [d for d in hits if "dev" in d.lower() or "valid" in d.lower()]
    for t in trs:
        sib = [d for d in dvs if os.path.dirname(d) == os.path.dirname(t)]
        if sib:
            MIND_TRAIN, MIND_DEV = t, sib[0]
            break
if not (MIND_TRAIN and MIND_DEV):
    import hf_mind
    print("không thấy dataset MIND-small trong /kaggle/input -> tải từ Hugging Face")
    os.makedirs(LOCAL_BASE, exist_ok=True)
    MIND_TRAIN, MIND_DEV = hf_mind.ensure_mind("small", os.path.join(LOCAL_BASE, "mind_small_hf"), repo=HF_REPO, token=HF_TOKEN, extra_roots=["/kaggle/input"], log=print)
assert MIND_TRAIN and MIND_DEV and MIND_TRAIN != MIND_DEV, "cần cặp MINDsmall_train + MINDsmall_dev của CÙNG một dataset"
count = lambda p: sum(1 for _ in open(p, encoding="utf-8"))
print("train:", MIND_TRAIN, "| behaviors:", f"{count(MIND_TRAIN + '/behaviors.tsv'):,}")
print("dev  :", MIND_DEV, "| behaviors:", f"{count(MIND_DEV + '/behaviors.tsv'):,}  (MIND-small: ~157k / ~73k)")
os.makedirs(WORK, exist_ok=True)
'''

DATA_LARGE = '''import hf_mind
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM: {vm.total / 1e9:.0f} GB total, {vm.available / 1e9:.0f} GB free; this notebook may use {RAM_GB:.0f} GB (the container limit when it is lower)")
except Exception:
    pass
print(f"disk free at {LOCAL_BASE}: {shutil.disk_usage(LOCAL_BASE).free / 1e9:.0f} GB")
os.makedirs(WORK, exist_ok=True)
MIND_TRAIN, MIND_DEV = hf_mind.ensure_mind("large", os.path.join(LOCAL_BASE, "mind_large_hf"), repo=HF_REPO, token=HF_TOKEN, extra_roots=["/kaggle/input"], log=print)
print("train:", MIND_TRAIN, "| dev:", MIND_DEV)
'''

RUNS_TEMPLATE = """import torch
import suite, common, final_report
common.CLOCK.reset()                                   # chạy lại cell trong cùng kernel: đồng hồ ngân sách bắt đầu lại

def reference_encoder(name, max_len):
    \"\"\"Tham chiếu Sentence-Transformers (chỉ 64 bài) để bắt lỗi pooling/EOS/prefix. Trả None nếu không dùng được.\"\"\"
    if not USE_REFERENCE:
        return lambda texts: None
    def ref(texts):
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
            st.max_seq_length = max_len
            out = st.encode(list(texts), normalize_embeddings=True)
            del st
            gc.collect(); torch.cuda.empty_cache()
            return out
        except Exception as e:
            print("reference encoder unavailable, skipping the pooling guard:", repr(e)[:120])
            return None
    return ref

COMMON = ["--mind-train", MIND_TRAIN, "--mind-dev", MIND_DEV, "--cache-dir", CACHE, "--device", DEVICE, "--cold-target", str(COLD_TARGET)]
def budget_args():
    \"\"\"Mỗi suite.main khởi động lại đồng hồ của nó: cho mỗi run phần ngân sách CÒN LẠI của notebook (BUDGET_H tính từ ô cấu hình).\"\"\"
    return ["--budget-hours", "%.3f" % max(BUDGET_H - (time.time() - T_NB) / 3600, 0.25)]
@SCALE@
TAG, TAG2, TAG3 = "qwen3", "bge", "cat"
READERS = ["--um-seeds", "2", "--um-seeds-focus", "2", "--um-extra", "prior:frozen", "--um-dim", "128"]
# run 1: the second encoder -- zero-shot scorers, and (BGE_READERS) the retrieval views + learned readers with a wider head
ARGS2 = [*COMMON, *SCALE, "--encoder", ENCODER2, "--work", WORK + "/" + TAG2, "--max-len", "256", "--text-modes", "cta", "--hist-lens", "100", "--km-ks", "5", "--seeds", "1"]
STAGES2 = "controls,pooling,moredata"
if BGE_READERS:
    ARGS2 += ["--um-views", "frozen,text,knn_rag", *READERS]
    STAGES2 += ",graph,usermodel"
    if FT:                                             # the top layers of the encoder unfrozen under the reader; hidden states of the lower layers on the local disk (not on Drive)
        ARGS2 += ["--ft-layers", str(FT_LAYERS), "--ft-steps", str(FT_STEPS), "--ft-minutes", str(FT_MINUTES), "--ft-maxlen", str(FT_MAXLEN),
                  "--ft-cache-dir", os.path.join(LOCAL_BASE, "ft_hidden")]
        STAGES2 += ",finetune"
# run 2: the Qwen3 and BGE-large vectors side by side (read from the cache of the single encoders): retrieval views + learned readers only
ARGS3 = [*COMMON, *SCALE, "--encoder", "concat:" + ENCODER + "+" + ENCODER2, "--work", WORK + "/" + TAG3, "--max-len", "256", "--seeds", "1", "--um-views", "frozen,knn_rag", *READERS]
STAGES3 = "controls,graph,usermodel"
# run 3: the main encoder -- every technique, then the linear supermodel and the stacker (which reads the exported scores of the runs above)
ARGS = [*COMMON, *SCALE, "--encoder", ENCODER, "--work", WORK + "/" + TAG, "--max-len", "256", "--q-max-len", "384", "--hist-k", "10", "--seeds", "1",
        "--text-modes", TEXT_MODES, "--hist-lens", "100", "--km-ks", "5", "--hq-subset", "1",
        "--um-views", "frozen,text,knn_rag", "--um-seeds", "2", "--um-seeds-focus", "2", "--um-extra", "prior:frozen,prior:text", "--um-dim", "128",
        "--stack-folds", str(STACK_FOLDS), "--stack-backend", STACK_BACKEND, "--stack-save-scores", str(SAVE_SCORES), "--stack-free-memory", "1", "--restack", str(RESTACK),
        "--stack-cv", STACK_CV, "--stack-configs", STACK_CONFIGS, "--stack-blend", str(STACK_BLEND)]
STAGES = "controls,pooling,moredata,graph,usermodel,supermodel,stack"       # rẻ và quan trọng trước; supermodel và stacker dùng những gì đã có
if QUICK:                                              # cỡ đồ chơi: mọi họ stage chạy một lần
    EXTRA_QUICK = ["--pairs", "20000", "--betas", "0.5", "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5", "--um-epochs", "2", "--um-seeds", "1", "--um-seeds-focus", "1",
                   "--km-ks", "2", "--stack-rounds", "60", "--stack-configs", "clean,clean_d5",
                   "--ft-layers", "1", "--ft-maxlen", "32", "--ft-steps", "40", "--ft-batch", "8", "--ft-refresh", "10", "--ft-eval-every", "20", "--ft-minutes", "5"]
    ARGS, ARGS2, ARGS3 = ARGS + EXTRA_QUICK, ARGS2 + EXTRA_QUICK, ARGS3 + EXTRA_QUICK
S2 = S3 = None
"""

SCALE_LARGE = """SCALE = ["--lean-data", "1", "--val-max", str(VAL_MAX), "--test-max", str(TEST_MAX), "--train-sub", str(TRAIN_SUB), "--save-arrays", "0",
         "--skip-user-rag", "1", "--um-epochs", str(UM_EPOCHS)]"""
SCALE_SMALL = """SCALE = ["--lean-data", "1"]                           # cùng split / popularity / negative với bộ đọc V10 (đã kiểm), và cần cho đặc trưng phơi nhiễm của stacker"""


def cell_config(profile):
    large = profile == "large"
    head = (CONFIG_HEAD.replace("@BUDGET@", "10" if large else "6").replace("@BUDGET_NOTE@", "; Kaggle giết notebook ở 12 giờ, Colab miễn phí thường sớm hơn" if large else "")
            .replace("@CHECK@", "0" if large else "1").replace("@CONTROLS@", "0" if large else "1").replace("@FT@", "1" if large else "0").replace("@CHECK_NOTE@", "; MIND-large không có số V10" if large else ""))
    return code(head + (LOW_RAM_NOTE + CONFIG_LARGE if large else CONFIG_SMALL) + CONFIG_TAIL)


def cell_runs(profile):
    return code(RUNS_TEMPLATE.replace("@SCALE@", SCALE_LARGE if profile == "large" else SCALE_SMALL))


PAIRS = '''Q = TAG
pairs = [# --- the stacker against what it is made of, against the linear supermodel and against the v15 recipe on the same components
         ((Q, "stack_llm"), (Q, "super_zs")), ((Q, "stack_llm"), (Q, "super_llm")), ((Q, "stack_meta"), (Q, "super_llm")), ((Q, "stack_meta"), (Q, "stack_llm")),
         ((Q, "stack_meta"), (Q, "stack_meta_v15cfg")), ((Q, "stack_llm"), (Q, "txt_cta_lse")), ((Q, "stack_llm"), (Q, "um_knn_rag")), ((Q, "stack_llm"), (Q, "frozen")),
         ((Q, "stack_pop"), (Q, "stack_meta")), ((Q, "stack_pop"), (Q, "super_llm+pop")), ((Q, "stack_pop"), (Q, "frozen+pop")),
         # --- the linear supermodel (as in v14, selection now on the cold-weighted validation mean)
         ((Q, "super_zs"), (Q, "super_base")), ((Q, "super_llm"), (Q, "super_zs")), ((Q, "super_llm"), (Q, "txt_cta_lse")), ((Q, "super_llm+pop"), (Q, "super_llm")),
         # --- single techniques, each against its own control
         ((Q, "pool_lse"), (Q, "frozen")), ((Q, "txt_cta_lse"), (Q, "pool_lse")), ((Q, "txt_cta_lse"), (Q, "txt_ctr_lse")), ((Q, "hist100_lse"), (Q, "pool_lse")),
         ((Q, "um_text"), (Q, "um_frozen")), ((Q, "um_knn_rag"), (Q, "um_frozen")), ((Q, "um_prior_frozen"), (Q, "frozen")),
         # --- readers and retrieval views on the other vectors
         (("bge", "um_frozen"), ("bge", "pool_lse")), (("bge", "um_knn_rag"), ("bge", "um_frozen")), (("bge", "um_frozen"), (Q, "um_frozen")),
         (("cat", "um_frozen"), ("bge", "um_frozen")), (("cat", "um_frozen"), (Q, "um_frozen")), (("cat", "um_knn_rag"), ("cat", "um_frozen")), (("cat", "um_knn_rag"), (Q, "um_knn_rag")),
         # --- the fine-tuned top layers against the matched frozen control (the gate), against the ordinary frozen reader and against the best frozen reader so far
         (("bge", "ft_reader"), ("bge", "ft_base")), (("bge", "ft_reader"), ("bge", "um_frozen")), (("bge", "ft_base"), ("bge", "um_frozen")), (("bge", "ft_reader"), ("cat", "um_frozen"))]
direct = suite.cross_compare(SUITES, TAGS, pairs, WORK + "/direct_comparisons.md")
'''

HEADLINE = '''def vkey(n):
    r = S.row(n)
    return r.get("val_ndcg@10_w", r["val_ndcg@10"])
def best_by_val(names):
    ok = [n for n in names if n in S.cache and S.row(n)["n_test"] == len(S.data.test)]
    return max(ok, key=vkey) if ok else None   # chosen on the (cold-weighted) validation mean only
lines_ = ["# Headline (test = %s impressions; each family's representative is chosen on the cold-weighted VALIDATION mean)" % f"{len(S.data.test):,}", "",
          "| row | family | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC | Δ nDCG@10 vs the supermodel's best offline row [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
sm_rungs = [r for r in (S.kv.get("supermodel") or {}).get("rungs", []) if "apex" not in r["name"] and r["name"] in S.cache and S.row(r["name"])["n_test"] == len(S.data.test)]
sup = max((r["name"] for r in sm_rungs if not r["pop"]), key=vkey, default=None)
sup_pop = max((r["name"] for r in sm_rungs if r["pop"]), key=vkey, default=None)
for n, ref in (("frozen", None), (sup, None), (best_by_val(["stack_llm"]), sup), (best_by_val(["stack_meta"]), sup), (best_by_val(["stack_meta_v15cfg"]), sup),
               ("frozen+pop", None), (sup_pop, None), (best_by_val(["stack_pop"]), sup_pop)):
    if not n:
        continue
    r = S.row(n)
    d = suite.R.compare(S.cache[n]["test"], S.cache[ref]["test"], "ndcg@10") if ref else None
    lines_.append(f"| `{n}` | {r['family']} | {r['auc']:.4f} | {r['mrr']:.4f} | {r['ndcg@5']:.4f} | {r['ndcg@10']:.4f} | {r.get('cold_auc', float('nan')):.4f} | {r.get('warm_auc', float('nan')):.4f} | "
                  + (f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" if d else "-") + " |")
lines_ += ["", "Offline rows (`stack_llm`, `stack_meta`, `stack_meta_v15cfg`, `super_*`) are comparable with offline papers up to the dev/test difference; `+pop` / `stack_pop` count the clicks of earlier impressions of the test day (streaming) and are not.",
           "`stack_meta` also reads the exposure of each candidate in the impression log (label-free); `stack_llm` does not.  `stack_meta_v15cfg` is the v15 recipe (user folds, every feature) on the same components."]
headline = "\\n".join(lines_)
open(os.path.join(WORK, "HEADLINE.md"), "w", encoding="utf-8").write(headline + "\\n")
print(headline)
'''


def build(profile):
    large = profile == "large"
    cells = [md(LARGE_INTRO if large else SMALL_INTRO),
             md("### 0. Phụ thuộc và GPU"), code(DEPS),
             md("### 1. Modules (ghi ra thư mục làm việc)")]
    cells += [writefile(m) for m in MODULES]
    cells += [md("### 2. Cấu hình"), cell_config(profile),
              md("### 3. Dữ liệu — " + ("MIND-large tải từ Hugging Face (cần Internet)" if large else "MIND-small: dataset trong /kaggle/input nếu có, nếu không thì tải từ Hugging Face")),
              code(DATA_LARGE if large else DATA_SMALL),
              md("### 4. Cấu hình ba run (BGE-large, vector nối, encoder chính)"), cell_runs(profile)]
    if not large:
        cells += [md("### 5. Kiểm tra giao thức (1–2 phút, không phải so sánh): BGE-small `frozen` phải khớp V10"),
                  code('''S_CHECK = None
if PROTOCOL_CHECK:
    try:
        suite.EQUIV_REFERENCE = None
        S_CHECK = suite.main([*COMMON, *budget_args(), "--encoder", CHECK_ENCODER, "--work", WORK + "/bge_check", "--max-len", "256", "--stages", "controls",
                              "--repro", "0.6241,0.3909", "--repro-pop", "0.6533,0.4026,0.6769,0.4330"])
        S_CHECK.release()
    except Exception as e:                       # the check must never stop the main run
        import traceback; traceback.print_exc()
        print("!! protocol check failed to run:", repr(e)[:200])
    gc.collect(); torch.cuda.empty_cache()
''')]
    else:
        cells += [code("S_CHECK = None   # MIND-large: V10 has no reference numbers\n")]
    n = 6 if not large else 5
    cells += [md(f"### {n}. Run 1 — BGE-large: controls → pooling → moredata → graph → usermodel (reader + RAG, đầu 128 chiều)"),
              code('''if ENCODER2:
    suite.EQUIV_REFERENCE = reference_encoder(ENCODER2, 256)
    try:
        S2 = suite.main([*ARGS2, *budget_args(), "--stages", STAGES2])
        S2.release()                                   # RAM: the finished run keeps its rows and per-impression results, not its data
    except Exception as e:                       # a failure here must not cost the main run: it only loses this encoder's scores
        import traceback; traceback.print_exc()
        print("!! second-encoder run failed:", repr(e)[:200], "-> the main run goes on without it")
    gc.collect(); torch.cuda.empty_cache()
'''),
              md(f"### {n + 1}. Run 2 — vector Qwen3 ⊕ BGE-large nối nhau: controls → graph → usermodel (reader + RAG, đầu 128 chiều)"),
              code('''if CONCAT:
    suite.EQUIV_REFERENCE = None                       # the pooling guard compares ONE encoder with its reference; the parts were guarded in their own runs
    try:
        S3 = suite.main([*ARGS3, *budget_args(), "--stages", STAGES3])
        S3.release()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("!! composite-encoder run failed:", repr(e)[:200], "-> the main run goes on without it")
    gc.collect(); torch.cuda.empty_cache()
'''),
              md(f"### {n + 2}. Run 3 — encoder chính (Qwen3-Embedding): controls → pooling → moredata → graph → usermodel (+ reader `prior`) → **supermodel** → **stack**"),
              code('''SUPER_DIRS = [d for d, s in ((WORK + "/" + TAG2, S2), (WORK + "/" + TAG3, S3)) if s is not None]
suite.EQUIV_REFERENCE = reference_encoder(ENCODER, 256)
S = suite.main([*ARGS, "--super-dirs", ",".join(SUPER_DIRS), *budget_args(), "--stages", STAGES])
gc.collect(); torch.cuda.empty_cache()
'''),
              md(f"### {n + 3}. Leaderboard, so sánh trực tiếp từng cặp, forest plot"),
              code('''SUITES, TAGS = zip(*[(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG)) if s is not None])
SUITES, TAGS = list(SUITES), list(TAGS)
rows = suite.leaderboard(SUITES, TAGS, (TAG, "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
rows_pop = suite.leaderboard(SUITES, TAGS, (TAG, "frozen+pop"), WORK + "/leaderboard_pop.md", plot=WORK + "/forest_pop.png", pop=True)
''' + PAIRS),
              md(f"### {n + 4}. Kết quả chính, báo cáo cuối và file gộp để gửi lại"),
              code(HEADLINE + '''print(final_report.build(S, WORK + "/FINAL_REPORT.md", WORK + "/final_tables.json", extra_suites=[S_CHECK] if S_CHECK is not None else ()))
gp = os.path.join(WORK, TAG2, "ft_gate.md")
if os.path.exists(gp):
    print("=" * 30, "fine-tune gate", "=" * 30)
    print(open(gp, encoding="utf-8").read())
p = os.path.join(WORK, TAG, "results.md")
if os.path.exists(p):
    print("=" * 30, TAG, "=" * 30)
    print(open(p, encoding="utf-8").read())
noise = ["# Seed-to-seed spread of the trained rows (test nDCG@10 of each seed; sample std with n-1; the paired CIs in the other tables do NOT include it)", "",
         "| run | row | seeds | nDCG@10 per seed | std |", "|---|---|---|---|---|"]
for tag_ in (TAG, TAG2, TAG3):
    pj = os.path.join(WORK, tag_, "results.json")
    if os.path.exists(pj):
        for r in json.load(open(pj, encoding="utf-8")):
            if r.get("ndcg10_per_seed") and len(r["ndcg10_per_seed"]) > 1:
                noise.append(f"| {tag_} | {r['name']} | {len(r['ndcg10_per_seed'])} | {', '.join(f'{x:.4f}' for x in r['ndcg10_per_seed'])} | {r.get('ndcg10_std', float('nan')):.4f} |")
open(os.path.join(WORK, "noise.md"), "w", encoding="utf-8").write("\\n".join(noise) + "\\n")
print("\\n".join(noise))
print("tổng thời gian của notebook: %.2f h" % ((time.time() - T_NB) / 3600))
parts = [("HEADLINE.md", WORK), ("FINAL_REPORT.md", WORK), ("ft_gate.md", WORK + "/" + TAG2), ("leaderboard.md", WORK), ("leaderboard_pop.md", WORK), ("direct_comparisons.md", WORK), ("noise.md", WORK),
         ("timings.json", WORK + "/" + TAG), ("memory.json", WORK + "/" + TAG), ("results.md", WORK + "/" + TAG)] + [(f, WORK + "/" + t) for t, s in ((TAG2, S2), (TAG3, S3)) if s is not None for f in ("timings.json", "memory.json", "results.md")]
os.makedirs(RESULTS, exist_ok=True)
SUMMARY = os.path.join(RESULTS, "SUMMARY_v17.md")
with open(SUMMARY, "w", encoding="utf-8") as out:
    for fn, d in parts:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out.write(f"\\n\\n## {fn} ({os.path.basename(d)})\\n\\n" + open(p, encoding="utf-8").read())
for fn in ("forest.png", "forest_pop.png", "FINAL_REPORT.md", "final_tables.json", "HEADLINE.md", "leaderboard.md", "leaderboard_pop.md", "direct_comparisons.md", "noise.md"):
    if os.path.exists(os.path.join(WORK, fn)) and os.path.abspath(WORK) != os.path.abspath(RESULTS):
        shutil.copy(os.path.join(WORK, fn), os.path.join(RESULTS, fn))
sc = os.path.join(WORK, TAG, "stack_scores.npz")
if os.path.exists(sc):
    shutil.copy(sc, os.path.join(RESULTS, "stack_scores.npz"))                # out-of-fold and test scores: lets the validation -> test gap be analysed later without a re-run
print("-> gửi lại file:", SUMMARY, f"({os.path.getsize(SUMMARY) / 1024:.0f} KB)")
if IN_COLAB:
    try:
        from google.colab import files
        files.download(SUMMARY)
    except Exception as e:
        print("(tự tải về không được:", repr(e)[:80], ") mở ngăn Files bên trái và tải", SUMMARY)
'''),
              md("""### Cách đọc kết quả

**Bước 0 — kiểm tra bộ đo:** (bản small) dòng `frozen == V10 bge_zeroshot` trong mục 0 của `FINAL_REPORT.md` phải là MATCH. Nếu không, dừng lại.

| Bạn thấy | Nghĩa là |
|---|---|
| `stack_meta` > `stack_meta_v15cfg` với CI ghép cặp loại trừ 0 **và** hiệu "test − validation" của `stack_meta` nhỏ hơn của đối chứng | các thay đổi v16 (fold theo thời gian, chọn trên bài mới, bỏ/làm thô cột, trộn cấu hình) làm đúng việc chúng được xây ra để làm |
| chỉ một trong hai điều trên đúng | giả thuyết "v15 học thuộc ngày validation" chưa được lần chạy này phân xử |
| `stack_llm` > `super_llm` (hoặc `super_zs`) với CI loại trừ 0 | bộ xếp hạng học được **hơn** ghép tuyến tính trên cùng các thành phần (cả hai chỉ dùng nội dung + lịch sử) |
| `stack_meta` > `stack_llm` | độ phơi nhiễm / đếm theo bài (vẫn offline) có thêm thông tin; nhớ rằng đặc trưng phơi nhiễm đọc danh sách ứng viên của chính impression test |
| `stack_pop` > `stack_meta` | popularity online (streaming) thêm giá trị; không so được với số offline của bài báo |
| `bge/um_frozen` > `qwen3/um_frozen`, `cat/um_*` > `bge/um_*` | reader học trên encoder thứ hai / vector nối có thêm thông tin so với Qwen3 một mình (cùng đầu 128 chiều ở cả ba, nên chênh lệch là của encoder; so `qwen3/um_frozen` với 0,4006 của v15 — đầu 64 chiều — để thấy riêng tác động của độ rộng đầu) |
| mục 1b: bảng configuration | cấu hình nào thắng trên holdout tiến theo thời gian (bài mới); các cấu hình cách nhau vài phần nghìn thì chỉ là nhiễu của một lần huấn luyện |
| mục 1b: val (out-of-fold) cao hơn test nhiều | stacker vẫn khớp ngày validation; tin cột test và các Δ ghép cặp |
| `ft_gate.md`: **PASS** và `ft_reader` − `ft_base` có CI loại trừ 0 trên test | mở khoá hai lớp cuối giúp thật (so với cùng reader huấn luyện lâu bằng nhưng đóng băng encoder): đáng chạy bản đầy đủ; xem `bge:ft_reader` trong supermodel / stacker |
| `ft_gate.md`: **INCONCLUSIVE** | có hướng tăng nhưng dưới ngưỡng: xem đường cong, nếu còn tăng ở cuối thì huấn luyện lâu hơn (`V11_FT_STEPS`, `V11_FT_MINUTES`) hoặc mở nhiều lớp hơn (`V11_FT_LAYERS=4`) |
| `ft_gate.md`: **FAIL** | không có lợi ích đo được ở quy mô này: dừng hướng fine-tune, giữ reader đóng băng. Nhớ: một seed, chỉ phía ứng viên có gradient, 18 000 bước (≈ nửa epoch) |
| `ft_base` so với `bge:um_frozen` | tác động của việc cắt văn bản ở 96 token và của một seed khác (cùng reader, cùng encoder đóng băng) |

**Giới hạn:** một dataset, một ngày test; stacker học trên **một** ngày validation (fold theo khối thời gian của ngày đó: bài xuất hiện ở cả hai bên của một khối vẫn có thể được học từ khối bên cạnh);
nhiễu huấn luyện của reader và của XGBoost (GPU không tất định bit-exact) không nằm trong CI ghép cặp; lựa chọn cấu hình dựa trên MỘT lần huấn luyện mỗi cấu hình trên holdout tiến (có nhiễu); số công bố (mục 4 của FINAL_REPORT, chỉ MIND-large) lấy từ kết quả tìm kiếm, chưa mở PDF, và đo trên tập test của benchmark chứ không phải `MINDlarge_dev`."""),
              ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                                         "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    for profile, fn in NOTEBOOKS.items():
        nb = build(profile)
        out = os.path.join(HERE, fn)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", out, "cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
