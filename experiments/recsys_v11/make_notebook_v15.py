"""Build the two self-contained Kaggle notebooks of v15 from the module files:

  v15_stacker.ipynb                 MIND-small from a Kaggle input dataset
  v15_mindlarge_hf_stacker.ipynb    the same pipeline on MIND-large downloaded from Hugging Face

Compared with v14 (``make_notebook.py``): a second encoder run (BGE-large: zero-shot scorers only) whose scores join the Qwen3-Embedding ones, the history-free click
prior as one more reader, no LoRA tower (measured: it added nothing at the budget a free GPU allows), and -- the point of this version -- a gradient-boosted RANKER
(``stacker.py``) over all the exported scores plus label-free features, next to the linear supermodel of v14.  Neither trains NRMS / NAML / Fastformer.
Bump the notebook names when the content changes: Kaggle caches notebooks by file name."""
import json
import os

import make_notebook as V14
from make_notebook import code, md, writefile  # re-used cell helpers

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = {"small": "v15_stacker.ipynb", "large": "v15_mindlarge_hf_stacker.ipynb"}
MODULES = ["data.py", "metrics.py", "common.py", "rep_eval.py", "semgraph.py", "adapt.py", "fields.py", "pooling.py", "sid.py", "llm_user.py", "llm_gen.py",
           "refdata.py", "refmodels.py", "usermodel.py", "stages_ext.py", "stages_ref.py", "supermodel.py", "featscore.py", "stacker.py", "final_report.py", "data_lean.py",
           "hf_mind.py", "suite.py"]

# ------------------------------------------------------------------------------------------------ text
WHY = """**Vì sao có bản này (số đo thật của v14 trên MIND-large, `RESULTS_v14_large.md`):** `super_zs` (ghép tuyến tính tốt nhất không popularity) đạt AUC 0,6762 / nDCG@10 0,4277, thấp hơn các mô hình PLM công bố khoảng 0,03 AUC.
Ba điều đã đo: (1) **validation "ấm" hơn test** (82% click của test là bài chưa thấy ở train): reader / RAG hơn `frozen` +0,03…+0,07 nDCG@10 trên validation nhưng chỉ +0,02…−0,007 trên test, nên trọng số **một bộ cho tất cả** khớp trên validation tin reader quá mức;
(2) thêm reader vào ghép tuyến tính làm test **giảm** (−0,0014); (3) encoder lớn hơn **không** phải nguyên nhân chính (MIND-small: bge-large ≈ bge-small). Hướng sửa rút ra từ số đo: để mô hình biết **bài này lạnh/mới/được hiển thị nhiều hay ít** rồi mới quyết định tin thành phần nào."""

STACK_HOW = """**Bộ xếp hạng (stacker) làm thế nào — không nhìn nhãn test để chọn gì:**
1. Mọi stage vẫn đo từng kỹ thuật riêng lẻ (CI ghép cặp so với đối chứng của nó) và giữ **điểm thô** của từng thành phần; có thêm **encoder thứ hai (BGE-large)** chỉ cho các scorer zero-shot, và reader **`prior`** (chỉ nhìn nội dung bài: "loại bài nào hay được click").
2. Với mỗi ứng viên, bảng đặc trưng gồm: điểm của mọi thành phần (z-score trong impression, vài scorer thêm giá trị thô) · đặc trưng lịch sử/slate tính từ vector đóng băng (độ giống click gần nhất, `in_hist`, tỉ lệ category, độ giống các ứng viên khác trong slate…) ·
   **meta**: số lần train đã hiển thị / click bài, CTR của category, và **độ phơi nhiễm trong log impression** (bao lâu kể từ lần đầu bài xuất hiện, đã hiển thị bao nhiêu lần, trong 1 / 6 / 24 giờ qua — *không dùng nhãn*) · (biến thể cuối) **popularity online** của V10.
3. Mô hình: **XGBoost** (GPU nếu có; dự phòng LightGBM, rồi scikit-learn), mục tiêu `rank:ndcg` hoặc `binary:logistic` (chọn trên phần giữ lại của fold 0), huấn luyện **chỉ trên impression validation**, 4 fold chia theo **người dùng**; điểm validation báo cáo là **out-of-fold**;
   điểm test = trung bình (sau z-score trong impression) của 4 mô hình fold, tính **một lần** ở cuối. Test không bao giờ được dùng để chọn mục tiêu, số cây hay biến thể.
4. Ba biến thể: `stack_llm` (điểm + lịch sử/slate) · `stack_meta` (+ meta, **vẫn offline**: dùng được để so với bảng xếp hạng) · `stack_pop` (+ popularity online: khung *streaming*, **không** so được với số offline của bài báo).
   Lưu ý minh bạch: các đặc trưng phơi nhiễm đọc **danh sách ứng viên của chính impression test** (không đọc nhãn). Đó là thông tin có sẵn khi dự đoán, nhưng một hệ thống chỉ biết lịch sử người dùng và nội dung bài thì không có nó; vì vậy `stack_llm` (không dùng chúng) và `stack_meta` được báo riêng.
5. Các kiểm tra tự động (selftest): bảng đặc trưng == tính lại độc lập; chạy lại **toàn bộ** với nhãn test bị xáo → điểm test của `stack_llm`/`stack_meta` **giống hệt**; điểm out-of-fold của một fold không đổi khi chỉ lật nhãn của chính fold đó (đã thử cố ý gây rò: các test bắt được)."""

SMALL_INTRO = f"""# V15 — Bộ xếp hạng học được (GBDT) trên điểm LLM + đặc trưng nhãn-tự-do · MIND-small · Qwen3-Embedding-0.6B + BGE-large (ước ~2,5 giờ trên T4, chưa đo)

Cùng giao thức V10 (validation theo ngày, `MINDsmall_dev` chỉ là test, CI 95% ghép cặp, lát cold/warm) và cùng supermodel tuyến tính của v14, **cộng** bộ xếp hạng ở dưới. Không huấn luyện NRMS / NAML / Fastformer.

{WHY}

{STACK_HOW}

{V14.TECHNIQUES}

**Cách chạy:** (1) Settings → Accelerator **GPU (T4/P100)**, Internet **On**. (2) *Add Input* → dataset MIND có **cả** `MINDsmall_train` và `MINDsmall_dev` (ví dụ `thinhhuynh3108/mindsmall`). (3) **Run All**.
Chạy thử cỡ đồ chơi (vài phút): đổi `"0"` thành `"1"` ở dòng `QUICK = ...` của ô *Cấu hình* (hoặc `V11_QUICK=1`). Công tắc khác: `V11_ENCODER2=` (rỗng) bỏ encoder thứ hai; `V11_CONTROLS=0` bỏ đối chứng "category của bài khác"; `V11_PROTOCOL_CHECK=0` bỏ bước kiểm tra giao thức; `V11_STACK_BACKEND=lightgbm|sklearn|xgboost`.
**Kiểm tra giao thức (1–2 phút):** BGE-small `frozen` phải khớp V10 `bge_zeroshot` (AUC 0.6241 / nDCG@10 0.3909); nếu `MISMATCH` thì dừng, không tin số nào.
**Thời gian:** *ước tính* từ số đo thật v13/v14, chưa đo ở bản này. `BUDGET_H` (mặc định 6) là ngân sách đồng hồ chung; stage nào không còn thời gian thì bị bỏ qua và chạy lại cell sẽ resume.
**Nhận lại:** `SUMMARY_v15.md` (FINAL_REPORT mục 1b là bảng stacker) + log nếu có lỗi."""

LARGE_INTRO = f"""# V15 — Bộ xếp hạng học được (GBDT) trên MIND-large tải từ Hugging Face · Qwen3-Embedding-0.6B + BGE-large (ước ~3–5 giờ trên T4, chưa đo)

Tiếp theo lần chạy thật của v14 (2,9 giờ, mọi stage xong, RAM đỉnh 12,6 GB). Cùng giao thức V10 trên `MINDlarge_dev` (376 471 impression, chạm một lần), cùng supermodel tuyến tính của v14, **cộng** bộ xếp hạng. Không huấn luyện NRMS / NAML / Fastformer, không tower LoRA.

{WHY}

{STACK_HOW}

**Dữ liệu.** `hf_mind.py` tải `MINDlarge_train.zip` và `MINDlarge_dev.zip` từ Hugging Face dataset `Recommenders/MIND` (đã chạy được ở v14: 531 MB + 104 MB trong 12 giây), chỉ giải nén `news.tsv` và `behaviors.tsv`; `MINDlarge_test.zip` (không nhãn) không bao giờ được dùng.
Cần **Internet On**. **Khác bản small (đều ghi vào log và `FINAL_REPORT.md`):** bộ đọc `data_lean.py` (cùng split / popularity / negative với `MindData.from_mind`; thêm đặc trưng phơi nhiễm và đếm theo bài); validation giữ ngẫu nhiên **150 000** impression (v14: 60 000; stacker cần nhiều hàng để huấn luyện);
các stage có huấn luyện dùng **1,2 triệu** mẫu train (v14: 600 000), `--um-epochs 5`; `user_rag` bỏ qua; không ghi `test_arrays.npz`; thư mục làm việc ở `/kaggle/temp` (hoặc `/tmp`).

**Thời gian (ước tính từ số đo thật của v14: E0 Qwen3 1 046 s, text `cta` 1 196 s, controls + pooling + graph + phần còn lại của moredata ≈ 520 s, usermodel 4 574 s cho 19 lần huấn luyện reader):** Qwen3 ≈ 0,8 giờ trước usermodel · BGE-large ≈ 0,5–0,75 giờ · usermodel (≈7 reader × ~7 phút + 2 prior) ≈ 1 giờ ·
supermodel ≈ 5 phút · **stacker** ≈ 20–60 phút trên GPU (xây bảng ~5,5 triệu hàng × ~120 đặc trưng ≈ 2,6 GB RAM; ~14 lần huấn luyện XGBoost; trước đó nó giải phóng mẫu train và bộ nhớ RAG). Nếu không có GPU cho XGBoost, nó tự rơi về CPU và **giới hạn 3 triệu hàng huấn luyện** (chậm hơn, vẫn chạy được). Tổng ≈ 3–5 giờ (v14 đo 2,9 giờ cho phần tương đương; ước tính này chưa đo); `BUDGET_H=10` là trần.
**Cách chạy:** Settings → GPU, Internet **On** → **Run All**. Chạy thử cỡ đồ chơi: `QUICK = ...` → `"1"` (vẫn tải MIND-large và mã hoá cả catalogue). Hết RAM/thời gian: giảm `V11_TRAIN_SUB`, `V11_VAL_MAX`, hoặc `V11_ENCODER2=` (bỏ BGE-large).
**Nhận lại:** `SUMMARY_v15.md` (mục 1b của FINAL_REPORT là bảng stacker, mục 4 là số công bố — thứ cấp) + log. Bản sao ở `/kaggle/working/results_large`."""

CONFIG_HEAD = """import os, glob, sys, gc, time, json
T_NB = time.time()                                                    # giờ bắt đầu của notebook: ngân sách BUDGET_H tính từ đây cho CẢ hai run
QUICK = bool(int(os.environ.get("V11_QUICK", "0")))                  # True: cỡ đồ chơi để bắt lỗi môi trường
BUDGET_H = float(os.environ.get("V11_BUDGET_H", "@BUDGET@"))          # ngân sách đồng hồ chung (giờ)@BUDGET_NOTE@
ENCODER = os.environ.get("V11_ENCODER", "Qwen/Qwen3-Embedding-0.6B")  # encoder chính: mọi kỹ thuật, reader, RAG
ENCODER2 = os.environ.get("V11_ENCODER2", "BAAI/bge-large-en-v1.5")   # encoder thứ hai (chỉ scorer zero-shot); '' = bỏ
CONTROLS = bool(int(os.environ.get("V11_CONTROLS", "@CONTROLS@")))            # 1: thêm đối chứng "category của bài khác" (txt_ctr*): mã hoá thêm một lần cả catalogue
TEXT_MODES = os.environ.get("V11_TEXT_MODES", "cta,ctr" if CONTROLS else "cta")
PROTOCOL_CHECK = bool(int(os.environ.get("V11_PROTOCOL_CHECK", "@CHECK@")))   # BGE-small frozen phải khớp V10 (1-2 phút)@CHECK_NOTE@
CHECK_ENCODER = os.environ.get("V11_CHECK_ENCODER", "BAAI/bge-small-en-v1.5")
DEVICE = os.environ.get("V11_DEVICE", "auto")
USE_REFERENCE = bool(int(os.environ.get("V11_REFERENCE", "1")))       # so HFEncoder với Sentence-Transformers trên 64 bài trước khi chạy
STACK_FOLDS = int(os.environ.get("V11_STACK_FOLDS", "3" if QUICK else "4"))
STACK_BACKEND = os.environ.get("V11_STACK_BACKEND", "auto")           # auto: xgboost (GPU nếu có) -> lightgbm -> sklearn
SAVE_SCORES = int(os.environ.get("V11_SAVE_SCORES", "0"))            # 1: ghi stack_scores.npz (điểm out-of-fold và test của từng biến thể, ~100 MB trên MIND-large)
"""

CONFIG_SMALL = """MIND_TRAIN = os.environ.get("V11_MIND_TRAIN")                      # để None -> tự dò trong /kaggle/input
MIND_DEV = os.environ.get("V11_MIND_DEV")
WORK = ("/kaggle/working/v15_out" if os.path.isdir("/kaggle/working") else "v15_out") + ("_quick" if QUICK else "")   # QUICK không bao giờ lẫn vào bản đầy đủ
CACHE = "/kaggle/temp/v15_cache" if os.path.isdir("/kaggle/temp") else os.path.join(WORK, "cache")
RESULTS = WORK
"""

CONFIG_LARGE = """HF_REPO = os.environ.get("V11_HF_REPO", "Recommenders/MIND")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("V11_HF_TOKEN") or None
if HF_TOKEN is None:                                                  # only needed if the Hub repository turns out to be gated: add a Kaggle secret named HF_TOKEN
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN") or None
    except Exception:
        HF_TOKEN = None
VAL_MAX = int(os.environ.get("V11_VAL_MAX", "5000" if QUICK else "150000"))     # validation giữ ngẫu nhiên từng này impression (0 = tất cả)
TEST_MAX = int(os.environ.get("V11_TEST_MAX", "20000" if QUICK else "0"))       # 0 = toàn bộ MINDlarge_dev (376k impression)
TRAIN_SUB = int(os.environ.get("V11_TRAIN_SUB", "50000" if QUICK else "1200000"))  # mẫu train cho các stage có huấn luyện (0 = tất cả)
UM_EPOCHS = int(os.environ.get("V11_UM_EPOCHS", "2" if QUICK else "5"))
import shutil
def pick_base(need_gb=15):
    \"\"\"Thư mục làm việc: ưu tiên /kaggle/temp, rồi /tmp, rồi /kaggle/working -- cái đầu tiên còn đủ chỗ.\"\"\"
    cands = [d for d in ("/kaggle/temp", "/tmp", "/kaggle/working") if os.path.isdir(d)]
    for d in cands:
        if shutil.disk_usage(d).free / 1e9 >= need_gb:
            return d
    return cands[0] if cands else "."
BASE = os.environ.get("V11_BASE") or pick_base()
WORK = os.path.join(BASE, "v15_large_out") + ("_quick" if QUICK else "")
RESULTS = ("/kaggle/working/results_large" if os.path.isdir("/kaggle/working") else "results_large") + ("_quick" if QUICK else "")
CACHE = os.path.join(BASE, "v15_large_cache")
"""

CONFIG_TAIL = """sys.path.insert(0, os.getcwd())
print(dict(QUICK=QUICK, BUDGET_H=BUDGET_H, ENCODER=ENCODER, ENCODER2=ENCODER2, TEXT_MODES=TEXT_MODES, PROTOCOL_CHECK=PROTOCOL_CHECK, STACK_BACKEND=STACK_BACKEND, WORK=WORK))
"""

DEPS = '''%pip install -q "transformers>=4.51" peft sentence-transformers
import importlib, subprocess, sys
for pkg in ("xgboost", "lightgbm"):                      # có sẵn trên ảnh Kaggle; cài nếu thiếu (stacker rơi về scikit-learn nếu cả hai đều không dùng được)
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True).stdout[-300:])
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout or "no GPU visible")
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

COMMON = ["--mind-train", MIND_TRAIN, "--mind-dev", MIND_DEV, "--cache-dir", CACHE, "--device", DEVICE]
def budget_args():
    \"\"\"Mỗi suite.main khởi động lại đồng hồ của nó: cho mỗi run phần ngân sách CÒN LẠI của notebook (BUDGET_H tính từ ô cấu hình).\"\"\"
    return ["--budget-hours", "%.3f" % max(BUDGET_H - (time.time() - T_NB) / 3600, 0.25)]
@SCALE@
TAG, TAG2 = "qwen3", "bge"
# run B: the second encoder -- zero-shot scorers only; its exported scores join the stacker and the supermodel
ARGS2 = [*COMMON, *SCALE, "--encoder", ENCODER2, "--work", WORK + "/" + TAG2, "--max-len", "256", "--text-modes", "cta", "--hist-lens", "100", "--km-ks", "5", "--seeds", "1"]
STAGES2 = "controls,pooling,moredata"
# run A: the main encoder -- every technique, then the linear supermodel and the stacker
ARGS = [*COMMON, *SCALE, "--encoder", ENCODER, "--work", WORK + "/" + TAG, "--max-len", "256", "--q-max-len", "384", "--hist-k", "10", "--seeds", "1",
        "--text-modes", TEXT_MODES, "--hist-lens", "100", "--km-ks", "5", "--hq-subset", "1",
        "--um-views", "frozen,text,knn_rag", "--um-seeds", "2", "--um-seeds-focus", "2", "--um-extra", "prior:frozen,prior:text",
        "--super-dirs", (WORK + "/" + TAG2) if ENCODER2 else "", "--stack-folds", str(STACK_FOLDS), "--stack-backend", STACK_BACKEND, "--stack-save-scores", str(SAVE_SCORES),
        "--stack-free-memory", "1"]
STAGES = "controls,pooling,moredata,graph,usermodel,supermodel,stack"       # rẻ và quan trọng trước; supermodel và stacker dùng những gì đã có
if QUICK:                                              # cỡ đồ chơi: mọi họ stage chạy một lần
    ARGS += ["--pairs", "20000", "--betas", "0.5", "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5", "--um-epochs", "2", "--um-seeds", "1", "--um-seeds-focus", "1",
             "--km-ks", "2", "--stack-rounds", "60"]
"""

SCALE_LARGE = """SCALE = ["--lean-data", "1", "--val-max", str(VAL_MAX), "--test-max", str(TEST_MAX), "--train-sub", str(TRAIN_SUB), "--save-arrays", "0",
         "--skip-user-rag", "1", "--um-epochs", str(UM_EPOCHS)]"""
SCALE_SMALL = """SCALE = ["--lean-data", "1"]                           # cùng split / popularity / negative với bộ đọc V10 (đã kiểm), và cần cho đặc trưng phơi nhiễm của stacker"""


def cell_config(profile):
    large = profile == "large"
    head = (CONFIG_HEAD.replace("@BUDGET@", "10" if large else "6").replace("@BUDGET_NOTE@", "; Kaggle giết notebook ở 12 giờ" if large else "")
            .replace("@CHECK@", "0" if large else "1").replace("@CONTROLS@", "0" if large else "1").replace("@CHECK_NOTE@", "; MIND-large không có số V10" if large else ""))
    return code(head + (CONFIG_LARGE if large else CONFIG_SMALL) + CONFIG_TAIL)


def cell_runs(profile):
    return code(RUNS_TEMPLATE.replace("@SCALE@", SCALE_LARGE if profile == "large" else SCALE_SMALL))


PAIRS = '''Q = TAG
pairs = [# --- the stacker against what it is made of, and against the linear supermodel
         ((Q, "stack_llm"), (Q, "super_zs")), ((Q, "stack_llm"), (Q, "super_llm")), ((Q, "stack_meta"), (Q, "super_llm")), ((Q, "stack_meta"), (Q, "stack_llm")),
         ((Q, "stack_llm"), (Q, "txt_cta_lse")), ((Q, "stack_llm"), (Q, "um_knn_rag")), ((Q, "stack_llm"), (Q, "frozen")),
         ((Q, "stack_pop"), (Q, "stack_meta")), ((Q, "stack_pop"), (Q, "super_llm+pop")), ((Q, "stack_pop"), (Q, "frozen+pop")),
         # --- the linear supermodel (as in v14)
         ((Q, "super_zs"), (Q, "super_base")), ((Q, "super_llm"), (Q, "super_zs")), ((Q, "super_llm"), (Q, "txt_cta_lse")), ((Q, "super_llm+pop"), (Q, "super_llm")),
         # --- single techniques, each against its own control
         ((Q, "pool_lse"), (Q, "frozen")), ((Q, "txt_cta_lse"), (Q, "pool_lse")), ((Q, "txt_cta_lse"), (Q, "txt_ctr_lse")), ((Q, "hist100_lse"), (Q, "pool_lse")),
         ((Q, "um_text"), (Q, "um_frozen")), ((Q, "um_knn_rag"), (Q, "um_frozen")), ((Q, "um_prior_frozen"), (Q, "frozen"))]
direct = suite.cross_compare([S], [Q], pairs, WORK + "/direct_comparisons.md")
'''

HEADLINE = '''def best_by_val(names):
    ok = [n for n in names if n in S.cache and S.row(n)["n_test"] == len(S.data.test)]
    return max(ok, key=lambda n: S.row(n)["val_ndcg@10"]) if ok else None   # chosen on validation only
lines_ = ["# Headline (test = %s impressions; each family's representative is chosen on VALIDATION)" % f"{len(S.data.test):,}", "",
          "| row | family | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC | Δ nDCG@10 vs the supermodel's best offline row [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
sup = best_by_val([r["name"] for r in (S.kv.get("supermodel") or {}).get("rungs", []) if not r["pop"] and "apex" not in r["name"]])
sup_pop = best_by_val([r["name"] for r in (S.kv.get("supermodel") or {}).get("rungs", []) if r["pop"] and "apex" not in r["name"]])
for n, ref in (("frozen", None), (sup, None), (best_by_val(["stack_llm"]), sup), (best_by_val(["stack_meta"]), sup), ("frozen+pop", None), (sup_pop, None), (best_by_val(["stack_pop"]), sup_pop)):
    if not n:
        continue
    r = S.row(n)
    d = suite.R.compare(S.cache[n]["test"], S.cache[ref]["test"], "ndcg@10") if ref else None
    lines_.append(f"| `{n}` | {r['family']} | {r['auc']:.4f} | {r['mrr']:.4f} | {r['ndcg@5']:.4f} | {r['ndcg@10']:.4f} | {r.get('cold_auc', float('nan')):.4f} | {r.get('warm_auc', float('nan')):.4f} | "
                  + (f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" if d else "-") + " |")
lines_ += ["", "Offline rows (`stack_llm`, `stack_meta`, `super_*`) are comparable with offline papers up to the dev/test difference; `+pop` / `stack_pop` count the clicks of earlier impressions of the test day (streaming) and are not.",
           "`stack_meta` also reads the exposure of each candidate in the impression log (label-free); `stack_llm` does not."]
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
              md("### 3. Dữ liệu — " + ("MIND-large tải từ Hugging Face (cần Internet On)" if large else "tìm MIND trong /kaggle/input (cần cặp train + dev của CÙNG một dataset)")), V14.cell_data(profile),
              md("### 4. Cấu hình hai run (encoder thứ hai, rồi encoder chính)"), cell_runs(profile)]
    if not large:
        cells += [md("### 5. Kiểm tra giao thức (1–2 phút, không phải so sánh): BGE-small `frozen` phải khớp V10"),
                  code('''S_CHECK = None
if PROTOCOL_CHECK:
    try:
        suite.EQUIV_REFERENCE = None
        S_CHECK = suite.main([*COMMON, *budget_args(), "--encoder", CHECK_ENCODER, "--work", WORK + "/bge_check", "--max-len", "256", "--stages", "controls",
                              "--repro", "0.6241,0.3909", "--repro-pop", "0.6533,0.4026,0.6769,0.4330"])
    except Exception as e:                       # the check must never stop the main run
        import traceback; traceback.print_exc()
        print("!! protocol check failed to run:", repr(e)[:200])
    gc.collect(); torch.cuda.empty_cache()
''')]
    else:
        cells += [code("S_CHECK = None   # MIND-large: V10 has no reference numbers\n")]
    n = 6 if not large else 5
    cells += [md(f"### {n}. Encoder thứ hai (BGE-large): controls → pooling → moredata (chỉ điểm zero-shot, để stacker và supermodel dùng)"),
              code('''S2 = None
if ENCODER2:
    suite.EQUIV_REFERENCE = reference_encoder(ENCODER2, 256)
    try:
        S2 = suite.main([*ARGS2, *budget_args(), "--stages", STAGES2])
    except Exception as e:                       # a failure here must not cost the main run: it only loses the second encoder's scores
        import traceback; traceback.print_exc()
        print("!! second-encoder run failed:", repr(e)[:200], "-> the main run goes on without it")
        ARGS[ARGS.index("--super-dirs") + 1] = ""
    gc.collect(); torch.cuda.empty_cache()
'''),
              md(f"### {n + 1}. Encoder chính (Qwen3-Embedding): controls → pooling → moredata → graph → usermodel (+ reader `prior`) → **supermodel** → **stack**"),
              code('''suite.EQUIV_REFERENCE = reference_encoder(ENCODER, 256)
S = suite.main([*ARGS, *budget_args(), "--stages", STAGES])
gc.collect(); torch.cuda.empty_cache()
'''),
              md(f"### {n + 2}. Leaderboard, so sánh trực tiếp từng cặp, forest plot"),
              code('''suites, tags = ([S2, S], [TAG2, TAG]) if S2 is not None else ([S], [TAG])
rows = suite.leaderboard(suites, tags, (TAG, "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
rows_pop = suite.leaderboard(suites, tags, (TAG, "frozen+pop"), WORK + "/leaderboard_pop.md", plot=WORK + "/forest_pop.png", pop=True)
''' + PAIRS),
              md(f"### {n + 3}. Kết quả chính, báo cáo cuối và file gộp để gửi lại"),
              code(HEADLINE + '''print(final_report.build(S, WORK + "/FINAL_REPORT.md", WORK + "/final_tables.json", extra_suites=[S_CHECK] if S_CHECK is not None else ()))
p = os.path.join(WORK, TAG, "results.md")
if os.path.exists(p):
    print("=" * 30, TAG, "=" * 30)
    print(open(p, encoding="utf-8").read())
noise = ["# Seed-to-seed spread of the trained rows (test nDCG@10 of each seed; sample std with n-1; the paired CIs in the other tables do NOT include it)", "",
         "| row | seeds | nDCG@10 per seed | std |", "|---|---|---|---|"]
pj = os.path.join(WORK, TAG, "results.json")
if os.path.exists(pj):
    for r in json.load(open(pj, encoding="utf-8")):
        if r.get("ndcg10_per_seed") and len(r["ndcg10_per_seed"]) > 1:
            noise.append(f"| {r['name']} | {len(r['ndcg10_per_seed'])} | {', '.join(f'{x:.4f}' for x in r['ndcg10_per_seed'])} | {r.get('ndcg10_std', float('nan')):.4f} |")
open(os.path.join(WORK, "noise.md"), "w", encoding="utf-8").write("\\n".join(noise) + "\\n")
print("\\n".join(noise))
print("tổng thời gian của notebook: %.2f h" % ((time.time() - T_NB) / 3600))
parts = [("HEADLINE.md", WORK), ("FINAL_REPORT.md", WORK), ("leaderboard.md", WORK), ("leaderboard_pop.md", WORK), ("direct_comparisons.md", WORK), ("noise.md", WORK),
         ("timings.json", WORK + "/" + TAG), ("results.md", WORK + "/" + TAG)] + ([("timings.json", WORK + "/" + TAG2)] if S2 is not None else [])
os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "SUMMARY_v15.md"), "w", encoding="utf-8") as out:
    for fn, d in parts:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out.write(f"\\n\\n## {fn} ({os.path.basename(d)})\\n\\n" + open(p, encoding="utf-8").read())
import shutil
for fn in ("forest.png", "forest_pop.png", "FINAL_REPORT.md", "final_tables.json", "HEADLINE.md", "leaderboard.md", "leaderboard_pop.md", "direct_comparisons.md", "noise.md"):
    if os.path.exists(os.path.join(WORK, fn)) and os.path.abspath(WORK) != os.path.abspath(RESULTS):
        shutil.copy(os.path.join(WORK, fn), os.path.join(RESULTS, fn))
print("-> gửi lại file:", os.path.join(RESULTS, "SUMMARY_v15.md"), f"({os.path.getsize(os.path.join(RESULTS, 'SUMMARY_v15.md')) / 1024:.0f} KB)")
'''),
              md("""### Cách đọc kết quả

**Bước 0 — kiểm tra bộ đo:** (bản small) dòng `frozen == V10 bge_zeroshot` trong mục 0 của `FINAL_REPORT.md` phải là MATCH. Nếu không, dừng lại.

| Bạn thấy | Nghĩa là |
|---|---|
| `stack_llm` > `super_llm` (hoặc `super_zs`) với CI ghép cặp loại trừ 0 | bộ xếp hạng học được **hơn** ghép tuyến tính trên cùng các thành phần (cả hai chỉ dùng nội dung + lịch sử) |
| `stack_meta` > `stack_llm` | độ phơi nhiễm / đếm theo bài (vẫn offline) có thêm thông tin; nhớ rằng đặc trưng phơi nhiễm đọc danh sách ứng viên của chính impression test |
| `stack_pop` > `stack_meta` | popularity online (streaming) thêm giá trị; không so được với số offline của bài báo |
| cold AUC của stacker tăng so với `super_*` | nó dùng được tín hiệu "bài lạnh/mới" (82% click test trên MIND-large là bài lạnh) |
| val (out-of-fold) cao hơn test nhiều | validation ấm hơn test (đã đo ở v14); tin cột test và các Δ ghép cặp |
| mục 1b: nhóm `scores` chiếm gain lớn nhất | stacker chủ yếu tin các điểm LLM; nhóm `meta` lớn ⇒ nó dựa nhiều vào độ phơi nhiễm / đếm theo bài |

**Giới hạn:** một dataset, một ngày test; stacker học trên **một** ngày validation (4 fold chia theo người dùng, nên không rò người dùng nhưng vẫn là cùng một ngày: hiệu ứng riêng của ngày đó có thể được học);
nhiễu huấn luyện của reader và của XGBoost (GPU không tất định bit-exact) không nằm trong CI ghép cặp; số công bố (mục 4 của FINAL_REPORT, chỉ MIND-large) lấy từ kết quả tìm kiếm, chưa mở PDF, và đo trên tập test của benchmark chứ không phải `MINDlarge_dev`."""),
              ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                                         "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}


def main():
    for profile, fn in NOTEBOOKS.items():
        nb = build(profile)
        out = os.path.join(HERE, fn)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", out, "cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
