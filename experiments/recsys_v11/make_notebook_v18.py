"""Build the two MIND-large notebooks of v18 from the module files.  Each fits one Kaggle GPU quota (machine "GPU T4 x2"; the encoders use both GPUs, the
rest one):

  v18a_mindlarge_q4b.ipynb           ~8.5-10 h (BUDGET_H 11.2): a larger LLM encoder, Qwen3-Embedding-4B, with an IN-RUN CONTROL.  The v16b pipeline runs first
                                     (BGE-large, Qwen3 + BGE-large, Qwen3-0.6B with supermodel + stacker = the control); then a run on the 4B vectors and a run on
                                     the 4B + BGE-large vectors; then the supermodel and the stacker run again with those components (the treatment).  Same
                                     impressions, so every fused row has a paired CI against its control, and pre-registered gates decide.
  v18b_mindlarge_strongreader.ipynb  ~5.5 h (BUDGET_H 6.0): the v16b pipeline with the readers of the Qwen3 + BGE-large run trained harder (all 1.8M training
                                     samples instead of 1.2M, up to 6 epochs, learning rate halved after every epoch from the second, finer early stopping) and a new
                                     reader 'sca' (a self-attention block over the history before the candidate-aware attention, the NRMS idea inside our reader).
                                     The fused rows are compared with v16b (another run, same test split); sca vs ca is paired inside the run.

Why these two (measured, RESULTS_v16b_large.md / v17): the readers were still improving when they stopped (several seeds at the 5-epoch cap; v17's frozen-encoder
control trained longer at a lower rate reached 0.4208 vs 0.4152 for the same reader), and the best component is the reader on two encoders side by side
(Qwen3-0.6B + BGE-large 0.4223 > BGE-large 0.4152 > Qwen3-0.6B 0.4050).  Content gains moved the linear fusion (+0.0105) much more than the stacker (+0.0008),
so v18a reports both levels and compares the treatment with a control of the same notebook.
Bump the notebook names when the content changes: Kaggle caches notebooks by file name."""
import json
import os

import make_notebook_v17 as V17
from make_notebook import code, md, writefile  # re-used cell helpers

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = {"a": "v18a_mindlarge_q4b.ipynb", "b": "v18b_mindlarge_strongreader.ipynb"}
MODULES = V17.MODULES[:-1] + ["report_v18.py", "suite.py"]
assert V17.MODULES[-1] == "suite.py"

# ------------------------------------------------------------------------------------------------ text
BEST = """**Kết quả tốt nhất hiện tại (v16b, `RESULTS_v16b_large.md`, test = `MINDlarge_dev` 376 471 impression):** offline `stack_meta` **0,4516** nDCG@10 (hàng chính đăng ký trước) và `stack_meta_v15cfg` **0,4543** (đối chứng công thức v15 — cao nhất trong các hàng offline); online `stack_pop` **0,4731**; chỉ nội dung + lịch sử `super_llm` **0,4373**. Reader tốt nhất: vector nối Qwen3-0,6B ⊕ BGE-large `cat:um_frozen` 0,4223 / `cat:um_knn_rag` 0,4226."""

A_TEXT = """**V18a — encoder LLM lớn hơn (Qwen3-Embedding-4B) có đối chứng ngay trong notebook.**
*Câu hỏi:* vector của một encoder LLM 4B (đọc "Category. Subcategory. title. abstract", cách đã giúp Qwen3 ở hướng F) có mang thêm thông tin cho reader, cho ghép tuyến tính và cho stacker không?
*Thiết kế:* (1) chạy lại đúng pipeline v16b — BGE-large, vector nối Qwen3-0,6B ⊕ BGE-large, Qwen3-0,6B với supermodel + stacker — đây là **đối chứng** (cũng cho biết nhiễu giữa hai lần chạy so với số v16b); (2) run `q4b`: Qwen3-Embedding-4B (trọng số fp16, embedding và norm giữ fp32 — cùng phép tính autocast như các encoder khác, một nửa bộ nhớ; mã hoá chia cho 2 GPU T4), các scorer zero-shot, kNN-RAG và reader 128 chiều; (3) run `cat4`: vector nối 4B ⊕ BGE-large (cùng văn bản "cta", lấy từ cache); (4) **chạy lại supermodel và stacker** của run chính với cả các thành phần mới — đây là **nhánh thí nghiệm**. Hai nhánh đọc cùng validation / test nên mọi hàng ghép có CI ghép cặp.
*Cổng (viết trước khi chạy, `gate_v18a.md`):* G1 nội dung `super_llm`, G2 offline `stack_meta`, G3 online `stack_pop`: **PASS** nếu Δ (thí nghiệm − đối chứng) có CI 95% ghép cặp nằm trên 0. "Vượt kết quả tốt nhất" chỉ được nói khi gate PASS **và** số của nhánh thí nghiệm cao hơn số v16b tương ứng (0,4373 / 0,4543 / 0,4731).
*Thời gian (ước tính, dựa trên số đo v16b):* đối chứng ≈ 4,3 giờ; mã hoá 104 k bài bằng 4B ≈ 1–2 giờ trên 2 GPU (chưa đo); reader + run nối ≈ 1,3 giờ; supermodel + stacker lại ≈ 1,2 giờ. Tổng ≈ 8,5–10 giờ, `BUDGET_H` = 11,2 (Kaggle dừng ở 12 giờ). Nhánh 4B chỉ bắt đầu khi còn ≥ 3,2 giờ, run nối 4B khi còn ≥ 2,2 giờ, nhánh thí nghiệm khi còn ≥ 1,4 giờ; stacker của nó chỉ bắt đầu khi còn ≥ 70 phút. Hết giờ thì phần đối chứng vẫn có đủ."""

B_TEXT = """**V18b — reader được huấn luyện kỹ hơn + reader mới `sca`.**
*Câu hỏi:* reader của v16b có đang bị huấn luyện thiếu không, và thêm một lớp self-attention kiểu NRMS trước attention theo ứng viên có giúp không?
*Bằng chứng:* nhiều seed của v16b dừng ở trần 5 epoch khi validation còn tăng; ở v17, nhánh đối chứng đóng băng encoder (`ft_base`, cùng reader, huấn luyện tiếp với learning rate thấp hơn) đạt 0,4208 so với 0,4152 của reader thường.
*Thay đổi (chỉ ở run vector nối Qwen3-0,6B ⊕ BGE-large, thành phần quan trọng nhất của v16b):* toàn bộ 1,8 triệu mẫu train thay vì 1,2 triệu; tối đa 6 epoch; learning rate nhân 0,5 sau mỗi epoch kể từ epoch thứ 2; ngưỡng cải thiện của early stopping 0,0002 thay vì 0,001; thêm reader `sca` trên view kNN-RAG (2 seed). `sca` = reader theo ứng viên của v16b, thêm **một khối self-attention đa đầu có residual trên lịch sử** trước attention theo ứng viên; phép chiếu ra của khối khởi tạo bằng 0 nên lúc đầu `sca` chính là reader cũ.
*So sánh:* `sca` với reader thường trên cùng view, cùng công thức huấn luyện: **ghép cặp, trong run**. Reader thường và các hàng ghép (supermodel, stacker) so với **số v16b** (run khác, cùng tập test): không có CI ghép cặp; chênh dưới ~0,003 (hàng ghép) hoặc ~0,005 (reader, 2 seed) nằm trong nhiễu giữa hai lần chạy.
*Thời gian (ước tính):* v16b 4,49 giờ + phần huấn luyện thêm ≈ 0,8–1 giờ → ≈ 5,3–5,6 giờ; `BUDGET_H` = 6,0 (quota 7 giờ); stacker chỉ bắt đầu khi còn ≥ 65 phút."""

HOW_RUN = """**Cách chạy:** (Kaggle) Settings → Accelerator **GPU T4 x2**, Internet **On** → **Save Version / Run All**. MIND-large tải từ Hugging Face (`Recommenders/MIND`), các encoder tải từ Hugging Face Hub. Không cần input dataset.
**Kiểm tra môi trường trước (vài phút):** đặt `V11_QUICK=1` và encoder nhỏ (`V11_ENCODER=BAAI/bge-small-en-v1.5`, `V11_ENCODER2=BAAI/bge-small-en-v1.5`@QUICK_EXTRA@) — mọi stage chạy một lần ở cỡ đồ chơi.
**Nhận lại:** `SUMMARY_v18@P@.md` trong `/kaggle/working/results_v18@P@` (bảng chính, cổng, so sánh trực tiếp, thời gian, RAM) + log của notebook."""


def intro(p):
    head = {"a": "# V18a — Qwen3-Embedding-4B (đọc category + tiêu đề + tóm tắt) có đối chứng trong notebook · MIND-large · Kaggle GPU T4 x2 (ước 8,5–10 giờ)",
            "b": "# V18b — reader huấn luyện kỹ hơn + reader `sca` (self-attention trên lịch sử) · MIND-large · Kaggle GPU T4 x2 (ước 5,3–5,8 giờ)"}[p]
    quick_extra = ", `V11_ENCODER4=BAAI/bge-small-en-v1.5`" if p == "a" else ""
    return "\n\n".join([head, BEST, A_TEXT if p == "a" else B_TEXT, HOW_RUN.replace("@P@", p).replace("@QUICK_EXTRA@", quick_extra),
                        "Giao thức như v16b: validation theo ngày (giữ ngẫu nhiên 150 000 impression), `MINDlarge_dev` chỉ là test và chỉ chạm một lần cho mỗi hàng, "
                        "CI 95% ghép cặp theo impression (chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện), lát cold / warm. Không dùng nhãn test để chọn gì."])


CONFIG = """import os, glob, sys, gc, time, json, shutil
T_NB = time.time()                                                    # giờ bắt đầu: ngân sách BUDGET_H tính từ đây cho mọi run của notebook
IN_KAGGLE = os.path.isdir("/kaggle/working")
QUICK = bool(int(os.environ.get("V11_QUICK", "0")))                  # True: cỡ đồ chơi để bắt lỗi môi trường
BUDGET_H = float(os.environ.get("V11_BUDGET_H", "@BUDGET@"))          # ngân sách đồng hồ chung (giờ); Kaggle dừng phiên GPU ở 12 giờ
ENCODER = os.environ.get("V11_ENCODER", "Qwen/Qwen3-Embedding-0.6B")  # encoder chính: run có supermodel + stacker
ENCODER2 = os.environ.get("V11_ENCODER2", "BAAI/bge-large-en-v1.5")   # encoder thứ hai
@PROFILE@
DEVICE = os.environ.get("V11_DEVICE", "auto")
USE_REFERENCE = bool(int(os.environ.get("V11_REFERENCE", "1")))       # so HFEncoder với Sentence-Transformers trên 64 bài trước khi chạy
STACK_FOLDS = int(os.environ.get("V11_STACK_FOLDS", "3" if QUICK else "4"))
STACK_BACKEND = os.environ.get("V11_STACK_BACKEND", "auto")           # auto: xgboost (GPU nếu có) -> lightgbm -> sklearn
SAVE_SCORES = int(os.environ.get("V11_SAVE_SCORES", "1"))
STACK_CV = os.environ.get("V11_STACK_CV", "time")
STACK_CONFIGS = os.environ.get("V11_STACK_CONFIGS", "v15,clean,clean_d5,clean_d3,noprior_d5")
STACK_BLEND = int(os.environ.get("V11_STACK_BLEND", "2"))
COLD_TARGET = float(os.environ.get("V11_COLD_TARGET", "0.8"))
STACK_MIN_S = int(os.environ.get("V11_STACK_MIN_S", "300" if QUICK else "@STACKMIN@"))   # stacker của run cuối chỉ bắt đầu khi còn từng này giây (v16b: 3 311 s)
sys.path.insert(0, os.getcwd())
import common                                                        # đã được ghi ra ở bước 1
RAM_GB = common.ram_total_gb()
RAM_TIER = "high" if (RAM_GB == 0 or RAM_GB >= 24) else ("mid" if RAM_GB >= 16 else "low")
HF_REPO = os.environ.get("V11_HF_REPO", "Recommenders/MIND")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("V11_HF_TOKEN") or None
if HF_TOKEN is None:                                                  # chỉ cần nếu repo trên Hub đòi đăng nhập: Kaggle secret tên HF_TOKEN
    try:
        HF_TOKEN = __import__("kaggle_secrets").UserSecretsClient().get_secret("HF_TOKEN") or None
    except Exception:
        HF_TOKEN = None
def pick_base(need_gb=40):
    \"\"\"Thư mục cục bộ còn đủ chỗ cho dữ liệu, cache vector và mô hình: /kaggle/temp, /tmp, /kaggle/working.\"\"\"
    cands = [d for d in ("/kaggle/temp", "/tmp", "/content", "/kaggle/working") if os.path.isdir(d)]
    for d in cands:
        if shutil.disk_usage(d).free / 1e9 >= need_gb:
            return d
    return max(cands, key=lambda d: shutil.disk_usage(d).free) if cands else "."
BASE = os.environ.get("V11_BASE") or pick_base()
VAL_MAX = int(os.environ.get("V11_VAL_MAX", "5000" if QUICK else {"high": "150000", "mid": "100000", "low": "50000"}[RAM_TIER]))
TEST_MAX = int(os.environ.get("V11_TEST_MAX", "20000" if QUICK else "0"))
TRAIN_SUB = int(os.environ.get("V11_TRAIN_SUB", "50000" if QUICK else {"high": "1200000", "mid": "800000", "low": "400000"}[RAM_TIER]))
UM_EPOCHS = int(os.environ.get("V11_UM_EPOCHS", "2" if QUICK else "5"))
WORK = os.path.join(BASE, "v18@P@_out") + ("_quick" if QUICK else "")
CACHE = os.path.join(BASE, "v18_cache")
RESULTS = ("/kaggle/working/results_v18@P@" if IN_KAGGLE else "results_v18@P@") + ("_quick" if QUICK else "")
if RAM_TIER != "high":
    print("!! RAM dưới 24 GB: quy mô được giảm tự động (validation / mẫu train nhỏ hơn v16b), số KHÔNG so thẳng được với v16b")
print(f"môi trường: {'Kaggle' if IN_KAGGLE else 'khác'} | RAM {RAM_GB:.0f} GB ({RAM_TIER}) | thư mục làm việc {BASE} (trống {shutil.disk_usage(BASE).free / 1e9:.0f} GB)")
print(dict(QUICK=QUICK, BUDGET_H=BUDGET_H, ENCODER=ENCODER, ENCODER2=ENCODER2, VAL_MAX=VAL_MAX, TRAIN_SUB=TRAIN_SUB, UM_EPOCHS=UM_EPOCHS, WORK=WORK@PRINT@))
"""

PROFILE_A = """ENCODER4 = os.environ.get("V11_ENCODER4", "Qwen/Qwen3-Embedding-4B")  # encoder lớn của nhánh thí nghiệm ('' = chỉ chạy đối chứng)
Q4B_MIN_H = float(os.environ.get("V11_Q4B_MIN_H", "0" if QUICK else "3.2"))     # giờ còn lại tối thiểu để bắt đầu nhánh 4B
CAT4_MIN_H = float(os.environ.get("V11_CAT4_MIN_H", "0" if QUICK else "2.2"))   # ... để bắt đầu run vector nối 4B + BGE-large
TREAT_MIN_H = float(os.environ.get("V11_TREAT_MIN_H", "0" if QUICK else "1.4"))   # ... để chạy lại supermodel + stacker (nhánh thí nghiệm)"""
PRINT_A = ", ENCODER4=ENCODER4"

PROFILE_B = """PLUS_EPOCHS = os.environ.get("V11_PLUS_EPOCHS", "2" if QUICK else "6")          # reader của run vector nối: tối đa từng này epoch
PLUS_DECAY = os.environ.get("V11_PLUS_DECAY", "0.5")                            # learning rate nhân hệ số này sau mỗi epoch kể từ epoch 2
PLUS_EXTRA = os.environ.get("V11_PLUS_EXTRA", "prior:frozen,sca:knn_rag")       # reader thêm ở run vector nối (sca = theo ứng viên + self-attention trên lịch sử)"""
PRINT_B = ", PLUS_EPOCHS=PLUS_EPOCHS, PLUS_DECAY=PLUS_DECAY, PLUS_EXTRA=PLUS_EXTRA"

DATA = '''import hf_mind
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM: {vm.total / 1e9:.0f} GB total, {vm.available / 1e9:.0f} GB free; this notebook may use {RAM_GB:.0f} GB (the container limit when it is lower)")
except Exception:
    pass
os.makedirs(WORK, exist_ok=True)
MIND_TRAIN, MIND_DEV = hf_mind.ensure_mind("large", os.path.join(BASE, "mind_large_hf"), repo=HF_REPO, token=HF_TOKEN, extra_roots=["/kaggle/input"], log=print)
print("train:", MIND_TRAIN, "| dev:", MIND_DEV)
'''

RUNS_COMMON = """import torch
import suite, common, final_report, report_v18
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

def hours_left():
    return BUDGET_H - (time.time() - T_NB) / 3600

def budget_args(reserve_h=0.0):
    \"\"\"Mỗi suite.main khởi động lại đồng hồ của nó: cho mỗi run phần ngân sách CÒN LẠI của notebook, trừ phần giữ cho các run sau.\"\"\"
    return ["--budget-hours", "%.3f" % max(hours_left() - reserve_h, 0.25)]

def run_safely(label, argv, release=True):
    \"\"\"Một run lỗi không được làm mất các run khác: in lỗi và trả None.\"\"\"
    try:
        s = suite.main(argv)
        if release:
            s.release()                                # RAM: run đã xong giữ các hàng và kết quả từng impression, không giữ dữ liệu
        return s
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"!! run {label} failed:", repr(e)[:300])
        return None
    finally:
        gc.collect(); torch.cuda.empty_cache()

class OnlyStages:
    \"\"\"Chỉ các hàng của vài stage của một run (để run thí nghiệm không lặp lại các hàng nó lấy lại từ đối chứng trong bảng chung).\"\"\"
    def __init__(self, s, stages):
        self.records = [r for r in s.records if r.get("stage") in stages]
        self.cache, self.row = s.cache, s.row

COMMON = ["--mind-train", MIND_TRAIN, "--mind-dev", MIND_DEV, "--cache-dir", CACHE, "--device", DEVICE, "--cold-target", str(COLD_TARGET)]
SCALE = ["--lean-data", "1", "--val-max", str(VAL_MAX), "--test-max", str(TEST_MAX), "--train-sub", str(TRAIN_SUB), "--save-arrays", "0",
         "--skip-user-rag", "1", "--um-epochs", str(UM_EPOCHS)]
TAG, TAG2, TAG3 = "qwen3", "bge", "cat"
READERS = ["--um-seeds", "2", "--um-seeds-focus", "2", "--um-extra", "prior:frozen", "--um-dim", "128"]
# run 1 (v16b): BGE-large -- zero-shot scorers, retrieval views and learned readers
ARGS2 = [*COMMON, *SCALE, "--encoder", ENCODER2, "--work", WORK + "/" + TAG2, "--max-len", "256", "--text-modes", "cta", "--hist-lens", "100", "--km-ks", "5", "--seeds", "1",
         "--um-views", "frozen,text,knn_rag", *READERS]
STAGES2 = "controls,pooling,moredata,graph,usermodel"
# run 2 (v16b): the Qwen3 and BGE-large vectors side by side -- retrieval views + learned readers only
ARGS3 = [*COMMON, *SCALE, "--encoder", "concat:" + ENCODER + "+" + ENCODER2, "--work", WORK + "/" + TAG3, "--max-len", "256", "--seeds", "1", "--um-views", "frozen,knn_rag", *READERS]
STAGES3 = "controls,graph,usermodel"
# run 3 (v16b): the main encoder -- every technique, then the linear supermodel and the stacker over the exported scores of the runs above
ARGS = [*COMMON, *SCALE, "--encoder", ENCODER, "--work", WORK + "/" + TAG, "--max-len", "256", "--q-max-len", "384", "--hist-k", "10", "--seeds", "1",
        "--text-modes", "cta", "--hist-lens", "100", "--km-ks", "5", "--hq-subset", "1",
        "--um-views", "frozen,text,knn_rag", "--um-seeds", "2", "--um-seeds-focus", "2", "--um-extra", "prior:frozen,prior:text", "--um-dim", "128",
        "--stack-folds", str(STACK_FOLDS), "--stack-backend", STACK_BACKEND, "--stack-save-scores", str(SAVE_SCORES), "--stack-free-memory", "1",
        "--stack-cv", STACK_CV, "--stack-configs", STACK_CONFIGS, "--stack-blend", str(STACK_BLEND)]
STAGES = "controls,pooling,moredata,graph,usermodel,supermodel,stack"
EXTRA_QUICK = ["--pairs", "20000", "--betas", "0.5", "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5", "--um-seeds", "1", "--um-seeds-focus", "1",
               "--km-ks", "2", "--stack-rounds", "60", "--stack-configs", "clean,clean_d5"]
S = S2 = S3 = None
"""

RUNS_A = """# run 4 (v18a): Qwen3-Embedding-4B on 'Category. Subcategory. title. abstract' (fp16 linear layers, fp32 embeddings / norms; the encoding is split over the GPUs)
TAG4, TAG5 = "q4b", "cat4"
ARGS4 = [*COMMON, *SCALE, "--encoder", ENCODER4, "--work", WORK + "/" + TAG4, "--max-len", "256", "--base-text", "cta", "--enc-dtype", "auto", "--seeds", "1", "--km-ks", "5",
         "--um-views", "frozen,knn_rag", *READERS]
STAGES4 = "controls,pooling,graph,usermodel"
# run 5 (v18a): the 4B and BGE-large vectors side by side, both on the same 'cta' text (BGE's from the cache of run 1's moredata stage, the 4B's from run 4)
ARGS5 = [*COMMON, *SCALE, "--encoder", "concat:" + ENCODER4 + "+" + ENCODER2, "--work", WORK + "/" + TAG5, "--max-len", "256", "--base-text", "cta", "--seeds", "1",
         "--um-views", "frozen,knn_rag", *READERS]
STAGES5 = "controls,graph,usermodel"
if QUICK:                                              # cỡ đồ chơi: mọi họ stage chạy một lần
    ARGS, ARGS2, ARGS3, ARGS4, ARGS5 = ARGS + EXTRA_QUICK, ARGS2 + EXTRA_QUICK, ARGS3 + EXTRA_QUICK, ARGS4 + EXTRA_QUICK, ARGS5 + EXTRA_QUICK
S4 = S5 = S_TREAT = None
CTRL_DIR = WORK + "/ctrl_snapshot"
print("budget:", BUDGET_H, "h; runs: bge -> cat -> qwen3 (control: supermodel + stack) -> q4b -> cat4 -> treatment (supermodel + stack again)")
"""

RUNS_B = """# v18b: the readers of the Qwen3 + BGE-large run trained harder (all training samples, more epochs, decaying learning rate, finer early stopping) + reader 'sca'
TAG3 = "catplus"
ARGS3 = [*COMMON, *SCALE, "--encoder", "concat:" + ENCODER + "+" + ENCODER2, "--work", WORK + "/" + TAG3, "--max-len", "256", "--seeds", "1", "--um-views", "frozen,knn_rag",
         "--um-seeds", "2", "--um-seeds-focus", "2", "--um-extra", PLUS_EXTRA, "--um-dim", "128",
         "--train-sub", "0", "--um-epochs", PLUS_EPOCHS, "--um-lr-decay", PLUS_DECAY, "--um-decay-after", "2", "--um-min-delta", "0.0002"]
if QUICK:
    ARGS, ARGS2, ARGS3 = ARGS + EXTRA_QUICK, ARGS2 + EXTRA_QUICK, ARGS3 + EXTRA_QUICK
print("budget:", BUDGET_H, "h; runs: bge (v16b) -> catplus (readers trained harder + sca) -> qwen3 (supermodel + stack)")
"""

CELL_BGE = '''suite.EQUIV_REFERENCE = reference_encoder(ENCODER2, 256)
S2 = run_safely("bge", [*ARGS2, *budget_args(), "--stages", STAGES2])
'''

CELL_CAT = '''suite.EQUIV_REFERENCE = None                           # the pooling guard compares ONE encoder with its reference; the parts are guarded in their own runs
S3 = run_safely("cat", [*ARGS3, *budget_args(), "--stages", STAGES3])
'''

CELL_MAIN_A = '''SUPER_DIRS = [d for d, s in ((WORK + "/" + TAG2, S2), (WORK + "/" + TAG3, S3)) if s is not None]
suite.EQUIV_REFERENCE = reference_encoder(ENCODER, 256)
S = run_safely("qwen3 (control)", [*ARGS, "--super-dirs", ",".join(SUPER_DIRS), *budget_args(), "--stages", STAGES], release=False)
if S is not None:                                      # the control's reports are written before its data are released (the final report reads the data)
    os.makedirs(CTRL_DIR, exist_ok=True)
    try:
        print(final_report.build(S, CTRL_DIR + "/FINAL_REPORT_ctrl.md", CTRL_DIR + "/final_tables_ctrl.json"))
    except Exception as e:
        print("!! control report failed:", repr(e)[:200])
    for fn in ("results.md", "results.json", "timings.json", "memory.json", "stack_scores.npz", "ladder.md"):
        p = os.path.join(WORK, TAG, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(CTRL_DIR, fn.replace(".", "_ctrl.", 1)))
    S.release()
    gc.collect(); torch.cuda.empty_cache()
print("giờ còn lại sau phần đối chứng: %.2f" % hours_left())
'''

CELL_Q4B = '''if S is None or not ENCODER4:
    print("nhánh 4B bỏ qua:", "không có đối chứng" if S is None else "ENCODER4 rỗng")
elif hours_left() < Q4B_MIN_H:
    print(f"nhánh 4B bỏ qua: còn {hours_left():.2f} giờ < {Q4B_MIN_H} giờ cần cho mã hoá 4B + reader + run nối + supermodel/stacker lại")
else:
    suite.EQUIV_REFERENCE = None                       # same tokenizer / last-token pooling as Qwen3-Embedding-0.6B (guarded in run 3); a 4B reference in fp32 does not fit a T4
    S4 = run_safely("q4b", [*ARGS4, *budget_args(reserve_h=CAT4_MIN_H), "--stages", STAGES4])
print("giờ còn lại: %.2f" % hours_left())
'''

CELL_CAT4 = '''if S4 is None:
    print("run vector nối 4B + BGE-large bỏ qua: run 4B không có")
elif hours_left() < CAT4_MIN_H:
    print(f"run vector nối 4B + BGE-large bỏ qua: còn {hours_left():.2f} giờ < {CAT4_MIN_H}")
else:
    suite.EQUIV_REFERENCE = None
    S5 = run_safely("cat4", [*ARGS5, *budget_args(reserve_h=TREAT_MIN_H), "--stages", STAGES5])
print("giờ còn lại: %.2f" % hours_left())
'''

CELL_TREAT = '''NEW_DIRS = [WORK + "/" + t for t, s in ((TAG4, S4), (TAG5, S5)) if s is not None and os.path.isdir(os.path.join(WORK, t, "scores"))]
if S is None or not NEW_DIRS:
    print("nhánh thí nghiệm bỏ qua:", "không có đối chứng" if S is None else "không có thành phần mới")
elif hours_left() < TREAT_MIN_H:
    print(f"nhánh thí nghiệm bỏ qua: còn {hours_left():.2f} giờ < {TREAT_MIN_H}")
else:
    suite.EQUIV_REFERENCE = None                       # the main encoder's vectors come from the cache (guarded in run 3)
    S_TREAT = run_safely("treatment", [*ARGS, "--super-dirs", ",".join(SUPER_DIRS + NEW_DIRS), "--rerun", "supermodel,stack", "--min-stage-s", str(STACK_MIN_S),
                                       *budget_args(), "--stages", STAGES], release=False)
'''

CELL_MAIN_B = '''SUPER_DIRS = [d for d, s in ((WORK + "/" + TAG2, S2), (WORK + "/" + TAG3, S3)) if s is not None]
suite.EQUIV_REFERENCE = reference_encoder(ENCODER, 256)
S = run_safely("qwen3", [*ARGS, "--super-dirs", ",".join(SUPER_DIRS), "--min-stage-s", str(STACK_MIN_S), *budget_args(), "--stages", STAGES], release=False)
'''

REPORT_A = '''SUITES_TAGS = [(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG), (S4, TAG4), (S5, TAG5)) if s is not None]
if S_TREAT is not None:
    SUITES_TAGS.append((OnlyStages(S_TREAT, ("supermodel", "stack")), "treat"))
SUITES, TAGS = [s for s, _ in SUITES_TAGS], [t for _, t in SUITES_TAGS]
os.makedirs(RESULTS, exist_ok=True)
if S is not None:
    suite.leaderboard(SUITES, TAGS, (TAG, "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
    suite.leaderboard(SUITES, TAGS, (TAG, "frozen+pop"), WORK + "/leaderboard_pop.md", plot=WORK + "/forest_pop.png", pop=True)
Q, T = TAG, "treat"
pairs = [# --- the treatment (+ 4B components) against the control (v16b pipeline of this notebook), row by row
         ((T, "super_base"), (Q, "super_base")), ((T, "super_zs"), (Q, "super_zs")), ((T, "super_llm"), (Q, "super_llm")), ((T, "stack_llm"), (Q, "stack_llm")),
         ((T, "stack_meta"), (Q, "stack_meta")), ((T, "stack_meta_v15cfg"), (Q, "stack_meta_v15cfg")), ((T, "super_llm+pop"), (Q, "super_llm+pop")), ((T, "stack_pop"), (Q, "stack_pop")),
         # --- the 4B encoder against Qwen3-0.6B on the same text ('cta'), and against BGE-large
         (("q4b", "pool_lse"), (Q, "txt_cta_lse")), (("q4b", "um_frozen"), (Q, "um_text")), (("q4b", "um_frozen"), ("bge", "um_frozen")), (("q4b", "um_knn_rag"), ("bge", "um_knn_rag")),
         (("q4b", "um_knn_rag"), ("q4b", "um_frozen")), (("q4b", "um_prior_frozen"), (Q, "um_prior_frozen")),
         # --- 4B + BGE-large side by side against Qwen3-0.6B + BGE-large (the best reader of v16b) and against the 4B alone
         (("cat4", "um_frozen"), ("cat", "um_frozen")), (("cat4", "um_knn_rag"), ("cat", "um_knn_rag")), (("cat4", "um_frozen"), ("q4b", "um_frozen")), (("cat4", "um_knn_rag"), ("cat4", "um_frozen")),
         # --- the control against itself in v16b's ordering (sanity: should reproduce RESULTS_v16b_large.md section 1)
         ((Q, "stack_meta"), (Q, "super_llm")), ((Q, "stack_meta"), (Q, "stack_meta_v15cfg")), ((Q, "stack_pop"), (Q, "stack_meta")), (("cat", "um_frozen"), ("bge", "um_frozen"))]
direct = suite.cross_compare(SUITES, TAGS, pairs, WORK + "/direct_comparisons.md") if S is not None else []
fusion_md = gate_md = ""
if S is not None and S_TREAT is not None:
    fusion_md, fusion = report_v18.fusion_table(S, S_TREAT, WORK + "/HEADLINE_v18a.md")
    gate_md, gates = report_v18.gate_v18a(S, S_TREAT, WORK + "/gate_v18a.md")
    print(fusion_md); print(); print(gate_md)
    try:
        print(final_report.build(S_TREAT, WORK + "/FINAL_REPORT_treat.md", WORK + "/final_tables_treat.json"))
    except Exception as e:
        print("!! treatment report failed:", repr(e)[:200])
elif S is not None:
    fusion_md, _ = report_v18.fusion_table(S, S, WORK + "/HEADLINE_v18a.md")
    print("!! no treatment: the table shows the control only (Δ = 0 by construction)")
print(report_v18.epochs_table([(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG), (S4, TAG4), (S5, TAG5))], WORK + "/epochs.md"))
'''

REPORT_B = '''SUITES_TAGS = [(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG)) if s is not None]
SUITES, TAGS = [s for s, _ in SUITES_TAGS], [t for _, t in SUITES_TAGS]
os.makedirs(RESULTS, exist_ok=True)
if S is not None:
    suite.leaderboard(SUITES, TAGS, (TAG, "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
    suite.leaderboard(SUITES, TAGS, (TAG, "frozen+pop"), WORK + "/leaderboard_pop.md", plot=WORK + "/forest_pop.png", pop=True)
Q, C = TAG, TAG3
pairs = [# --- inside this run: the new reader against the candidate-aware reader on the same view, same training recipe (paired)
         ((C, "um_sca_knn_rag"), (C, "um_knn_rag")), ((C, "um_knn_rag"), (C, "um_frozen")),
         # --- the harder-trained readers on the Qwen3 + BGE-large vectors against the v16b-recipe readers of the other runs
         ((C, "um_frozen"), ("bge", "um_frozen")), ((C, "um_knn_rag"), ("bge", "um_knn_rag")), ((C, "um_frozen"), (Q, "um_frozen")),
         # --- the fused rows (as in v16b)
         ((Q, "stack_meta"), (Q, "super_llm")), ((Q, "stack_meta"), (Q, "stack_meta_v15cfg")), ((Q, "stack_pop"), (Q, "stack_meta")), ((Q, "super_llm"), (Q, "super_base"))]
direct = suite.cross_compare(SUITES, TAGS, pairs, WORK + "/direct_comparisons.md") if S is not None else []
vs_rows = [("super_base", S, "super_base", "super_base"), ("super_llm", S, "super_llm", "super_llm"), ("stack_llm", S, "stack_llm", "stack_llm"),
           ("stack_meta", S, "stack_meta", "stack_meta"), ("stack_meta_v15cfg", S, "stack_meta_v15cfg", "stack_meta_v15cfg"), ("super_llm+pop", S, "super_llm+pop", "super_llm+pop"),
           ("stack_pop", S, "stack_pop", "stack_pop"), ("catplus:um_frozen", S3, "um_frozen", "cat:um_frozen"), ("catplus:um_knn_rag", S3, "um_knn_rag", "cat:um_knn_rag"),
           ("catplus:um_sca_knn_rag", S3, "um_sca_knn_rag", "cat:um_knn_rag"), ("bge:um_frozen", S2, "um_frozen", "bge:um_frozen"), ("bge:um_knn_rag", S2, "um_knn_rag", "bge:um_knn_rag"),
           ("qwen3:um_frozen", S, "um_frozen", "qwen3:um_frozen")]
fusion_md, _ = report_v18.vs_v16b_table([r for r in vs_rows if r[1] is not None], WORK + "/HEADLINE_v18b.md")
print(fusion_md)
gate_md = ""
print(report_v18.epochs_table([(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG))], WORK + "/epochs.md"))
if S is not None:
    try:
        print(final_report.build(S, WORK + "/FINAL_REPORT.md", WORK + "/final_tables.json"))
    except Exception as e:
        print("!! final report failed:", repr(e)[:200])
'''

SUMMARY = '''noise = ["# Seed-to-seed spread of the trained rows (test nDCG@10 of each seed; sample std with n-1; the paired CIs do NOT include it)", "",
         "| run | row | seeds | nDCG@10 per seed | std |", "|---|---|---|---|---|"]
for tag_ in [t for t in (TAG, TAG2, TAG3, "q4b", "cat4") if os.path.isdir(os.path.join(WORK, t))]:
    pj = os.path.join(WORK, tag_, "results.json")
    if os.path.exists(pj):
        for r in json.load(open(pj, encoding="utf-8")):
            if r.get("ndcg10_per_seed") and len(r["ndcg10_per_seed"]) > 1:
                noise.append(f"| {tag_} | {r['name']} | {len(r['ndcg10_per_seed'])} | {', '.join(f'{x:.4f}' for x in r['ndcg10_per_seed'])} | {r.get('ndcg10_std', float('nan')):.4f} |")
open(os.path.join(WORK, "noise.md"), "w", encoding="utf-8").write("\\n".join(noise) + "\\n")
print("tổng thời gian của notebook: %.2f h" % ((time.time() - T_NB) / 3600))
parts = [(f"HEADLINE_v18@P@.md", WORK), ("gate_v18a.md", WORK), ("direct_comparisons.md", WORK), ("epochs.md", WORK), ("noise.md", WORK), ("FINAL_REPORT_treat.md", WORK),
         ("FINAL_REPORT.md", WORK), ("FINAL_REPORT_ctrl.md", WORK + "/ctrl_snapshot"), ("leaderboard.md", WORK), ("leaderboard_pop.md", WORK)]
for t in [t for t in (TAG, TAG2, TAG3, "q4b", "cat4") if os.path.isdir(os.path.join(WORK, t))]:
    parts += [(f, WORK + "/" + t) for f in ("timings.json", "memory.json", "results.md")]
SUMMARY = os.path.join(RESULTS, "SUMMARY_v18@P@.md")
with open(SUMMARY, "w", encoding="utf-8") as out:
    out.write(f"# SUMMARY v18@P@ ({'QUICK' if QUICK else 'full'}; {(time.time() - T_NB) / 3600:.2f} h)\\n")
    for fn, d in parts:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out.write(f"\\n\\n## {fn} ({os.path.basename(d)})\\n\\n" + open(p, encoding="utf-8").read())
for fn in ("forest.png", "forest_pop.png", "HEADLINE_v18@P@.md", "gate_v18a.md", "direct_comparisons.md", "epochs.md", "noise.md", "leaderboard.md", "leaderboard_pop.md",
           "FINAL_REPORT.md", "FINAL_REPORT_treat.md", "final_tables.json", "final_tables_treat.json"):
    if os.path.exists(os.path.join(WORK, fn)) and os.path.abspath(WORK) != os.path.abspath(RESULTS):
        shutil.copy(os.path.join(WORK, fn), os.path.join(RESULTS, fn))
for src, dst in ((os.path.join(WORK, TAG, "stack_scores.npz"), "stack_scores.npz"), (os.path.join(WORK, "ctrl_snapshot", "stack_scores_ctrl.npz"), "stack_scores_ctrl.npz"),
                 (os.path.join(WORK, "ctrl_snapshot", "FINAL_REPORT_ctrl.md"), "FINAL_REPORT_ctrl.md")):
    if os.path.exists(src):
        shutil.copy(src, os.path.join(RESULTS, dst))
print("-> gửi lại file:", SUMMARY, f"({os.path.getsize(SUMMARY) / 1024:.0f} KB)")
'''

READ_A = """### Cách đọc kết quả (v18a)

| Bạn thấy (trong `gate_v18a.md` / `HEADLINE_v18a.md`) | Nghĩa là |
|---|---|
| G1 **PASS** | vector 4B thêm thông tin cho ghép tuyến tính chỉ dùng nội dung + lịch sử (cùng impression, CI ghép cặp) |
| G2 **PASS** và `treatment` > 0,4543 | encoder 4B làm hệ thống offline tốt hơn cả đối chứng trong notebook lẫn kết quả tốt nhất trước đây |
| G2 **NO GAIN** dù G1 PASS | như v16b: cải thiện nội dung không lên được stacker (đặc trưng phơi nhiễm / đếm theo bài đã giải thích phần đó) |
| `q4b:um_frozen` − `qwen3:um_text` > 0 | cùng văn bản 'cta', encoder 4B tốt hơn 0,6B khi làm đầu vào cho reader |
| `cat4:*` − `cat:*` > 0 | thay Qwen3-0,6B bằng 4B trong vector nối có ích |
| hàng `control` khác `v16b` vài phần nghìn | nhiễu giữa hai lần chạy (seed reader, XGBoost GPU) — đó là thước đo cho mọi so sánh với số của run khác |

**Giới hạn:** một ngày test; CI chỉ tính nhiễu mẫu test; 2 seed mỗi reader; thời gian mã hoá của 4B và RAM của run nối 3 584 chiều chưa được đo trước."""

READ_B = """### Cách đọc kết quả (v18b)

| Bạn thấy | Nghĩa là |
|---|---|
| `catplus:um_sca_knn_rag` − `catplus:um_knn_rag` > 0 (CI ghép cặp) | thêm self-attention kiểu NRMS trên lịch sử giúp reader theo ứng viên |
| `catplus:um_*` cao hơn số v16b (0,4223 / 0,4226) ≥ 0,005 | reader v16b bị huấn luyện thiếu: nhiều dữ liệu, nhiều epoch và learning rate giảm dần có ích (xem `epochs.md`: epoch tốt nhất có nằm ở trần không) |
| `stack_meta` / `stack_meta_v15cfg` / `stack_pop` cao hơn số v16b ≥ 0,003 | hệ thống tốt hơn; chênh nhỏ hơn thì chưa phân biệt được với nhiễu giữa hai lần chạy |

**Giới hạn:** so với v16b là so giữa hai lần chạy (không có CI ghép cặp); chỉ reader của run vector nối được huấn luyện lại."""


def config_cell(p):
    s = (CONFIG.replace("@BUDGET@", "11.2" if p == "a" else "6.0").replace("@STACKMIN@", "4200" if p == "a" else "3900").replace("@P@", p)
         .replace("@PROFILE@", PROFILE_A if p == "a" else PROFILE_B).replace("@PRINT@", PRINT_A if p == "a" else PRINT_B))
    return code(s)


def build(p):
    a = p == "a"
    cells = [md(intro(p)), md("### 0. Phụ thuộc và GPU"), code(V17.DEPS), md("### 1. Modules (ghi ra thư mục làm việc)")]
    cells += [writefile(m) for m in MODULES]
    cells += [md("### 2. Cấu hình"), config_cell(p), md("### 3. Dữ liệu — MIND-large tải từ Hugging Face (cần Internet)"), code(DATA),
              md("### 4. Cấu hình các run"), code(RUNS_COMMON + (RUNS_A if a else RUNS_B)),
              md("### 5. Run 1 — BGE-large (như v16b): controls → pooling → moredata → graph → usermodel"), code(CELL_BGE)]
    if a:
        cells += [md("### 6. Run 2 — vector Qwen3-0,6B ⊕ BGE-large (như v16b): controls → graph → usermodel"), code(CELL_CAT),
                  md("### 7. Run 3 — Qwen3-0,6B + supermodel + stacker = **ĐỐI CHỨNG** (pipeline v16b)"), code(CELL_MAIN_A),
                  md("### 8. Run 4 — Qwen3-Embedding-4B trên văn bản 'cta': controls → pooling → graph → usermodel"), code(CELL_Q4B),
                  md("### 9. Run 5 — vector 4B ⊕ BGE-large (văn bản 'cta', từ cache): controls → graph → usermodel"), code(CELL_CAT4),
                  md("### 10. **NHÁNH THÍ NGHIỆM** — supermodel + stacker của run 3 chạy lại với các thành phần 4B"), code(CELL_TREAT),
                  md("### 11. Bảng chính, cổng, so sánh trực tiếp"), code(REPORT_A)]
    else:
        cells += [md("### 6. Run 2 — vector Qwen3-0,6B ⊕ BGE-large, reader huấn luyện kỹ hơn + reader `sca`"), code(CELL_CAT.replace('"cat"', '"catplus"')),
                  md("### 7. Run 3 — Qwen3-0,6B + supermodel + stacker (như v16b, đọc các thành phần của run 1 và run 2)"), code(CELL_MAIN_B),
                  md("### 8. Bảng chính, so sánh trực tiếp"), code(REPORT_B)]
    cells += [md("### Tổng hợp và file gửi lại"), code(SUMMARY.replace("@P@", p)), md(READ_A if a else READ_B)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                                         "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    for p, fn in NOTEBOOKS.items():
        nb = build(p)
        out = os.path.join(HERE, fn)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", out, "cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
