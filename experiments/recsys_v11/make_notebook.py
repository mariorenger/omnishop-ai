"""Build the two self-contained Kaggle notebooks of v14 from the module files:

  v14_supermodel.ipynb              MIND-small from a Kaggle input dataset; ONE model (Qwen3-Embedding-0.6B) with every technique, fused into the supermodel
  v14_mindlarge_hf_supermodel.ipynb the same pipeline on MIND-large downloaded from Hugging Face (does Kaggle run it?)

Neither trains NRMS / NAML / Fastformer (v13 and v7 did; their numbers are printed as a static reference table).
Bump the notebook names when the content changes: Kaggle caches notebooks by file name."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = {"small": "v14_supermodel.ipynb", "large": "v14_mindlarge_hf_supermodel.ipynb"}
MODULES = ["data.py", "metrics.py", "common.py", "rep_eval.py", "semgraph.py", "adapt.py", "fields.py", "pooling.py", "sid.py", "llm_user.py", "llm_gen.py",
           "refdata.py", "refmodels.py", "usermodel.py", "stages_ext.py", "stages_ref.py", "supermodel.py", "featscore.py", "stacker.py", "final_report.py", "data_lean.py",
           "hf_mind.py", "suite.py"]       # suite.py imports stacker.py (v15); the committed v14 notebooks keep the code of the run that was measured -- do not regenerate them


def lines(s):
    out = s.splitlines(keepends=True)
    return out or [""]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(t)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(t)}


def writefile(fn):
    with open(os.path.join(HERE, fn), encoding="utf-8") as f:
        return code(f"%%writefile {fn}\n" + f.read())


# ------------------------------------------------------------------------------------------------ text
TECHNIQUES = """| Nhóm | Kỹ thuật (đều trên cùng một encoder) | Stage |
|---|---|---|
| Dữ liệu | text có **category + subcategory** (`cta`), đọc **100 click** thay vì 50 (MIND không có body bài) | `moredata` |
| Pooling | **late-interaction** (log-sum-exp, τ chọn trên validation), **k-means interest** (multi-interest không huấn luyện) | `pooling` |
| RAG | RAG theo entity / kNN, GraphRAG-lite (cộng đồng bài, người dùng tương tự) làm view cho reader | `graph` |
| Học trên vector đóng băng | reader học được (V10 `llmenc_ca`), view `text`, view RAG, **multi-interest** (`mi1` đối chứng, `mi4`) | `usermodel` |
| LLM làm user encoder | tower **LoRA** (causal), trên tập con 5 000 / 30 000 impression | `llm_user` |
| Popularity | gộp online kiểu `bge_zs_pop` của V10 (khung *streaming*, không so được với bài báo) | bản `+pop` |
| **Supermodel** | ghép tất cả bằng trọng số khớp trên validation | `supermodel` |"""

SUPER_HOW = """**Supermodel làm thế nào (không nhìn test để chọn gì):**
1. Mỗi kỹ thuật được đo **riêng lẻ** trước (CI ghép cặp so với đối chứng của nó), và điểm thô của từng thành phần được giữ lại (`<work>/scores`).
2. **Nền** = mô hình đơn tốt nhất theo **validation** trong phía LLM (zero-shot hoặc reader học được).
3. Thêm lần lượt từng **nhóm**: các scorer zero-shot khác → các reader học được → (tower LLM, trên tập con của nó) → popularity. Ở mỗi bậc trọng số được khớp lại **trên validation**
   (coordinate ascent trên nDCG@10, nền giữ trọng số 1, lưới nhỏ); test chạm **một lần** cho mỗi bậc. Điểm mỗi thành phần được z-score trong từng impression (công thức của `bge_zs_pop`).
4. Có **bảng bỏ-một-nhóm** từ mô hình cuối (mỗi nhóm đóng góp bao nhiêu, CI ghép cặp) và bảng trọng số.
Supermodel **không** chứa NRMS / NAML / Fastformer: nó là "phía LLM"; muốn biết nó có hơn các baseline đó không, xem bảng tham chiếu tĩnh trong `FINAL_REPORT.md` (số đã đo ở v7/v13, **không** phải của lần chạy này)."""

SMALL_INTRO = f"""# V14 Supermodel — MỘT mô hình (Qwen3-Embedding-0.6B) + đủ kỹ thuật trên MIND-small (1 lần Run All, ước ~2 giờ trên T4)

Notebook này chạy **đúng một mô hình**. Nó **không** huấn luyện NRMS / NAML / Fastformer, không quét nhiều encoder (BGE-small chỉ chạy 1–2 phút để kiểm tra giao thức), không LoRA/DoRA cho encoder, LLM judge hay KAR,
nên nhẹ hơn v13 (đo thật 8,2 giờ) khoảng 3–4 lần (ước tính, chưa đo ở bản này). Giao thức vẫn là V10: validation theo ngày, `MINDsmall_dev` chỉ là test, CI 95% ghép cặp, lát cold/warm.

{TECHNIQUES}

{SUPER_HOW}

**Cách chạy:** (1) Settings → Accelerator **GPU (T4/P100)**, Internet **On**. (2) *Add Input* → dataset MIND có **cả** `MINDsmall_train` và `MINDsmall_dev` (ví dụ `thinhhuynh3108/mindsmall`). (3) **Run All**.
Lần đầu có thể chạy thử cỡ đồ chơi (vài phút, bắt lỗi môi trường): ở ô *Cấu hình* đổi `"0"` thành `"1"` trong dòng `QUICK = ...` (hoặc đặt biến môi trường `V11_QUICK=1`). Các công tắc khác cũng ở ô đó (đổi giá trị mặc định trong `os.environ.get("V11_...", ...)`): `V11_CONTROLS=0` bỏ đối chứng "category của bài khác" (tiết kiệm ~10 phút);
`V11_TOWER_SEEDS=2` thêm một seed cho tower (+40 phút); `V11_PROTOCOL_CHECK=0` bỏ bước kiểm tra giao thức.

**Kiểm tra giao thức (1–2 phút, không phải so sánh):** trước hết chạy `controls` với BGE-small để xác nhận `frozen` khớp V10 `bge_zeroshot` (AUC 0.6241 / nDCG@10 0.3909) và `popularity` / `frozen+pop` khớp V10.
Nếu in `MISMATCH` thì dừng và sửa trước khi tin bất kỳ con số nào của mô hình chính.

**Thời gian (ước, chưa đo ở bản này; dựa trên số đo thật của v13 trên T4):** mã hoá 65k bài bằng Qwen3: E0 ≈ 10 phút, text `cta` ≈ 11 phút, text `ctr` (đối chứng) ≈ 11 phút · pooling + moredata ≈ 8 phút · graph ≈ 3 phút · user model ≈ 30 phút (19 lần huấn luyện reader × ~90 s) ·
tower LoRA ≈ 40 phút/seed · supermodel ≈ 5 phút. `BUDGET_H` (mặc định 6) là ngân sách đồng hồ chung: vòng lặp dài tự dừng và stage nào không còn thời gian thì bị bỏ qua (kết quả ghi sau từng stage; chạy lại cell sẽ resume).

**Nhận lại:** `FINAL_REPORT.md` (supermodel, thang kỹ thuật, bỏ-một-nhóm, từng kỹ thuật, bảng tham chiếu NRMS/NAML/Fastformer), `leaderboard*.md`, `direct_comparisons.md`, `forest*.png` và **một file gộp `SUMMARY_v14.md`** — gửi file đó (cùng log nếu có lỗi)."""

LARGE_INTRO = f"""# V14 Supermodel trên MIND-large tải từ Hugging Face — Kaggle có chạy nổi không? (1 lần Run All, ước ~4 giờ trên T4, chưa đo)

Cùng một pipeline với `v14_supermodel` (MỘT mô hình Qwen3-Embedding-0.6B + đủ kỹ thuật + supermodel, không baseline), nhưng dữ liệu là **MIND-large** (2,2 triệu impression train, 376 nghìn impression dev có nhãn, khoảng 100–130 nghìn bài; mỗi lần mã hoá cả catalogue bằng Qwen3 ước ~20–25 phút).
Đây là một **thử nghiệm khả thi**: mục tiêu đầu tiên là biết Kaggle (T4, ~30 GB RAM) có chạy hết không và mỗi stage tốn bao lâu; kết quả điểm số là thứ hai.

**Dữ liệu.** `hf_mind.py` tải `MINDlarge_train.zip` và `MINDlarge_dev.zip` từ Hugging Face dataset `Recommenders/MIND` (nơi thư viện Microsoft Recommenders đang lấy MIND; Azure blob gốc không còn mở công khai), chỉ giải nén `news.tsv` và `behaviors.tsv`.
**Bố cục của repo trên Hub chưa được kiểm tra từ máy phát triển** (sandbox bị chặn huggingface.co), nên bộ tải liệt kê repo, tìm file theo tên, có đường dự phòng (thư mục đã giải nén, `/kaggle/input`, HTTP thuần) và in rõ mọi bước.
Nếu không tải được nó dừng ngay (trong vài phút đầu, chưa tốn giờ GPU) với danh sách file thật của repo — gửi log đó cho tôi; hoặc *Add Input* một dataset Kaggle có `MINDlarge_train` và `MINDlarge_dev` (đã giải nén hoặc zip), bộ tải tự tìm trong `/kaggle/input` và dùng nó thay vì tải. `MINDlarge_test.zip` (không có nhãn) không bao giờ được dùng. Cần **Internet On** (nếu repo trên Hub đòi đăng nhập: tạo Kaggle secret tên `HF_TOKEN`, ô Cấu hình tự đọc).

**Giao thức giống bản small:** validation = những ngày cuối của train (≥ 10% impression), test = `MINDlarge_dev` (chạm một lần), cold/warm, CI ghép cặp. Khác biệt do quy mô (đều ghi vào log và `FINAL_REPORT.md`):
bộ đọc `data_lean.py` (cùng split / popularity / negative với `MindData.from_mind`, đã kiểm trên fixture, nhưng ít RAM hơn), validation giữ ngẫu nhiên 60 000 impression để chọn siêu tham số nhanh (`V11_VAL_MAX`),
các stage có huấn luyện dùng ngẫu nhiên 600 000 mẫu train (`V11_TRAIN_SUB`; vết "đã thấy ở train" và split vẫn tính trên toàn bộ), `user_rag` bỏ qua (cần ma trận 750k người dùng × 1024-d), không ghi `test_arrays.npz`, thư mục làm việc ở `/kaggle/temp`.

{TECHNIQUES}

{SUPER_HOW}

**Không có baseline và không có kiểm tra V10 trên MIND-large** (V10 chưa có số cho large). Bảng NRMS / NAML / Fastformer chỉ tồn tại cho MIND-small nên không in ở đây.

**Cách chạy:** Settings → GPU, Internet **On** → **Run All** (không cần Add Input). Chạy thử cỡ đồ chơi (đổi `"0"` thành `"1"` ở dòng `QUICK = ...` của ô *Cấu hình*) vẫn tải MIND-large và mã hoá cả catalogue (~40 phút) nhưng mọi thứ sau đó rất nhỏ; mục đích là bắt lỗi môi trường. Nếu hết RAM / thời gian: giảm `V11_TRAIN_SUB`, `V11_VAL_MAX`, hoặc đặt `V11_TEST_MAX=150000`.
**Nhận lại:** `SUMMARY_v14.md` (+ log). Kết quả cũng được sao vào `/kaggle/working/results_large` để khỏi lẫn với các file nặng ở `/kaggle/temp`."""


CONFIG_HEAD = """import os, glob, sys, gc, time, json
QUICK = bool(int(os.environ.get("V11_QUICK", "0")))                  # True: cỡ đồ chơi để bắt lỗi môi trường
BUDGET_H = float(os.environ.get("V11_BUDGET_H", "@BUDGET@"))          # ngân sách đồng hồ chung (giờ)@BUDGET_NOTE@
ENCODER = os.environ.get("V11_ENCODER", "Qwen/Qwen3-Embedding-0.6B")  # MỘT mô hình
CONTROLS = bool(int(os.environ.get("V11_CONTROLS", "@CONTROLS@")))            # 1: thêm đối chứng "category của bài khác" (txt_ctr*): mã hoá thêm một lần cả catalogue
TEXT_MODES = os.environ.get("V11_TEXT_MODES", "cta,ctr" if CONTROLS else "cta")
TOWER_SEEDS = int(os.environ.get("V11_TOWER_SEEDS", "1"))             # số seed của tower LoRA (mỗi seed ~40 phút)
PROTOCOL_CHECK = bool(int(os.environ.get("V11_PROTOCOL_CHECK", "@CHECK@")))   # BGE-small frozen phải khớp V10 (1-2 phút)@CHECK_NOTE@
CHECK_ENCODER = os.environ.get("V11_CHECK_ENCODER", "BAAI/bge-small-en-v1.5")
DEVICE = os.environ.get("V11_DEVICE", "auto")
USE_REFERENCE = bool(int(os.environ.get("V11_REFERENCE", "1")))       # so HFEncoder với Sentence-Transformers trên 64 bài trước khi chạy
"""

CONFIG_SMALL = """MIND_TRAIN = os.environ.get("V11_MIND_TRAIN")                      # để None -> tự dò trong /kaggle/input
MIND_DEV = os.environ.get("V11_MIND_DEV")
WORK = ("/kaggle/working/v14_out" if os.path.isdir("/kaggle/working") else "v14_out") + ("_quick" if QUICK else "")   # QUICK không bao giờ lẫn vào bản đầy đủ
CACHE = "/kaggle/temp/v14_cache" if os.path.isdir("/kaggle/temp") else os.path.join(WORK, "cache")
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
VAL_MAX = int(os.environ.get("V11_VAL_MAX", "5000" if QUICK else "60000"))      # validation giữ ngẫu nhiên từng này impression (0 = tất cả)
TEST_MAX = int(os.environ.get("V11_TEST_MAX", "20000" if QUICK else "0"))       # 0 = toàn bộ MINDlarge_dev (376k impression)
TRAIN_SUB = int(os.environ.get("V11_TRAIN_SUB", "50000" if QUICK else "600000"))  # mẫu train cho các stage có huấn luyện (0 = tất cả)
BASE = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
WORK = os.path.join(BASE, "v14_large_out") + ("_quick" if QUICK else "")
RESULTS = ("/kaggle/working/results_large" if os.path.isdir("/kaggle/working") else "results_large") + ("_quick" if QUICK else "")
CACHE = os.path.join(BASE, "v14_large_cache")
"""

CONFIG_TAIL = """sys.path.insert(0, os.getcwd())
print(dict(QUICK=QUICK, BUDGET_H=BUDGET_H, ENCODER=ENCODER, TEXT_MODES=TEXT_MODES, TOWER_SEEDS=TOWER_SEEDS, PROTOCOL_CHECK=PROTOCOL_CHECK, WORK=WORK))
"""


def cell_config(profile):
    large = profile == "large"
    head = (CONFIG_HEAD.replace("@BUDGET@", "10" if large else "6").replace("@BUDGET_NOTE@", "; Kaggle giết notebook ở 12 giờ" if large else "")
            .replace("@CHECK@", "0" if large else "1").replace("@CONTROLS@", "0" if large else "1").replace("@CHECK_NOTE@", "; MIND-large không có số V10" if large else ""))
    return code(head + (CONFIG_LARGE if large else CONFIG_SMALL) + CONFIG_TAIL)


def cell_data(profile):
    if profile == "large":
        return code('''import hf_mind, shutil
try:
    import psutil
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.0f} GB total, {psutil.virtual_memory().available / 1e9:.0f} GB free")
except Exception:
    pass
print(f"disk free at {BASE}: {shutil.disk_usage(BASE).free / 1e9:.0f} GB")
os.makedirs(WORK, exist_ok=True)
MIND_TRAIN, MIND_DEV = hf_mind.ensure_mind("large", os.path.join(BASE, "mind_large_hf"), repo=HF_REPO, token=HF_TOKEN, extra_roots=["/kaggle/input"], log=print)
print("train:", MIND_TRAIN, "| dev:", MIND_DEV)
''')
    return code('''if not MIND_TRAIN:
    hits = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/news.tsv", recursive=True)})
    trs = [d for d in hits if "train" in d.lower()]
    dvs = [d for d in hits if "dev" in d.lower() or "valid" in d.lower()]
    for t in trs:
        sib = [d for d in dvs if os.path.dirname(d) == os.path.dirname(t)]
        if sib:
            MIND_TRAIN, MIND_DEV = t, sib[0]
            break
assert MIND_TRAIN and MIND_DEV and MIND_TRAIN != MIND_DEV, (
    "Hãy Add Input một dataset MIND có CẢ MINDsmall_train và MINDsmall_dev (ví dụ thinhhuynh3108/mindsmall).")
count = lambda p: sum(1 for _ in open(p, encoding="utf-8"))
print("train:", MIND_TRAIN, "| behaviors:", f"{count(MIND_TRAIN + '/behaviors.tsv'):,}")
print("dev  :", MIND_DEV, "| behaviors:", f"{count(MIND_DEV + '/behaviors.tsv'):,}  (MIND-small: ~157k / ~73k)")
''')


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

COMMON = ["--mind-train", MIND_TRAIN, "--mind-dev", MIND_DEV, "--cache-dir", CACHE, "--device", DEVICE, "--budget-hours", str(BUDGET_H)]
TOWER_MODES = ",".join(["causal"] + ["causal@%d" % i for i in range(1, TOWER_SEEDS)])
TAG = "qwen3"
ARGS = [*COMMON, "--encoder", ENCODER, "--work", WORK + "/" + TAG, "--max-len", "256", "--q-max-len", "384", "--hist-k", "10", "--seeds", "1",
        "--text-modes", TEXT_MODES, "--hist-lens", "100", "--km-ks", "5", "--hq-subset", "1",
        "--um-views", "frozen,text,knn_rag,entity_rag", "--um-seeds", "2", "--um-seeds-focus", "3", "--um-extra", "mi1:frozen,mi4:frozen,mi4:text",
        "--llm-user-modes", TOWER_MODES]
@SCALE@
STAGES = "controls,pooling,moredata,graph,usermodel,llm_user,supermodel"      # rẻ và quan trọng trước, tower (đắt nhất) sau, supermodel dùng những gì đã có
if QUICK:                                              # cỡ đồ chơi: mọi họ stage chạy một lần
    ARGS += ["--pairs", "20000", "--betas", "0.5", "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5", "--llm-user-steps", "60", "--llm-user-bs", "8",
             "--llm-val-n", "400", "--llm-test-n", "1000", "--um-epochs", "2", "--um-seeds", "1", "--um-seeds-focus", "1",
             "--um-extra", "mi1:frozen,mi4:frozen,mi4:text", "--km-ks", "2"]
"""

SCALE_LARGE = """SCALE = ["--lean-data", "1", "--val-max", str(VAL_MAX), "--test-max", str(TEST_MAX), "--train-sub", str(TRAIN_SUB), "--save-arrays", "0",
         "--skip-user-rag", "1", "--um-epochs", "6"]
ARGS += SCALE"""


def cell_runs(profile):
    return code(RUNS_TEMPLATE.replace("@SCALE@", SCALE_LARGE if profile == "large" else "SCALE = []"))


PAIRS = '''Q = TAG
pairs = [# --- pooling and the data: each against its own control
         ((Q, "pool_lse"), (Q, "frozen")), ((Q, "pool_km5"), (Q, "pool_lse")), ((Q, "pool_km5"), (Q, "frozen")),
         ((Q, "txt_cta"), (Q, "frozen")), ((Q, "txt_cta_lse"), (Q, "pool_lse")), ((Q, "txt_ctr_lse"), (Q, "pool_lse")), ((Q, "txt_cta_lse"), (Q, "txt_ctr_lse")),
         ((Q, "hist100_lse"), (Q, "pool_lse")), ((Q, "hist100_mean"), (Q, "frozen")),
         # --- learned readers
         ((Q, "um_text"), (Q, "um_frozen")), ((Q, "um_mi1_frozen"), (Q, "um_frozen")), ((Q, "um_mi4_frozen"), (Q, "um_mi1_frozen")), ((Q, "um_mi4_text"), (Q, "um_mi4_frozen")),
         ((Q, "um_knn_rag"), (Q, "um_frozen")), ((Q, "um_entity_rag"), (Q, "um_frozen")), ((Q, "um_frozen"), (Q, "pool_lse")), ((Q, "um_text"), (Q, "txt_cta_lse")),
         # --- tower
         ((Q, "ut_causal+meanpool"), (Q, "pool_lse")), ((Q, "ut_causal"), (Q, "frozen")), ((Q, "ut_causal+meanpool"), (Q, "um_frozen")),
         # --- the supermodel against what it is made of
         ((Q, "super_zs"), (Q, "super_base")), ((Q, "super_llm"), (Q, "super_zs")), ((Q, "super_llm"), (Q, "txt_cta_lse")), ((Q, "super_llm"), (Q, "um_text")),
         ((Q, "super_llm"), (Q, "pool_lse")), ((Q, "super_apex"), (Q, "super_llm")),
         ((Q, "super_llm+pop"), (Q, "super_llm")), ((Q, "super_llm+pop"), (Q, "txt_cta_lse+pop")), ((Q, "super_llm+pop"), (Q, "pool_lse+pop")), ((Q, "super_llm+pop"), (Q, "frozen+pop")),
         ((Q, "super_apex+pop"), (Q, "super_llm+pop")), ((Q, "txt_cta_lse+pop"), (Q, "pool_lse+pop"))]
direct = suite.cross_compare([S], [Q], pairs, WORK + "/direct_comparisons.md")
'''


def build(profile):
    large = profile == "large"
    cells = [md(LARGE_INTRO if large else SMALL_INTRO),
             md("### 0. Phụ thuộc và GPU"),
             code('''%pip install -q "transformers>=4.51" peft sentence-transformers
import subprocess
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout or "no GPU visible")
'''),
             md("### 1. Modules (ghi ra thư mục làm việc)")]
    cells += [writefile(m) for m in MODULES]
    cells += [md("### 2. Cấu hình"), cell_config(profile),
              md("### 3. Dữ liệu — " + ("MIND-large tải từ Hugging Face (cần Internet On)" if large else "tìm MIND trong /kaggle/input (cần cặp train + dev của CÙNG một dataset)")), cell_data(profile),
              md("### 4. Cấu hình run (MỘT mô hình)"), cell_runs(profile)]
    if not large:
        cells += [md("### 5. Kiểm tra giao thức (1–2 phút, không phải so sánh): BGE-small `frozen` phải khớp V10"),
                  code('''S_CHECK = None
if PROTOCOL_CHECK:
    try:
        suite.EQUIV_REFERENCE = None
        S_CHECK = suite.main([*COMMON, "--encoder", CHECK_ENCODER, "--work", WORK + "/bge_check", "--max-len", "256", "--stages", "controls",
                              "--repro", "0.6241,0.3909", "--repro-pop", "0.6533,0.4026,0.6769,0.4330"])
    except Exception as e:                       # the check must never stop the main run
        import traceback; traceback.print_exc()
        print("!! protocol check failed to run:", repr(e)[:200])
    gc.collect(); torch.cuda.empty_cache()
''')]
    else:
        cells += [code("S_CHECK = None   # MIND-large: V10 has no reference numbers\n")]
    n = 6 if not large else 5
    cells += [md(f"### {n}. MỘT mô hình, đủ kỹ thuật: controls → pooling → moredata → graph → usermodel → tower → **supermodel**"),
              code('''suite.EQUIV_REFERENCE = reference_encoder(ENCODER, 256)
S = suite.main([*ARGS, "--stages", STAGES])
gc.collect(); torch.cuda.empty_cache()
'''),
              md(f"### {n + 1}. Leaderboard (control = `frozen` của chính mô hình), so sánh trực tiếp từng kỹ thuật, forest plot"),
              code('''rows = suite.leaderboard([S], [TAG], (TAG, "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
rows_pop = suite.leaderboard([S], [TAG], (TAG, "frozen+pop"), WORK + "/leaderboard_pop.md", plot=WORK + "/forest_pop.png", pop=True)
''' + PAIRS),
              md(f"### {n + 2}. Báo cáo cuối (supermodel, thang kỹ thuật, bỏ-một-nhóm, từng kỹ thuật) và file gộp để gửi lại"),
              code('''print(final_report.build(S, WORK + "/FINAL_REPORT.md", WORK + "/final_tables.json", extra_suites=[S_CHECK] if S_CHECK is not None else ()))
p = os.path.join(WORK, TAG, "results.md")
if os.path.exists(p):
    print("=" * 30, TAG, "=" * 30)
    print(open(p, encoding="utf-8").read())
# seed noise of every trained row (a gap between two trained rows smaller than this is not a finding); std = sample std (n-1)
noise = ["# Seed-to-seed spread of the trained rows (test nDCG@10 of each seed; sample std with n-1; the paired CIs in the other tables do NOT include it)", "",
         "| row | seeds | nDCG@10 per seed | std | mean cosine between interests | attention TV between interests |", "|---|---|---|---|---|---|"]
pj = os.path.join(WORK, TAG, "results.json")
if os.path.exists(pj):
    for r in json.load(open(pj, encoding="utf-8")):
        if r.get("ndcg10_per_seed") and len(r["ndcg10_per_seed"]) > 1:
            f = lambda v: "-" if not v else ", ".join("-" if x is None else f"{x:.2f}" for x in v)
            noise.append(f"| {r['name']} | {len(r['ndcg10_per_seed'])} | {', '.join(f'{x:.4f}' for x in r['ndcg10_per_seed'])} | {r.get('ndcg10_std', float('nan')):.4f} | {f(r.get('interest_cos'))} | {f(r.get('attn_tv'))} |")
open(os.path.join(WORK, "noise.md"), "w", encoding="utf-8").write("\\n".join(noise) + "\\n")
print("\\n".join(noise))
print("tổng thời gian: %.2f h" % (common.CLOCK.elapsed() / 3600))
parts = [("FINAL_REPORT.md", WORK), ("leaderboard.md", WORK), ("leaderboard_pop.md", WORK), ("direct_comparisons.md", WORK), ("noise.md", WORK), ("timings.json", WORK + "/" + TAG),
         ("results.md", WORK + "/" + TAG)]
os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "SUMMARY_v14.md"), "w", encoding="utf-8") as out:
    for fn, d in parts:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out.write(f"\\n\\n## {fn}\\n\\n" + open(p, encoding="utf-8").read())
import shutil
for fn in ("forest.png", "forest_pop.png", "FINAL_REPORT.md", "final_tables.json", "leaderboard.md", "leaderboard_pop.md", "direct_comparisons.md", "noise.md"):
    if os.path.exists(os.path.join(WORK, fn)) and os.path.abspath(WORK) != os.path.abspath(RESULTS):
        shutil.copy(os.path.join(WORK, fn), os.path.join(RESULTS, fn))
print("-> gửi lại file:", os.path.join(RESULTS, "SUMMARY_v14.md"), f"({os.path.getsize(os.path.join(RESULTS, 'SUMMARY_v14.md')) / 1024:.0f} KB)")
'''),
              md("""### Cách đọc kết quả

**Bước 0 — kiểm tra bộ đo:** (bản small) dòng `frozen == V10 bge_zeroshot` trong mục 0 của `FINAL_REPORT.md` phải là MATCH; `popularity` / `frozen+pop` cũng vậy. Nếu không, dừng lại.

| Bạn thấy | Nghĩa là |
|---|---|
| `super_llm` > `super_base` với CI ghép cặp loại trừ 0 | ghép các kỹ thuật **có ích hơn** mô hình đơn tốt nhất (trọng số khớp trên validation, test chạm một lần) |
| `super_zs` ≈ `super_base` nhưng `super_llm` > `super_zs` | phần tăng đến từ reader học được, không phải từ việc ghép các scorer zero-shot |
| bảng bỏ-một-nhóm: bỏ nhóm X làm test tụt rõ (CI loại trừ 0) | nhóm X thực sự đóng góp trong mô hình ghép; nếu bỏ mà không tụt, nhóm đó thừa |
| `txt_cta_lse` > `txt_ctr_lse` (CI loại trừ 0) | category thật có thông tin chủ đề; nếu `txt_ctr_lse` cũng tăng ngang, phần tăng là do thêm chữ/định dạng (đối chứng này bật mặc định ở bản small, tắt ở bản MIND-large; đổi bằng `V11_CONTROLS`) |
| val tăng nhiều nhưng test không tăng | khớp trọng số quá mức; tin test, và nhìn khoảng cách val–test ở bảng trọng số |
| `super_apex` chỉ trên tập con tower | so với `super_llm` trên **đúng các impression đó** (bảng rung-tới-rung); không so con số tuyệt đối với dòng chạy toàn bộ |
| `+pop` | khung *streaming*: `pop` đếm click của các impression trước đó trong chính ngày test, không so được với số offline của bài báo |

**Giới hạn:** một dataset, một ngày test; supermodel ghép các thành phần mà bản thân chúng chọn trên cùng validation (trọng số có ~10 tham số trên hàng chục nghìn impression: val–test cho biết mức khớp quá);
reader / tower là tổng hợp nhiều seed ở mức điểm nhưng vẫn có nhiễu huấn luyện không nằm trong CI ghép cặp; bảng NRMS / NAML / Fastformer là số của lần chạy khác (khác seed, không ghép cặp)."""),
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
