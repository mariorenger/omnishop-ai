"""Build the self-contained Kaggle notebook ``v13_suite_moredata.ipynb`` from the module files.

Bump NOTEBOOK when the content changes: Kaggle caches notebooks by file name."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = "v13_suite_moredata.ipynb"
MODULES = ["data.py", "metrics.py", "common.py", "rep_eval.py", "semgraph.py", "adapt.py", "fields.py", "pooling.py", "sid.py",
           "llm_user.py", "llm_gen.py", "refdata.py", "refmodels.py", "usermodel.py", "stages_ext.py", "stages_ref.py", "suite.py"]


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


cells = [md("""# V11 suite v13 — thêm dữ liệu (category, tên entity, lịch sử dài), multi-interest, baseline rút gọn còn Fastformer (1 lần Run All, ước ~7,5–8 giờ; `V11_FOCUS=1` ước ~2–2,5 giờ)

Mọi biến thể được đo trên **đúng giao thức V10** (validation theo ngày, `MINDsmall_dev` chỉ dùng 1 lần làm test), có **CI 95% ghép cặp** so với control,
tách **tin mới (cold) / tin cũ (warm)**, hiệu chỉnh **Bonferroni**, và chọn đại diện từng hướng **trên validation** (không chọn trên test).
Toàn bộ stage cũ của v7 (H1–H10, quét encoder) giữ nguyên, và `E0` vẫn là `title. abstract` nên `frozen` vẫn phải khớp V10 (AUC 0.6241 / nDCG@10 0.3909).

**Vì sao có bản v13.** Ở v7, vector đóng băng + pooling kiểu late-interaction (`pool_lse`) ngang baseline huấn luyện đúng cách, còn các kỹ thuật phức tạp hơn (RAG, GraphRAG, Semantic ID, LoRA, LLM judge) cho ≤ 0,005 hoặc âm.
v13 hỏi hai câu còn bỏ ngỏ, **từng kỹ thuật riêng lẻ** (chưa ghép thành mô hình lớn): (1) phần dữ liệu chưa từng đưa vào encoder có giúp không? (2) nhiều interest có hơn một vector người dùng không?

| | Nội dung | Stage |
|---|---|---|
| **F** | **Thêm dữ liệu.** *MIND-small không có body của bài* (`news.tsv` chỉ có category, subcategory, title, abstract, URL, entity); abstract đã nằm trong `E0` từ v1. Chưa bao giờ đưa vào encoder: **category/subcategory**, **tên entity**, **hơn 50 click gần nhất của người đọc**. Các hàng: `txt_t` (chỉ title ⇒ phần abstract đóng góp bao nhiêu), `txt_cta` (+ category), `txt_ctae` (+ tên entity), `txt_views` (title / abstract / category mã hoá riêng rồi ghép có trọng số, kiểu NAML nhưng vector đóng băng), mỗi biến thể với mean-pool (so với `frozen`) và late-interaction `*_lse` (so với `pool_lse`); `hist100_*`, `hist200_*` (đọc 100 / 200 click thay vì 50). Biến thể thắng trên validation thành view `text` cho các reader học được. | `moredata` |
| **H11** | **Multi-interest.** Không huấn luyện: `pool_km{2,3,5}` (gom click của người đọc thành K interest bằng k-means; K=1 chính là mean-pool, K ≥ số click chính là `pool_lse`, nên nằm giữa hai đầu đã đo). Học được: `um_mi1_*` (một vector, **đối chứng**), `um_mi4_*` (4 interest, điểm = soft-max qua interest), `um_mi4d_*` (thêm phạt "bất đồng" giữa các interest (bình phương cosine)); log in độ giống nhau giữa các interest để biết có bị "sập" về một vector không. | `pooling`, `usermodel` |
| **R** | **Baseline huấn luyện chỉ còn Fastformer** theo công thức chuẩn: `fastformer_ref` (**trung bình 2 seed**, + `+pop`) và `fastformer_ref_refit` (huấn luyện lại trên train_core + validation với số epoch trung bình đã chọn, số so được với bài báo). NRMS / NAML / thang ablation **tắt mặc định** (`V11_REF_ALL=1` bật lại); số đo v7 của chúng ở `RESULTS_v7.md`. | `ref_ff`, `ref_ff_refit` |
| H1–H10 | như v7 (history-as-text, LoRA/DoRA, RAG, GraphRAG, tower LoRA, KAR, Semantic ID, pooling, LLM judge, user model) | như v7 |

**Lưu ý khi so với baseline.** Fastformer ở đây chỉ đọc **title** (cài đặt của `refmodels.py`, cùng đầu vào với NRMS; chưa đối chiếu với bài báo gốc). Vì NAML, mô hình duy nhất ở v7 dùng cả abstract và category, đã bị bỏ, phép so cùng đầu vào là `txt_t` / `txt_t_lse` (vector chỉ title) với `fastformer_ref`; các hàng dùng abstract/category thì nhìn thấy nhiều dữ liệu hơn baseline.

**Cách chạy:** (1) Settings → Accelerator **GPU (T4/P100)**, Internet **On**. (2) *Add Input* → dataset MIND có **cả** `MINDsmall_train` và `MINDsmall_dev`
(ví dụ `thinhhuynh3108/mindsmall`). (3) **Run All**. Lần đầu nên đặt `QUICK = True` để bắt lỗi môi trường rồi chạy đầy đủ; chỉ thử BGE: `V11_RUN_B=0`; bỏ phần baseline: `V11_REF=0`;
**chỉ phần mới** (moredata + multi-interest + Fastformer, bỏ run B và các stage cũ): `V11_FOCUS=1`.
GloVe-300 tự tải từ Hugging Face (~480 MB); nếu bạn đã Add Input một dataset GloVe hoặc muốn chỉ đường dẫn: `V11_GLOVE_PATH=/kaggle/input/.../glove.6B.300d.txt`. Nếu mọi nguồn GloVe đều hỏng,
baseline vẫn chạy với vector từ ngẫu nhiên và các hàng được gắn nhãn `_noglove` (đừng so chúng với số công bố).

**Thời gian.** *Đã đo* ở v7 trên Tesla T4 (cả notebook 7,52 giờ): A-core 18,5 phút · baseline (8 mô hình) 114 phút, trong đó NRMS 16, NAML 18, **Fastformer 29** (chậm nhất), refit NRMS 15 · run B (Qwen3) 3,6 giờ, trong đó tower LoRA 2,5 giờ · A-phần còn lại 1,5 giờ (LoRA/DoRA 62 phút).
*Ước (chưa đo)* cho phần mới: `moredata` ≈ 4 phút (BGE) + ≈ 14 phút (Qwen3, 2 biến thể text); `pool_km` ≈ 3 phút mỗi run; user model thêm ≈ 25 phút (BGE) + ≈ 12 phút (Qwen3) do thêm view `text`, reader multi-interest và 3 seed trên `frozen`/`text`;
Fastformer 2 seed + refit ≈ 85 phút thay cho 114 phút. Tổng ≈ 7,5–8 giờ; `V11_FOCUS=1` ≈ 2–2,5 giờ. Thứ tự chạy: **A-core → baseline → B (Qwen3) → A-phần còn lại**.
`BUDGET_H` (mặc định 9.5) là ngân sách đồng hồ chung: mọi vòng lặp dài tự dừng, stage nào không còn thời gian thì bị bỏ qua (kết quả luôn được ghi sau từng stage; chạy lại cell sẽ **resume** các stage đã xong).

**Kết quả:** `results.md` của từng run, **DECISION SUMMARY** (đại diện từng hướng, chọn theo validation, Bonferroni; baseline *không* tham gia như một "hướng"), **HEAD-TO-HEAD**, **LEADERBOARD** chung (có cả baseline) + `forest.png`,
`direct_comparisons.md`, `test_arrays.npz` (metric theo từng impression), và **một file gộp `SUMMARY_v13.md`** — gửi file đó (cùng log nếu có lỗi) để đọc kết quả.

Mọi biến thể zero-shot và `fastformer_ref` đều có **bản sinh đôi `+pop`** (gộp z-score với popularity online của V10, công thức đúng của `bge_zs_pop`; với các hàng `*_lse` thì đối chứng là `pool_lse+pop`).
**Lưu ý quan trọng:** `pop` đếm click của các impression *trước đó trong chính ngày test* (phản hồi streaming) nên các hàng `+pop` **không so được** với số offline trong bài báo; bảng *content only* mới so được.

Dòng đầu của kết quả là **reproduction check**: `frozen` phải khớp V10 `bge_zeroshot` (AUC 0.6241, nDCG@10 0.3909). Nếu in `MISMATCH` thì dừng và sửa trước khi tin bất kỳ dòng nào. Tương tự, `popularity` phải khớp (AUC 0.6533, nDCG@10 0.4026) và `frozen+pop` khớp `bge_zs_pop` (0.6769 / 0.4330)."""),
         md("### 0. Phụ thuộc và GPU"),
         code('''%pip install -q "transformers>=4.51" peft sentence-transformers
import subprocess
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout or "no GPU visible")
'''),
         md("### 1. Modules (ghi ra thư mục làm việc)")]
cells += [writefile(m) for m in MODULES]
cells += [
    md("### 2. Cấu hình"),
    code('''import os, glob, sys, gc, time
QUICK = bool(int(os.environ.get("V11_QUICK", "0")))              # True: bản rút gọn để bắt lỗi môi trường trên dữ liệu thật
BUDGET_H = float(os.environ.get("V11_BUDGET_H", "9.5"))          # ngân sách đồng hồ chung cho cả notebook (giờ)
ENCODER_A = os.environ.get("V11_ENCODER", "BAAI/bge-small-en-v1.5")        # run A: encoder nhỏ, đúng control của V10
ENCODER_B = os.environ.get("V11_ENCODER_B", "Qwen/Qwen3-Embedding-0.6B")   # run B: LLM embedding (decoder-only)
GEN_MODEL = os.environ.get("V11_GEN_MODEL", "Qwen/Qwen3-1.7B")             # LLM sinh văn bản / giám khảo
FOCUS = bool(int(os.environ.get("V11_FOCUS", "0")))              # 1: chỉ phần mới (moredata, multi-interest, Fastformer): bỏ run B và các stage cũ, ước ~2-2,5 giờ
RUN_B = bool(int(os.environ.get("V11_RUN_B", "0" if FOCUS else "1")))
RUN_REF = bool(int(os.environ.get("V11_REF", "1")))              # baseline Fastformer chuẩn (2 seed + refit, ước ~85 phút)
REF_ALL = bool(int(os.environ.get("V11_REF_ALL", "0")))          # 1: chạy thêm NRMS / NAML / refit NRMS / thang ablation / nrms_ref@1 như v7 (+~1,4 giờ)
GLOVE_PATH = os.environ.get("V11_GLOVE_PATH", "")                # file GloVe-300 (txt/zip) nếu đã Add Input; để trống -> tự tìm trong /kaggle/input rồi tải từ Hugging Face
USE_REFERENCE = bool(int(os.environ.get("V11_REFERENCE", "1")))   # so HFEncoder với Sentence-Transformers trước khi chạy (đặt 0 để bỏ qua)
SWEEP = os.environ.get("V11_SWEEP", "BAAI/bge-base-en-v1.5" if QUICK else "BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5,Qwen/Qwen3-Embedding-0.6B")
MIND_TRAIN = os.environ.get("V11_MIND_TRAIN")                    # để None -> tự dò trong /kaggle/input
MIND_DEV = os.environ.get("V11_MIND_DEV")
DEVICE = os.environ.get("V11_DEVICE", "auto")
WORK = ("/kaggle/working/v11_out" if os.path.isdir("/kaggle/working") else "v11_out") + ("_quick" if QUICK else "")   # QUICK never mixes with a full run
CACHE = "/kaggle/temp/v11_cache" if os.path.isdir("/kaggle/temp") else os.path.join(WORK, "cache")   # embedding bài báo: dùng chung giữa các run
sys.path.insert(0, os.getcwd())
print(dict(QUICK=QUICK, FOCUS=FOCUS, BUDGET_H=BUDGET_H, ENCODER_A=ENCODER_A, ENCODER_B=ENCODER_B, GEN_MODEL=GEN_MODEL, RUN_B=RUN_B, RUN_REF=RUN_REF, REF_ALL=REF_ALL, WORK=WORK))
'''),
    md("### 3. Tìm MIND (cần cặp train + dev của CÙNG một dataset)"),
    code('''if not MIND_TRAIN:
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
'''),
    md("""### 4. Cấu hình các run

* **Run A** = BGE-small (đúng control V10): H1–H4, H7, H8, H11 (k-means interest), F (`moredata`), H2 (LoRA/DoRA), H10 (+ reader multi-interest), quét encoder. Chạy làm **nhiều lần gọi** vào cùng thư mục, mỗi lần **resume** từ `state.pkl`:
  *A-core* (rẻ, quan trọng, có cả phần mới) → **baseline (R)** → *run B* → *A-phần còn lại* (LoRA/DoRA, user model, sweep). Thứ tự chỉ để ưu tiên ngân sách thời gian.
* **Run R** = Fastformer chuẩn (2 seed) + refit, cùng thư mục `bge` để các hàng nằm chung bảng với control (cùng impression ⇒ CI ghép cặp). `V11_REF_ALL=1` thêm NRMS / NAML / thang ablation.
* **Run B** = Qwen3-Embedding-0.6B + LLM sinh văn bản: stage rẻ và phần mới chạy trước (pooling, moredata với `t`,`cta`, user model), rồi H5 (tower), H9 judge, H6 augment, H4b — các stage đắt nhất để sau.
"""),
    code('''import torch
import suite, common
common.CLOCK.reset()                                   # chạy lại cell trong cùng kernel: đồng hồ ngân sách bắt đầu lại

def reference_encoder(name, max_len):
    """Tham chiếu Sentence-Transformers (chỉ 64 bài) để bắt lỗi pooling/EOS/prefix. Trả None nếu không dùng được."""
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
repro = ["--repro", "0.6241,0.3909", "--um-repro", "0.6494,0.3997", "--repro-pop", "0.6533,0.4026,0.6769,0.4330"] if "bge-small-en-v1.5" in ENCODER_A else []
# Fastformer chuẩn: 2 seed (trung bình) để baseline có phương sai thấp hơn một lần chạy; 70 phút = trần 35 phút mỗi seed (v7 đo: 29 phút/seed trên T4)
REF_ARGS = ["--ref-glove-path", GLOVE_PATH, "--ref-seeds", "2", "--ref-model-minutes", "70"]
if REF_ALL:
    REF_ARGS += ["--ref-repro", "0.6181,0.3685"]       # NRMS của V10 (kaggle_run.ipynb): harness check của rung đầu thang ablation
if QUICK:
    REF_ARGS += ["--ref-train-frac", "0.1", "--ref-max-epochs", "2", "--ref-seeds", "1"]

A_BASE = ["--encoder", ENCODER_A, "--work", WORK + "/bge", "--max-len", "256", "--q-max-len", "384", "--hist-k", "10",
          "--sweep-encoders", SWEEP, "--um-seeds-focus", "3", *repro, *REF_ARGS]
B_BASE = ["--encoder", ENCODER_B, "--work", WORK + "/qwen3", "--max-len", "256", "--q-max-len", "384", "--hist-k", "10", "--seeds", "1",
          "--hq-plain", "0", "--hq-subset", "1", "--gen-model", GEN_MODEL, "--um-views", "frozen,head,entity_rag,text", "--um-seeds-focus", "2",
          "--um-extra", "ff:frozen,mi1:frozen,mi4:frozen,mi4d:frozen", "--text-modes", "t,cta", "--hist-lens", "100"]
A_QUICK = ["--seeds", "1", "--pairs", "20000", "--pairs-enc", "3000", "--enc-steps", "60", "--epochs-head", "1", "--betas", "0.5",
           "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5", "--bs-enc", "16", "--hq-plain", "0", "--sid-epochs", "1", "--sid-steps", "300",
           "--sid-val-n", "500", "--um-epochs", "2", "--um-seeds", "1", "--um-seeds-focus", "1", "--um-views", "frozen,head,text",
           "--um-extra", "ff:frozen,mi1:frozen,mi4:frozen,mi4d:frozen,mi4:text", "--text-modes", "t,cta,views", "--hist-lens", "100", "--km-ks", "2",
           "--llm-user-steps", "60", "--llm-user-bs", "16", "--llm-val-n", "400", "--llm-test-n", "1000"]
B_QUICK = ["--seeds", "1", "--pairs", "20000", "--epochs-head", "1", "--betas", "0.5", "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5",
           "--llm-user-modes", "causal,soft", "--llm-user-steps", "60", "--llm-user-bs", "8", "--llm-val-n", "400", "--llm-test-n", "1000",
           "--gen-val-n", "60", "--gen-test-n", "150", "--augment-minutes", "4", "--rerank-val-n", "40", "--rerank-test-n", "100",
           "--rerank-minutes", "4", "--gr-max-comm", "30", "--um-epochs", "2", "--um-seeds", "1", "--um-seeds-focus", "1", "--um-views", "frozen,text",
           "--um-extra", "ff:frozen,mi4:frozen", "--text-modes", "t", "--km-ks", "2"]
A1 = "controls,pooling,moredata,graph,head,histquery,sid"
A2 = A1 + ",encoder,llm_user,usermodel,sweep"
REF_STAGES = "ref_ff,ref_ff_refit" + (",ref_nrms,ref_naml,ref_nrms_refit,ref_l0,ref_l1,ref_l2,ref_nrms_s1" if REF_ALL else "")      # priority order: the headline model, its refit, then the optional v7 baselines
B_STAGES = "controls,pooling,moredata,graph,head,usermodel,histquery,llm_user,rerank,augment,graphrag_llm"                            # cheap and new first, the 2,5-hour tower and the generation stages after
if QUICK:                                              # smoke test: every stage family runs once, at toy sizes (Louvain only inside graphrag_llm)
    A1 = "controls,pooling,moredata,head,histquery,sid"
    A2 = A1 + ",encoder,llm_user,usermodel"
    REF_STAGES = "ref_ff,ref_ff_refit" + (",ref_nrms,ref_naml,ref_nrms_refit,ref_l0,ref_l1" if REF_ALL else "")
    B_STAGES = "controls,pooling,moredata,usermodel,histquery,llm_user,rerank,augment,graphrag_llm"
if FOCUS:                                              # only what is new: moredata, multi-interest, the Fastformer reference
    A1, A2 = "controls,pooling,moredata", "controls,pooling,moredata,usermodel"
    B_STAGES = "controls,pooling,moredata,usermodel"
A_ARGS, B_ARGS = [*COMMON, *A_BASE, *(A_QUICK if QUICK else [])], [*COMMON, *B_BASE, *(B_QUICK if QUICK else [])]
if FOCUS:                                              # the user-model views that exist without the old stages
    A_ARGS += ["--um-views", "frozen,text", "--um-extra", "ff:frozen,mi1:frozen,mi4:frozen,mi4d:frozen,mi4:text"]
    B_ARGS += ["--um-views", "frozen,text", "--um-extra", "ff:frozen,mi1:frozen,mi4:frozen,mi4d:frozen"]
'''),
    md("### 5. Run A — phần lõi (H1, H2-head, H3, H4, H7, H8, H11 k-means interest, F `moredata`)"),
    code('''suite.EQUIV_REFERENCE = reference_encoder(ENCODER_A, 256)
S = suite.main([*A_ARGS, "--stages", A1])
gc.collect(); torch.cuda.empty_cache()
'''),
    md("""### 6. Run R — baseline Fastformer theo công thức chuẩn: 2 seed + refit (cùng thư mục `bge`; `V11_REF_ALL=1` thêm NRMS / NAML / thang ablation)

Mỗi mô hình là một stage; nếu GloVe không tải được, mô hình vẫn chạy với vector ngẫu nhiên và mang hậu tố `_noglove`."""),
    code('''if RUN_REF:
    try:
        suite.EQUIV_REFERENCE = None                    # embedding đã kiểm ở run A
        S = suite.main([*A_ARGS, "--stages", REF_STAGES])
    except Exception as e:
        import traceback; traceback.print_exc()
        print("!! RUN R FAILED:", repr(e)[:300], "-> continuing with the remaining runs")
    gc.collect(); torch.cuda.empty_cache()
'''),
    md("### 7. Run B — LLM embedding (pooling, moredata, user model rẻ chạy trước) rồi LLM sinh văn bản (H5 tower causal/bidir/soft, H9 judge, H6 augment, H4b GraphRAG-LLM, …)"),
    code('''SB = None
if RUN_B:
    try:
        suite.EQUIV_REFERENCE = reference_encoder(ENCODER_B, 256)
        SB = suite.main([*B_ARGS, "--stages", B_STAGES])
    except Exception as e:                       # e.g. model download failed: keep the A results and still run the rest
        import traceback; traceback.print_exc()
        print("!! RUN B FAILED:", repr(e)[:300], "-> continuing with the remaining runs")
    gc.collect(); torch.cuda.empty_cache()
'''),
    md("### 8. Run A — phần còn lại (LoRA/DoRA, user model học được + reader Fastformer / multi-interest, quét encoder); resume từ state.pkl"),
    code('''suite.EQUIV_REFERENCE = reference_encoder(ENCODER_A, 256)
try:
    S = suite.main([*A_ARGS, "--stages", A2])
except Exception as e:
    import traceback; traceback.print_exc()
    print("!! RUN A (second part) FAILED:", repr(e)[:300], "-> the results of the first part are still in", WORK + "/bge")
gc.collect(); torch.cuda.empty_cache()
'''),
    md("### 9. Leaderboard chung (control toàn cục = BGE-small frozen, đúng V10, có cả baseline) và forest plot; bảng thứ hai là các biến thể gộp với popularity (control = `frozen+pop` = V10 `bge_zs_pop`)"),
    code('''runs, tags = [S] + ([SB] if SB is not None else []), ["bge-small"] + (["qwen3-0.6b"] if SB is not None else [])
rows = suite.leaderboard(runs, tags, ("bge-small", "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
rows_pop = suite.leaderboard(runs, tags, ("bge-small", "frozen+pop"), WORK + "/leaderboard_pop.md", plot=WORK + "/forest_pop.png", pop=True)
Q, B = "qwen3-0.6b", "bge-small"
pairs = [# --- H8 / H11: where do k-means interests sit between the mean-pool control and late interaction?
         ((B, "pool_lse"), (B, "frozen")), ((B, "pool_km2"), (B, "pool_lse")), ((B, "pool_km3"), (B, "pool_lse")), ((B, "pool_km5"), (B, "pool_lse")),
         ((B, "pool_km3"), (B, "frozen")), ((Q, "pool_km3"), (Q, "pool_lse")),
         # --- F: what do the extra fields add, under each pooling rule? (txt_t = title only: the same input as the Fastformer reference)
         ((B, "txt_t"), (B, "frozen")), ((B, "txt_cta"), (B, "frozen")), ((B, "txt_ctae"), (B, "txt_cta")), ((B, "txt_views"), (B, "txt_cta")),
         ((B, "txt_t_lse"), (B, "pool_lse")), ((B, "txt_cta_lse"), (B, "pool_lse")), ((B, "txt_ctae_lse"), (B, "txt_cta_lse")), ((B, "txt_views_lse"), (B, "txt_cta_lse")),
         ((B, "hist100_lse"), (B, "pool_lse")), ((B, "hist200_lse"), (B, "pool_lse")), ((B, "hist100_mean"), (B, "frozen")),
         ((Q, "txt_t"), (Q, "frozen")), ((Q, "txt_cta"), (Q, "frozen")), ((Q, "txt_t_lse"), (Q, "pool_lse")), ((Q, "txt_cta_lse"), (Q, "pool_lse")),
         ((Q, "pool_lse"), (Q, "frozen")),
         # --- the trained Fastformer reference (2 seeds; the refit saw the validation days too, like the published numbers)
         ((B, "fastformer_ref_refit"), (B, "fastformer_ref")), ((B, "frozen"), (B, "fastformer_ref")), ((B, "pool_lse"), (B, "fastformer_ref")),
         ((B, "txt_t_lse"), (B, "fastformer_ref")), ((B, "txt_cta_lse"), (B, "fastformer_ref")),
         ((B, "pool_lse"), (B, "fastformer_ref_refit")), ((B, "txt_t_lse"), (B, "fastformer_ref_refit")), ((B, "txt_cta_lse"), (B, "fastformer_ref_refit")),
         ((B, "fastformer_ref+pop"), (B, "frozen+pop")), ((B, "pool_lse+pop"), (B, "fastformer_ref+pop")), ((B, "txt_cta_lse+pop"), (B, "pool_lse+pop")),
         ((B, "um_frozen"), (B, "fastformer_ref")), ((B, "um_ff_frozen"), (B, "fastformer_ref")),
         # --- H11 learned readers: single vector (mi1) vs several interests (mi4, mi4d) vs the V10 reader vs the best zero-shot rule
         ((B, "um_mi1_frozen"), (B, "um_frozen")), ((B, "um_mi4_frozen"), (B, "um_mi1_frozen")), ((B, "um_mi4d_frozen"), (B, "um_mi4_frozen")),
         ((B, "um_mi4_frozen"), (B, "um_frozen")), ((B, "um_mi4_frozen"), (B, "um_ff_frozen")), ((B, "um_mi4_frozen"), (B, "pool_lse")),
         ((B, "um_mi4_frozen"), (B, "fastformer_ref")), ((B, "um_mi4_frozen"), (B, "pool_km3")),
         ((B, "um_text"), (B, "um_frozen")), ((B, "um_mi4_text"), (B, "um_mi4_frozen")), ((B, "um_mi4_text"), (B, "pool_lse")),
         ((B, "pool_lse"), (B, "um_frozen")), ((B, "um_ff_frozen"), (B, "um_frozen")), ((B, "um_ff_knn_rag"), (B, "um_knn_rag")),
         ((B, "um_knn_rag"), (B, "pool_lse")),
         ((Q, "um_mi1_frozen"), (Q, "um_frozen")), ((Q, "um_mi4_frozen"), (Q, "um_mi1_frozen")), ((Q, "um_mi4d_frozen"), (Q, "um_mi4_frozen")),
         ((Q, "um_mi4_frozen"), (Q, "pool_lse")), ((Q, "um_ff_frozen"), (Q, "um_frozen")), ((Q, "um_text"), (Q, "um_frozen")),
         # --- H5 tower and the rest, carried over from v7
         ((Q, "ut_causal+meanpool"), (B, "pool_lse")), ((Q, "ut_causal"), (B, "ut_native")), ((Q, "ut_causal+meanpool"), (B, "ut_native+meanpool")),
         ((Q, "ut_bidir"), (Q, "ut_causal")), ((Q, "ut_soft"), (Q, "ut_causal")), ((Q, "ut_bidir"), (Q, "ut_soft")),
         ((Q, "ut_causal@1"), (Q, "ut_causal")),                                   # same recipe, different seed: the training-noise floor of the tower
         ((Q, "ut_causal+meanpool"), (B, "um_knn_rag")), ((Q, "ut_causal+meanpool"), (B, "fastformer_ref")), ((Q, "ut_causal+meanpool"), (B, "fastformer_ref_refit")),
         ((Q, "ut_causal+meanpool+pop"), (B, "pool_lse+pop")), ((Q, "ut_causal+meanpool+pop"), (B, "frozen+pop")), ((B, "pool_lse+pop"), (B, "frozen+pop")),
         ((Q, "ut_causal+meanpool+pop"), (B, "fastformer_ref+pop")),
         # --- only with V11_REF_ALL=1 (rows missing otherwise are skipped): the v7 baselines
         ((B, "nrms_ref"), (B, "ladder0_v10recipe")), ((B, "nrms_ref@1"), (B, "nrms_ref")), ((B, "nrms_ref_refit"), (B, "nrms_ref")),
         ((B, "naml_ref"), (B, "nrms_ref")), ((B, "fastformer_ref"), (B, "nrms_ref")), ((B, "pool_lse"), (B, "nrms_ref_refit")), ((B, "pool_lse"), (B, "naml_ref"))]
direct = suite.cross_compare(runs, tags, pairs, WORK + "/direct_comparisons.md")
'''),
    md("### 10. Kết quả đầy đủ và file gộp để gửi lại"),
    code('''for sub in ("bge", "qwen3"):
    p = os.path.join(WORK, sub, "results.md")
    if os.path.exists(p):
        print("=" * 30, sub, "=" * 30)
        print(open(p, encoding="utf-8").read())
lad = os.path.join(WORK, "bge", "ladder.md")
if os.path.exists(lad):
    print(open(lad, encoding="utf-8").read())
# seed noise of every trained row (a gap between two trained rows smaller than this is not a finding) and the multi-interest collapse diagnostics
import json
noise = ["# Seed-to-seed spread of the trained rows (test nDCG@10 of each seed; the paired CIs in the other tables do NOT include it)", "",
         "| run | row | seeds | nDCG@10 per seed | std | mean cosine between interests | attention TV between interests |", "|---|---|---|---|---|---|---|"]
for sub in ("bge", "qwen3"):
    p = os.path.join(WORK, sub, "results.json")
    if os.path.exists(p):
        for r in json.load(open(p, encoding="utf-8")):
            if r.get("ndcg10_per_seed") and len(r["ndcg10_per_seed"]) > 1:
                f = lambda v: "-" if not v else ", ".join("-" if x is None else f"{x:.2f}" for x in v)
                noise.append(f"| {sub} | {r['name']} | {len(r['ndcg10_per_seed'])} | {', '.join(f'{x:.4f}' for x in r['ndcg10_per_seed'])} | {r.get('ndcg10_std', float('nan')):.4f} | {f(r.get('interest_cos'))} | {f(r.get('attn_tv'))} |")
open(os.path.join(WORK, "noise.md"), "w", encoding="utf-8").write("\\n".join(noise) + "\\n")
print("\\n".join(noise))
try:
    audit = S.kv.get("data_audit")
    if audit:
        json.dump(audit, open(os.path.join(WORK, "bge", "data_audit.json"), "w"), indent=1)
        print("data audit:", audit)
except Exception as e:
    print("no data audit:", repr(e)[:80])
print("tổng thời gian: %.2f h" % (common.CLOCK.elapsed() / 3600))
print("files:", sorted(os.listdir(WORK)))
# one file with every table needed to read the run (attach it, plus the log if something failed)
parts = [("leaderboard.md", WORK), ("leaderboard_pop.md", WORK), ("direct_comparisons.md", WORK), ("noise.md", WORK), ("data_audit.json", WORK + "/bge"), ("ladder.md", WORK + "/bge"),
         ("results.md", WORK + "/bge"), ("results.md", WORK + "/qwen3"), ("timings.json", WORK + "/bge"), ("timings.json", WORK + "/qwen3")]
with open(os.path.join(WORK, "SUMMARY_v13.md"), "w", encoding="utf-8") as out:
    for fn, d in parts:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out.write(f"\\n\\n## {os.path.relpath(p, WORK)}\\n\\n" + open(p, encoding="utf-8").read())
print("-> gửi lại file:", os.path.join(WORK, "SUMMARY_v13.md"), f"({os.path.getsize(os.path.join(WORK, 'SUMMARY_v13.md')) / 1024:.0f} KB)")
'''),
    md("""### Cách đọc kết quả → hướng luận văn

**Bước 0 — kiểm tra bộ đo trước khi tin bất cứ hàng nào:** `frozen` khớp V10 (MATCH), `popularity` và `frozen+pop` khớp V10; `um_frozen` trong 0,01 của `llmenc_ca` của V10; `fastformer_ref` nằm gần số v7 đã đo (AUC ≈ 0,6455 / nDCG@10 ≈ 0,3926 với 1 seed, sai khác giữa hai seed cỡ 0,007) và dưới số bài báo (Fastformer ≈ 0,66 / 0,41; con số lấy từ kết quả tìm kiếm, hãy đối chiếu PDF trước khi trích dẫn);
`fastformer_ref_refit` mới là số cùng điều kiện huấn luyện với bài báo (toàn bộ ngày train). Nếu `fastformer_ref` thấp hơn nhiều, đọc nhật ký epoch và xem GloVe có được dùng không (hậu tố `_noglove`).

| Kết quả (CI loại trừ 0 trên test, sau Bonferroni nếu là đại diện hướng) | Ý nghĩa |
|---|---|
| `txt_t` ≈ `frozen`, `txt_t_lse` ≈ `pool_lse` | abstract không thêm gì khi chỉ có title đã đủ (hoặc ngược lại: `frozen` > `txt_t` nghĩa là abstract giúp — đọc dấu của Δ) |
| `txt_cta*` > `frozen` / `pool_lse` | **category giúp**: tín hiệu chủ đề tường minh mà vector title+abstract chưa nén hết; kiểm tra thêm ở cột cold AUC (bài mới có category nhưng chưa có click) |
| `txt_ctae*` ≈ `txt_cta*` | tên entity đã nằm trong title/abstract nên không thêm thông tin |
| `txt_views*` > `txt_cta*` | tách title / abstract / category thành các view có trọng số tốt hơn nhét chung một chuỗi |
| `hist100_*`, `hist200_*` ≈ `frozen` / `pool_lse` | click cũ hơn 50 không thêm gì (nhiều người đọc có ít hơn 50 click; xem `data_audit`) |
| `pool_km{K}` > `pool_lse` | gom click thành vài interest tốt hơn coi mỗi click là một interest; nếu < `pool_lse`, soft-max trên từng click đã là multi-interest tốt nhất |
| `um_mi4_*` > `um_mi1_*` (CI loại trừ 0 **và** hơn nhiễu seed ở `noise.md`) | **nhiều interest hơn một vector** trong reader học được |
| `um_mi4_*` ≈ `um_mi1_*` và `attention TV` ≈ 0 / `interest cosine` ≈ 1 | các interest đã sập về một: kết luận chưa đủ, thử `um_mi4d_*` (có regulariser) |
| `um_mi4_*` ≈ `um_mi1_*` và các interest khác nhau rõ | kết luận âm sạch: nhiều interest không giúp trên MIND-small với vector đóng băng |
| `um_mi4_*` hoặc `pool_km*` **thắng `fastformer_ref_refit`** | claim "vượt baseline huấn luyện chuẩn, cùng điều kiện" (nhưng nhớ: Fastformer chỉ đọc title) |
| `*_lse` / `um_*` chỉ thắng `fastformer_ref` (không refit) | chưa đủ: baseline thiệt thòi vì không được huấn luyện trên các ngày validation |
| `fastformer_ref`: hai seed chênh X | chênh nhỏ hơn X giữa hai mô hình **huấn luyện** không phải phát hiện (CI ghép cặp chỉ tính nhiễu mẫu test) |
| Thắng ở bảng *content only* nhưng **không** thắng ở bảng *with popularity* | kỹ thuật chỉ bù chỗ popularity đã làm — hợp cho **cold-start**, không hợp cho xếp hạng chung |
| H1–H10 | đọc như v7: `RESULTS_v7.md` ghi số đo và kết luận (late-interaction pooling là kết quả sạch nhất; RAG/GraphRAG/Semantic ID/judge/augment ≤ 0,005 hoặc âm; mask của tower chưa kết luận được vì nhiễu seed) |
| Không hướng mới nào | **Kết quả âm có giá trị**: bản v7 đã gần trần của vector đóng băng trên MIND-small ⇒ MIND-large / EB-NeRD (bộ có body thật) |

Lưu ý: dòng chạy trên **tập con** (cột `n test` nhỏ hơn 73k) so sánh với control **trên đúng các impression đó** (cột `ref nDCG@10`); con số tuyệt đối của chúng không so trực tiếp với dòng chạy toàn bộ.
Hàng `um_*` trung bình 3 seed (view `frozen`, `text`) hoặc 2 seed (các view khác); `fastformer_ref` trung bình 2 seed. Baseline huấn luyện trên `train_core` (≈ 80,7% số impression của MINDsmall_train; 10% ngày cuối giữ làm validation) và chọn epoch trên validation;
`fastformer_ref_refit` huấn luyện lại trên cả validation với số epoch trung bình đã chọn (test vẫn chỉ chạm một lần; cột validation của hàng này lặp lại lượt chọn epoch, chỉ cột test có ý nghĩa)."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, NOTEBOOK)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
