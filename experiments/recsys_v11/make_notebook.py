"""Build the self-contained Kaggle notebook ``v11_full_suite_v7.ipynb`` from the module files.

Bump NOTEBOOK when the content changes: Kaggle caches notebooks by file name."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = "v11_full_suite_v7.ipynb"
MODULES = ["data.py", "metrics.py", "common.py", "rep_eval.py", "semgraph.py", "adapt.py", "pooling.py", "sid.py",
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


cells = [md("""# V11 full suite — so sánh mọi hướng LLM **và các baseline huấn luyện chuẩn** cho news recommendation trên MIND-small (1 lần Run All, ước ~7–8,5 giờ)

Mọi biến thể được đo trên **đúng giao thức V10** (validation theo ngày, `MINDsmall_dev` chỉ dùng 1 lần làm test), có **CI 95% ghép cặp** so với control,
tách **tin mới (cold) / tin cũ (warm)**, hiệu chỉnh **Bonferroni**, và chọn đại diện từng hướng **trên validation** (không chọn trên test).

**Vì sao có bản v7.** Ở V10, NRMS / NAML / Fastformer đạt AUC 0,60–0,64 và nDCG@10 0,36–0,40 — thấp hơn các bài báo công bố trên MIND-small (NRMS ≈ 0,66 / 0,41) khoảng 0,03–0,05 ở mọi chỉ số.
Nguyên nhân khả dĩ nằm ở công thức huấn luyện của V10 (64-d, vector từ ngẫu nhiên, 20 từ đầu của title+abstract và từ hiếm bị đổi thành PAD, 4 negative bốc một lần cho mọi epoch, lr 1e-3), không phải ở bộ đo.
Nên "hơn Fastformer của V10" chưa nói lên nhiều. Bản này thêm **baseline huấn luyện đúng công thức đã công bố** (cùng split, cùng evaluator) và một **thang ablation** để biết khoảng cách đến từ đâu.

| | Hướng (paper line) | Stage |
|---|---|---|
| **R** | **NRMS / NAML / Fastformer** huấn luyện theo công thức chuẩn (title regex 30 từ, GloVe-300, negative lấy lại mỗi epoch, lr 1e-4, mask padding) — hàng `nrms_ref`, `naml_ref`, `fastformer_ref` (+ `+pop`); `nrms_ref@1` = seed khác (sàn nhiễu huấn luyện); `nrms_ref_refit` = huấn luyện lại trên train_core **+ validation** với số epoch mà `nrms_ref` đã chọn (số so được với bài báo, vốn huấn luyện trên toàn bộ train; cột validation của hàng này lặp lại lượt chọn epoch, chỉ cột test có ý nghĩa) | `ref_nrms`, `ref_naml`, `ref_ff`, `ref_nrms_refit`, `ref_nrms_s1` |
| **R** | **Thang ablation** NRMS: `ladder0_v10recipe` (tái hiện công thức V10, phải gần NRMS của V10) → `ladder1_data` → `ladder2_recipe` → `nrms_ref` (+GloVe), bảng `ladder.md` | `ref_l0`, `ref_l1`, `ref_l2` |
| H1 | history-as-text, query có instruction (LLM làm user encoder, zero-shot) | `histquery` |
| H2 | co-click contrastive adaptation: head, **LoRA / DoRA**, **Matryoshka**, hard-negative theo impression (LLM2Rec KDD'25) | `head`, `encoder` |
| H3 | RAG: entity / kNN retrieval-augmented article representation | `graph` |
| H4 | GraphRAG-lite (Louvain, similar-user) và **H4b: tóm tắt community do LLM viết** (Microsoft GraphRAG) | `graph`, `graphrag_llm` |
| H5 | **LLM user tower tinh chỉnh LoRA** với mask **causal / bidirectional / soft** (nghiên cứu mask của arXiv 2602.10622; bidirectional kiểu LLM2Vec, COLM'24). `soft` = scheduler tuyến tính λ: 0→1, *không phải* Gradient-Guided Soft Masking của bài | `llm_user` |
| H6 | **LLM-augmented**: mở rộng bài báo + hồ sơ độc giả do LLM viết (KAR, RecSys'24) | `augment` |
| H7 | **Semantic IDs**: RQ-KMeans (như OneRec) + SID-GPT kiểu TIGER (NeurIPS'23) dùng làm *slate ranker*; tokenizer học được như LETTER (CIKM'24) **chưa** cài | `sid` |
| H8 | pooling: recency / **late interaction** (ColBERT) | `pooling` |
| H9 | **LLM judge zero-shot** xếp lại top-k (pointwise Yes/No; dòng LLMRank ECIR'24; LLM4Rerank WWW'25 dạng CoT **chưa** cài) | `rerank` |
| H10 | user model học được trên từng biểu diễn: V10 `llmenc_ca` (`um_*`) **và** user encoder Fastformer / NRMS / additive trên vector BGE đóng băng (`um_ff_*`, `um_nrms_*`) | `usermodel` |
| — | quét encoder: TF-IDF/LSA, BGE small/base/large, Qwen3-Embedding | `sweep` |

**Cách chạy:** (1) Settings → Accelerator **GPU (T4/P100)**, Internet **On**. (2) *Add Input* → dataset MIND có **cả** `MINDsmall_train` và `MINDsmall_dev`
(ví dụ `thinhhuynh3108/mindsmall`). (3) **Run All**. Lần đầu nên đặt `QUICK = True` để bắt lỗi môi trường rồi chạy đầy đủ; chỉ thử BGE: `V11_RUN_B=0`; bỏ phần baseline: `V11_REF=0`.
GloVe-300 tự tải từ Hugging Face (~480 MB); nếu bạn đã Add Input một dataset GloVe hoặc muốn chỉ đường dẫn: `V11_GLOVE_PATH=/kaggle/input/.../glove.6B.300d.txt`. Nếu mọi nguồn GloVe đều hỏng,
baseline vẫn chạy với vector từ ngẫu nhiên và các hàng được gắn nhãn `_noglove` (đừng so chúng với số công bố).

**Thời gian.** *Đã đo* ở lần chạy đầy đủ v4 trên Tesla T4 (chỉ dùng 1 GPU): **3,2 giờ** — A-core 17 phút · B 88 phút · A-phần còn lại 75 phút. *Ước (chưa đo)*: bản v6 (thêm mode tower bidir/soft, seed phụ, tower BGE, hàng `+pop`, test tower 30k, quét thêm encoder) ≈ 5,5–6 giờ;
phần baseline mới (8 mô hình) ≈ 1,7–2,2 giờ (mỗi mô hình bị chặn ở 35 phút, NAML nặng nhất vì conv trên 80 token/bài). Thứ tự chạy: **A-core → baseline → B (Qwen3) → A-phần còn lại**.
`BUDGET_H` (mặc định 9.5) là ngân sách đồng hồ chung: mọi vòng lặp dài tự dừng, stage nào không còn thời gian thì bị bỏ qua (kết quả luôn được ghi sau từng stage; chạy lại cell sẽ **resume** các stage đã xong, mỗi mô hình baseline là một stage riêng).

**Kết quả:** `results.md` của từng run, **DECISION SUMMARY** (đại diện từng hướng, chọn theo validation, Bonferroni; baseline *không* tham gia như một "hướng"), **HEAD-TO-HEAD**, **LEADERBOARD** chung (có cả baseline) + `forest.png`,
`ladder.md`, `direct_comparisons.md`, `test_arrays.npz` (metric theo từng impression), và **một file gộp `SUMMARY_v11.md`** — gửi file đó (cùng log nếu có lỗi) để đọc kết quả.

Mọi biến thể zero-shot và 3 baseline chính đều có **bản sinh đôi `+pop`** (gộp z-score với popularity online của V10, công thức đúng của `bge_zs_pop`).
**Lưu ý quan trọng:** `pop` đếm click của các impression *trước đó trong chính ngày test* (phản hồi streaming) nên các hàng `+pop` **không so được** với số offline trong bài báo; bảng *content only* mới so được.

Dòng đầu của kết quả là **reproduction check**: `frozen` phải khớp V10 `bge_zeroshot` (AUC 0.6241, nDCG@10 0.3909). Nếu in `MISMATCH` thì dừng và sửa trước khi tin bất kỳ dòng nào. Tương tự, `popularity` phải khớp (AUC 0.6533, nDCG@10 0.4026) và `frozen+pop` khớp `bge_zs_pop` (0.6769 / 0.4330).
Với baseline: `ladder0_v10recipe` phải gần NRMS của V10 (0.6181 / 0.3685, sai lệch < 0.015); nếu không, hãy đọc log epoch trước khi tin thang ablation."""),
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
RUN_B = bool(int(os.environ.get("V11_RUN_B", "1")))
RUN_REF = bool(int(os.environ.get("V11_REF", "1")))              # baseline NRMS/NAML/Fastformer chuẩn + thang ablation (ước ~1,5-2 giờ, chưa đo)
GLOVE_PATH = os.environ.get("V11_GLOVE_PATH", "")                # file GloVe-300 (txt/zip) nếu đã Add Input; để trống -> tự tìm trong /kaggle/input rồi tải từ Hugging Face
USE_REFERENCE = bool(int(os.environ.get("V11_REFERENCE", "1")))   # so HFEncoder với Sentence-Transformers trước khi chạy (đặt 0 để bỏ qua)
SWEEP = os.environ.get("V11_SWEEP", "BAAI/bge-base-en-v1.5" if QUICK else "BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5,Qwen/Qwen3-Embedding-0.6B")
MIND_TRAIN = os.environ.get("V11_MIND_TRAIN")                    # để None -> tự dò trong /kaggle/input
MIND_DEV = os.environ.get("V11_MIND_DEV")
DEVICE = os.environ.get("V11_DEVICE", "auto")
WORK = ("/kaggle/working/v11_out" if os.path.isdir("/kaggle/working") else "v11_out") + ("_quick" if QUICK else "")   # QUICK never mixes with a full run
CACHE = "/kaggle/temp/v11_cache" if os.path.isdir("/kaggle/temp") else os.path.join(WORK, "cache")   # embedding bài báo: dùng chung giữa các run
sys.path.insert(0, os.getcwd())
print(dict(QUICK=QUICK, BUDGET_H=BUDGET_H, ENCODER_A=ENCODER_A, ENCODER_B=ENCODER_B, GEN_MODEL=GEN_MODEL, RUN_REF=RUN_REF, WORK=WORK))
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

* **Run A** = BGE-small (đúng control V10): H1–H4, H7, H8, H2 (LoRA/DoRA), H10, quét encoder. Chạy làm **nhiều lần gọi** vào cùng thư mục, mỗi lần **resume** từ `state.pkl`:
  *A-core* (rẻ, quan trọng) → **baseline (R)** → *run B* → *A-phần còn lại* (LoRA/DoRA, user model, sweep). Thứ tự chỉ để ưu tiên ngân sách thời gian.
* **Run R** = NRMS / NAML / Fastformer chuẩn + thang ablation, cùng thư mục `bge` để các hàng nằm chung bảng với control (cùng impression ⇒ CI ghép cặp).
* **Run B** = Qwen3-Embedding-0.6B + LLM sinh văn bản: H5 (tower), H6, H9, H4b và lặp lại các stage rẻ với embedding LLM để xem kết luận có chuyển giao không.
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
# V10's own NRMS (kaggle_run.ipynb): AUC 0.6181, nDCG@10 0.3685 -> harness check of the first ablation rung
REF_ARGS = ["--ref-glove-path", GLOVE_PATH, "--ref-repro", "0.6181,0.3685"] + (["--ref-train-frac", "0.1", "--ref-max-epochs", "2"] if QUICK else [])

A_BASE = ["--encoder", ENCODER_A, "--work", WORK + "/bge", "--max-len", "256", "--q-max-len", "384", "--hist-k", "10",
          "--sweep-encoders", SWEEP, *repro, *REF_ARGS]
B_BASE = ["--encoder", ENCODER_B, "--work", WORK + "/qwen3", "--max-len", "256", "--q-max-len", "384", "--hist-k", "10", "--seeds", "1",
          "--hq-plain", "0", "--hq-subset", "1", "--gen-model", GEN_MODEL, "--um-views", "frozen,head,entity_rag", "--um-extra", "ff:frozen"]
A_QUICK = ["--seeds", "1", "--pairs", "20000", "--pairs-enc", "3000", "--enc-steps", "60", "--epochs-head", "1", "--betas", "0.5",
           "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5", "--bs-enc", "16", "--hq-plain", "0", "--sid-epochs", "1", "--sid-steps", "300",
           "--sid-val-n", "500", "--um-epochs", "2", "--um-seeds", "1", "--um-views", "frozen,head", "--um-extra", "ff:frozen,nrms:frozen",
           "--llm-user-steps", "60", "--llm-user-bs", "16", "--llm-val-n", "400", "--llm-test-n", "1000"]
B_QUICK = ["--seeds", "1", "--pairs", "20000", "--epochs-head", "1", "--betas", "0.5", "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5",
           "--llm-user-modes", "causal,soft", "--llm-user-steps", "60", "--llm-user-bs", "8", "--llm-val-n", "400", "--llm-test-n", "1000",
           "--gen-val-n", "60", "--gen-test-n", "150", "--augment-minutes", "4", "--rerank-val-n", "40", "--rerank-test-n", "100",
           "--rerank-minutes", "4", "--gr-max-comm", "30", "--um-epochs", "2", "--um-seeds", "1", "--um-views", "frozen"]
A1 = "controls,pooling,graph,head,histquery,sid"
A2 = A1 + ",encoder,llm_user,usermodel,sweep"
REF_STAGES = "ref_nrms,ref_naml,ref_ff,ref_nrms_refit,ref_l0,ref_l1,ref_l2,ref_nrms_s1"       # priority order: headline models, refit, the ladder, then the noise-floor seed
B_STAGES = "controls,histquery,llm_user,rerank,augment,graphrag_llm,pooling,graph,head,usermodel"
if QUICK:                                              # smoke test: every stage family runs once, at toy sizes (Louvain only inside graphrag_llm)
    A1 = "controls,pooling,head,histquery,sid"
    A2 = A1 + ",encoder,llm_user,usermodel"
    REF_STAGES = "ref_nrms,ref_naml,ref_ff,ref_nrms_refit,ref_l0,ref_l1"
    B_STAGES = "controls,histquery,llm_user,rerank,augment,graphrag_llm,usermodel"
A_ARGS, B_ARGS = [*COMMON, *A_BASE, *(A_QUICK if QUICK else [])], [*COMMON, *B_BASE, *(B_QUICK if QUICK else [])]
'''),
    md("### 5. Run A — phần lõi (H1, H2-head, H3, H4, H7, H8)"),
    code('''suite.EQUIV_REFERENCE = reference_encoder(ENCODER_A, 256)
S = suite.main([*A_ARGS, "--stages", A1])
gc.collect(); torch.cuda.empty_cache()
'''),
    md("""### 6. Run R — baseline NRMS / NAML / Fastformer theo công thức chuẩn và thang ablation (cùng thư mục `bge`)

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
    md("### 7. Run B — LLM embedding + LLM sinh văn bản (H5 tower causal/bidir/soft, H9 judge, H6 augment, H4b GraphRAG-LLM, …)"),
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
    md("### 8. Run A — phần còn lại (LoRA/DoRA, user model học được + reader Fastformer/NRMS, quét encoder); resume từ state.pkl"),
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
pairs = [((Q, "ut_causal+meanpool"), (B, "pool_lse")), ((Q, "ut_causal"), (B, "ut_native")), ((Q, "ut_causal+meanpool"), (B, "ut_native+meanpool")),
         ((Q, "ut_bidir"), (Q, "ut_causal")), ((Q, "ut_soft"), (Q, "ut_causal")), ((Q, "ut_bidir"), (Q, "ut_soft")),
         ((Q, "ut_causal@1"), (Q, "ut_causal")),                                   # same recipe, different seed: the training-noise floor of the tower
         ((B, "pool_lse"), (B, "um_frozen")), ((B, "um_knn_rag"), (B, "pool_lse")), ((Q, "ut_causal+meanpool"), (B, "um_knn_rag")),
         ((Q, "ut_causal+meanpool+pop"), (B, "pool_lse+pop")), ((Q, "ut_causal+meanpool+pop"), (B, "frozen+pop")), ((B, "pool_lse+pop"), (B, "frozen+pop")),
         # --- trained reference baselines: do the zero-shot / LLM methods beat properly trained NRMS / NAML / Fastformer?
         ((B, "nrms_ref"), (B, "ladder0_v10recipe")), ((B, "nrms_ref@1"), (B, "nrms_ref")),         # recipe effect; baseline noise floor
         ((B, "nrms_ref_refit"), (B, "nrms_ref")), ((B, "pool_lse"), (B, "nrms_ref_refit")), ((Q, "ut_causal+meanpool"), (B, "nrms_ref_refit")),   # the literature-comparable baseline (all training days)
         ((B, "naml_ref"), (B, "nrms_ref")), ((B, "fastformer_ref"), (B, "nrms_ref")),
         ((B, "frozen"), (B, "nrms_ref")), ((B, "frozen"), (B, "fastformer_ref")), ((B, "pool_lse"), (B, "nrms_ref")), ((B, "pool_lse"), (B, "fastformer_ref")),
         ((B, "um_frozen"), (B, "nrms_ref")), ((B, "um_knn_rag"), (B, "nrms_ref")), ((B, "um_knn_rag"), (B, "fastformer_ref")),
         ((Q, "ut_causal+meanpool"), (B, "nrms_ref")), ((Q, "ut_causal+meanpool"), (B, "fastformer_ref")),
         ((B, "nrms_ref+pop"), (B, "frozen+pop")), ((B, "pool_lse+pop"), (B, "nrms_ref+pop")), ((Q, "ut_causal+meanpool+pop"), (B, "nrms_ref+pop")),
         # --- the same attention modules on top of frozen BGE vectors
         ((B, "um_ff_frozen"), (B, "um_frozen")), ((B, "um_nrms_frozen"), (B, "um_frozen")), ((B, "um_ff_frozen"), (B, "um_nrms_frozen")),
         ((B, "um_ff_frozen"), (B, "fastformer_ref")), ((B, "um_ff_knn_rag"), (B, "um_knn_rag")), ((Q, "um_ff_frozen"), (Q, "um_frozen"))]
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
print("tổng thời gian: %.2f h" % (common.CLOCK.elapsed() / 3600))
print("files:", sorted(os.listdir(WORK)))
# one file with every table needed to read the run (attach it, plus the log if something failed)
parts = [("leaderboard.md", WORK), ("leaderboard_pop.md", WORK), ("direct_comparisons.md", WORK), ("ladder.md", WORK + "/bge"),
         ("results.md", WORK + "/bge"), ("results.md", WORK + "/qwen3"), ("timings.json", WORK + "/bge"), ("timings.json", WORK + "/qwen3")]
with open(os.path.join(WORK, "SUMMARY_v11.md"), "w", encoding="utf-8") as out:
    for fn, d in parts:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out.write(f"\\n\\n## {os.path.relpath(p, WORK)}\\n\\n" + open(p, encoding="utf-8").read())
print("-> gửi lại file:", os.path.join(WORK, "SUMMARY_v11.md"), f"({os.path.getsize(os.path.join(WORK, 'SUMMARY_v11.md')) / 1024:.0f} KB)")
'''),
    md("""### Cách đọc kết quả → hướng luận văn

**Bước 0 — kiểm tra bộ đo và baseline trước khi tin bất cứ hàng nào:** `frozen` khớp V10 (MATCH); `ladder0_v10recipe` gần NRMS của V10 (0.6181 / 0.3685); hàng `nrms_ref` nằm trong khoảng bài báo công bố cho MIND-small
(NRMS ≈ AUC 0,66 / nDCG@10 0,41; NAML ≈ 0,666–0,671 / 0,412–0,415 — con số lấy từ kết quả tìm kiếm, hãy đối chiếu PDF trước khi trích dẫn). Nếu `nrms_ref` thấp hơn nhiều so với khoảng đó,
baseline vẫn còn dưới chuẩn (xem nhật ký epoch, GloVe có được dùng không: hậu tố `_noglove`).

| Kết quả (CI loại trừ 0 trên test, sau Bonferroni) | Hướng nên theo |
|---|---|
| Biến thể LLM/zero-shot **thắng cả `nrms_ref`/`fastformer_ref`** (bảng `direct_comparisons.md`, content-only) | claim "vượt baseline đã huấn luyện chuẩn" — mạnh hơn nhiều so với "vượt Fastformer của V10" |
| Chỉ thắng baseline của V10, **thua** `nrms_ref` | claim cũ không đứng vững; báo cáo trung thực: biểu diễn LLM đóng băng ≈ baseline huấn luyện nhưng chưa vượt |
| `nrms_ref@1` vs `nrms_ref` chênh X | chênh lệch nhỏ hơn X giữa hai mô hình **huấn luyện** không phải phát hiện (sàn nhiễu seed; CI ghép cặp chỉ tính nhiễu mẫu test) |
| `ladder.md`: rung nào tăng nhiều nhất | nguồn chính của khoảng cách V10 ↔ bài báo (thứ tự rung ảnh hưởng đến Δ, có tương tác) |
| `um_ff_frozen` / `um_nrms_frozen` vs `um_frozen` | attention module nào là đáng giá khi đã có vector LLM (Fastformer / self-attention / candidate-aware) |
| **H5** (`ut_*`) thắng, nhất là `ut_soft`/`ut_bidir` > `ut_causal` | "Tinh chỉnh LLM làm user encoder + lịch trình attention mask" — đóng góp mới nhất, đúng dòng 2026 |
| **H7** (`sid_gpt*`) thắng, nhất là trên *cold* | "Semantic ID cho cold-start tin tức; generative model dùng làm slate ranker" |
| **H3/H4/H4b** thắng trên *cold* | "Retrieval-/Graph-augmented representation cho tin mới", mở rộng bằng LLM đọc ngữ cảnh |
| **H6/H9** thắng | "LLM-as-augmenter / LLM-as-reranker" (chi phí suy luận cao — nói rõ trong luận văn) |
| **H2** thắng | "Collaborative-aware adaptation của embedding LLM" |
| **H10** (`um_*`) không còn thắng | cải thiện biểu diễn bị user model học được "nuốt" — kết luận quan trọng cho luận văn |
| Thắng ở bảng *content only* nhưng **không** thắng ở bảng *with popularity* | kỹ thuật chỉ bù chỗ popularity đã làm — hợp cho **cold-start** (bài chưa có CTR), không hợp cho xếp hạng chung |
| Thắng cả hai bảng | đóng góp thật, không bị popularity "nuốt" |
| Không hướng nào | **Kết quả âm có giá trị**: mean-pool của embedding đóng băng đã gần trần trên MIND-small ⇒ MIND-large / EB-NeRD |

Lưu ý: dòng chạy trên **tập con** (cột `n test` nhỏ hơn 73k) so sánh với control **trên đúng các impression đó** (cột `ref nDCG@10`); con số tuyệt đối của chúng không so trực tiếp với dòng chạy toàn bộ.
Baseline huấn luyện trên `train_core` (≈ 80,7% số impression của MINDsmall_train; 10% ngày cuối giữ làm validation) và chọn epoch trên validation — các bài báo huấn luyện trên toàn bộ train, nên so với bài báo là hơi bất lợi cho baseline; `nrms_ref_refit` bù lại bằng cách huấn luyện lại trên cả validation với số epoch đã chọn (test vẫn chỉ chạm một lần)."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, NOTEBOOK)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
