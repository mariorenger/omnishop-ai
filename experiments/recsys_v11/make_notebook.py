"""Build the self-contained Kaggle notebook ``v11_full_suite_v3.ipynb`` from the module files.

Bump NOTEBOOK when the content changes: Kaggle caches notebooks by file name."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = "v11_full_suite_v3.ipynb"
MODULES = ["data.py", "metrics.py", "common.py", "rep_eval.py", "semgraph.py", "adapt.py", "pooling.py", "sid.py",
           "llm_user.py", "llm_gen.py", "usermodel.py", "stages_ext.py", "suite.py"]


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


cells = [md("""# V11 full suite — so sánh mọi hướng LLM cho news recommendation trên MIND-small (1 lần Run All, ~7–9 giờ)

Mọi biến thể được đo trên **đúng giao thức V10** (validation theo ngày, `MINDsmall_dev` chỉ dùng 1 lần làm test), có **CI 95% ghép cặp** so với control,
tách **tin mới (cold) / tin cũ (warm)**, hiệu chỉnh **Bonferroni**, và chọn đại diện từng hướng **trên validation** (không chọn trên test).

| | Hướng (paper line) | Stage |
|---|---|---|
| H1 | history-as-text, query có instruction (LLM làm user encoder, zero-shot) | `histquery` |
| H2 | co-click contrastive adaptation: head, **LoRA / DoRA**, **Matryoshka**, hard-negative theo impression (LLM2Rec KDD'25) | `head`, `encoder` |
| H3 | RAG: entity / kNN retrieval-augmented article representation | `graph` |
| H4 | GraphRAG-lite (Louvain, similar-user) và **H4b: tóm tắt community do LLM viết** (Microsoft GraphRAG) | `graph`, `graphrag_llm` |
| H5 | **LLM user tower tinh chỉnh LoRA** với mask **causal / bidirectional / soft** (nghiên cứu mask của arXiv 2602.10622; bidirectional kiểu LLM2Vec, COLM'24). `soft` = scheduler tuyến tính λ: 0→1, *không phải* Gradient-Guided Soft Masking của bài | `llm_user` |
| H6 | **LLM-augmented**: mở rộng bài báo + hồ sơ độc giả do LLM viết (KAR, RecSys'24) | `augment` |
| H7 | **Semantic IDs**: RQ-KMeans (như OneRec) + SID-GPT kiểu TIGER (NeurIPS'23) dùng làm *slate ranker*; tokenizer học được như LETTER (CIKM'24) **chưa** cài | `sid` |
| H8 | pooling: recency / **late interaction** (ColBERT) | `pooling` |
| H9 | **LLM judge zero-shot** xếp lại top-k (pointwise Yes/No; dòng LLMRank ECIR'24; LLM4Rerank WWW'25 dạng CoT **chưa** cài) | `rerank` |
| H10 | user model học được (V10 `llmenc_ca`) trên từng biểu diễn — kiểm tra cải thiện có còn khi có user model | `usermodel` |
| — | quét encoder: TF-IDF/LSA, BGE small/base/large, Qwen3-Embedding | `sweep` |

**Cách chạy:** (1) Settings → Accelerator **GPU (T4/P100)**, Internet **On**. (2) *Add Input* → dataset MIND có **cả** `MINDsmall_train` và `MINDsmall_dev`
(ví dụ `thinhhuynh3108/mindsmall`). (3) **Run All**. Lần đầu nên đặt `QUICK = True` (~60–90 phút, chủ yếu là encode toàn bộ 65k bài hai lần, Louvain và tải model) để bắt lỗi môi trường rồi chạy đầy đủ; muốn thử nhanh hơn nữa: `V11_RUN_B=0` (~30 phút, chỉ BGE).

**Thời gian ước tính trên T4 (CHƯA đo — ước lượng từ FLOPs):** A-core ≈ 1.1–1.5 h · B (Qwen3-Embedding + LLM) ≈ 3.8–4.8 h · A-phần còn lại ≈ 1.6–2.2 h ⇒ **~6.5–8.5 h**.
`BUDGET_H` (mặc định 9.5) là ngân sách đồng hồ chung: mọi vòng lặp dài tự dừng, stage nào không còn thời gian thì bị bỏ qua (kết quả luôn được ghi sau từng stage;
chạy lại cell sẽ **resume** các stage đã xong).

**Kết quả:** `results.md` của từng run, **DECISION SUMMARY** (đại diện từng hướng, chọn theo validation, Bonferroni), **HEAD-TO-HEAD**, **LEADERBOARD** chung + `forest.png`,
`test_arrays.npz` (metric theo từng impression để phân tích ghép cặp tiếp mà không chạy lại).

Dòng đầu của kết quả là **reproduction check**: `frozen` phải khớp V10 `bge_zeroshot` (AUC 0.6241, nDCG@10 0.3909). Nếu in `MISMATCH` thì dừng và sửa trước khi tin bất kỳ dòng nào."""),
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
QUICK = bool(int(os.environ.get("V11_QUICK", "0")))              # True: bản rút gọn (~60-90 phút) để bắt lỗi môi trường trên dữ liệu thật
BUDGET_H = float(os.environ.get("V11_BUDGET_H", "9.5"))          # ngân sách đồng hồ chung cho cả notebook (giờ)
ENCODER_A = os.environ.get("V11_ENCODER", "BAAI/bge-small-en-v1.5")        # run A: encoder nhỏ, đúng control của V10
ENCODER_B = os.environ.get("V11_ENCODER_B", "Qwen/Qwen3-Embedding-0.6B")   # run B: LLM embedding (decoder-only)
GEN_MODEL = os.environ.get("V11_GEN_MODEL", "Qwen/Qwen3-1.7B")             # LLM sinh văn bản / giám khảo
RUN_B = bool(int(os.environ.get("V11_RUN_B", "1")))
USE_REFERENCE = bool(int(os.environ.get("V11_REFERENCE", "1")))   # so HFEncoder với Sentence-Transformers trước khi chạy (đặt 0 để bỏ qua)
SWEEP = os.environ.get("V11_SWEEP", "BAAI/bge-base-en-v1.5" if QUICK else "BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5,Qwen/Qwen3-Embedding-0.6B")
MIND_TRAIN = os.environ.get("V11_MIND_TRAIN")                    # để None -> tự dò trong /kaggle/input
MIND_DEV = os.environ.get("V11_MIND_DEV")
DEVICE = os.environ.get("V11_DEVICE", "auto")
WORK = ("/kaggle/working/v11_out" if os.path.isdir("/kaggle/working") else "v11_out") + ("_quick" if QUICK else "")   # QUICK never mixes with a full run
CACHE = "/kaggle/temp/v11_cache" if os.path.isdir("/kaggle/temp") else os.path.join(WORK, "cache")   # embedding bài báo: dùng chung giữa các run
sys.path.insert(0, os.getcwd())
print(dict(QUICK=QUICK, BUDGET_H=BUDGET_H, ENCODER_A=ENCODER_A, ENCODER_B=ENCODER_B, GEN_MODEL=GEN_MODEL, WORK=WORK))
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

* **Run A** = BGE-small (đúng control V10): H1–H4, H7, H8, H2 (LoRA/DoRA), H10, quét encoder. Chạy làm **2 lần gọi**: *A-core* (rẻ, quan trọng) → *run B* → *A-phần còn lại*
  (LoRA/DoRA, user model, sweep). Lần gọi thứ hai **resume** từ `state.pkl`, nên thứ tự này chỉ để ưu tiên ngân sách thời gian cho phần LLM.
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
repro = ["--repro", "0.6241,0.3909", "--um-repro", "0.6494,0.3997"] if "bge-small-en-v1.5" in ENCODER_A else []

A_BASE = ["--encoder", ENCODER_A, "--work", WORK + "/bge", "--max-len", "256", "--q-max-len", "384", "--hist-k", "10",
          "--sweep-encoders", SWEEP, *repro]
B_BASE = ["--encoder", ENCODER_B, "--work", WORK + "/qwen3", "--max-len", "256", "--q-max-len", "384", "--hist-k", "10", "--seeds", "1",
          "--hq-plain", "0", "--hq-subset", "1", "--gen-model", GEN_MODEL, "--um-views", "frozen,head,entity_rag"]
A_QUICK = ["--seeds", "1", "--pairs", "20000", "--pairs-enc", "3000", "--enc-steps", "60", "--epochs-head", "1", "--betas", "0.5",
           "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5", "--bs-enc", "16", "--hq-plain", "0", "--sid-epochs", "1", "--sid-steps", "300",
           "--sid-val-n", "500", "--um-epochs", "2", "--um-seeds", "1", "--um-views", "frozen,head"]
B_QUICK = ["--seeds", "1", "--pairs", "20000", "--epochs-head", "1", "--betas", "0.5", "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5",
           "--llm-user-modes", "causal,soft", "--llm-user-steps", "60", "--llm-user-bs", "8", "--llm-val-n", "400", "--llm-test-n", "1000",
           "--gen-val-n", "60", "--gen-test-n", "150", "--augment-minutes", "4", "--rerank-val-n", "40", "--rerank-test-n", "100",
           "--rerank-minutes", "4", "--gr-max-comm", "30", "--um-epochs", "2", "--um-seeds", "1", "--um-views", "frozen"]
A1 = "controls,pooling,graph,head,histquery,sid"
A2 = A1 + ",encoder,usermodel,sweep"
B_STAGES = "controls,histquery,llm_user,rerank,augment,graphrag_llm,pooling,graph,head,usermodel"
if QUICK:                                              # smoke test: every stage family runs once, at toy sizes (Louvain only inside graphrag_llm)
    A1 = "controls,pooling,head,histquery,sid"
    A2 = A1 + ",encoder,usermodel"
    B_STAGES = "controls,histquery,llm_user,rerank,augment,graphrag_llm,usermodel"
A_ARGS, B_ARGS = [*COMMON, *A_BASE, *(A_QUICK if QUICK else [])], [*COMMON, *B_BASE, *(B_QUICK if QUICK else [])]
'''),
    md("### 5. Run A — phần lõi (H1, H2-head, H3, H4, H7, H8)"),
    code('''suite.EQUIV_REFERENCE = reference_encoder(ENCODER_A, 256)
S = suite.main([*A_ARGS, "--stages", A1])
gc.collect(); torch.cuda.empty_cache()
'''),
    md("### 6. Run B — LLM embedding + LLM sinh văn bản (H5 tower causal/bidir/soft, H9 judge, H6 augment, H4b GraphRAG-LLM, …)"),
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
    md("### 7. Run A — phần còn lại (LoRA/DoRA, user model học được, quét encoder); resume từ state.pkl"),
    code('''suite.EQUIV_REFERENCE = reference_encoder(ENCODER_A, 256)
try:
    S = suite.main([*A_ARGS, "--stages", A2])
except Exception as e:
    import traceback; traceback.print_exc()
    print("!! RUN A (second part) FAILED:", repr(e)[:300], "-> the results of the first part are still in", WORK + "/bge")
gc.collect(); torch.cuda.empty_cache()
'''),
    md("### 8. Leaderboard chung (một control toàn cục = BGE-small frozen, đúng V10) và forest plot"),
    code('''runs, tags = [S] + ([SB] if SB is not None else []), ["bge-small"] + (["qwen3-0.6b"] if SB is not None else [])
rows = suite.leaderboard(runs, tags, ("bge-small", "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
'''),
    md("### 9. Kết quả đầy đủ"),
    code('''for sub in ("bge", "qwen3"):
    p = os.path.join(WORK, sub, "results.md")
    if os.path.exists(p):
        print("=" * 30, sub, "=" * 30)
        print(open(p, encoding="utf-8").read())
print("tổng thời gian: %.2f h" % (common.CLOCK.elapsed() / 3600))
print("files:", sorted(os.listdir(WORK)))
'''),
    md("""### Cách đọc kết quả → hướng luận văn

| Kết quả (CI loại trừ 0 trên test, sau Bonferroni) | Hướng nên theo |
|---|---|
| **H5** (`ut_*`) thắng, nhất là `ut_soft`/`ut_bidir` > `ut_causal` | "Tinh chỉnh LLM làm user encoder + lịch trình attention mask" — đóng góp mới nhất, đúng dòng 2026 |
| **H7** (`sid_gpt*`) thắng, nhất là trên *cold* | "Semantic ID cho cold-start tin tức; generative model dùng làm slate ranker" |
| **H3/H4/H4b** thắng trên *cold* | "Retrieval-/Graph-augmented representation cho tin mới", mở rộng bằng LLM đọc ngữ cảnh |
| **H6/H9** thắng | "LLM-as-augmenter / LLM-as-reranker" (chi phí suy luận cao — nói rõ trong luận văn) |
| **H2** thắng | "Collaborative-aware adaptation của embedding LLM" |
| **H10** (`um_*`) không còn thắng | cải thiện biểu diễn bị user model học được "nuốt" — kết luận quan trọng cho luận văn |
| Không hướng nào | **Kết quả âm có giá trị**: mean-pool của embedding đóng băng đã gần trần trên MIND-small ⇒ MIND-large / EB-NeRD |

Lưu ý: dòng chạy trên **tập con** (cột `n test` nhỏ hơn 73k) so sánh với control **trên đúng các impression đó** (cột `ref nDCG@10`); con số tuyệt đối của chúng không so trực tiếp với dòng chạy toàn bộ."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, NOTEBOOK)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
