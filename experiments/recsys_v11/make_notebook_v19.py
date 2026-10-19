"""Build the two MIND-large notebooks of v19.  They run side by side on one Kaggle account (30-hour weekly GPU quota, two sessions at once, "GPU T4 x2"),
~10.5 hours each:

  v19a_mindlarge_plusreaders.ipynb  the v16b pipeline (in-run CONTROL: BGE-large, Qwen3-0.6B + BGE-large, Qwen3-0.6B with supermodel + stacker) + the readers
                                    trained harder (the v18b recipe without 'sca': all training samples, 6 epochs, decaying learning rate) on BGE-large and on
                                    Qwen3 + BGE-large, then the supermodel and the stacker again with them (TREATMENT).  The LLM-text methods H6 (KAR-style
                                    LLM expansions of articles and LLM-written reader profiles) and H9 (an LLM judging candidates, Yes/No) run on subsets inside
                                    the main run (they are not inputs of the fusion).
  v19b_mindlarge_h5tower.ipynb      the same control + H5: the LoRA-tuned LLM user tower that reads the reader's last 10 headlines AS TEXT, trained for ~3 hours
                                    (v14 on MIND-large: 800 steps, equal to the zero-shot pool_lse with validation still rising) and scored on the WHOLE validation and
                                    test split, so it is exported as a reader for the supermodel and the stacker (TREATMENT); H1 (the same LLM, zero-shot,
                                    history as text) on a subset.

v18b's lesson is built in: only the stacker has the large time gate (--min-stack-s), never the supermodel, and every run's exported scores are written to the
notebook's output (results_v19x/scores/<run>/) so that a later fusion-only run can combine both treatments without re-training anything.
Bump the notebook names when the content changes: Kaggle caches notebooks by file name."""
import json
import os

import make_notebook_v17 as V17
import make_notebook_v18 as V18
from make_notebook import code, md, writefile  # re-used cell helpers

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = {"a": "v19a_mindlarge_plusreaders.ipynb", "b": "v19b_mindlarge_h5tower.ipynb"}
MODULES = V18.MODULES

# ------------------------------------------------------------------------------------------------ text
BEST = """**Kết quả tốt nhất hiện tại (test = `MINDlarge_dev`, 376 471 impression):** offline **0,4570** (`stack_meta_v15cfg` của v18a, có thêm thành phần Qwen3-4B; v16b: 0,4543, hàng chính `stack_meta` 0,4516); online `stack_pop` **0,4731** (v16b); chỉ nội dung + lịch sử `super_llm` **0,4373** (v16b). Reader tốt nhất: **0,4316** (`catplus:um_knn_rag` của v18b, huấn luyện kỹ hơn; v16b: 0,4226). Pipeline tất định: đối chứng của v18a tái lập v16b đến 4 chữ số."""

A_TEXT = """**V19a — reader huấn luyện kỹ hơn đi vào hệ thống (có đối chứng trong notebook) + H6 / H9.**
*Câu hỏi:* v18b cho reader tốt nhất +0,0090 (0,4226 → 0,4316) nhưng supermodel và stacker không chạy (hết ngân sách). Mức tăng đó có lên được `super_llm` / `stack_meta` / `stack_pop` không?
*Thiết kế:* (1) pipeline v16b = **đối chứng**; (2) `catplus`: reader trên vector nối Qwen3-0,6B ⊕ BGE-large huấn luyện kỹ hơn (toàn bộ 2,71 triệu mẫu, 6 epoch, learning rate × 0,5 mỗi epoch từ epoch 2, ngưỡng 0,0002; bỏ `sca` vì không hơn); (3) `bgeplus`: cùng công thức cho reader trên BGE-large; (4) supermodel + stacker của run chính chạy lại **có thêm** các reader mới = **nhánh thí nghiệm** (cùng impression → CI ghép cặp).
*LLM sinh văn bản (chạy trên tập con, chỉ đo từng kỹ thuật, không vào phần ghép):* **H6** — LLM (Qwen3-1,7B) viết mở rộng cho bài và hồ sơ người đọc (kiểu KAR); **H9** — LLM đọc lịch sử và trả lời Có / Không cho từng ứng viên trong top-10. Trên MIND-small cả hai đều âm; đây là lần đầu đo trên MIND-large. Chạy toàn bộ test thì cần ~50 giờ, nên chỉ chạy trên tập con.
*Cổng (viết trước):* G1 `super_llm`, G2 `stack_meta`, G3 `stack_pop`: PASS nếu CI 95% ghép cặp của Δ (thí nghiệm − đối chứng) nằm trên 0; "vượt kết quả tốt nhất" khi PASS và số cao hơn 0,4373 / 0,4570 / 0,4731.
*Thời gian (từ số đo v18a/v18b):* đối chứng ≈ 4,1 giờ + H6/H9 ≈ 1,4 giờ; `catplus` ≈ 1,75 giờ; `bgeplus` ≈ 1,7 giờ; ghép lại ≈ 1,3 giờ → ≈ 10,4 giờ, `BUDGET_H` 11,3."""

B_TEXT = """**V19b — H5: tháp người dùng LLM đọc lịch sử dạng văn bản, huấn luyện ~3 giờ, chấm toàn bộ, vào hệ thống (có đối chứng trong notebook) + H1.**
*Câu hỏi:* LLM (Qwen3-0,6B + LoRA) đọc 10 tiêu đề gần nhất của người dùng như một đoạn văn và học ra vector người dùng — khi được huấn luyện đủ — có hơn reader trên vector đóng băng không, và có giúp hệ thống không? Trên MIND-large (v14) tháp mới được huấn luyện 800 bước (0,5% dữ liệu): ngang `pool_lse`, validation còn tăng.
*Thiết kế:* (1) pipeline v16b = **đối chứng**; (2) trong run chính: **H1** (cùng LLM, zero-shot, lịch sử thành câu truy vấn; tập con 60 000 impression test) và **H5** (LoRA r=16, causal, huấn luyện **~180 phút**: số bước và lịch cosine được khớp theo thời gian đo được ở 20 bước đầu; đánh giá giữa chừng trên 20 000 impression validation; cuối cùng chấm **toàn bộ** validation và test, chia cho 2 GPU) — H5 được xuất như một reader; (3) supermodel + stacker chạy lại có H5 = **nhánh thí nghiệm**.
*So sánh ghép cặp trong run:* `ut_causal` với `pool_lse`, `txt_cta_lse`, `um_frozen`, `um_knn_rag` (Qwen3) và `cat:um_knn_rag` (reader tốt nhất của v16b); H5 với H1 (cái mà huấn luyện thêm vào).
*Cổng (viết trước):* như v19a (G1 / G2 / G3, so với 0,4373 / 0,4570 / 0,4731).
*Thời gian:* đối chứng ≈ 4,3 giờ + H1 ≈ 0,3 + H5 ≈ 3 (huấn luyện) + 1 (chấm toàn bộ) + ghép lại ≈ 1,3 → ≈ 10 giờ, `BUDGET_H` 11,3 (thời gian huấn luyện H5 tự rút ngắn nếu phần trước chạy chậm)."""


def intro(p):
    head = {"a": "# V19a — reader huấn luyện kỹ hơn vào supermodel + stacker (đối chứng trong notebook) + LLM sinh văn bản H6 / H9 · MIND-large · Kaggle T4 x2 (~10,4 giờ)",
            "b": "# V19b — H5: tháp người dùng LLM đọc lịch sử dạng văn bản (~3 giờ huấn luyện, chấm toàn bộ) vào supermodel + stacker (đối chứng trong notebook) + H1 · MIND-large · Kaggle T4 x2 (~10 giờ)"}[p]
    how = V18.HOW_RUN.replace("v18@P@", "v19" + p).replace("@P@", p).replace("@QUICK_EXTRA@", "")
    return "\n\n".join([head, BEST, A_TEXT if p == "a" else B_TEXT, how,
                        "Giao thức như v16b: validation theo ngày (giữ ngẫu nhiên 150 000 impression), `MINDlarge_dev` chỉ là test và chỉ chạm một lần cho mỗi hàng, "
                        "CI 95% ghép cặp theo impression (chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện), lát cold / warm. Không dùng nhãn test để chọn gì. "
                        "Điểm của mọi thành phần được ghi ra `results_v19" + p + "/scores/<run>/` để ghép lại sau mà không phải huấn luyện lại."])


PROFILE_A = """PLUS_EPOCHS = os.environ.get("V11_PLUS_EPOCHS", "2" if QUICK else "6")          # reader huấn luyện kỹ hơn: tối đa từng này epoch
PLUS_DECAY = os.environ.get("V11_PLUS_DECAY", "0.5")                            # learning rate nhân hệ số này sau mỗi epoch kể từ epoch 2
LLM_TEXT = bool(int(os.environ.get("V11_LLM_TEXT", "1")))                       # H6 + H9 (LLM sinh văn bản) trên tập con trong run chính
GEN_MODEL = os.environ.get("V11_GEN_MODEL", "Qwen/Qwen3-1.7B")                  # LLM sinh văn bản của H6 / H9
PLUS_MIN_H = float(os.environ.get("V11_PLUS_MIN_H", "0" if QUICK else "3.2"))   # giờ còn lại tối thiểu để bắt đầu một run reader-kỹ-hơn (~1,7 giờ + nhánh thí nghiệm)
TREAT_MIN_H = float(os.environ.get("V11_TREAT_MIN_H", "0" if QUICK else "1.4"))  # ... để chạy lại supermodel + stacker (nhánh thí nghiệm)"""
PRINT_A = ", PLUS_EPOCHS=PLUS_EPOCHS, PLUS_DECAY=PLUS_DECAY, LLM_TEXT=LLM_TEXT, GEN_MODEL=GEN_MODEL"

PROFILE_B = """H5_MINUTES = float(os.environ.get("V11_H5_MINUTES", "1" if QUICK else "180"))    # thời gian huấn luyện tháp H5 (v14: 800 bước ~ 30 phút)
H5_EVAL_N = int(os.environ.get("V11_H5_EVAL_N", "400" if QUICK else "20000"))        # impression validation của các lần đánh giá giữa chừng
AFTER_H5_H = float(os.environ.get("V11_AFTER_H5_H", "0" if QUICK else "2.9"))       # giờ cần sau khi huấn luyện: chấm toàn bộ + H1 + supermodel/stacker lại
TREAT_MIN_H = float(os.environ.get("V11_TREAT_MIN_H", "0" if QUICK else "3.5"))     # giờ còn lại tối thiểu để bắt đầu nhánh thí nghiệm (H1 + H5 + ghép)"""
PRINT_B = ", H5_MINUTES=H5_MINUTES, H5_EVAL_N=H5_EVAL_N"


def config_cell(p):
    s = (V18.CONFIG.replace("v18", "v19").replace("@BUDGET@", "11.3").replace("@STACKMIN@", "4200").replace("@P@", p)
         .replace("@PROFILE@", PROFILE_A if p == "a" else PROFILE_B).replace("@PRINT@", PRINT_A if p == "a" else PRINT_B))
    return code(s)


RUNS_A = """# v19a: the readers trained harder (the v18b recipe without 'sca') on BGE-large and on Qwen3 + BGE-large; H6 / H9 inside the main run
PLUS = ["--train-sub", "0", "--um-epochs", PLUS_EPOCHS, "--um-lr-decay", PLUS_DECAY, "--um-decay-after", "2", "--um-min-delta", "0.0002"]
TAG2P, TAG3P = "bgeplus", "catplus"
ARGS2P = [*COMMON, *SCALE, "--encoder", ENCODER2, "--work", WORK + "/" + TAG2P, "--max-len", "256", "--seeds", "1", "--um-views", "frozen,knn_rag", *READERS, *PLUS]
ARGS3P = [*COMMON, *SCALE, "--encoder", "concat:" + ENCODER + "+" + ENCODER2, "--work", WORK + "/" + TAG3P, "--max-len", "256", "--seeds", "1",
          "--um-views", "frozen,knn_rag", *READERS, *PLUS]
STAGES_PLUS = "controls,graph,usermodel"
if LLM_TEXT:                                           # LLM-generated text (subsets; not inputs of the fusion)
    ARGS += ["--gen-model", GEN_MODEL]
    STAGES = "controls,pooling,moredata,graph,usermodel,augment,rerank,supermodel,stack"
if QUICK:                                              # cỡ đồ chơi: mọi họ stage chạy một lần
    EXTRA_GEN = ["--gen-bs", "8", "--gen-new-tokens", "8", "--gen-val-n", "40", "--gen-test-n", "80", "--augment-minutes", "2", "--rerank-k", "5",
                 "--rerank-val-n", "40", "--rerank-test-n", "80", "--rerank-minutes", "2"]
    ARGS = ARGS + EXTRA_QUICK + (EXTRA_GEN if LLM_TEXT else [])
    ARGS2, ARGS3, ARGS2P, ARGS3P = ARGS2 + EXTRA_QUICK, ARGS3 + EXTRA_QUICK, ARGS2P + EXTRA_QUICK, ARGS3P + EXTRA_QUICK
S2P = S3P = S_TREAT = None
CTRL_DIR = WORK + "/ctrl_snapshot"
print("budget:", BUDGET_H, "h; runs: bge -> cat -> qwen3 (control: supermodel + stack" + (", H6 / H9" if LLM_TEXT else "") + ") -> catplus -> bgeplus -> treatment")
"""

RUNS_B = """# v19b: H1 + H5 (the LLM user tower, history as text) are added to the main run after the control; H5 covers the whole split and is exported as a reader
H5_ARGS = ["--llm-user-modes", "causal", "--llm-val-n", "0", "--llm-test-n", "0", "--llm-eval-n", str(H5_EVAL_N), "--llm-user-steps", "100000",
           "--llm-user-bs", "8" if QUICK else "16", "--llm-enc-bs", "64", "--hq-plain", "0", "--hq-subset", "1",
           "--hq-val-n", "400" if QUICK else "20000", "--hq-test-n", "1000" if QUICK else "60000"]
ARGS = ARGS + H5_ARGS                                  # in BOTH calls of the main run (the saved state is resumed only when the arguments match)
STAGES_T = "controls,pooling,moredata,graph,usermodel,histquery,llm_user,supermodel,stack"
if QUICK:
    ARGS, ARGS2, ARGS3 = ARGS + EXTRA_QUICK, ARGS2 + EXTRA_QUICK, ARGS3 + EXTRA_QUICK
S_TREAT = None
CTRL_DIR = WORK + "/ctrl_snapshot"
print("budget:", BUDGET_H, "h; runs: bge -> cat -> qwen3 (control: supermodel + stack) -> treatment (H1 + H5 on the same run, then supermodel + stack again)")
"""

CELL_PLUS = '''for tag_, argv_ in ((TAG3P, ARGS3P), (TAG2P, ARGS2P)):
    if S is None:
        print(f"{tag_} bỏ qua: không có đối chứng")
        continue
    if hours_left() < PLUS_MIN_H:
        print(f"{tag_} bỏ qua: còn {hours_left():.2f} giờ < {PLUS_MIN_H} giờ")
        continue
    suite.EQUIV_REFERENCE = None                       # the vectors come from the cache of runs 1-2 (guarded there)
    s_ = run_safely(tag_, [*argv_, *budget_args(reserve_h=TREAT_MIN_H), "--stages", STAGES_PLUS])
    if tag_ == TAG3P:
        S3P = s_
    else:
        S2P = s_
    print("giờ còn lại: %.2f" % hours_left())
'''

CELL_TREAT_A = '''NEW_DIRS = [WORK + "/" + t for t, s in ((TAG3P, S3P), (TAG2P, S2P)) if s is not None and os.path.isdir(os.path.join(WORK, t, "scores"))]
if S is None or not NEW_DIRS:
    print("nhánh thí nghiệm bỏ qua:", "không có đối chứng" if S is None else "không có reader mới")
elif hours_left() < TREAT_MIN_H:
    print(f"nhánh thí nghiệm bỏ qua: còn {hours_left():.2f} giờ < {TREAT_MIN_H}")
else:
    suite.EQUIV_REFERENCE = None                       # the main encoder's vectors come from the cache (guarded in run 3)
    S_TREAT = run_safely("treatment", [*ARGS, "--super-dirs", ",".join(SUPER_DIRS + NEW_DIRS), "--rerun", "supermodel,stack", "--min-stack-s", str(STACK_MIN_S),
                                       *budget_args(), "--stages", STAGES], release=False)
'''

CELL_TREAT_B = '''if S is None:
    print("nhánh thí nghiệm bỏ qua: không có đối chứng")
elif hours_left() < TREAT_MIN_H:
    print(f"nhánh thí nghiệm bỏ qua: còn {hours_left():.2f} giờ < {TREAT_MIN_H}")
else:
    h5_min = max(1.0 if QUICK else 30.0, min(H5_MINUTES, (hours_left() - AFTER_H5_H) * 60))
    print(f"H5: huấn luyện {h5_min:.0f} phút (còn {hours_left():.2f} giờ của ngân sách)")
    suite.EQUIV_REFERENCE = None                       # the main encoder's vectors come from the cache (guarded in run 3)
    S_TREAT = run_safely("treatment (H1 + H5)", [*ARGS, "--llm-user-minutes", "%.1f" % h5_min, "--super-dirs", ",".join(SUPER_DIRS), "--rerun", "supermodel,stack",
                                                 "--min-stack-s", str(STACK_MIN_S), *budget_args(), "--stages", STAGES_T], release=False)
'''

FUSED_PAIRS = '''Q, T = TAG, "treat"
pairs = [# --- the treatment against the control (v16b pipeline of this notebook), row by row
         ((T, "super_base"), (Q, "super_base")), ((T, "super_zs"), (Q, "super_zs")), ((T, "super_llm"), (Q, "super_llm")), ((T, "stack_llm"), (Q, "stack_llm")),
         ((T, "stack_meta"), (Q, "stack_meta")), ((T, "stack_meta_v15cfg"), (Q, "stack_meta_v15cfg")), ((T, "super_llm+pop"), (Q, "super_llm+pop")), ((T, "stack_pop"), (Q, "stack_pop")),
         # --- the control in v16b's ordering (should reproduce RESULTS_v16b_large.md)
         ((Q, "stack_meta"), (Q, "super_llm")), ((Q, "stack_meta"), (Q, "stack_meta_v15cfg")), ((Q, "stack_pop"), (Q, "stack_meta"))]
'''

REPORT_A = '''SUITES_TAGS = [(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG), (S3P, TAG3P), (S2P, TAG2P)) if s is not None]
if S_TREAT is not None:
    SUITES_TAGS.append((OnlyStages(S_TREAT, ("supermodel", "stack")), "treat"))
SUITES, TAGS = [s for s, _ in SUITES_TAGS], [t for _, t in SUITES_TAGS]
os.makedirs(RESULTS, exist_ok=True)
if S is not None:
    suite.leaderboard(SUITES, TAGS, (TAG, "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
    suite.leaderboard(SUITES, TAGS, (TAG, "frozen+pop"), WORK + "/leaderboard_pop.md", plot=WORK + "/forest_pop.png", pop=True)
''' + FUSED_PAIRS + '''pairs += [# --- the readers trained harder against the v16b-recipe readers on the same vectors
          ((TAG3P, "um_frozen"), (TAG3, "um_frozen")), ((TAG3P, "um_knn_rag"), (TAG3, "um_knn_rag")), ((TAG3P, "um_prior_frozen"), (TAG3, "um_prior_frozen")),
          ((TAG2P, "um_frozen"), (TAG2, "um_frozen")), ((TAG2P, "um_knn_rag"), (TAG2, "um_knn_rag")), ((TAG2P, "um_prior_frozen"), (TAG2, "um_prior_frozen")),
          ((TAG3P, "um_knn_rag"), (TAG2P, "um_knn_rag")),
          # --- H6 / H9: LLM-generated text (subsets; the comparison uses the impressions both rows share)
          ((Q, "kar_item"), (Q, "frozen")), ((Q, "kar_user"), (Q, "frozen")), ((Q, "kar_item+user"), (Q, "frozen")), ((Q, "kar_user+meanpool"), (Q, "frozen")),
          ((Q, "rerank_llm"), (Q, "frozen")), ((Q, "rerank_fused"), (Q, "frozen")), ((Q, "rerank_fused"), (Q, "pool_lse")), ((Q, "kar_item"), (Q, "pool_lse"))]
direct = suite.cross_compare(SUITES, TAGS, pairs, WORK + "/direct_comparisons.md") if S is not None else []
fusion_md = gate_md = ""
if S is not None and S_TREAT is not None:
    fusion_md, fusion = report_v18.fusion_table(S, S_TREAT, WORK + "/HEADLINE_v19a.md",
                                                title="v19a: control (v16b pipeline, this notebook) vs treatment (+ readers trained harder on BGE-large and on Qwen3 + BGE-large), same impressions")
    gate_md, gates = report_v18.gate_v18a(S, S_TREAT, WORK + "/gate_v19a.md", bests=report_v18.BEST_V18,
                                          title="v19a gates (written before the run; decided on the paired test CI of the treatment against the in-run control)")
    print(fusion_md); print(); print(gate_md)
    try:
        print(final_report.build(S_TREAT, WORK + "/FINAL_REPORT_treat.md", WORK + "/final_tables_treat.json"))
    except Exception as e:
        print("!! treatment report failed:", repr(e)[:200])
elif S is not None:
    fusion_md, _ = report_v18.fusion_table(S, S, WORK + "/HEADLINE_v19a.md", title="v19a: control only (no treatment ran; Δ = 0 by construction)")
print(report_v18.epochs_table([(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG), (S3P, TAG3P), (S2P, TAG2P))], WORK + "/epochs.md"))
'''

REPORT_B = '''SUITES_TAGS = [(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG)) if s is not None]
if S_TREAT is not None:
    SUITES_TAGS.append((OnlyStages(S_TREAT, ("histquery", "llm_user", "supermodel", "stack")), "treat"))
SUITES, TAGS = [s for s, _ in SUITES_TAGS], [t for _, t in SUITES_TAGS]
os.makedirs(RESULTS, exist_ok=True)
if S is not None:
    suite.leaderboard(SUITES, TAGS, (TAG, "frozen"), WORK + "/leaderboard.md", plot=WORK + "/forest.png")
    suite.leaderboard(SUITES, TAGS, (TAG, "frozen+pop"), WORK + "/leaderboard_pop.md", plot=WORK + "/forest_pop.png", pop=True)
''' + FUSED_PAIRS + '''pairs += [# --- H5 (the LLM user tower) against the frozen-vector rows of the same encoder and the best v16b reader, on the whole test day
          ((T, "ut_causal"), (Q, "frozen")), ((T, "ut_causal"), (Q, "pool_lse")), ((T, "ut_causal"), (Q, "txt_cta_lse")), ((T, "ut_causal"), (Q, "um_frozen")),
          ((T, "ut_causal"), (Q, "um_knn_rag")), ((T, "ut_causal"), (TAG3, "um_knn_rag")), ((T, "ut_causal+meanpool"), (Q, "pool_lse")),
          # --- H1 (zero-shot, the same LLM) and what training adds (H5 - H1, on the impressions both cover)
          ((T, "histq_instr"), (Q, "frozen")), ((T, "histq_instr+meanpool"), (Q, "pool_lse")), ((T, "ut_causal"), (T, "histq_instr"))]
direct = suite.cross_compare(SUITES, TAGS, pairs, WORK + "/direct_comparisons.md") if S is not None else []
fusion_md = gate_md = ""
if S is not None and S_TREAT is not None:
    fusion_md, fusion = report_v18.fusion_table(S, S_TREAT, WORK + "/HEADLINE_v19b.md",
                                                title="v19b: control (v16b pipeline, this notebook) vs treatment (+ H5 LLM user tower, history as text), same impressions")
    gate_md, gates = report_v18.gate_v18a(S, S_TREAT, WORK + "/gate_v19b.md", bests=report_v18.BEST_V18,
                                          title="v19b gates (written before the run; decided on the paired test CI of the treatment against the in-run control)")
    print(fusion_md); print(); print(gate_md)
    try:
        r5 = S_TREAT.row("ut_causal")
        h5 = ["# H5 tower (ut_causal)", "", f"test nDCG@10 {r5['ndcg@10']:.4f} / AUC {r5['auc']:.4f} on {r5['n_test']:,} impressions; validation {r5['val_ndcg@10']:.4f}",
              f"steps {r5.get('steps')} of {r5.get('planned_steps')} planned (batch {r5.get('batch')}), best step {r5.get('best_step')}, "
              f"training {r5.get('train_s', 0) / 60:.0f} min, first loss {r5.get('first_loss')}, stopped by the budget: {r5.get('truncated')}"]
        open(WORK + "/h5_tower.md", "w", encoding="utf-8").write("\\n".join(h5) + "\\n")
        print("\\n".join(h5))
    except Exception as e:
        print("!! no H5 row:", repr(e)[:200])
    try:
        print(final_report.build(S_TREAT, WORK + "/FINAL_REPORT_treat.md", WORK + "/final_tables_treat.json"))
    except Exception as e:
        print("!! treatment report failed:", repr(e)[:200])
elif S is not None:
    fusion_md, _ = report_v18.fusion_table(S, S, WORK + "/HEADLINE_v19b.md", title="v19b: control only (no treatment ran; Δ = 0 by construction)")
print(report_v18.epochs_table([(s, t) for s, t in ((S2, TAG2), (S3, TAG3), (S, TAG))], WORK + "/epochs.md"))
'''

SUMMARY = '''RUN_TAGS = [t for t in (TAG, TAG2, TAG3, "catplus", "bgeplus") if os.path.isdir(os.path.join(WORK, t))]
noise = ["# Seed-to-seed spread of the trained rows (test nDCG@10 of each seed; sample std with n-1; the paired CIs do NOT include it)", "",
         "| run | row | seeds | nDCG@10 per seed | std |", "|---|---|---|---|---|"]
for tag_ in RUN_TAGS:
    pj = os.path.join(WORK, tag_, "results.json")
    if os.path.exists(pj):
        for r in json.load(open(pj, encoding="utf-8")):
            if r.get("ndcg10_per_seed") and len(r["ndcg10_per_seed"]) > 1:
                noise.append(f"| {tag_} | {r['name']} | {len(r['ndcg10_per_seed'])} | {', '.join(f'{x:.4f}' for x in r['ndcg10_per_seed'])} | {r.get('ndcg10_std', float('nan')):.4f} |")
open(os.path.join(WORK, "noise.md"), "w", encoding="utf-8").write("\\n".join(noise) + "\\n")
print("tổng thời gian của notebook: %.2f h" % ((time.time() - T_NB) / 3600))
parts = [("HEADLINE_v19@P@.md", WORK), ("gate_v19@P@.md", WORK), ("h5_tower.md", WORK), ("direct_comparisons.md", WORK), ("epochs.md", WORK), ("noise.md", WORK),
         ("FINAL_REPORT_treat.md", WORK), ("FINAL_REPORT_ctrl.md", WORK + "/ctrl_snapshot"), ("leaderboard.md", WORK), ("leaderboard_pop.md", WORK)]
for t in RUN_TAGS:
    parts += [(f, WORK + "/" + t) for f in ("timings.json", "memory.json", "results.md")]
os.makedirs(RESULTS, exist_ok=True)
SUMMARY = os.path.join(RESULTS, "SUMMARY_v19@P@.md")
with open(SUMMARY, "w", encoding="utf-8") as out:
    out.write(f"# SUMMARY v19@P@ ({'QUICK' if QUICK else 'full'}; {(time.time() - T_NB) / 3600:.2f} h)\\n")
    for fn, d in parts:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out.write(f"\\n\\n## {fn} ({os.path.basename(d)})\\n\\n" + open(p, encoding="utf-8").read())
for fn in ("forest.png", "forest_pop.png", "HEADLINE_v19@P@.md", "gate_v19@P@.md", "h5_tower.md", "direct_comparisons.md", "epochs.md", "noise.md", "leaderboard.md",
           "leaderboard_pop.md", "FINAL_REPORT_treat.md", "final_tables_treat.json"):
    if os.path.exists(os.path.join(WORK, fn)) and os.path.abspath(WORK) != os.path.abspath(RESULTS):
        shutil.copy(os.path.join(WORK, fn), os.path.join(RESULTS, fn))
for src, dst in ((os.path.join(WORK, TAG, "stack_scores.npz"), "stack_scores.npz"), (os.path.join(WORK, "ctrl_snapshot", "stack_scores_ctrl.npz"), "stack_scores_ctrl.npz"),
                 (os.path.join(WORK, "ctrl_snapshot", "FINAL_REPORT_ctrl.md"), "FINAL_REPORT_ctrl.md")):
    if os.path.exists(src):
        shutil.copy(src, os.path.join(RESULTS, dst))
# every run's exported component scores (+ its rows): a later fusion-only run reads them as --super-dirs without re-training anything
n_files = 0
for t in RUN_TAGS:
    sd, dst = os.path.join(WORK, t, "scores"), os.path.join(RESULTS, "scores", t)
    if not os.path.isdir(sd):
        continue
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(sd):
        if fn.endswith(".npz") and not fn.endswith(".tmp.npz"):
            shutil.copy(os.path.join(sd, fn), os.path.join(dst, fn))
            n_files += 1
    for fn in ("results.json", "results.md"):
        if os.path.exists(os.path.join(WORK, t, fn)):
            shutil.copy(os.path.join(WORK, t, fn), os.path.join(dst, fn))
print(f"-> {n_files} component score files written to {os.path.join(RESULTS, 'scores')}")
print("-> gửi lại file:", SUMMARY, f"({os.path.getsize(SUMMARY) / 1024:.0f} KB)")
'''

READ_A = """### Cách đọc kết quả (v19a)

| Bạn thấy (`gate_v19a.md` / `HEADLINE_v19a.md`) | Nghĩa là |
|---|---|
| `catplus:*` − `cat:*` và `bgeplus:*` − `bge:*` > 0 | huấn luyện kỹ hơn giúp reader (tái lập v18b ở `catplus`, mới ở BGE) |
| G1 **PASS** | reader mạnh hơn nâng ghép tuyến tính chỉ dùng nội dung + lịch sử |
| G2 / G3 **PASS** và số > 0,4570 / 0,4731 | hệ thống tốt hơn cả đối chứng trong notebook lẫn kết quả tốt nhất trước đây |
| `kar_*` / `rerank_*` ≤ hàng tham chiếu | như MIND-small: LLM sinh văn bản (H6 / H9) không giúp xếp hạng ở quy mô lớn (tập con: CI rộng hơn) |

**Giới hạn:** một ngày test; CI chỉ tính nhiễu mẫu test; 2 seed mỗi reader; H6 / H9 chỉ trên tập con."""

READ_B = """### Cách đọc kết quả (v19b)

| Bạn thấy | Nghĩa là |
|---|---|
| `ut_causal` > `um_knn_rag` / `cat:um_knn_rag` (CI ghép cặp, toàn bộ test) | tháp LLM đọc lịch sử dạng văn bản, khi được huấn luyện đủ, hơn reader trên vector đóng băng |
| `ut_causal` ≈ `pool_lse` | như v14: huấn luyện thêm vẫn chưa đủ (xem `h5_tower.md`: số bước, bước tốt nhất) |
| `ut_causal` − `histq_instr` > 0 | phần do huấn luyện mang lại so với cùng LLM zero-shot (H1) |
| G1 / G2 / G3 **PASS** | H5 thêm thông tin mà các thành phần khác chưa có, lên được ghép tuyến tính / stacker |

**Giới hạn:** một seed cho tháp; huấn luyện ~3 giờ vẫn chỉ thấy một phần nhỏ dữ liệu train; H1 trên tập con."""


def build(p):
    a = p == "a"
    cells = [md(intro(p)), md("### 0. Phụ thuộc và GPU"), code(V17.DEPS), md("### 1. Modules (ghi ra thư mục làm việc)")]
    cells += [writefile(m) for m in MODULES]
    cells += [md("### 2. Cấu hình"), config_cell(p), md("### 3. Dữ liệu — MIND-large tải từ Hugging Face (cần Internet)"), code(V18.DATA),
              md("### 4. Cấu hình các run"), code(V18.RUNS_COMMON + (RUNS_A if a else RUNS_B)),
              md("### 5. Run 1 — BGE-large (như v16b): controls → pooling → moredata → graph → usermodel"), code(V18.CELL_BGE),
              md("### 6. Run 2 — vector Qwen3-0,6B ⊕ BGE-large (như v16b): controls → graph → usermodel"), code(V18.CELL_CAT),
              md("### 7. Run 3 — Qwen3-0,6B + supermodel + stacker = **ĐỐI CHỨNG** (pipeline v16b" + (", cộng H6 / H9 trên tập con)" if a else ")")),
              code(V18.CELL_MAIN_A)]
    if a:
        cells += [md("### 8. Run 4 – 5 — reader huấn luyện kỹ hơn: `catplus` (vector nối) rồi `bgeplus` (BGE-large)"), code(CELL_PLUS),
                  md("### 9. **NHÁNH THÍ NGHIỆM** — supermodel + stacker của run 3 chạy lại với các reader mới"), code(CELL_TREAT_A),
                  md("### 10. Bảng chính, cổng, so sánh trực tiếp"), code(REPORT_A)]
    else:
        cells += [md("### 8. **NHÁNH THÍ NGHIỆM** — H1 + H5 (tháp người dùng LLM) trong run 3, rồi supermodel + stacker chạy lại với H5"), code(CELL_TREAT_B),
                  md("### 9. Bảng chính, cổng, so sánh trực tiếp"), code(REPORT_B)]
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
