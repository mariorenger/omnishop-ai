"""Build the three MIND-large notebooks of v20: the LLM-text directions at full scale, then fused into the best system so far (v19c's treatment).

  v20a_mindlarge_kar.ipynb         (account 1, "GPU T4 x2", ~6-7 h)  H6 at full scale: Qwen3-1.7B writes an expansion of EVERY article (split over both GPUs),
                                   embedded and added to the article vectors; rows on the whole test day; then the learned readers trained on the augmented
                                   vectors: Qwen3+KAR alone (`qwen3kar`) and side by side with BGE-large (`catkarplus`, the recipe of v19a's best reader `catplus`).
  v20b_mindlarge_sid_histq.ipynb   (account 2, ~3.5 h)  H7 Semantic IDs v2 (codes of the category-aware text, the generative ranker scored per code level,
                                   shuffled-history control) and H1 + a learned head (`histq_mlp`: the frozen LLM reads the history as text, residual MLP heads
                                   trained like a reader), both on the whole split.
  v20c_mindlarge_fusion.ipynb      (account of v19, after both)  fusion only: CONTROL = v19c's treatment (v16b components + v19a's readers trained harder + v19b's
                                   H5 tower), TREATMENT = + every v20 component; supermodel chosen on the cold-weighted validation (pre-registered, gates) and on
                                   the plain validation (pre-registered here as the secondary rule); stacker seed 1 when time allows; paired comparisons of the
                                   new components with the ones they should beat.  The v20 components reach it as a private dataset of flat files
                                   (``<a|b>__<run>__<component>.npz``): Kaggle does not mount another account's private notebook output.

Bump the notebook names when the content changes: Kaggle caches notebooks by file name."""
import json
import os

import make_notebook_v17 as V17
import make_notebook_v18 as V18
import make_notebook_v19 as V19
import make_notebook_v19c as V19C
from make_notebook import code, md, writefile  # re-used cell helpers

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = {"a": "v20a_mindlarge_kar.ipynb", "b": "v20b_mindlarge_sid_histq.ipynb", "c": "v20c_mindlarge_fusion.ipynb"}
MODULES = V18.MODULES[:-1] + ["stages_v20.py", "fusion_only.py", "suite.py"]
assert V18.MODULES[-1] == "suite.py"

BEST = """**Hệ tốt nhất hiện tại (v19c, `RESULTS_v19c_large.md`, test = `MINDlarge_dev` 376 471 impression):** chỉ nội dung + lịch sử `super_llm` **0,4380**; offline `stack_meta` **0,4531** (`stack_meta_v15cfg` 0,4562; trung bình 2 seed 0,4570); online `stack_pop` **0,4731** (nhiễu seed của stacker ±0,0008). Thành phần: pipeline v16b + reader huấn luyện kỹ hơn (v19a) + tháp H5 (v19b)."""

PROTOCOL = ("Giao thức như v16b: validation theo ngày (giữ ngẫu nhiên 150 000 impression), `MINDlarge_dev` chỉ là test và chỉ chạm một lần cho mỗi hàng, "
            "CI 95% ghép cặp theo impression (chỉ tính nhiễu mẫu test, không tính nhiễu huấn luyện), lát cold / warm. Không dùng nhãn test để chọn gì.")

A_TEXT = """# V20a — H6 ở quy mô đủ: LLM viết mở rộng cho MỌI bài, rồi reader học trên vector đã mở rộng · MIND-large · Kaggle T4 x2 (~6–7 giờ)

**Vì sao:** H6 (LLM viết thêm kiến thức cho bài, kiểu KAR) mới đo trên tập con rất nhỏ (MIND-small +0,0079 có ý nghĩa; MIND-large +0,0043 trên 2 500 impression, CI rộng) và chưa bao giờ vào reader hay bộ ghép.

**Thiết kế:**
1. Run `kar` (Qwen3-Embedding-0,6B): Qwen3-1,7B viết 1–2 câu cho **toàn bộ 104 151 bài** (tiêu đề + tóm tắt vào; chia đều cho 2 GPU, có cache để chạy tiếp được); đoạn mở rộng được mã hoá bằng chính encoder; vector bài mới = chuẩn hoá(E0 + α·E_mở_rộng), α chọn trên validation. Các hàng trên **toàn bộ** validation / test: `kar_item` (mean pooling), `kar_item_lse` (late interaction), `kar_exp` (chỉ vector đoạn mở rộng). Toàn bộ đoạn mở rộng được lưu ra `kar_expansions.jsonl`.
2. Run `catkarplus`: reader trên vector nối (Qwen3 + KAR) ⊕ BGE-large, **đúng công thức** của reader tốt nhất v19a (`catplus`: toàn bộ mẫu, 6 epoch, learning rate giảm dần) — so ghép cặp với `catplus` ở v20c cho biết riêng phần LLM-text thêm vào.
3. Run `qwen3kar`: reader trên vector Qwen3 + KAR một mình (so với reader Qwen3 cũ).

Điểm của mọi thành phần được ghi ra `results_v20a/scores/<run>/` để v20c ghép vào hệ tốt nhất."""

B_TEXT = """# V20b — SID v2 (chấm theo tầng mã) + H1 có lớp học được (`histq_mlp`) · MIND-large · Kaggle T4 x2 (~3,5 giờ)

**SID v2 (H7):** v7 cộng log-xác suất của cả 3 tầng mã và ra ngang ngẫu nhiên (0,2898 so với 0,2897), dù cài đặt học được quy luật cài sẵn (selftest). Giả thuyết: với bài mới (82% click ở test), tầng mã mịn gần như ngẫu nhiên và lấn át tầng chủ đề. v2: mã từ vector Qwen3 đọc "chuyên mục + tiêu đề + tóm tắt"; chấm **theo từng tầng** (chỉ tầng 1, tầng 1–2, cả 3; có / không trừ xác suất không cần lịch sử), chọn trên validation; đối chứng **lịch sử xáo trộn** (mỗi người đọc lịch sử của người khác) để kiểm mô hình có dùng lịch sử không; chạy trên MIND-large, toàn bộ test.

**H1 + lớp học được (`histq_mlp`):** H1 cho LLM (Qwen3-Embedding-0,6B, đóng băng) đọc 10 tiêu đề gần nhất như một đoạn văn có câu chỉ dẫn — zero-shot, không học gì. Ở đây thêm hai đầu MLP dạng residual (cho vector lịch sử và vector bài), huấn luyện như reader (1 bài click + 4 bài bỏ qua, cross-entropy, dừng sớm theo validation), 2 seed. Ở bước 0 hai đầu là phép đồng nhất, nên hàng bắt đầu đúng bằng H1: hiệu `histq_mlp` − `histq_instr` là phần một lớp học được mang lại, hiệu với H5 (v19b) là phần do tinh chỉnh chính LLM.

Điểm của mọi thành phần được ghi ra `results_v20b/scores/<run>/` để v20c ghép vào hệ tốt nhất."""

HOW_AB = """**Cách chạy:** Settings → Accelerator **GPU T4 x2**, Internet **On** → **Save Version / Run All**. MIND-large và các mô hình tải từ Hugging Face. **Kiểm tra môi trường trước (vài phút):** `V11_QUICK=1` với encoder nhỏ. **Nhận lại:** `SUMMARY_v20@P@.md` trong `/kaggle/working/results_v20@P@` + log."""

PROFILE_A = """GEN_MODEL = os.environ.get("V11_GEN_MODEL", "Qwen/Qwen3-1.7B")                  # LLM viết đoạn mở rộng cho bài
GEN_BS = os.environ.get("V11_GEN_BS", "8" if QUICK else "48")
KAR_TOKENS = os.environ.get("V11_KAR_TOKENS", "8" if QUICK else "56")            # tối đa từng này token mỗi đoạn mở rộng (1-2 câu)
PLUS_EPOCHS = os.environ.get("V11_PLUS_EPOCHS", "2" if QUICK else "6")          # reader: công thức của catplus (v19a)
PLUS_DECAY = os.environ.get("V11_PLUS_DECAY", "0.5")
READER_MIN_H = float(os.environ.get("V11_READER_MIN_H", "0" if QUICK else "1.9"))  # giờ còn lại tối thiểu để bắt đầu một run reader"""
PRINT_A = ", GEN_MODEL=GEN_MODEL, KAR_TOKENS=KAR_TOKENS, PLUS_EPOCHS=PLUS_EPOCHS"

PROFILE_B = """SID_TEXT = os.environ.get("V11_SID_TEXT", "cta")                              # văn bản của vector dùng để tạo mã SID
SID_EPOCHS = os.environ.get("V11_SID_EPOCHS", "1" if QUICK else "4")
HM_TRAIN_N = os.environ.get("V11_HM_TRAIN_N", "3000" if QUICK else "300000")     # số lịch sử train được mã hoá cho histq_mlp"""
PRINT_B = ", SID_TEXT=SID_TEXT, SID_EPOCHS=SID_EPOCHS, HM_TRAIN_N=HM_TRAIN_N"

RUNS_A = """import stages_v20
TAGK, TAGCK, TAGQK = "kar", "catkarplus", "qwen3kar"
PLUS = ["--train-sub", "0", "--um-epochs", PLUS_EPOCHS, "--um-lr-decay", PLUS_DECAY, "--um-decay-after", "2", "--um-min-delta", "0.0002"]
ARGSK = [*COMMON, *SCALE, "--encoder", ENCODER, "--work", WORK + "/" + TAGK, "--max-len", "256", "--seeds", "1", "--km-ks", "5",
         "--gen-model", GEN_MODEL, "--gen-bs", GEN_BS, "--gen-new-tokens", KAR_TOKENS]
STAGESK = "controls,pooling,kar_full"
READ_ARGS = ["--max-len", "256", "--seeds", "1", "--um-views", "frozen,knn_rag", *READERS, *PLUS]
STAGES_R = "controls,graph,usermodel"
if QUICK:
    ARGSK, READ_ARGS = ARGSK + EXTRA_QUICK, READ_ARGS + EXTRA_QUICK
SK = SCK = SQK = AUG = None
print("budget:", BUDGET_H, "h; runs: kar (expansions of every article + kar rows) -> catkarplus (readers on (Qwen3+KAR) + BGE-large) -> qwen3kar (readers on Qwen3+KAR)")
"""

CELL_KAR = '''suite.EQUIV_REFERENCE = reference_encoder(ENCODER, 256)
SK = run_safely("kar", [*ARGSK, *budget_args(), "--stages", STAGESK], release=False)
if SK is not None:
    AUG = SK.kv.get("kar_full", {}).get("aug_npz")
    print("kar_full:", SK.kv.get("kar_full"))
    os.makedirs(RESULTS, exist_ok=True)
    p_ = os.path.join(WORK, TAGK, "kar_expansions.jsonl")
    if os.path.exists(p_):
        shutil.copy(p_, os.path.join(RESULTS, "kar_expansions.jsonl"))
    SK.release()
    gc.collect(); torch.cuda.empty_cache()
print("vector đã mở rộng:", AUG, "| giờ còn lại %.2f" % hours_left())
'''

CELL_CATK = '''if AUG is None or not os.path.exists(AUG):
    print("catkarplus bỏ qua: không có vector đã mở rộng")
elif hours_left() < READER_MIN_H:
    print(f"catkarplus bỏ qua: còn {hours_left():.2f} giờ < {READER_MIN_H}")
else:
    ids_a, va = stages_v20.load_npz_vectors(AUG)
    ids_b, vb = stages_v20.catalogue_vectors(MIND_TRAIN, MIND_DEV, ENCODER2, CACHE, "cuda" if torch.cuda.is_available() else "cpu", max_len=256)
    CAT_NPZ = stages_v20.concat_npz([(ids_a, va), (ids_b, vb)], os.path.join(WORK, "emb_catkar.npz"), "(Qwen3 + KAR) | BGE-large")
    del va, vb
    gc.collect(); torch.cuda.empty_cache()
    suite.EQUIV_REFERENCE = None
    SCK = run_safely(TAGCK, [*COMMON, *SCALE, "--encoder", ENCODER, "--news-emb", CAT_NPZ, "--work", WORK + "/" + TAGCK, *READ_ARGS, *budget_args(),
                             "--stages", STAGES_R])
print("giờ còn lại: %.2f" % hours_left())
'''

CELL_QK = '''if AUG is None or not os.path.exists(AUG):
    print("qwen3kar bỏ qua: không có vector đã mở rộng")
elif hours_left() < READER_MIN_H:
    print(f"qwen3kar bỏ qua: còn {hours_left():.2f} giờ < {READER_MIN_H}")
else:
    suite.EQUIV_REFERENCE = None
    SQK = run_safely(TAGQK, [*COMMON, *SCALE, "--encoder", ENCODER, "--news-emb", AUG, "--work", WORK + "/" + TAGQK, *READ_ARGS, *budget_args(), "--stages", STAGES_R])
print("giờ còn lại: %.2f" % hours_left())
'''

REPORT_A = '''SUITES_TAGS = [(s, t) for s, t in ((SK, TAGK), (SCK, TAGCK), (SQK, TAGQK)) if s is not None]
SUITES, TAGS = [s for s, _ in SUITES_TAGS], [t for _, t in SUITES_TAGS]
os.makedirs(RESULTS, exist_ok=True)
pairs = [((TAGK, "kar_item"), (TAGK, "frozen")), ((TAGK, "kar_item_lse"), (TAGK, "pool_lse")), ((TAGK, "kar_exp"), (TAGK, "frozen")),
         ((TAGK, "kar_item_lse"), (TAGK, "kar_item")), ((TAGQK, "um_frozen"), (TAGK, "kar_item")), ((TAGQK, "um_knn_rag"), (TAGQK, "um_frozen")),
         ((TAGCK, "um_frozen"), (TAGQK, "um_frozen")), ((TAGCK, "um_knn_rag"), (TAGQK, "um_knn_rag")), ((TAGCK, "um_knn_rag"), (TAGCK, "um_frozen"))]
direct = suite.cross_compare(SUITES, TAGS, pairs, WORK + "/direct_comparisons.md") if SUITES else []
if SK is not None:
    kf = SK.kv.get("kar_full", {})
    lines = ["# H6 at full scale (kar_full)", "", f"* articles expanded: {kf.get('expanded')} of {kf.get('articles')} ({kf.get('model')}, at most {kf.get('max_new_tokens')} new tokens) "
             f"in {kf.get('gen_s', 0) / 60:.0f} min on {kf.get('devices')} GPU(s); embedded in {kf.get('embed_s', 0) / 60:.1f} min; alpha {kf.get('alpha')}", ""]
    for nm in ("frozen", "pool_lse", "kar_item", "kar_item_lse", "kar_exp"):
        try:
            r_ = SK.row(nm)
            lines.append(f"* `{nm}`: test nDCG@10 {r_['ndcg@10']:.4f}, AUC {r_['auc']:.4f}, cold AUC {r_.get('cold_auc', float('nan')):.4f} ({r_['n_test']:,} impressions; cfg {r_['cfg']})")
        except KeyError:
            pass
    open(WORK + "/kar_full.md", "w", encoding="utf-8").write("\\n".join(lines) + "\\n")
    print("\\n".join(lines))
print(report_v18.epochs_table([(s, t) for s, t in ((SCK, TAGCK), (SQK, TAGQK)) if s is not None], WORK + "/epochs.md"))
'''

RUNS_B = """TAGS_ = "sidq"
ARGSS = [*COMMON, *SCALE, "--encoder", ENCODER, "--work", WORK + "/" + TAGS_, "--max-len", "256", "--q-max-len", "384", "--hist-k", "10", "--seeds", "1", "--km-ks", "5",
         "--sid-text", SID_TEXT, "--sid-epochs", SID_EPOCHS, "--sid-bs", "256", "--sid-val-n", "300" if QUICK else "3000", "--sid-shuf-n", "200" if QUICK else "20000",
         "--hm-train-n", HM_TRAIN_N, "--hm-val-n", "300" if QUICK else "20000", "--llm-enc-bs", "64"]
STAGESS = "controls,pooling,sid2,histq_mlp"
if QUICK:
    ARGSS = ARGSS + EXTRA_QUICK
SS = None
print("budget:", BUDGET_H, "h; run sidq: controls -> pooling -> sid2 (Semantic IDs v2) -> histq_mlp (H1 + learned head)")
"""

CELL_S = '''suite.EQUIV_REFERENCE = reference_encoder(ENCODER, 256)
SS = run_safely("sidq", [*ARGSS, *budget_args(), "--stages", STAGESS], release=False)
print("giờ còn lại: %.2f" % hours_left())
'''

REPORT_B = '''SUITES_TAGS = [(SS, TAGS_)] if SS is not None else []
SUITES, TAGS = [s for s, _ in SUITES_TAGS], [t for _, t in SUITES_TAGS]
os.makedirs(RESULTS, exist_ok=True)
Q = TAGS_
pairs = [((Q, "sid_profile"), (Q, "frozen")), ((Q, "sid_gpt"), (Q, "frozen")), ((Q, "sid_gpt"), (Q, "sid_gpt_sumlevels")), ((Q, "sid_gpt_shuffled"), (Q, "sid_gpt")),
         ((Q, "histq_instr"), (Q, "frozen")), ((Q, "histq_mlp"), (Q, "histq_instr")), ((Q, "histq_mlp"), (Q, "frozen")), ((Q, "histq_mlp"), (Q, "pool_lse"))]
direct = suite.cross_compare(SUITES, TAGS, pairs, WORK + "/direct_comparisons.md") if SUITES else []
if SS is not None:
    lines = ["# v20b rows", ""]
    for nm in ("frozen", "pool_lse", "sid_profile", "sid_gpt", "sid_gpt_sumlevels", "sid_gpt_shuffled", "histq_instr", "histq_mlp"):
        try:
            r_ = SS.row(nm)
            lines.append(f"* `{nm}`: test nDCG@10 {r_['ndcg@10']:.4f}, AUC {r_['auc']:.4f}, cold AUC {r_.get('cold_auc', float('nan')):.4f} ({r_['n_test']:,} impressions; cfg {r_['cfg']})"
                         + (f"; per seed {r_['ndcg10_per_seed']}, best epochs {r_.get('best_epochs')}" if r_.get("ndcg10_per_seed") else ""))
        except KeyError:
            pass
    lines += ["", f"SID: {SS.kv.get('sid2')}", f"histq_mlp: {SS.kv.get('histq_mlp')}"]
    open(WORK + "/rows_v20b.md", "w", encoding="utf-8").write("\\n".join(lines) + "\\n")
    print("\\n".join(lines))
    SS.release()
'''


def summary_ab(p, tags):
    s = V19.SUMMARY.replace('RUN_TAGS = [t for t in (TAG, TAG2, TAG3, "catplus", "bgeplus") if os.path.isdir(os.path.join(WORK, t))]',
                            f"RUN_TAGS = [t for t in {tags!r} if os.path.isdir(os.path.join(WORK, t))]")
    assert "RUN_TAGS = [t for t in (" in s
    s = s.replace('parts = [("HEADLINE_v19@P@.md", WORK),', '''parts = [("kar_full.md", WORK), ("rows_v20b.md", WORK), ("HEADLINE_v19@P@.md", WORK),''')
    s = s.replace('for fn in ("forest.png",', 'for fn in ("kar_full.md", "rows_v20b.md", "forest.png",')
    return s.replace("v19@P@", "v20" + p)


def config_cell(p, budget):
    s = (V18.CONFIG.replace("v18", "v20").replace("@BUDGET@", budget).replace("@STACKMIN@", "4200").replace("@P@", p)
         .replace("@PROFILE@", PROFILE_A if p == "a" else PROFILE_B).replace("@PRINT@", PRINT_A if p == "a" else PRINT_B))
    return code(s)


def build_ab(p):
    a = p == "a"
    head = [md("\n\n".join([A_TEXT if a else B_TEXT, BEST, HOW_AB.replace("@P@", p), PROTOCOL])), md("### 0. Phụ thuộc và GPU"), code(V17.DEPS),
            md("### 1. Modules (ghi ra thư mục làm việc)")]
    cells = head + [writefile(m) for m in MODULES]
    cells += [md("### 2. Cấu hình"), config_cell(p, "11.3" if a else "8"), md("### 3. Dữ liệu — MIND-large tải từ Hugging Face (cần Internet)"), code(V18.DATA),
              md("### 4. Cấu hình các run"), code(V18.RUNS_COMMON + (RUNS_A if a else RUNS_B))]
    if a:
        cells += [md("### 5. Run `kar` — LLM viết mở rộng cho mọi bài (2 GPU), các hàng H6 trên toàn bộ test, vector đã mở rộng"), code(CELL_KAR),
                  md("### 6. Run `catkarplus` — reader trên (Qwen3 + KAR) ⊕ BGE-large, công thức của `catplus`"), code(CELL_CATK),
                  md("### 7. Run `qwen3kar` — reader trên Qwen3 + KAR"), code(CELL_QK),
                  md("### 8. So sánh trực tiếp"), code(REPORT_A)]
    else:
        cells += [md("### 5. Run `sidq` — SID v2 + H1 có lớp học được"), code(CELL_S), md("### 6. So sánh trực tiếp"), code(REPORT_B)]
    cells += [md("### Tổng hợp và file gửi lại (điểm thành phần ở `results_v20" + p + "/scores/<run>/`)"),
              code(summary_ab(p, ("kar", "catkarplus", "qwen3kar") if a else ("sidq",)))]
    return cells


# ------------------------------------------------------------------------------------------------ v20c
C_INTRO = """# V20c — ghép mọi thành phần LLM-text mới vào hệ tốt nhất (v19c), không huấn luyện lại · đối chứng = hệ tốt nhất · MIND-large · Kaggle T4 x2 (~3–5 giờ)

**Đầu vào:** output của `v19a-mindlarge-plusreaders` và `v19b-mindlarge-h5tower` (`kernel_sources`) + một dataset riêng tư chứa điểm thành phần của v20a / v20b dưới dạng file phẳng `<a|b>__<run>__<thành phần>.npz` (Kaggle không gắn output riêng tư của tài khoản khác). Notebook dừng ngay nếu thiếu đầu vào.

**Các nhánh (cùng validation / test → CI ghép cặp):**
1. `ctrl_s0` — **đối chứng = hệ tốt nhất hiện tại** (thí nghiệm của v19c: pipeline v16b + reader kỹ hơn + tháp H5). Pipeline tất định nên phải tái lập v19c đến 4 chữ số (`control_vs_v19c.md`).
2. `treat_s0` — **thí nghiệm**: + H6 toàn bộ (`kar_item`, `kar_item_lse`, reader `catkarplus`, `qwen3kar`) + SID v2 (`sid_gpt`, `sid_profile`) + `histq_mlp`.
3. `ctrl_cw0`, `treat_cw0` — supermodel chọn theo validation **không trọng số**. v19c chỉ coi đây là khám phá; ở v20c quy tắc này được **đăng ký trước** như quy tắc phụ: báo cáo cả hai, cổng chính vẫn là quy tắc có trọng số.
4. `ctrl_s1`, `treat_s1` — stacker seed 1 (nhiễu seed + hàng trung bình 2 seed), chỉ khi còn đủ giờ.

**Cổng (viết trước):** G1 `super_llm`, G2 `stack_meta`, G3 `stack_pop` ở seed 0: PASS nếu CI 95% ghép cặp của Δ (thí nghiệm − đối chứng) nằm trên 0. Đối chứng ở đây chính là hệ tốt nhất, nên PASS = **mô hình mới nhất tốt hơn hệ tốt nhất**.

**Kèm theo:** `components_v20c.md` — so sánh ghép cặp từng thành phần mới với thành phần nó cần vượt (`catkarplus` với `catplus`, `qwen3kar` với reader Qwen3, `kar_item_lse` với `pool_lse` / `txt_cta_lse`, `histq_mlp` với H1 / H5 / reader, `sid_gpt` với mean pooling).

""" + PROTOCOL

PROFILE_C = V19C.PROFILE_C + """
V20_DATA = os.environ.get("V11_V20_DATA", "")                                  # thư mục dataset điểm thành phần v20 (trống: tự tìm trong V19_INPUT)"""

INPUTS_C = '''import fusion_only
A_ROOT = fusion_only.find_scores("a", [V19_INPUT], V19A_SCORES or None)
B_ROOT = fusion_only.find_scores("b", [V19_INPUT], V19B_SCORES or None)
V20_ROOTS = fusion_only.unflatten_inputs(V20_DATA or V19_INPUT, os.path.join(WORK, "v20_inputs"))
print("điểm của v19a:", A_ROOT, "| v19b:", B_ROOT, "| v20:", V20_ROOTS)
ARMS = ["ctrl_s0", "treat_s0"] + (["ctrl_cw0", "treat_cw0"] if EXPLORE else []) + (["ctrl_s1", "treat_s1"] if SEED_ARMS else [])
PREP = fusion_only.prepare_v20(WORK, A_ROOT, B_ROOT, [V20_ROOTS[k] for k in ("a", "b") if k in V20_ROOTS], arms=ARMS)
if len(V20_ROOTS) < 2:
    PREP["missing"].append("v20a / v20b components (dataset of flat files <a|b>__<run>__<component>.npz)")
if PREP["missing"]:
    raise SystemExit("thiếu đầu vào: " + ", ".join(PREP["missing"]) + " -> notebook dừng ở đây (chưa tốn giờ GPU).")
print("tháp H5:", PREP["tower"], "| thành phần mới:", PREP["new"])
for t_, c_ in PREP["checks"].items():
    print(f"bản của v19a so với bản của v19b, run {t_}: {c_['identical']}/{c_['shared']} file giống hệt (lệch lớn nhất {c_['max_abs_diff']:.3g})")
for a_, d_ in PREP["arms"].items():
    print(f"{a_}: {d_['components']} thành phần của run chính + super-dirs {[os.path.basename(x) for x in d_['super']]}")
'''

COMP_PAIRS = '''RUNS_DIR = os.path.join(WORK, "runs")
M_ = PREP["arms"]["treat_s0"]["work"]
rn = lambda t: os.path.join(RUNS_DIR, t)
COMP = [((rn("catkarplus"), "um_knn_rag"), (rn("catplus"), "um_knn_rag")), ((rn("catkarplus"), "um_frozen"), (rn("catplus"), "um_frozen")),
        ((rn("qwen3kar"), "um_frozen"), (M_, "um_frozen")), ((rn("qwen3kar"), "um_knn_rag"), (M_, "um_knn_rag")),
        ((rn("kar"), "kar_item"), (M_, "frozen")), ((rn("kar"), "kar_item_lse"), (M_, "pool_lse")), ((rn("kar"), "kar_item_lse"), (M_, "txt_cta_lse")),
        ((rn("sidq"), "sid_gpt"), (M_, "frozen")), ((rn("sidq"), "sid_profile"), (M_, "frozen")),
        ((rn("sidq"), "histq_mlp"), (rn("sidq"), "histq_instr")), ((rn("sidq"), "histq_mlp"), (M_, "ut_causal")), ((rn("sidq"), "histq_mlp"), (M_, "um_frozen")),
        ((M_, "ut_causal"), (rn("sidq"), "histq_instr")), ((rn("catkarplus"), "um_knn_rag"), (M_, "ut_causal"))]
try:
    print(fusion_only.component_table(S_EVAL, COMP, WORK + "/components_v20c.md")[0] if S_EVAL is not None else "(no run with data loaded: component table skipped)")
except Exception as e:
    print("!! component table failed:", repr(e)[:300])
'''


def report_c():
    s = V19C.REPORT.replace("v19c", "v20c")
    s = s.replace('BESTS = dict(getattr(report_v18, "BEST_V19", report_v18.BEST_V18))', 'BESTS = {"super_llm": 0.4380, "stack_meta": 0.4570, "stack_pop": 0.4731}')
    s = s.replace('title="v20c: control (v16b components) vs treatment (+ v19a\'s readers trained harder + v19b\'s H5 tower), same impressions, stacker seed 0"',
                  'title="v20c: control (= v19c\'s treatment, the best system so far) vs treatment (+ the v20 LLM-text components), same impressions, stacker seed 0"')
    old_vs = ('    print(report_v18.vs_v16b_table([(n, S_c0, n, n) for n in report_v18.FUSED], WORK + "/control_vs_v16b.md",\n'
              '                                   title="v20c control vs v16b (the same components, another run: a deterministic pipeline gives 0)", label="v20c control")[0])\n')
    new_vs = ('    print(report_v18.vs_v16b_table([(n, S_c0, n, n) for n in report_v18.FUSED], WORK + "/control_vs_v19c.md",\n'
              '                                   title="v20c control vs v19c\'s treatment (the same components, another run: a deterministic pipeline gives 0)", label="v20c control",\n'
              '                                   ref_numbers=report_v18.V19C_TREAT, ref_label="v19c treatment")[0])\n')
    assert old_vs in s, "report cell changed"
    s = s.replace(old_vs, new_vs)
    old_ex = '    print(fusion_only.explore_table(S_ARM, WORK + "/exploratory_v20c.md")[0])\n'
    new_ex = ('    print(fusion_only.explore_table(S_ARM, WORK + "/plainval_v20c.md", header=["# v20c: supermodel chosen on the plain validation mean (--cold-target 0) next to the '
              'cold-weighted rule", "", "Pre-registered in v20c as the SECONDARY rule (v19c found it after looking at v19a\'s test day); the gates use the cold-weighted rule.", ""])[0])\n')
    assert old_ex in s
    s = s.replace(old_ex, new_ex)
    old_rel = "if S_EVAL is not None:\n    S_EVAL.release()\n"
    assert old_rel in s
    return s.replace(old_rel, COMP_PAIRS + old_rel)


def summary_c():
    s = V19C.SUMMARY.replace("v19c", "v20c").replace('"exploratory_v20c.md"', '"plainval_v20c.md"').replace('"control_vs_v16b.md"', '"control_vs_v19c.md", "components_v20c.md"')
    s = s.replace('inputs = ["# Inputs", "", f"v19a scores: {A_ROOT}", f"v19b scores: {B_ROOT}", f"H5 tower: {PREP[\'tower\']}", ""]',
                  'inputs = ["# Inputs", "", f"v19a scores: {A_ROOT}", f"v19b scores: {B_ROOT}", f"v20 scores: {V20_ROOTS}", f"H5 tower: {PREP[\'tower\']}", '
                  'f"new components: {PREP[\'new\']}", ""]')
    assert "v20 scores" in s
    return s


def build_c():
    cfg = (V18.CONFIG.replace("v18", "v20").replace("@BUDGET@", "4.6").replace("@STACKMIN@", "4200").replace("@P@", "c")
           .replace("@PROFILE@", PROFILE_C).replace("@PRINT@", V19C.PRINT_C))
    cells = [md(C_INTRO), md("### 0. Phụ thuộc và GPU"), code(V17.DEPS), md("### 1. Modules (ghi ra thư mục làm việc)")]
    cells += [writefile(m) for m in MODULES]
    cells += [md("### 2. Cấu hình"), code(cfg),
              md("### 3. Đầu vào — điểm thành phần của v19a, v19b và v20a / v20b (dừng ngay nếu thiếu)"), code(INPUTS_C),
              md("### 4. Dữ liệu — MIND-large tải từ Hugging Face (cần Internet)"), code(V18.DATA),
              md("### 5. Cấu hình các nhánh"), code(V18.RUNS_COMMON + V19C.RUNS_C),
              md("### 6. Nhánh **ĐỐI CHỨNG** (`ctrl_s0`) = hệ tốt nhất hiện tại (thí nghiệm của v19c)"), code(V19C.CELL_CTRL),
              md("### 7. Nhánh **THÍ NGHIỆM** (`treat_s0`) — + mọi thành phần LLM-text của v20"), code(V19C.CELL_TREAT),
              md("### 8. Quy tắc phụ (đăng ký trước): supermodel chọn theo validation không trọng số (`ctrl_cw0`, `treat_cw0`)"), code(V19C.CELL_EXPLORE),
              md("### 9. Stacker với seed 1 (`ctrl_s1`, `treat_s1`) — khi còn đủ giờ"), code(V19C.CELL_SEEDS),
              md("### 10. Bảng chính, cổng, seed, quy tắc phụ, từng thành phần mới"), code(report_c()),
              md("### Tổng hợp và file gửi lại"), code(summary_c())]
    return cells


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                                         "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    for p, fn in NOTEBOOKS.items():
        nb = notebook(build_c() if p == "c" else build_ab(p))
        out = os.path.join(HERE, fn)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", out, "cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
