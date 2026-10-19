"""Build the fusion-only notebook v19c (MIND-large, Kaggle "GPU T4 x2", ~3-5.5 hours): the supermodel and the stacker over the components of BOTH v19 notebooks,
without training any component again.

  v19c_mindlarge_fusion.ipynb  inputs: the outputs of v19a (the v16b runs + the readers trained harder, `catplus` / `bgeplus`) and v19b (the main run with the H5
                               LLM user tower), mounted under /kaggle/input (kernel_sources).  Arms, each a supermodel + stacker over linked score files:
                                 ctrl_s0   the v16b components (in-run CONTROL; should reproduce v16b / the v19 controls exactly)
                                 treat_s0  + v19a's readers trained harder + v19b's H5 tower (TREATMENT)
                                 ctrl_s1 / treat_s1   the stacker again with seed 1 (it subsamples rows and columns): the stacker's own training noise, measured,
                                           and the mean of the two seeds (an ensemble of two stackers) as rows of their own.
                               Gates as in v19 (G1 super_llm, G2 stack_meta, G3 stack_pop on the paired CI at seed 0), plus the robustness check over the seeds.

Bump the notebook name when the content changes: Kaggle caches notebooks by file name."""
import json
import os

import make_notebook_v17 as V17
import make_notebook_v18 as V18
from make_notebook import code, md, writefile  # re-used cell helpers

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = "v19c_mindlarge_fusion.ipynb"
MODULES = V18.MODULES[:-1] + ["fusion_only.py", "suite.py"]
assert V18.MODULES[-1] == "suite.py"

# ------------------------------------------------------------------------------------------------ text
INTRO = """# V19c — ghép (supermodel + stacker) mọi thành phần của v19a và v19b, không huấn luyện lại · có đối chứng và 2 seed của stacker · MIND-large · Kaggle T4 x2 (~3–5,5 giờ)

**Đầu vào:** output của hai notebook `v19a-mindlarge-plusreaders` (pipeline v16b + reader huấn luyện kỹ hơn `catplus` / `bgeplus`) và `v19b-mindlarge-h5tower` (run chính có tháp người dùng LLM H5). Kaggle gắn chúng dưới `/kaggle/input` (khai trong `kernel_sources`; chạy tay: *Add Input → Notebook output*). Điểm của từng thành phần được **liên kết**, không sao chép; notebook dừng ngay (vài giây, chưa tải gì) nếu thiếu một phần.

**Câu hỏi:** v19a đo reader kỹ hơn, v19b đo H5, mỗi cái riêng với đối chứng của nó. Gộp cả hai vào supermodel + stacker thì hệ thống có vượt kết quả tốt nhất không?

**Các nhánh (cùng validation / test → CI ghép cặp):**
1. `ctrl_s0` — **đối chứng**: chỉ thành phần của pipeline v16b (BGE-large, vector nối Qwen3-0,6B ⊕ BGE-large, Qwen3-0,6B). Pipeline tất định nên nhánh này phải tái lập v16b và đối chứng của v19 đến 4 chữ số.
2. `treat_s0` — **thí nghiệm**: + reader huấn luyện kỹ hơn của v19a + tháp H5 của v19b (H5 là một reader phủ toàn bộ split).
3. `ctrl_cw0`, `treat_cw0` — **khám phá, không phải cổng** (~25 phút): chỉ supermodel, chọn nền và trọng số theo validation **không trọng số** (`--cold-target 0`) thay vì validation có trọng số bài lạnh. Lý do (đo ở v19a, sau khi đã xem test): validation có trọng số xếp reader huấn luyện kỹ hơn **dưới** reader cũ (0,4532 so với 0,4573), test thì ngược lại (+0,0091), nên supermodel giữ nền cũ. Vì quy tắc này được chọn sau khi xem test, mọi mức tăng ở đây chỉ là gợi ý cho một lần chạy đăng ký trước, không phải kết quả.
4. `ctrl_s1`, `treat_s1` — stacker chạy lại với **seed 1**. Stacker lấy mẫu con 70% hàng và 70% cột, nên seed là một nguồn nhiễu huấn luyện mà CI ghép cặp không tính (v18a: thêm cột làm các hàng stacker dịch −0,0019 … +0,0026). Hai nhánh này **đo trực tiếp** nhiễu đó (`ctrl_s1` − `ctrl_s0`: cùng thành phần, khác seed) và cho thêm **hàng trung bình 2 seed** (ghép hai stacker; so với trung bình 2 seed của đối chứng). Chỉ chạy khi còn đủ giờ.

**Cổng (viết trước khi chạy):** G1 `super_llm`, G2 `stack_meta`, G3 `stack_pop` ở seed 0: PASS nếu CI 95% ghép cặp của Δ (thí nghiệm − đối chứng) nằm trên 0. "Vượt kết quả tốt nhất" khi PASS và số cao hơn số tốt nhất trước đó; chắc chắn hơn khi Δ cũng dương ở seed 1 và ở trung bình 2 seed, và lớn hơn nhiễu seed.

**Thời gian (từ số đo v18a):** tải dữ liệu + mã hoá Qwen3-0,6B ≈ 0,3 giờ; mỗi nhánh seed 0 ≈ 1,2–1,5 giờ (supermodel ~5 phút, stacker 55–75 phút); mỗi nhánh seed 1 ≈ 1,1–1,4 giờ (chỉ stacker). Hai nhánh khám phá ≈ 0,4 giờ. Tổng ≈ 6 giờ, `BUDGET_H` 7,5.

**Cách chạy:** Settings → Accelerator **GPU T4 x2**, Internet **On** (MIND-large và Qwen3-Embedding-0,6B tải từ Hugging Face) → **Save Version / Run All**. **Nhận lại:** `SUMMARY_v19c.md` trong `/kaggle/working/results_v19c` + log.

Giao thức như v16b: validation theo ngày (giữ ngẫu nhiên 150 000 impression), `MINDlarge_dev` chỉ là test và chỉ chạm một lần cho mỗi hàng, không dùng nhãn test để chọn gì."""

PROFILE_C = """V19_INPUT = os.environ.get("V11_V19_INPUT", "/kaggle/input")                 # nơi Kaggle gắn output của v19a / v19b
V19A_SCORES = os.environ.get("V11_V19A_SCORES", "")                           # đường dẫn thẳng tới results_v19a/scores (trống: tự tìm)
V19B_SCORES = os.environ.get("V11_V19B_SCORES", "")
SEED_ARMS = bool(int(os.environ.get("V11_SEED_ARMS", "1")))                   # stacker chạy lại với seed 1 (nhiễu của stacker + hàng trung bình 2 seed)
EXPLORE = bool(int(os.environ.get("V11_EXPLORE", "1")))                       # khám phá: supermodel chọn theo validation không trọng số (--cold-target 0)
ARM_MIN_H = float(os.environ.get("V11_ARM_MIN_H", "0" if QUICK else "1.6"))   # giờ còn lại tối thiểu để bắt đầu một nhánh seed 1"""
PRINT_C = ", V19_INPUT=V19_INPUT, SEED_ARMS=SEED_ARMS, EXPLORE=EXPLORE"

INPUTS = '''import fusion_only
A_ROOT = fusion_only.find_scores("a", [V19_INPUT], V19A_SCORES or None)
B_ROOT = fusion_only.find_scores("b", [V19_INPUT], V19B_SCORES or None)
print("điểm của v19a:", A_ROOT, "| điểm của v19b:", B_ROOT)
ARMS = ["ctrl_s0", "treat_s0"] + (["ctrl_cw0", "treat_cw0"] if EXPLORE else []) + (["ctrl_s1", "treat_s1"] if SEED_ARMS else [])
PREP = fusion_only.prepare(WORK, A_ROOT, B_ROOT, arms=ARMS)
if PREP["missing"]:
    raise SystemExit("thiếu đầu vào: " + ", ".join(PREP["missing"]) + " -> notebook dừng ở đây (chưa tốn giờ GPU). Gắn output của cả hai notebook v19a / v19b.")
print("tháp H5:", PREP["tower"])
for t_, c_ in PREP["checks"].items():
    print(f"bản của v19a so với bản của v19b, run {t_}: {c_['identical']}/{c_['shared']} file giống hệt (lệch lớn nhất {c_['max_abs_diff']:.3g})"
          + (f"; khác: {c_['differ'][:6]}" if c_["differ"] else ""))
for a_, d_ in PREP["arms"].items():
    print(f"{a_}: {d_['components']} thành phần của run chính + super-dirs {[os.path.basename(x) for x in d_['super']]}")
'''

RUNS_C = """def with_work(argv, path):
    out = list(argv)
    out[out.index("--work") + 1] = path
    return out

if QUICK:
    ARGS = ARGS + EXTRA_QUICK
SEED_OF = {"ctrl_s0": 0, "treat_s0": 0, "ctrl_cw0": 0, "treat_cw0": 0, "ctrl_s1": 1, "treat_s1": 1}
S_ARM = {}

def run_arm(arm, last=False):
    \"\"\"Một nhánh = một suite.main trên thư mục của nó: seed 0 chạy supermodel + stacker, seed 1 chỉ stacker (supermodel không có yếu tố ngẫu nhiên),
    nhánh khám phá _cw0 chỉ supermodel với --cold-target 0.\"\"\"
    a = PREP["arms"][arm]
    stages = "supermodel,stack" if arm.endswith("_s0") else ("supermodel" if arm.endswith("_cw0") else "stack")
    extra = ["--cold-target", "0"] if arm.endswith("_cw0") else []
    suite.EQUIV_REFERENCE = reference_encoder(ENCODER, 256) if arm == ARMS[0] else None    # E0 được kiểm một lần; các nhánh sau đọc nó từ cache
    argv = [*with_work(ARGS, a["work"]), "--super-dirs", ",".join(a["super"]), "--stack-seed", str(SEED_OF[arm]), "--min-stack-s", str(STACK_MIN_S),
            *extra, *budget_args(), "--stages", stages]
    s = S_ARM[arm] = run_safely(arm, argv, release=False)
    if s is not None and arm == "treat_s0":            # the one-run report reads the data: written before they are released
        try:
            print(final_report.build(s, WORK + "/FINAL_REPORT_treat.md", WORK + "/final_tables_treat.json"))
        except Exception as e:
            print("!! treatment report failed:", repr(e)[:200])
    if s is not None and not last:
        s.release()                                    # RAM: the rows and the per-impression results stay, the data go
        gc.collect(); torch.cuda.empty_cache()
    print(f"{arm}: {'xong' if s is not None else 'LỖI'}; giờ còn lại {hours_left():.2f}")
    return s

print("budget:", BUDGET_H, "h; arms:", ARMS)
"""

CELL_CTRL = '''run_arm("ctrl_s0")
'''

CELL_TREAT = '''run_arm("treat_s0", last=not (SEED_ARMS or EXPLORE))
'''

CELL_EXPLORE = '''if EXPLORE:
    for arm in ("ctrl_cw0", "treat_cw0"):
        if hours_left() < (0 if QUICK else 0.5):
            print(f"{arm} bỏ qua: còn {hours_left():.2f} giờ")
            continue
        run_arm(arm)
'''

CELL_SEEDS = '''if SEED_ARMS:
    for arm in ("ctrl_s1", "treat_s1"):
        if hours_left() < ARM_MIN_H:
            print(f"{arm} bỏ qua: còn {hours_left():.2f} giờ < {ARM_MIN_H}")
            continue
        run_arm(arm, last=(arm == "treat_s1"))
'''

REPORT = '''S_c0, S_t0, S_c1, S_t1 = (S_ARM.get(k) for k in ("ctrl_s0", "treat_s0", "ctrl_s1", "treat_s1"))
S_EVAL = S_t1 if S_t1 is not None and getattr(S_t1, "es", None) is not None else (S_t0 if S_t0 is not None and getattr(S_t0, "es", None) is not None else None)
BESTS = dict(getattr(report_v18, "BEST_V19", report_v18.BEST_V18))
print("kết quả tốt nhất trước đó:", BESTS)
os.makedirs(RESULTS, exist_ok=True)
S_cp, S_tp = S_ARM.get("ctrl_cw0"), S_ARM.get("treat_cw0")
SUITES_TAGS = [(s, t) for s, t in ((S_c0, "ctrl"), (S_t0, "treat"), (S_c1, "ctrl_s1"), (S_t1, "treat_s1"), (S_cp, "ctrl_cw0"), (S_tp, "treat_cw0")) if s is not None]
SUITES, TAGS = [s for s, _ in SUITES_TAGS], [t for _, t in SUITES_TAGS]
pairs = [(("treat", n), ("ctrl", n)) for n in report_v18.FUSED]
pairs += [((a, n), (b, n)) for n in fusion_only.STACK_ROWS for a, b in (("treat_s1", "ctrl_s1"), ("ctrl_s1", "ctrl"), ("treat_s1", "treat"))]
pairs += [((a, n), (b, n)) for n in fusion_only.SUPER_ROWS for a, b in (("ctrl_cw0", "ctrl"), ("treat_cw0", "treat"), ("treat_cw0", "ctrl_cw0"))]
direct = suite.cross_compare(SUITES, TAGS, pairs, WORK + "/direct_comparisons.md") if SUITES else []
fusion_md = gate_md = seed_md = ""
if S_c0 is not None and S_t0 is not None:
    fusion_md, fusion = report_v18.fusion_table(S_c0, S_t0, WORK + "/HEADLINE_v19c.md",
                                                title="v19c: control (v16b components) vs treatment (+ v19a's readers trained harder + v19b's H5 tower), same impressions, stacker seed 0")
    gate_md, gates = report_v18.gate_v18a(S_c0, S_t0, WORK + "/gate_v19c.md", bests=BESTS,
                                          title="v19c gates (written before the run; decided on the paired test CI of the treatment against the in-run control, stacker seed 0)")
    print(fusion_md); print(); print(gate_md)
if S_c0 is not None:
    print(report_v18.vs_v16b_table([(n, S_c0, n, n) for n in report_v18.FUSED], WORK + "/control_vs_v16b.md",
                                   title="v19c control vs v16b (the same components, another run: a deterministic pipeline gives 0)", label="v19c control")[0])
files = {k: [os.path.join(PREP["arms"][a]["work"], "stack_scores.npz") for a in (k + "_s0", k + "_s1") if a in PREP["arms"]] for k in ("ctrl", "treat")}
try:
    seed_md, seeds = fusion_only.seed_table(S_ARM, files, S_eval=S_EVAL if SEED_ARMS else None, path=WORK + "/seeds_v19c.md", bests=BESTS)
    print(seed_md)
except Exception as e:
    print("!! seed table failed:", repr(e)[:300])
if EXPLORE:
    print(fusion_only.explore_table(S_ARM, WORK + "/exploratory_v19c.md")[0])
print(fusion_only.supermodel_table(S_t0, WORK + "/supermodel_treat.md"))
print(fusion_only.supermodel_table(S_c0, WORK + "/supermodel_ctrl.md", title="Supermodel of the control: components and weights chosen on validation"))
print(fusion_only.stack_table(S_t0, WORK + "/stack_treat.md"))
print(fusion_only.stack_table(S_c0, WORK + "/stack_ctrl.md", title="Stacker of the control: features and what the trees use"))
if S_EVAL is not None:
    S_EVAL.release()
'''

SUMMARY = '''inputs = ["# Inputs", "", f"v19a scores: {A_ROOT}", f"v19b scores: {B_ROOT}", f"H5 tower: {PREP['tower']}", ""]
for t_, c_ in PREP["checks"].items():
    inputs.append(f"* v19a's copy vs v19b's copy of run `{t_}`: {c_['identical']}/{c_['shared']} files identical, largest difference {c_['max_abs_diff']:.3g}"
                  + (f" (differ: {', '.join(c_['differ'])})" if c_["differ"] else ""))
for a_, d_ in PREP["arms"].items():
    inputs.append(f"* arm `{a_}`: {d_['components']} components of the main run + runs {[os.path.basename(x) for x in d_['super']]}; "
                  + ("done" if S_ARM.get(a_) is not None else "NOT RUN / FAILED"))
open(os.path.join(WORK, "inputs.md"), "w", encoding="utf-8").write("\\n".join(inputs) + "\\n")
print("tổng thời gian của notebook: %.2f h" % ((time.time() - T_NB) / 3600))
parts = [(f, WORK) for f in ("HEADLINE_v19c.md", "gate_v19c.md", "seeds_v19c.md", "exploratory_v19c.md", "control_vs_v16b.md", "direct_comparisons.md", "inputs.md", "supermodel_treat.md",
                             "supermodel_ctrl.md", "stack_treat.md", "stack_ctrl.md", "FINAL_REPORT_treat.md")]
for a_ in ARMS:
    parts += [(f, PREP["arms"][a_]["work"]) for f in ("timings.json", "memory.json", "results.md")]
SUMMARY = os.path.join(RESULTS, "SUMMARY_v19c.md")
with open(SUMMARY, "w", encoding="utf-8") as out:
    out.write(f"# SUMMARY v19c ({'QUICK' if QUICK else 'full'}; {(time.time() - T_NB) / 3600:.2f} h)\\n")
    for fn, d in parts:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out.write(f"\\n\\n## {fn} ({os.path.basename(os.path.dirname(d)) if fn in ('timings.json', 'memory.json', 'results.md') else os.path.basename(d)})\\n\\n"
                      + open(p, encoding="utf-8").read())
for fn in ("HEADLINE_v19c.md", "gate_v19c.md", "seeds_v19c.md", "exploratory_v19c.md", "control_vs_v16b.md", "direct_comparisons.md", "inputs.md", "supermodel_treat.md", "supermodel_ctrl.md",
           "stack_treat.md", "stack_ctrl.md", "FINAL_REPORT_treat.md", "final_tables_treat.json"):
    if os.path.exists(os.path.join(WORK, fn)) and os.path.abspath(WORK) != os.path.abspath(RESULTS):
        shutil.copy(os.path.join(WORK, fn), os.path.join(RESULTS, fn))
for a_ in ARMS:                                        # the stackers' scores (val out-of-fold + test of every variant) and the rows of every arm, for later analysis
    w_ = PREP["arms"][a_]["work"]
    for fn in ("stack_scores.npz", "results.json", "results.md"):
        if os.path.exists(os.path.join(w_, fn)):
            os.makedirs(os.path.join(RESULTS, a_), exist_ok=True)
            shutil.copy(os.path.join(w_, fn), os.path.join(RESULTS, a_, fn))
print("-> gửi lại file:", SUMMARY, f"({os.path.getsize(SUMMARY) / 1024:.0f} KB)")
'''

READ = """### Cách đọc kết quả (v19c)

| Bạn thấy | Nghĩa là |
|---|---|
| `control_vs_v16b.md`: mọi hiệu = 0 | đối chứng tái lập v16b (và đối chứng của v19): đầu vào được liên kết đúng |
| G2 / G3 **PASS** và số > số tốt nhất trước đó | gộp reader kỹ hơn + H5 làm hệ thống tốt hơn cả đối chứng trong notebook lẫn kết quả trước đây |
| `seeds_v19c.md`: Δ dương ở seed 0, seed 1 và trung bình 2 seed, lớn hơn nhiễu seed | mức tăng không phải do huấn luyện lại stacker |
| hàng trung bình 2 seed của thí nghiệm > số tốt nhất | ghép hai stacker là một công thức mới, tốt hơn; đối chứng của nó là trung bình 2 seed của đối chứng |
| `exploratory_v19c.md`: chọn theo validation không trọng số nâng `super_llm` | quy tắc chọn đang giấu mức tăng của reader mới — chỉ là gợi ý (quy tắc được chọn sau khi xem test), cần một lần chạy đăng ký trước |

**Giới hạn:** một ngày test; CI chỉ tính nhiễu mẫu test (nhiễu seed của stacker được đo riêng, không gộp vào CI); reader và tháp H5 không được huấn luyện lại (nhiễu seed của chúng không có ở đây)."""


def config_cell():
    s = (V18.CONFIG.replace("v18", "v19").replace("@BUDGET@", "7.5").replace("@STACKMIN@", "4200").replace("@P@", "c")
         .replace("@PROFILE@", PROFILE_C).replace("@PRINT@", PRINT_C))
    return code(s)


def build():
    cells = [md(INTRO), md("### 0. Phụ thuộc và GPU"), code(V17.DEPS), md("### 1. Modules (ghi ra thư mục làm việc)")]
    cells += [writefile(m) for m in MODULES]
    cells += [md("### 2. Cấu hình"), config_cell(),
              md("### 3. Đầu vào — điểm thành phần của v19a và v19b (dừng ngay nếu thiếu)"), code(INPUTS),
              md("### 4. Dữ liệu — MIND-large tải từ Hugging Face (cần Internet)"), code(V18.DATA),
              md("### 5. Cấu hình các nhánh"), code(V18.RUNS_COMMON + RUNS_C),
              md("### 6. Nhánh **ĐỐI CHỨNG** (`ctrl_s0`) — supermodel + stacker trên thành phần của pipeline v16b"), code(CELL_CTRL),
              md("### 7. Nhánh **THÍ NGHIỆM** (`treat_s0`) — + reader huấn luyện kỹ hơn (v19a) + tháp H5 (v19b)"), code(CELL_TREAT),
              md("### 8. Khám phá (`ctrl_cw0`, `treat_cw0`) — supermodel chọn theo validation không trọng số (không phải cổng)"), code(CELL_EXPLORE),
              md("### 9. Stacker với seed 1 (`ctrl_s1`, `treat_s1`) — nhiễu huấn luyện của stacker và hàng trung bình 2 seed"), code(CELL_SEEDS),
              md("### 10. Bảng chính, cổng, seed, khám phá, so sánh trực tiếp"), code(REPORT),
              md("### Tổng hợp và file gửi lại"), code(SUMMARY), md(READ)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                                         "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    nb = build()
    out = os.path.join(HERE, NOTEBOOK)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", out, "cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
