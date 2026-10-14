"""Figures for the report (PNG, 200 dpi) from facts.FACTS -- the slides use native charts built from the same numbers."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from facts import FACTS as F  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/fig"
FS = float(os.environ.get("FIG_FS", "1"))      # font multiplier (slides use larger text)
FW = float(os.environ.get("FIG_FW", "1"))      # figure-size multiplier
os.makedirs(OUT, exist_ok=True)

NAVY, RED, GRAY, LIGHT, GOLD, GREEN, TEAL = "#003366", "#C02034", "#8C8C8C", "#7A9CC6", "#D99A2B", "#3A7D44", "#2E7D8C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9 * FS, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#222222", "xtick.color": "#333333", "ytick.color": "#333333",
                     "axes.titleweight": "bold", "axes.titlesize": 10 * FS})


def vn(x, nd=4, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


_subplots = plt.subplots


def _subplots_scaled(*a, **k):
    if "figsize" in k:
        k["figsize"] = (k["figsize"][0] * FW, k["figsize"][1] * FW)
    return _subplots(*a, **k)


plt.subplots = _subplots_scaled


def comma(*axes_axis):
    """Comma decimal separator on the given numeric axes (Vietnamese convention)."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _p: (f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:g}").replace(".", ",").replace("-", "−"))
    for axis in axes_axis:
        axis.set_major_formatter(fmt)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ 1. pipeline diagram
def box(ax, x, y, w, h, title, body, fc, tc="white", fs=7.4, bfs=6.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=fs, color=tc, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.068, body, ha="center", va="top", fontsize=bfs, color=tc, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#444444"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=0, shrinkB=0))


def fig_pipeline(name="fig01_pipeline.png", W=7.4, H=4.6):
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(-0.015, 1.015); ax.set_ylim(-0.01, 1.0); ax.axis("off")
    box(ax, 0.00, 0.62, 0.20, 0.36, "Dữ liệu MIND", "news.tsv: tiêu đề, abstract,\nchuyên mục, entity\nbehaviors.tsv: lịch sử\n+ impression (có nhãn)", NAVY)
    box(ax, 0.00, 0.18, 0.20, 0.38, "Chia theo thời gian", "train_core\nvalidation = ngày cuối\ncủa train\ntest = dev chính thức\n(chạm một lần)", "#3B4F8C")
    box(ax, 0.25, 0.62, 0.21, 0.36, "Mã hoá đóng băng", "BGE-small / BGE-large\n(CLS)\nQwen3-Embedding-0.6B\n(last-token)\nnối Qwen3 ⊕ BGE-large", TEAL)
    box(ax, 0.25, 0.18, 0.21, 0.38, "Đặc trưng không nhãn", "tuổi bài, số lần hiển thị\ntrước đó (1/6/24 giờ),\nđếm theo bài đến hôm\ntrước, CTR chuyên mục\n(+ pop: streaming)", "#5B6B8F")
    comps = [("Zero-shot", "mean-pool, LSE, top-k,\nk-means, chuyên mục\ntrong văn bản"),
             ("Reader học được", "chú ý theo ứng viên,\nFastformer/NRMS,\nđa sở thích, prior"),
             ("RAG / GraphRAG", "entity, kNN,\ncộng đồng Louvain"),
             ("LLM sinh / xếp hạng", "lịch sử-văn bản, KAR,\ngiám khảo 1,7B, tower\nLoRA, Semantic ID")]
    for i, (t, b) in enumerate(comps):
        box(ax, 0.51, 0.765 - i * 0.205, 0.215, 0.19, t, b, LIGHT, tc="#10204A", fs=7.8, bfs=6.7)
    box(ax, 0.775, 0.62, 0.225, 0.36, "Ghép tuyến tính", "z-score trong impression\ntrọng số khớp trên\nvalidation (v16: có trọng\nsố bài lạnh 80%)", RED)
    box(ax, 0.775, 0.18, 0.225, 0.38, "Stacker GBDT", "XGBoost trên điểm +\nngữ cảnh lịch sử/slate\n+ phơi nhiễm (+ pop)\nCV theo khối thời gian\n(v16)", "#8E1C21")
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.13, boxstyle="round,pad=0.006,rounding_size=0.012", fc="#EEF1F7", ec="#B8C2D9"))
    ax.text(0.5, 0.065, "Đánh giá: AUC · MRR · nDCG@5/10  |  CI 95% ghép cặp theo impression  |  lát bài lạnh/ấm  |  "
            "nhiễu seed  |  đối chiếu với V10", ha="center", va="center", fontsize=6.9 * FS if W < 9 else 9.5, color="#10204A")
    arrow(ax, 0.20, 0.80, 0.25, 0.80); arrow(ax, 0.10, 0.62, 0.10, 0.56); arrow(ax, 0.20, 0.37, 0.25, 0.37)
    for i in range(4):
        arrow(ax, 0.46, 0.80, 0.51, 0.86 - i * 0.205)
    for i in range(4):
        arrow(ax, 0.725, 0.86 - i * 0.205, 0.775, 0.80 if i < 2 else 0.37)
    arrow(ax, 0.46, 0.37, 0.775, 0.30)
    save(fig, name)


# ------------------------------------------------------------------ 2. cold share
def fig_cold(name="fig02_cold_share.png"):
    d, L = F["data_small"], F["data_large"]
    labels = ["MIND-small\ntest", "MIND-large\nvalidation (ngày cuối train)", "MIND-large\ntest (dev chính thức)"]
    vals = [d["cold_click_share_test_pct"], L["cold_click_share_val_pct"], L["cold_click_share_test_pct"]]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    bars = ax.barh(labels, vals, color=[NAVY, GOLD, RED], height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%".replace(".", ","), va="center", fontsize=9 * FS, fontweight="bold")
    ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_xlabel("Tỉ lệ lượt nhấp rơi vào bài chưa xuất hiện khi huấn luyện (%)")
    ax.grid(axis="x", color="#E5E5E5"); ax.set_axisbelow(True)
    comma(ax.xaxis)
    save(fig, name)


# ------------------------------------------------------------------ 3. validation vs test gain (v14)
def fig_valtest(name="fig03_val_vs_test_v14.png"):
    rows = [r for r in F["large_v14_val_test"] if r["row"] != "random_vec"]
    lab = [r["row"] for r in rows]
    import numpy as np
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    b1 = ax.bar(x - w / 2, [r["val"] for r in rows], w, color=GOLD, label="validation − frozen")
    b2 = ax.bar(x + w / 2, [r["test"] for r in rows], w, color=NAVY, label="test − frozen")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + (0.0015 if v >= 0 else -0.0015), vn(v, 3, True), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7 * FS)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([l.replace("community_graphrag", "community_\ngraphrag") for l in lab], fontsize=8 * FS)
    ax.set_ylabel("Δ nDCG@10 so với frozen"); ax.legend(frameon=False, fontsize=8 * FS, loc="upper left")
    ax.set_ylim(-0.012, 0.075); ax.grid(axis="y", color="#E5E5E5"); ax.set_axisbelow(True)
    comma(ax.yaxis)
    save(fig, name)


# ------------------------------------------------------------------ 4. directions on MIND-small
def fig_directions(name="fig04_small_directions.png"):
    rows = F["small_directions"]
    rows = sorted(rows, key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    y = range(len(rows))
    cols = [GREEN if r["delta"] > 0.005 else (RED if r["delta"] < -0.005 else GRAY) for r in rows]
    ax.barh(list(y), [r["delta"] for r in rows], color=cols, height=0.6)
    for i, r in enumerate(rows):
        if r["ci"]:
            ax.errorbar(r["delta"], i, xerr=[[r["delta"] - r["ci"][0]], [r["ci"][1] - r["delta"]]], fmt="none", ecolor="#222222", capsize=2.5, lw=0.9)
        tx = r["delta"] + (0.0012 if r["delta"] >= 0 else -0.0012)
        ax.text(tx if not r["ci"] else (r["ci"][1] + 0.0012 if r["delta"] >= 0 else r["ci"][0] - 0.0012), i, vn(r["delta"], 4, True),
                va="center", ha="left" if r["delta"] >= 0 else "right", fontsize=7 * FS)
    ax.set_yticks(list(y)); ax.set_yticklabels([f"{r['code']} · {r['name']}" for r in rows], fontsize=7.6 * FS)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.axvspan(-0.007, 0.007, color="#F2F2F2", zorder=0)
    ax.text(0, -1.05, "vùng xám: dải nhiễu seed ±0,007", ha="center", fontsize=6.8 * FS, color="#666666")
    ax.set_ylim(-1.4, len(rows) - 0.4)
    ax.set_xlabel("Δ nDCG@10 so với đối chứng của chính kỹ thuật (MIND-small, CI 95% ghép cặp khi log có)")
    ax.set_xlim(-0.045, 0.03); ax.grid(axis="x", color="#E5E5E5"); ax.set_axisbelow(True)
    save(fig, name)


# ------------------------------------------------------------------ 5. MIND-small main comparison
def fig_small_main(name="fig05_small_main.png"):
    keep = ["NRMS – công thức V10 (số V10 báo)", "Fastformer chuẩn (TB 2 seed)", "NRMS chuẩn (GloVe, seed 0)", "NAML chuẩn",
            "Fastformer refit (+ngày validation)", "NRMS refit (+ngày validation)", "BGE-small đóng băng, mean-pool",
            "BGE-small + LSE (late interaction)", "Qwen3-Emb-0.6B + LSE", "BGE-small + LSE + chuyên mục + entity",
            "Qwen3-Emb-0.6B + LSE + chuyên mục", "Reader Qwen3 um_text (TB 2 seed)"]
    rows = [r for r in F["small_main"] if r["label"] in keep]
    rows = sorted(rows, key=lambda r: r["ndcg10"])
    colmap = {"base_old": "#B5B5B5", "base": GRAY, "zs": NAVY, "reader": TEAL}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.barh(range(len(rows)), [r["ndcg10"] for r in rows], color=[colmap[r["group"]] for r in rows], height=0.62)
    for i, r in enumerate(rows):
        ax.text(r["ndcg10"] + 0.0006, i, vn(r["ndcg10"]), va="center", fontsize=7.2 * FS)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["label"] for r in rows], fontsize=7.6 * FS)
    for v, lab in ((0.4121, "NRMS công bố 0,4121"),):
        ax.axvline(v, color=RED, lw=0.9, ls="--"); ax.text(v + 0.0004, -0.9, lab, color=RED, fontsize=6.8 * FS)
    ax.set_xlim(0.36, 0.418); ax.set_xlabel("nDCG@10 trên MINDsmall_dev (73.152 impression, chỉ nội dung)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRAY, label="baseline huấn luyện chuẩn"), Patch(color=NAVY, label="LLM zero-shot"),
                       Patch(color=TEAL, label="reader học được"), Patch(color="#B5B5B5", label="công thức V10 cũ")],
              frameon=False, fontsize=7 * FS, loc="lower right")
    ax.grid(axis="x", color="#E5E5E5"); ax.set_axisbelow(True)
    comma(ax.xaxis)
    save(fig, name)


# ------------------------------------------------------------------ 6. MIND-large ladder
def fig_large_ladder(name="fig06_large_ladder.png"):
    rows = F["large_ladder"]
    colmap = {"zs": NAVY, "reader": TEAL, "lin": RED, "meta": "#8E1C21", "pop": GOLD}
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.barh(range(len(rows)), [r["nd"] for r in rows], color=[colmap[r["group"]] for r in rows], height=0.62)
    for i, r in enumerate(rows):
        ax.text(r["nd"] + 0.0008, i, vn(r["nd"]), va="center", fontsize=7.4 * FS)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["row"] for r in rows], fontsize=7.8 * FS)
    ax.invert_yaxis(); ax.set_xlim(0.36, 0.49)
    ax.set_xlabel("nDCG@10 trên MINDlarge_dev (376.471 impression)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NAVY, label="zero-shot"), Patch(color=TEAL, label="reader học được"),
                       Patch(color=RED, label="ghép / GBDT, chỉ nội dung + lịch sử"), Patch(color="#8E1C21", label="+ phơi nhiễm (không nhãn)"),
                       Patch(color=GOLD, label="streaming (đọc click trước đó trong ngày test)")], frameon=False, fontsize=6.8 * FS, loc="upper right")
    ax.grid(axis="x", color="#E5E5E5"); ax.set_axisbelow(True)
    comma(ax.xaxis)
    save(fig, name)


# ------------------------------------------------------------------ 7. readers by encoder
def fig_readers(name="fig07_readers_by_encoder.png"):
    import numpy as np
    R = F["large_v16b_readers"]
    encs = [("qwen3", "Qwen3-Emb-0.6B"), ("bge", "BGE-large (335M)"), ("cat", "Nối Qwen3 ⊕ BGE")]
    kinds = [("um_frozen", "reader trên vector đóng băng"), ("um_knn_rag", "reader + view kNN-RAG")]
    x = np.arange(len(kinds)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for j, (e, el) in enumerate(encs):
        vals = [R[k][e][0] for k, _ in kinds]
        b = ax.bar(x + (j - 1) * w, vals, w, color=[GRAY, NAVY, RED][j], label=el)
        for i, (k, _) in enumerate(kinds):
            m, s0, s1 = R[k][e]
            ax.plot([x[i] + (j - 1) * w] * 2, [min(s0, s1), max(s0, s1)], color="#111111", lw=1.2)
            ax.text(x[i] + (j - 1) * w, max(s0, s1) + 0.0007, vn(m), ha="center", fontsize=7 * FS)
    ax.set_xticks(x); ax.set_xticklabels([kl for _, kl in kinds]); ax.set_ylim(0.395, 0.43)
    ax.set_ylabel("nDCG@10 test (TB 2 seed; vạch = khoảng 2 seed)"); ax.legend(frameon=False, fontsize=7.5 * FS, ncol=3, loc="upper left")
    ax.grid(axis="y", color="#E5E5E5"); ax.set_axisbelow(True)
    comma(ax.yaxis)
    save(fig, name)


# ------------------------------------------------------------------ 8. stacker honesty
def fig_stacker_cv(name="fig08_stacker_cv.png"):
    import numpy as np
    rows = F["large_v16b_cv"]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    for j, (k, lab, c) in enumerate((("val", "validation out-of-fold", GOLD), ("val_new", "validation, bài mới", LIGHT), ("test", "test", NAVY))):
        vals = [r[k] for r in rows]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=c, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.001, vn(v), ha="center", fontsize=6.6 * FS, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([r["row"].replace(" (", "\n(") for r in rows], fontsize=7.2 * FS)
    ax.set_ylim(0.35, 0.52); ax.set_ylabel("nDCG@10"); ax.legend(frameon=False, fontsize=7.5 * FS, ncol=3, loc="upper right")
    ax.grid(axis="y", color="#E5E5E5"); ax.set_axisbelow(True)
    comma(ax.yaxis)
    save(fig, name)


# ------------------------------------------------------------------ 9. published comparison
def fig_published(name="fig09_published.png"):
    ours = [("super_llm (v16b)\nnội dung + lịch sử", 0.6863, 0.4373, NAVY, ""),
            ("stack_meta (v16b)\n+ phơi nhiễm log", 0.7013, 0.4516, "#8E1C21", "//"),
            ("stack_pop (v16b)\nstreaming", 0.7196, 0.4731, GOLD, "xx")]
    pub = [(p["label"].replace(" (", "\n("), p["auc"], p["ndcg10"], GRAY, "") for p in F["published_large"]]
    rows = pub[:3] + ours
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for ax, k, lim, ttl in ((axs[0], 1, (0.66, 0.73), "AUC"), (axs[1], 2, (0.40, 0.48), "nDCG@10")):
        ys = range(len(rows))
        for i, r in enumerate(rows):
            ax.barh(i, r[k], color=r[3], hatch=r[4], edgecolor="white", height=0.6)
            ax.text(r[k] + (lim[1] - lim[0]) * 0.01, i, vn(r[k]), va="center", fontsize=7 * FS)
        ax.set_xlim(*lim); ax.set_title(ttl, fontsize=9 * FS); ax.grid(axis="x", color="#E5E5E5"); ax.set_axisbelow(True)
        if k == 1:
            ax.axvline(0.7268, color=RED, ls="--", lw=0.9); ax.text(0.7262, -0.55, "đầu bảng 0,7268", color=RED, fontsize=6.5 * FS, ha="right")
    axs[0].set_yticks(list(range(len(rows)))); axs[0].set_yticklabels([r[0] for r in rows], fontsize=7.2 * FS); axs[0].invert_yaxis()
    comma(axs[0].xaxis, axs[1].xaxis)
    fig.text(0.01, -0.04, "Số công bố đo trên tập test của leaderboard (thứ cấp, chưa mở PDF); số của ta đo trên MINDlarge_dev. "
             "Gạch chéo = dùng thêm thông tin mà các mô hình công bố không dùng.", fontsize=6.6 * FS, color="#444444")
    save(fig, name)


# ------------------------------------------------------------------ 10. gain shares
def fig_gain(name="fig10_gain_shares.png"):
    G = F["gain_shares"]
    rows = [(f"{v} · {m}", G[v][m]) for v in ("v15", "v16b") for m in ("stack_llm", "stack_meta", "stack_pop")]
    cols = {"scores": NAVY, "ctx": LIGHT, "meta": "#8E1C21", "pop": GOLD}
    names = {"scores": "điểm thành phần LLM", "ctx": "ngữ cảnh lịch sử / slate", "meta": "phơi nhiễm + đếm theo bài", "pop": "popularity streaming"}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (lab, d) in enumerate(rows):
        left = 0
        for g in ("scores", "ctx", "meta", "pop"):
            v = d.get(g, 0)
            if v:
                ax.barh(i, v, left=left, color=cols[g], height=0.6, label=names[g] if i == 2 else None)
                ax.text(left + v / 2, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=7 * FS, color="white" if g != "ctx" else "#10204A")
                left += v
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.6 * FS); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Tỉ trọng split gain của XGBoost theo nhóm đặc trưng")
    h, l = ax.get_legend_handles_labels(); ax.legend(h, l, frameon=False, fontsize=7 * FS, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    comma(ax.xaxis)
    save(fig, name)


if __name__ == "__main__":
    fig_pipeline()
    fig_pipeline("fig01_pipeline_wide.png", W=12.0, H=5.6)
    fig_cold(); fig_valtest(); fig_directions(); fig_small_main(); fig_large_ladder(); fig_readers(); fig_stacker_cv(); fig_published(); fig_gain()
    print("ok", sorted(os.listdir(OUT)))
