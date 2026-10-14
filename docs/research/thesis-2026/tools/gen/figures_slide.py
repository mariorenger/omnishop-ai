"""English, slide-sized figures (the user's deck is written in English) from the same facts as the report."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, os.path.dirname(__file__))
from facts import FACTS as F  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/fig_slide"
os.makedirs(OUT, exist_ok=True)
NAVY, RED, GRAY, LIGHT, GOLD, GREEN, TEAL, DRED = "#0F1E42", "#B91C1C", "#94A3B8", "#7A9CC6", "#D99A2B", "#2F7D4A", "#2E7D8C", "#7F1D1D"
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#475569", "axes.labelcolor": "#1E293B", "xtick.color": "#1E293B", "ytick.color": "#1E293B",
                     "axes.titleweight": "bold", "axes.titlesize": 14, "legend.fontsize": 11.5})


def d4(x, sign=False):
    return (f"{x:+.4f}" if sign else f"{x:.4f}").replace("-", "−")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt(axis, nd=2):
    axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.{nd}f}".replace("-", "−")))


# ------------------------------------------------------------------ pipeline (full width)
def box(ax, x, y, w, h, title, body, fc, tc="white", boff=0.11, tfs=13.5, bfs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top", fontsize=tfs, color=tc, fontweight="bold")
    ax.text(x + w / 2, y + h - boff, body, ha="center", va="top", fontsize=bfs, color=tc, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.4, shrinkA=0, shrinkB=0))


def line(ax, xs, ys):
    ax.plot(xs, ys, color="#334155", lw=1.4, solid_capstyle="butt")


def pipeline():
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.0); ax.axis("off")
    box(ax, 0.00, 0.60, 0.17, 0.38, "MIND data", "news.tsv: title, abstract,\ncategory, entities\nbehaviors.tsv: history\n+ labelled impressions", NAVY)
    box(ax, 0.00, 0.21, 0.17, 0.34, "Chronological split", "train_core\nvalidation = last day\ntest = official dev\n(touched once)", "#334E7E")
    box(ax, 0.21, 0.60, 0.19, 0.38, "Frozen encoders", "BGE-small / BGE-large\n(CLS pooling)\nQwen3-Embedding-0.6B\n(last-token pooling)\nconcat (Qwen3 + BGE)", TEAL)
    box(ax, 0.21, 0.21, 0.19, 0.34, "Label-free features", "article age, prior\nimpressions (1/6/24 h),\nitem counts to previous\nday, category CTR\n(+ pop: streaming)", "#475569")
    comps = [("Zero-shot scorers", "mean-pool, LSE, top-k,\nk-means, category text"), ("Learned readers", "candidate-aware attention,\nFastformer / NRMS, prior"),
             ("RAG / GraphRAG", "entity, kNN, Louvain\ncommunities + LLM summary"), ("LLM gen. / ranking", "history-as-text, KAR,\n1.7B judge, LoRA tower, SID")]
    ys = [0.815 - i * 0.19 for i in range(4)]; hh = 0.165
    for (t, b), y in zip(comps, ys):
        box(ax, 0.445, y, 0.215, hh, t, b, LIGHT, tc="#0F1E42", boff=0.075, tfs=13, bfs=11)
    box(ax, 0.715, 0.60, 0.285, 0.38, "Linear fusion (supermodel)", "per-impression z-scores,\nweights fit on validation\n(v16: cold-weighted 80%)", RED)
    box(ax, 0.715, 0.21, 0.285, 0.34, "GBDT stacker (XGBoost)", "component scores\n+ history / slate context\n+ exposure (+ pop)\ntime-block CV (v16)", DRED)
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.10, boxstyle="round,pad=0.005,rounding_size=0.012", fc="#F1F5F9", ec="#CBD5E1"))
    ax.text(0.5, 0.05, "Evaluation: AUC · MRR · nDCG@5/10  |  paired 95% CI per impression  |  cold / warm slices  |  measured seed noise  |  V10 reproduction checks",
            ha="center", va="center", fontsize=12, color="#0F1E42")
    arrow(ax, 0.17, 0.79, 0.21, 0.79); arrow(ax, 0.085, 0.60, 0.085, 0.55); arrow(ax, 0.17, 0.38, 0.21, 0.38)
    for y in ys:                                   # encoders -> every component family
        arrow(ax, 0.40, 0.79, 0.445, y + hh / 2)
    mids = [y + hh / 2 for y in ys]; bus = 0.688   # every family exports scores to both combiners
    for m in mids:
        line(ax, [0.66, bus], [m, m])
    line(ax, [bus, bus], [mids[-1], mids[0]])
    arrow(ax, bus, 0.79, 0.715, 0.79); arrow(ax, bus, 0.38, 0.715, 0.38)
    line(ax, [0.305, 0.305, 0.80], [0.21, 0.155, 0.155])   # label-free features -> stacker, routed below the components
    arrow(ax, 0.80, 0.155, 0.80, 0.21)
    ax.text(0.55, 0.163, "label-free exposure features (stack_meta / stack_pop only)", ha="center", va="bottom", fontsize=10.5, color="#475569", style="italic")
    save(fig, "s_pipeline.png")


def cold():
    d, L = F["data_small"], F["data_large"]
    labels = ["MIND-small\ntest", "MIND-large\nvalidation (last train day)", "MIND-large\ntest (official dev)"]
    vals = [d["cold_click_share_test_pct"], L["cold_click_share_val_pct"], L["cold_click_share_test_pct"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.barh(labels, vals, color=[NAVY, GOLD, RED], height=0.58)
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_xlabel("Share of clicks on articles unseen in training (%)")
    ax.grid(axis="x", color="#E2E8F0"); ax.set_axisbelow(True)
    save(fig, "s_cold.png")


def valtest():
    rows = [r for r in F["large_v14_val_test"] if r["row"] != "random_vec"]
    lab = ["pool_lse (zero-shot)", "txt_cta_lse (zero-shot)", "knn_rag (RAG)", "community_graphrag", "um_frozen (reader)", "um_knn_rag (reader)"]
    y = np.arange(len(rows)); h = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    b1 = ax.barh(y - h / 2, [r["val"] for r in rows], h, color=GOLD, label="validation − frozen")
    b2 = ax.barh(y + h / 2, [r["test"] for r in rows], h, color=NAVY, label="test − frozen")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_width()
            ax.text(v + (0.0012 if v >= 0 else -0.0012), b.get_y() + b.get_height() / 2, f"{v:+.3f}".replace("-", "−"),
                    ha="left" if v >= 0 else "right", va="center", fontsize=10.5)
    ax.axvline(0, color="#1E293B", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=11.5); ax.invert_yaxis()
    ax.set_xlabel("Δ nDCG@10 vs frozen mean-pool (MIND-large, v14)"); ax.legend(frameon=False, loc="upper right", fontsize=11)
    ax.set_xlim(-0.016, 0.078); fmt(ax.xaxis); ax.grid(axis="x", color="#E2E8F0"); ax.set_axisbelow(True)
    save(fig, "s_valtest.png")


EN_DIR = {
    "Late interaction LSE (BGE)": "Late interaction LSE (BGE)", "Late interaction top-k (BGE)": "Late interaction top-k (BGE)",
    "Chuyên mục trong văn bản (Qwen3, LSE)": "Category in text (Qwen3)", "Chuyên mục trong văn bản (BGE, LSE)": "Category in text (BGE)",
    "Tên entity (BGE, ctae vs cta, LSE)": "Entity names in text (BGE)", "Lịch sử 200 click (BGE, LSE)": "200-click history (BGE)",
    "Chỉ tiêu đề, bỏ abstract (Qwen3, LSE)": "Title only, no abstract (Qwen3)", "Đa sở thích k-means K=5 (BGE)": "k-means interests K=5 vs LSE",
    "Reader học được (BGE) so với LSE": "Learned reader vs LSE (BGE)", "Semantic ID (RQ-KMeans) profile": "Semantic-ID profile",
    "GraphRAG cộng đồng (Louvain)": "GraphRAG communities", "LoRA/DoRA co-click (tốt nhất)": "LoRA/DoRA co-click (best)",
    "RAG biểu diễn (tệ nhất)": "Representation RAG (worst)", "Lịch sử dạng văn bản (tệ nhất)": "History-as-text (worst)",
    "KAR: LLM mở rộng bài (n=2.500)": "KAR item expansion (n=2,500)", "KAR: hồ sơ độc giả do LLM viết": "KAR LLM-written reader profile",
    "LLM 1,7B làm giám khảo (n=2.000)": "LLM 1.7B judge (n=2,000)",
}


def directions():
    rows = sorted(F["small_directions"], key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    cols = [GREEN if r["delta"] > 0.005 else (RED if r["delta"] < -0.005 else GRAY) for r in rows]
    ax.barh(range(len(rows)), [r["delta"] for r in rows], color=cols, height=0.62)
    for i, r in enumerate(rows):
        if r["ci"]:
            ax.errorbar(r["delta"], i, xerr=[[r["delta"] - r["ci"][0]], [r["ci"][1] - r["delta"]]], fmt="none", ecolor="#111827", capsize=3, lw=1.0)
        edge = (r["ci"][1] if r["ci"] else r["delta"]) if r["delta"] >= 0 else (r["ci"][0] if r["ci"] else r["delta"])
        ax.text(edge + (0.0013 if r["delta"] >= 0 else -0.0013), i, d4(r["delta"], True), va="center", ha="left" if r["delta"] >= 0 else "right", fontsize=10)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r['code']} · {EN_DIR[r['name']]}" for r in rows], fontsize=11)
    ax.axvline(0, color="#1E293B", lw=0.9); ax.axvspan(-0.007, 0.007, color="#EEF2F7", zorder=0)
    ax.text(0, -1.25, "grey band: seed noise ±0.007", ha="center", fontsize=10, color="#475569"); ax.set_ylim(-1.6, len(rows) - 0.4)
    ax.set_xlabel("Δ nDCG@10 vs each technique's own control (MIND-small)")
    ax.set_xlim(-0.047, 0.031); fmt(ax.xaxis); ax.grid(axis="x", color="#E2E8F0"); ax.set_axisbelow(True)
    save(fig, "s_directions.png")


def ladder():
    names = ["Qwen3 frozen (mean-pool)", "BGE-large frozen", "Popularity (online)", "Qwen3 + LSE + category text", "Best reader (concat, kNN view)",
             "super_llm (linear fusion)", "stack_llm (GBDT)", "stack_meta (+ exposure)", "stack_pop (streaming)"]
    rows = F["large_ladder"]
    colmap = {"zs": NAVY, "reader": TEAL, "lin": RED, "meta": DRED, "pop": GOLD}
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(range(len(rows)), [r["nd"] for r in rows], color=[colmap[r["group"]] for r in rows], height=0.64)
    for i, r in enumerate(rows):
        ax.text(r["nd"] + 0.0012, i, f"{r['nd']:.4f}", va="center", fontsize=11)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=11.5); ax.invert_yaxis(); ax.set_xlim(0.36, 0.50)
    ax.set_xlabel("nDCG@10 on MINDlarge_dev (376,471 impressions)"); fmt(ax.xaxis)
    ax.legend(handles=[Patch(color=NAVY, label="zero-shot"), Patch(color=TEAL, label="learned reader"), Patch(color=RED, label="fusion, content + history only"),
                       Patch(color=DRED, label="+ label-free exposure"), Patch(color=GOLD, label="streaming (reads earlier test-day clicks)")],
              frameon=False, fontsize=10, loc="upper right")
    ax.grid(axis="x", color="#E2E8F0"); ax.set_axisbelow(True)
    save(fig, "s_ladder.png")


def readers():
    R = F["large_v16b_readers"]
    encs = [("qwen3", "Qwen3-Emb-0.6B"), ("bge", "BGE-large (335M)"), ("cat", "Concat (Qwen3 + BGE)")]
    kinds = [("um_frozen", "reader on frozen vectors"), ("um_knn_rag", "reader + kNN-RAG view")]
    x = np.arange(2); w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for j, (e, el) in enumerate(encs):
        vals = [R[k][e][0] for k, _ in kinds]
        ax.bar(x + (j - 1) * w, vals, w, color=[GRAY, NAVY, RED][j], label=el)
        for i, (k, _) in enumerate(kinds):
            m, s0, s1 = R[k][e]
            ax.plot([x[i] + (j - 1) * w] * 2, [min(s0, s1), max(s0, s1)], color="#111827", lw=1.6)
            ax.text(x[i] + (j - 1) * w, max(s0, s1) + 0.0012, f"{m:.4f}", ha="center", fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels([kl for _, kl in kinds]); ax.set_ylim(0.395, 0.436); fmt(ax.yaxis, 3)
    ax.set_ylabel("test nDCG@10 (mean of 2 seeds)"); ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=10.5)
    ax.grid(axis="y", color="#E2E8F0"); ax.set_axisbelow(True)
    save(fig, "s_readers.png")


def stackcv():
    rows = F["large_v16b_cv"]
    lab = ["v15 recipe\n(user folds)", "stack_llm\n(time blocks)", "stack_meta\n(time blocks)", "stack_pop\n(time blocks)"]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for j, (k, l, c) in enumerate((("val", "validation (out-of-fold)", GOLD), ("val_new", "validation, new articles", LIGHT), ("test", "test", NAVY))):
        vals = [r[k] for r in rows]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=c, label=l)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.3f}", ha="center", fontsize=9.5, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=11)
    ax.set_ylim(0.35, 0.535); ax.yaxis.set_major_locator(MultipleLocator(0.05)); fmt(ax.yaxis); ax.set_ylabel("nDCG@10")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=10.5)
    ax.grid(axis="y", color="#E2E8F0"); ax.set_axisbelow(True)
    save(fig, "s_stackcv.png")


def gain():
    G = F["gain_shares"]
    rows = [(f"{v} · {m}", G[v][m]) for v in ("v15", "v16b") for m in ("stack_llm", "stack_meta", "stack_pop")]
    cols = {"scores": NAVY, "ctx": LIGHT, "meta": DRED, "pop": GOLD}
    names = {"scores": "LLM component scores", "ctx": "history / slate context", "meta": "exposure + item counts", "pop": "streaming popularity"}
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for i, (lab, d) in enumerate(rows):
        left = 0
        for g in ("scores", "ctx", "meta", "pop"):
            v = d.get(g, 0)
            if v:
                ax.barh(i, v, left=left, color=cols[g], height=0.62, label=names[g] if i == 2 else None)
                ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=11, color="white" if g != "ctx" else "#0F1E42")
                left += v
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=11.5); ax.invert_yaxis()
    ax.set_xlim(0, 1); fmt(ax.xaxis, 1); ax.set_xlabel("Share of XGBoost split gain by feature group")
    h, l = ax.get_legend_handles_labels(); ax.legend(h, l, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.45, 1.25), fontsize=10.5)
    save(fig, "s_gain.png")


def published():
    rows = [("NRMS (MIND paper, ACL'20)", 0.6776, 0.4163, GRAY, ""), ("PLM-NR (SIGIR'21)", 0.7064, 0.4438, GRAY, ""), ("UNBERT (IJCAI'21)", 0.7068, 0.4478, GRAY, ""),
            ("ours: super_llm (content + history)", 0.6863, 0.4373, NAVY, ""), ("ours: stack_meta (+ log exposure)", 0.7013, 0.4516, DRED, "//"),
            ("ours: stack_pop (streaming)", 0.7196, 0.4731, GOLD, "xx")]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, k, lim, ttl in ((axs[0], 1, (0.66, 0.745), "AUC"), (axs[1], 2, (0.40, 0.485), "nDCG@10")):
        for i, r in enumerate(rows):
            ax.barh(i, r[k], color=r[3], hatch=r[4], edgecolor="white", height=0.62)
            xl = r[k] + (lim[1] - lim[0]) * 0.012
            if k == 1 and r[k] > 0.715:
                xl = 0.7290                                   # right of the leaderboard line
            ax.text(xl, i, f"{r[k]:.4f}", va="center", fontsize=10.5)
        ax.set_xlim(*lim); ax.set_title(ttl, pad=14); fmt(ax.xaxis, 2); ax.grid(axis="x", color="#E2E8F0"); ax.set_axisbelow(True)
    axs[0].axvline(0.7268, color=RED, ls="--", lw=1.1)
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([r[0] for r in rows], fontsize=11); axs[0].set_ylim(5.5, -0.95)
    axs[0].text(0.7262, -0.62, "leaderboard top 0.7268 (ensemble)", color=RED, fontsize=9.5, ha="right", va="center")
    save(fig, "s_published.png")


if __name__ == "__main__":
    pipeline(); cold(); valtest(); directions(); ladder(); readers(); stackcv(); gain(); published()
    print("ok", sorted(os.listdir(OUT)))
