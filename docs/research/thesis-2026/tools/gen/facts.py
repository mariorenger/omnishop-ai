"""Single source of truth for every number used in the report and the slides.

Every value is copied from the measured-run write-ups in experiments/recsys_v11/
(RESULTS_v7.md, RESULTS_v13.md, RESULTS_v14_large.md, RESULTS_v15_large.md,
RESULTS_v16_partial.md, RESULTS_v16b_large.md) or, for published numbers, from
secondary sources found by web search (marked "secondary"; PDFs not opened).
Metrics: AUC, MRR, nDCG@5, nDCG@10, cold AUC, warm AUC (None = not in the logs).
"""
import json
import sys

M = ("auc", "mrr", "ndcg5", "ndcg10", "cold_auc", "warm_auc")


def row(label, vals, note="", group=""):
    vals = list(vals) + [None] * (6 - len(vals))
    return {"label": label, **dict(zip(M, vals)), "note": note, "group": group}


FACTS = {
    # ------------------------------------------------------------------ data
    "data_small": {
        "articles": 65238, "test_impressions": 73152, "val_impressions": 30270,
        "categories": 18, "subcategories": 270, "abstract_words": 34.9, "abstract_empty_pct": 5.2,
        "title_words": 10.7, "entity_pct": 87.5, "entities_per_article": 2.4,
        "test_reader_clicks_mean": 32.3, "test_readers_over50_pct": 19.6, "test_reader_clicks_max": 444,
        "cold_click_share_test_pct": 85.9, "train_core_share_pct": 80.7,
    },
    "data_large": {
        "train_impressions": 2232748, "train_articles": 101527, "dev_impressions": 376471, "dev_articles": 72023,
        "articles_merged": 104151, "train_core": 1801231, "validation": 431517, "test": 376471,
        "train_samples": 2712612, "cold_click_share_test_pct": 82.2, "cold_click_share_val_pct": 18.1,
        "val_used_v14": 60000, "val_used_v15": 150000, "train_used_v14": 600000, "train_used_v15": 1200000,
    },
    # -------------------------------------------------------- MIND-small main
    "small_main": [
        row("NRMS – công thức V10 (số V10 báo)", (0.6181, 0.2809, 0.3020, 0.3685), group="base_old"),
        row("NAML – công thức V10", (0.6387, 0.3036, 0.3327, 0.3953), group="base_old"),
        row("Fastformer – công thức V10", (0.6022, 0.2798, 0.2998, 0.3642), group="base_old"),
        row("NRMS chuẩn (GloVe, seed 0)", (0.6392, 0.3037, 0.3293, 0.3937, 0.6471, 0.6400), group="base"),
        row("NRMS chuẩn (seed 1)", (0.6501, 0.3091, 0.3372, 0.4005, 0.6630, 0.6152), group="base"),
        row("NAML chuẩn", (0.6471, 0.3079, 0.3364, 0.4006, 0.6698, 0.5640), group="base"),
        row("Fastformer chuẩn (TB 2 seed)", (0.6426, 0.3000, 0.3289, 0.3927, 0.6623, 0.5869), group="base"),
        row("Fastformer refit (+ngày validation)", (0.6550, 0.3097, 0.3387, 0.4033, 0.6767, 0.5883), group="base"),
        row("NRMS refit (+ngày validation)", (0.6555, 0.3140, 0.3443, 0.4078, 0.6700, 0.6284), group="base"),
        row("BGE-small đóng băng, mean-pool", (0.6242, 0.3002, 0.3299, 0.3909, 0.6617, 0.4416), group="zs"),
        row("BGE-small + LSE (late interaction)", (0.6386, 0.3118, 0.3427, 0.4037, 0.6675, 0.5036), group="zs"),
        row("BGE-small + LSE + chuyên mục + entity", (0.6516, 0.3176, 0.3476, 0.4083, 0.6703, 0.5479), group="zs"),
        row("Qwen3-Emb-0.6B + LSE", (0.6349, 0.3025, 0.3300, 0.3912, 0.6632, 0.4939), group="zs"),
        row("Qwen3-Emb-0.6B + LSE + chuyên mục", (0.6556, 0.3201, 0.3509, 0.4094, 0.6637, 0.6180), group="zs"),
        row("Reader BGE um_knn_rag (1 seed)", (0.6531, 0.3090, 0.3419, 0.4033, 0.6824, 0.5365), group="reader"),
        row("Reader Qwen3 um_text (TB 2 seed)", (0.6535, 0.3167, 0.3492, 0.4104, 0.6885, 0.5102), group="reader"),
        row("Công bố: NRMS (Iana và cs., SIGIR'23)", (0.6658, 0.3144, 0.3499, 0.4121), group="pub"),
        row("Công bố: NAML (Iana và cs., SIGIR'23)", (0.6714, 0.3158, 0.3520, 0.4152), group="pub"),
    ],
    "small_pop": [
        row("Popularity (CTR online, không nội dung)", (0.6533, 0.3114, 0.3402, 0.4026)),
        row("BGE-small mean-pool + pop (V10 tốt nhất)", (0.6769, 0.3350, 0.3727, 0.4330, 0.7249, 0.4671)),
        row("NRMS chuẩn + pop", (0.6946, 0.3401, 0.3759, 0.4383)),
        row("Fastformer chuẩn + pop", (0.6909, 0.3400, 0.3763, 0.4381)),
        row("BGE-small top-k + pop", (0.6871, 0.3414, 0.3807, 0.4410, 0.7324, 0.4979)),
        row("BGE-small views_lse + pop", (0.6956, 0.3441, 0.3861, 0.4455, 0.7369, 0.5269)),
    ],
    "small_ladder": [
        {"rung": "ladder0: công thức V10 (harness mới)", "auc": 0.6156, "ndcg10": 0.3694, "delta": None, "ci": None},
        {"rung": "ladder1: + title-only, UNK, mask, negative mỗi epoch", "auc": 0.6102, "ndcg10": 0.3661, "delta": -0.0033, "ci": (-0.0051, -0.0015)},
        {"rung": "ladder2: + cỡ/lr chuẩn, vector ngẫu nhiên (chưa hội tụ)", "auc": 0.5818, "ndcg10": 0.3447, "delta": -0.0213, "ci": (-0.0229, -0.0197)},
        {"rung": "nrms_ref: + GloVe-300", "auc": 0.6392, "ndcg10": 0.3937, "delta": 0.0490, "ci": (0.0471, 0.0509)},
        {"rung": "nrms_ref_refit: + huấn luyện thêm các ngày validation", "auc": 0.6555, "ndcg10": 0.4078, "delta": 0.0141, "ci": (0.0126, 0.0156)},
    ],
    "small_seed_noise": [
        {"pair": "NRMS chuẩn seed 0 / seed 1", "a": 0.3937, "b": 0.4005, "gap": 0.0068},
        {"pair": "Tower Qwen3 causal seed 0 / 1", "a": 0.3944, "b": 0.3820, "gap": 0.0124},
        {"pair": "Tower Qwen3 causal+meanpool seed 0 / 1", "a": 0.4026, "b": 0.3923, "gap": 0.0103},
        {"pair": "Reader Qwen3 um_frozen seed 0 / 1 (v13)", "a": 0.3929, "b": 0.4063, "gap": 0.0134},
        {"pair": "Reader Qwen3 um_text seed 0 / 1 (v13)", "a": 0.4173, "b": 0.4036, "gap": 0.0137},
    ],
    # Δ nDCG@10 on MIND-small (73,152 test impressions unless n given), each against its own control
    "small_directions": [
        {"code": "H8", "name": "Late interaction LSE (BGE)", "delta": 0.0128, "ci": None, "ctrl": "mean-pool", "verdict": "+"},
        {"code": "H8", "name": "Late interaction top-k (BGE)", "delta": 0.0107, "ci": None, "ctrl": "mean-pool", "verdict": "+"},
        {"code": "F", "name": "Chuyên mục trong văn bản (Qwen3, LSE)", "delta": 0.0182, "ci": (0.0169, 0.0196), "ctrl": "LSE cùng encoder", "verdict": "+"},
        {"code": "F", "name": "Chuyên mục trong văn bản (BGE, LSE)", "delta": 0.0013, "ci": (0.0000, 0.0026), "ctrl": "LSE cùng encoder", "verdict": "≈"},
        {"code": "F", "name": "Tên entity (BGE, ctae vs cta, LSE)", "delta": 0.0033, "ci": (0.0025, 0.0041), "ctrl": "cta_lse", "verdict": "+ nhỏ"},
        {"code": "F", "name": "Lịch sử 200 click (BGE, LSE)", "delta": 0.0012, "ci": (0.0008, 0.0016), "ctrl": "50 click", "verdict": "≈"},
        {"code": "F", "name": "Chỉ tiêu đề, bỏ abstract (Qwen3, LSE)", "delta": -0.0321, "ci": (-0.0338, -0.0304), "ctrl": "title+abstract", "verdict": "−"},
        {"code": "H11", "name": "Đa sở thích k-means K=5 (BGE)", "delta": -0.0030, "ci": (-0.0039, -0.0021), "ctrl": "LSE", "verdict": "−"},
        {"code": "H10", "name": "Reader học được (BGE) so với LSE", "delta": -0.0080, "ci": (-0.0099, -0.0061), "ctrl": "LSE (không tham số)", "verdict": "−"},
        {"code": "H7", "name": "Semantic ID (RQ-KMeans) profile", "delta": 0.0035, "ci": None, "ctrl": "mean-pool", "verdict": "≈"},
        {"code": "H4", "name": "GraphRAG cộng đồng (Louvain)", "delta": 0.0043, "ci": None, "ctrl": "mean-pool", "verdict": "≈ (cold AUC −0,0031)"},
        {"code": "H2", "name": "LoRA/DoRA co-click (tốt nhất)", "delta": 0.0029, "ci": None, "ctrl": "mean-pool", "verdict": "≈ (cold AUC giảm)"},
        {"code": "H3", "name": "RAG biểu diễn (tệ nhất)", "delta": -0.0080, "ci": None, "ctrl": "mean-pool", "verdict": "−"},
        {"code": "H1", "name": "Lịch sử dạng văn bản (tệ nhất)", "delta": -0.0211, "ci": None, "ctrl": "mean-pool", "verdict": "−"},
        {"code": "H6", "name": "KAR: LLM mở rộng bài (n=2.500)", "delta": 0.0079, "ci": (0.0021, 0.0137), "ctrl": "mean-pool", "verdict": "+ (mất khi có pop)"},
        {"code": "H6", "name": "KAR: hồ sơ độc giả do LLM viết", "delta": -0.0308, "ci": None, "ctrl": "mean-pool", "verdict": "−"},
        {"code": "H9", "name": "LLM 1,7B làm giám khảo (n=2.000)", "delta": -0.0112, "ci": (-0.0216, -0.0008), "ctrl": "mean-pool", "verdict": "−"},
    ],
    "small_sid_gpt": {"sid_gpt": 0.2898, "random_vec": 0.2897},
    "small_encoder_sweep": [  # Δ nDCG@10 of mean-pool vs BGE-small mean-pool
        {"enc": "TF-IDF + LSA", "delta": -0.0443}, {"enc": "BGE-base", "delta": -0.0185},
        {"enc": "Qwen3-Embedding-0.6B", "delta": -0.0125}, {"enc": "BGE-large", "delta": 0.0015},
    ],
    "small_tower": [
        {"mode": "causal (seed 0)", "nd": 0.3944, "nd_mp": 0.4026},
        {"mode": "bidirectional", "nd": 0.3854, "nd_mp": 0.3965},
        {"mode": "soft (v7)", "nd": 0.3964, "nd_mp": 0.4043},
        {"mode": "soft (v13, chạy lại)", "nd": None, "nd_mp": 0.3956},
        {"mode": "causal (seed 1)", "nd": 0.3820, "nd_mp": 0.3923},
    ],
    "small_text_fields": [
        {"enc": "Qwen3", "row": "txt_t_lse (chỉ tiêu đề)", "val": 0.3446, "test": 0.3591, "dval": -0.0187, "dtest": -0.0321, "ci": (-0.0338, -0.0304)},
        {"enc": "Qwen3", "row": "txt_cta_lse (+ chuyên mục)", "val": 0.3762, "test": 0.4094, "dval": 0.0129, "dtest": 0.0182, "ci": (0.0169, 0.0196)},
        {"enc": "Qwen3", "row": "hist100_lse (100 click)", "val": 0.3644, "test": 0.3924, "dval": 0.0011, "dtest": 0.0013, "ci": (0.0009, 0.0016)},
        {"enc": "BGE", "row": "txt_t_lse (chỉ tiêu đề)", "val": 0.3706, "test": 0.3876, "dval": -0.0109, "dtest": -0.0161, "ci": (-0.0177, -0.0146)},
        {"enc": "BGE", "row": "txt_cta_lse (+ chuyên mục)", "val": 0.3719, "test": 0.4050, "dval": -0.0096, "dtest": 0.0013, "ci": (0.0000, 0.0026)},
        {"enc": "BGE", "row": "txt_ctae_lse (+ chuyên mục + entity)", "val": 0.3764, "test": 0.4083, "dval": -0.0051, "dtest": 0.0046, "ci": (0.0033, 0.0059)},
        {"enc": "BGE", "row": "hist200_lse (200 click)", "val": 0.3822, "test": 0.4049, "dval": 0.0007, "dtest": 0.0012, "ci": (0.0008, 0.0016)},
    ],
    "small_multi_interest": [
        {"enc": "BGE", "row": "um_frozen (1 vector, 3 seed)", "nd": 0.3968, "std": 0.0016, "cos": None},
        {"enc": "BGE", "row": "um_mi1 (đối chứng K=1)", "nd": 0.3906, "std": 0.0035, "cos": None},
        {"enc": "BGE", "row": "um_mi4 (K=4)", "nd": 0.3944, "std": 0.0011, "cos": "0,85–0,87"},
        {"enc": "BGE", "row": "um_mi4d (K=4 + phạt đa dạng)", "nd": 0.3972, "std": 0.0055, "cos": "0,34–0,39"},
        {"enc": "Qwen3", "row": "um_frozen (2 seed)", "nd": 0.3996, "std": 0.0067, "cos": None},
        {"enc": "Qwen3", "row": "um_mi1 (đối chứng K=1)", "nd": 0.4008, "std": 0.0018, "cos": None},
        {"enc": "Qwen3", "row": "um_mi4 (K=4)", "nd": 0.4029, "std": 0.0019, "cos": "0,91"},
        {"enc": "Qwen3", "row": "um_mi4d (K=4 + phạt đa dạng)", "nd": 0.3965, "std": 0.0032, "cos": "0,38–0,52"},
    ],
    # ---------------------------------------------------------------- MIND-large
    "large_v14_val_test": [  # gain over 'frozen' on validation vs on test (nDCG@10)
        {"row": "random_vec", "val": -0.0699, "test": -0.0906},
        {"row": "pool_lse", "val": 0.0232, "test": 0.0146},
        {"row": "txt_cta_lse", "val": 0.0370, "test": 0.0315},
        {"row": "knn_rag", "val": 0.0130, "test": -0.0037},
        {"row": "community_graphrag", "val": 0.0199, "test": -0.0066},
        {"row": "um_frozen", "val": 0.0596, "test": 0.0192},
        {"row": "um_knn_rag", "val": 0.0653, "test": 0.0302},
    ],
    "large_v14": [
        row("Qwen3 đóng băng, mean-pool", (0.6215, 0.2922, 0.3168, 0.3787, 0.6583, 0.4784)),
        row("Popularity (online, không nội dung)", (0.6540, 0.3126, 0.3415, 0.4040, 0.6986, 0.5213)),
        row("Qwen3 + LSE", (0.6352, 0.3046, 0.3319, 0.3933, 0.6662, 0.5179)),
        row("Qwen3 + LSE + chuyên mục", (0.6554, 0.3206, 0.3509, 0.4102, 0.6620, 0.6333)),
        row("Reader um_knn_rag (2 seed)", (0.6599, 0.3117, 0.3463, 0.4089, 0.6966, 0.5507)),
        row("Tower LoRA Qwen3 (n = 30.000)", (0.6384, 0.2986, 0.3277, 0.3904, 0.6663, 0.5454)),
        row("super_zs (ghép tuyến tính)", (0.6762, 0.3295, 0.3663, 0.4277, 0.7081, 0.5828)),
        row("super_llm (+3 reader, hist100)", (0.6756, 0.3283, 0.3646, 0.4263, 0.7031, 0.6028)),
    ],
    "large_v15": [
        row("Qwen3 đóng băng, mean-pool", (0.6215, 0.2922, 0.3168, 0.3787, 0.6583, 0.4784), group="zs"),
        row("BGE-large đóng băng, mean-pool", (0.6297, 0.3042, 0.3321, 0.3933, 0.6632, 0.4985), group="zs"),
        row("super_zs", (0.6702, 0.3229, 0.3586, 0.4205, 0.6893, 0.6338), group="lin"),
        row("super_llm (ghép tuyến tính)", (0.6774, 0.3286, 0.3654, 0.4268, 0.7004, 0.6207), group="lin"),
        row("stack_llm (GBDT: điểm + ngữ cảnh)", (0.6843, 0.3355, 0.3730, 0.4332, 0.7003, 0.6534), group="stack"),
        row("stack_meta (+ phơi nhiễm, đếm theo bài)", (0.6998, 0.3500, 0.3901, 0.4508, 0.7280, 0.6187), group="meta"),
        row("frozen + pop (streaming)", (0.6757, 0.3324, 0.3694, 0.4312, 0.7259, 0.5098), group="pop"),
        row("super_llm + pop (streaming)", (0.7114, 0.3574, 0.3998, 0.4599, 0.7446, 0.6192), group="pop"),
        row("stack_pop (streaming)", (0.7206, 0.3680, 0.4125, 0.4730, 0.7524, 0.6313), group="pop"),
    ],
    "large_v15_pairs": [
        {"pair": "stack_llm − super_llm", "d": 0.0064, "ci": (0.0059, 0.0069), "dauc": 0.0069, "dcold": -0.0001},
        {"pair": "stack_meta − super_llm", "d": 0.0240, "ci": (0.0234, 0.0246), "dauc": 0.0224, "dcold": 0.0276},
        {"pair": "stack_meta − stack_llm", "d": 0.0176, "ci": (0.0170, 0.0182), "dauc": 0.0155, "dcold": 0.0277},
        {"pair": "stack_pop − stack_meta", "d": 0.0222, "ci": (0.0216, 0.0227), "dauc": 0.0208, "dcold": 0.0244},
        {"pair": "stack_pop − super_llm+pop", "d": 0.0131, "ci": (0.0125, 0.0137), "dauc": 0.0093, "dcold": 0.0078},
    ],
    "large_v15_encoders": [  # BGE-large vs Qwen3 zero-shot, test nDCG@10 / AUC
        {"row": "frozen (mean-pool)", "bge": (0.3933, 0.6297), "qwen": (0.3787, 0.6215)},
        {"row": "pool_lse", "bge": (0.4047, 0.6459), "qwen": (0.3916, 0.6348)},
        {"row": "pool_topk", "bge": (0.4045, 0.6448), "qwen": (0.3930, 0.6341)},
        {"row": "pool_km5", "bge": (0.4023, 0.6402), "qwen": (0.3879, 0.6306)},
        {"row": "hist100_lse", "bge": (0.4057, 0.6471), "qwen": (0.3929, 0.6360)},
        {"row": "txt_cta_lse (chuyên mục)", "bge": (0.4022, 0.6484), "qwen": (0.4102, 0.6554)},
    ],
    "large_v15_valtest": [
        {"row": "stack_llm", "val": 0.4664, "test": 0.4332},
        {"row": "stack_meta", "val": 0.4909, "test": 0.4508},
        {"row": "stack_pop", "val": 0.4922, "test": 0.4730},
        {"row": "super_llm", "val": 0.4208, "test": 0.4268},
        {"row": "super_llm+pop", "val": 0.4394, "test": 0.4599},
        {"row": "frozen", "val": 0.3392, "test": 0.3787},
        {"row": "random_vec", "val": 0.2685, "test": 0.2881},
    ],
    "large_v16b": [
        {"label": "super_llm (ghép tuyến tính)", "auc": 0.6863, "ndcg10": 0.4373, "cold_auc": 0.7185, "v15": (0.6774, 0.4268), "group": "lin"},
        {"label": "stack_llm (GBDT, nội dung + lịch sử)", "auc": 0.6853, "ndcg10": 0.4369, "cold_auc": 0.7063, "v15": (0.6843, 0.4332), "group": "stack"},
        {"label": "stack_meta (+ phơi nhiễm; đăng ký trước)", "auc": 0.7013, "ndcg10": 0.4516, "cold_auc": 0.7248, "v15": (0.6998, 0.4508), "group": "meta"},
        {"label": "stack_meta_v15cfg (đối chứng)", "auc": 0.7029, "ndcg10": 0.4543, "cold_auc": 0.7285, "v15": None, "group": "meta"},
        {"label": "super_llm + pop (streaming)", "auc": 0.7133, "ndcg10": 0.4638, "cold_auc": 0.7539, "v15": (0.7114, 0.4599), "group": "pop"},
        {"label": "stack_pop (streaming)", "auc": 0.7196, "ndcg10": 0.4731, "cold_auc": 0.7526, "v15": (0.7206, 0.4730), "group": "pop"},
    ],
    "large_v16b_pairs": [
        {"pair": "stack_llm − super_llm", "d": -0.0004, "ci": (-0.0010, 0.0002)},
        {"pair": "stack_meta − super_llm", "d": 0.0143, "ci": (0.0137, 0.0149)},
        {"pair": "stack_meta − stack_llm", "d": 0.0147, "ci": (0.0142, 0.0153)},
        {"pair": "stack_pop − stack_meta", "d": 0.0215, "ci": (0.0209, 0.0220)},
        {"pair": "stack_pop − super_llm+pop", "d": 0.0093, "ci": (0.0087, 0.0099)},
        {"pair": "stack_meta − stack_meta_v15cfg", "d": -0.0027, "ci": (-0.0031, -0.0023)},
    ],
    "large_v16b_readers": {  # test nDCG@10 (mean of 2 seeds) and the two seeds
        "um_frozen": {"qwen3": (0.4050, 0.4091, 0.4009), "bge": (0.4152, 0.4186, 0.4117), "cat": (0.4223, 0.4229, 0.4216)},
        "um_knn_rag": {"qwen3": (0.4095, 0.4075, 0.4116), "bge": (0.4189, 0.4203, 0.4175), "cat": (0.4226, 0.4200, 0.4252)},
        "um_text": {"qwen3": (0.4092, 0.4085, 0.4099)},
    },
    "large_v16b_reader_pairs": [
        {"pair": "BGE-large − Qwen3 (um_frozen)", "d": 0.0102, "ci": (0.0095, 0.0108)},
        {"pair": "Nối − BGE-large (um_frozen)", "d": 0.0071, "ci": (0.0066, 0.0076)},
        {"pair": "Nối − Qwen3 (um_frozen)", "d": 0.0173, "ci": None},
        {"pair": "kNN-RAG − đóng băng (BGE-large)", "d": 0.0037, "ci": None},
        {"pair": "kNN-RAG − đóng băng (vector nối)", "d": 0.0003, "ci": (-0.0001, 0.0007)},
        {"pair": "Reader − LSE zero-shot (BGE-large)", "d": 0.0105, "ci": None},
    ],
    "large_v16b_cv": [
        {"row": "stack_meta_v15cfg (fold theo người dùng)", "val": 0.4922, "val_new": 0.3738, "test": 0.4543},
        {"row": "stack_llm (fold khối thời gian)", "val": 0.4456, "val_new": 0.4109, "test": 0.4369},
        {"row": "stack_meta (fold khối thời gian)", "val": 0.4588, "val_new": 0.4533, "test": 0.4516},
        {"row": "stack_pop (fold khối thời gian)", "val": 0.4682, "val_new": 0.4605, "test": 0.4731},
    ],
    "gain_shares": {  # XGBoost split-gain share by feature group
        "v15": {"stack_llm": {"scores": 0.60, "ctx": 0.40}, "stack_meta": {"scores": 0.47, "ctx": 0.32, "meta": 0.21},
                "stack_pop": {"scores": 0.39, "ctx": 0.27, "meta": 0.11, "pop": 0.22}},
        "v16b": {"stack_llm": {"scores": 0.66, "ctx": 0.34}, "stack_meta": {"scores": 0.53, "ctx": 0.27, "meta": 0.20},
                 "stack_pop": {"scores": 0.46, "ctx": 0.22, "meta": 0.08, "pop": 0.24}},
    },
    "large_v16_partial_concat_zs": [
        {"row": "Qwen3 đóng băng", "nd": 0.3787, "auc": 0.6215},
        {"row": "BGE-large đóng băng", "nd": 0.3933, "auc": 0.6297},
        {"row": "Vector nối Qwen3 ⊕ BGE-large", "nd": 0.3945, "auc": 0.6379},
    ],
    # ladder used in the figure (v16b where available)
    "large_ladder": [
        {"row": "Qwen3 đóng băng (mean-pool)", "nd": 0.3787, "group": "zs"},
        {"row": "BGE-large đóng băng", "nd": 0.3933, "group": "zs"},
        {"row": "Popularity (online)", "nd": 0.4040, "group": "pop"},
        {"row": "Qwen3 + LSE + chuyên mục", "nd": 0.4102, "group": "zs"},
        {"row": "Reader tốt nhất (vector nối, kNN)", "nd": 0.4226, "group": "reader"},
        {"row": "super_llm (ghép tuyến tính)", "nd": 0.4373, "group": "lin"},
        {"row": "stack_llm (GBDT)", "nd": 0.4369, "group": "lin"},
        {"row": "stack_meta (+ phơi nhiễm)", "nd": 0.4516, "group": "meta"},
        {"row": "stack_pop (streaming)", "nd": 0.4731, "group": "pop"},
    ],
    # --------------------------------------------------------------- published (secondary)
    "published_large": [
        {"label": "NRMS (Wu và cs., ACL'20)", "auc": 0.6776, "mrr": 0.3305, "ndcg5": 0.3594, "ndcg10": 0.4163},
        {"label": "PLM-NR (Wu và cs., SIGIR'21)", "auc": 0.7064, "mrr": 0.3539, "ndcg5": 0.3871, "ndcg10": 0.4438},
        {"label": "UNBERT (Zhang và cs., IJCAI'21)", "auc": 0.7068, "mrr": 0.3568, "ndcg5": 0.3913, "ndcg10": 0.4478},
        {"label": "Fastformer+PLM-NR (ensemble, đầu bảng)", "auc": 0.7268, "mrr": None, "ndcg5": None, "ndcg10": None},
    ],
    # ---------------------------------------------------------------- runs / compute
    "runs": [
        {"ver": "V10", "data": "MIND-small", "what": "Bộ đo gốc: NRMS/NAML/Fastformer, BGE zero-shot, llmenc_ca, popularity", "time": "—", "status": "đã đo"},
        {"ver": "v7", "data": "MIND-small", "what": "~16 họ kỹ thuật/~150 biến thể; baseline chuẩn NRMS/NAML/Fastformer; thang ablation", "time": "7,52 h (1×T4)", "status": "đã đo"},
        {"ver": "v13", "data": "MIND-small", "what": "Thêm dữ liệu (trường văn bản, 100/200 click), đa sở thích, Fastformer 2 seed + refit", "time": "8,22 h", "status": "đã đo"},
        {"ver": "v14", "data": "MIND-large", "what": "Một mô hình Qwen3 + đủ kỹ thuật; supermodel tuyến tính", "time": "2,90 h", "status": "đã đo"},
        {"ver": "v15", "data": "MIND-large", "what": "Stacker GBDT (XGBoost GPU); đặc trưng phơi nhiễm; encoder thứ hai BGE-large", "time": "2,73 h", "status": "đã đo"},
        {"ver": "v16", "data": "MIND-large", "what": "CV theo khối thời gian, reader BGE-large/vector nối", "time": "chết sau 1,6 h (hết RAM)", "status": "một phần"},
        {"ver": "v16b", "data": "MIND-large", "what": "Sửa knn_view (chia khối), đo RAM đỉnh thật, đầu reader 128 chiều cho 3 encoder", "time": "4,49 h; RAM đỉnh 21,2 GB", "status": "đã đo"},
        {"ver": "v17", "data": "MIND-large", "what": "Tinh chỉnh 2 lớp cuối BGE-large dưới reader (công thức PLM-NR), 2 nhánh khớp cặp, cổng dừng", "time": "ước 6–6,5 h", "status": "chờ chạy"},
    ],
    "compute": {"v15_stack_rows": "6,05 triệu × 72 cột", "v16b_stack_rows": "6,05 triệu × 99 cột (2,4 GB)", "test_rows": "14,09 triệu",
                "v16b_peak_ram_gb": 21.2, "container_gb": 32, "gpu_peak_gb": 5.0, "oom_bytes": "2 × [104.151, 20, 2048] float32 = 34,1 GB",
                "selftests": 73},
    "v17_prediction": {"pass": 35, "inconclusive": 25, "fail": 40, "gate": 0.006, "ft_reader_range": (0.418, 0.428),
                        "super_llm_gain": (0.002, 0.006), "stack_meta_gain": (0.000, 0.003)},
}


if __name__ == "__main__":
    json.dump(FACTS, sys.stdout if len(sys.argv) < 2 else open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
