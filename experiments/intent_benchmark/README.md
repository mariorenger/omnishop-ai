# Benchmark phân loại intent tin nhắn CSKH

8 nhãn: `tu_van`, `chot_don`, `dang_giao`, `huy_don`, `doi_tra`, `khieu_nai`, `thanh_toan`, `khac`.

- `train.csv`: 71 câu (8–12 câu/nhãn), `test.csv`: 43 câu khó hơn (không dấu, teencode, câu ngắn, nhãn gần nghĩa).
- `benchmark.py`: so sánh luật từ khoá hiện tại (`orchestrator.classify`), TF-IDF + LogisticRegression,
  và một encoder HuggingFace bất kỳ (`--model`, ví dụ Laya) + LogisticRegression.

```bash
pip install scikit-learn                       # baseline
pip install laya torch transformers            # Laya
python benchmark.py --laya                      # Laya zero-shot, checkpoint multilingual
python benchmark.py --model <hf_id_hoặc_thư_mục_local>   # encoder bất kỳ + LR
```

## Kết quả (2026-09-26, CPU)

| Cách | Số nhãn | Accuracy | Macro-F1 |
|---|---|---|---|
| Keyword (`orchestrator.classify`) | 3 (product/order/knowledge) | 0.488 | 0.491 |
| TF-IDF char+word + LR | 8 | 0.698 | 0.693 |
| Laya zero-shot (`convaiinnovations/laya`, multilingual) | 8 | chưa chạy: sandbox bị chặn huggingface.co | — |

Bộ dữ liệu rất nhỏ, chỉ để kiểm tra nhanh; cần vài trăm tin nhắn thật mỗi nhãn để có kết luận.

## Ghi chú về Laya

- Model: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), Apache-2.0; checkpoint
  `multilingual` (mmBERT-base, 322M) cho tiếng Việt. Gói pip `laya` không kèm trọng số, phải tải từ HuggingFace.
- Theo README của chính Laya: bản gốc zero-shot gần mức ngẫu nhiên trên bộ typed-decisions (0.35),
  intent MASSIVE ở ngôn ngữ ngoài tiếng Anh đạt 0.451; độ chính xác chủ yếu đến từ fine-tune
  (notebook fine-tune chạy trên Kaggle 2x T4).
