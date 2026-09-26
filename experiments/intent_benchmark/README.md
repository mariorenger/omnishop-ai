# Benchmark phân loại intent tin nhắn CSKH

8 nhãn: `tu_van`, `chot_don`, `dang_giao`, `huy_don`, `doi_tra`, `khieu_nai`, `thanh_toan`, `khac`.

- `train.csv`: 71 câu (8–12 câu/nhãn), `test.csv`: 43 câu khó hơn (không dấu, teencode, câu ngắn, nhãn gần nghĩa).
- `benchmark.py`: so sánh luật từ khoá hiện tại (`orchestrator.classify`), TF-IDF + LogisticRegression,
  và một encoder HuggingFace bất kỳ (`--model`, ví dụ Laya) + LogisticRegression.

```bash
pip install scikit-learn                       # baseline
pip install torch transformers                 # nếu chạy encoder
python benchmark.py --model <hf_id_hoặc_thư_mục_local>
```

## Kết quả (2026-09-26, CPU)

| Cách | Số nhãn | Accuracy | Macro-F1 |
|---|---|---|---|
| Keyword (`orchestrator.classify`) | 3 (product/order/knowledge) | 0.488 | 0.491 |
| TF-IDF char+word + LR | 8 | 0.698 | 0.693 |
| Encoder (Laya) + LR | 8 | chưa chạy: sandbox bị chặn huggingface.co | — |

Bộ dữ liệu rất nhỏ, chỉ để kiểm tra nhanh; cần vài trăm tin nhắn thật mỗi nhãn để có kết luận.
