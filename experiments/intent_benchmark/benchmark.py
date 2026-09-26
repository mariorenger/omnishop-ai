"""Benchmark phân loại intent tin nhắn CSKH.

So sánh 3 cách trên cùng train.csv / test.csv:
  1. keyword   — classify() hiện tại trong app/modules/orchestrator.py (luật từ khoá)
  2. tfidf     — TF-IDF n-gram ký tự + LogisticRegression (baseline không cần model)
  3. encoder   — encoder HuggingFace bất kỳ -> embedding -> LogisticRegression
  4. laya      — Laya zero-shot (câu hỏi `choice`, không train), checkpoint multilingual

Chạy:
  python benchmark.py                          # keyword + tfidf
  python benchmark.py --model <hf_id_or_path>  # thêm encoder (cần: pip install torch transformers)
  python benchmark.py --laya                   # thêm Laya (cần: pip install laya torch transformers)
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline, make_union

HERE = Path(__file__).parent
ROOT = HERE.parents[1]

# nhãn chi tiết -> 3 nhánh mà orchestrator hiện có (product / order / knowledge)
COARSE = {"tu_van": "product", "chot_don": "product", "dang_giao": "order", "huy_don": "order",
          "doi_tra": "order", "thanh_toan": "order", "khieu_nai": "knowledge", "khac": "knowledge"}


def load(name):
    with open(HERE / name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def normalize(t: str) -> str:
    return unicodedata.normalize("NFC", t.lower().strip())


def run_keyword(Xte, yte):
    sys.path.insert(0, str(ROOT))
    from app.modules.orchestrator import classify  # noqa: E402
    pred = [classify(t) for t in Xte]
    gold = [COARSE[y] for y in yte]
    return pred, gold


def run_tfidf(Xtr, ytr, Xte):
    clf = make_pipeline(
        make_union(
            TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
            TfidfVectorizer(preprocessor=normalize, analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
        ),
        LogisticRegression(C=10, max_iter=2000),
    )
    clf.fit(Xtr, ytr)
    return list(clf.predict(Xte))


def run_encoder(model_id, Xtr, ytr, Xte, prefix=""):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()

    @torch.no_grad()
    def embed(texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            b = tok([prefix + normalize(t) for t in texts[i:i + bs]], padding=True,
                    truncation=True, max_length=128, return_tensors="pt")
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1)                      # mean pooling
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out).numpy()

    t0 = time.perf_counter()
    Etr, Ete = embed(Xtr), embed(Xte)
    ms = (time.perf_counter() - t0) * 1000 / (len(Xtr) + len(Xte))
    clf = LogisticRegression(C=10, max_iter=2000).fit(Etr, ytr)
    return list(clf.predict(Ete)), ms


# mô tả từng nhãn cho câu hỏi `choice` của Laya
LAYA_CRITERIA = {
    "tu_van": "Khách hỏi thông tin sản phẩm: giá, size, màu, chất liệu, còn hàng, khuyến mãi",
    "chot_don": "Khách đồng ý mua, chốt đơn, gửi địa chỉ/số điện thoại để lên đơn",
    "dang_giao": "Khách hỏi tình trạng giao hàng, đơn đang ở đâu, bao giờ nhận được",
    "huy_don": "Khách muốn hủy đơn hoặc không lấy hàng nữa",
    "doi_tra": "Khách muốn đổi size/mẫu, trả hàng, hoàn tiền, hỏi chính sách đổi trả",
    "khieu_nai": "Khách phàn nàn, bực tức, chê chất lượng hàng hoặc dịch vụ",
    "thanh_toan": "Khách hỏi hoặc báo về thanh toán, chuyển khoản, số tài khoản, trả góp",
    "khac": "Chào hỏi, cảm ơn, xác nhận ngắn, hoặc câu hỏi chung về shop",
}


def run_laya(Xte, model_id, subfolder):
    import laya
    agent = laya.load(model_id, subfolder=subfolder or None)
    q = {"intent": {"type": "choice",
                    "instructions": "Tin nhắn của khách hàng gửi shop thuộc loại nào?",
                    "criteria": LAYA_CRITERIA}}
    t0 = time.perf_counter()
    res = agent.predict_batch([{"body": t} for t in Xte], q, batch_size=16)
    ms = (time.perf_counter() - t0) * 1000 / len(Xte)
    return [r["answers"]["intent"]["choice"] for r in res], ms


def report(title, X, gold, pred):
    print(f"\n=== {title} ===")
    print(f"accuracy={accuracy_score(gold, pred):.3f}  macro-F1={f1_score(gold, pred, average='macro'):.3f}")
    print(classification_report(gold, pred, zero_division=0))
    wrong = [(t, g, p) for t, g, p in zip(X, gold, pred) if g != p]
    if wrong:
        print(f"Sai {len(wrong)}/{len(X)}:")
        for t, g, p in wrong:
            print(f"  {t!r:60}  đúng={g:10} đoán={p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model id hoặc thư mục local của encoder (vd. Laya)")
    ap.add_argument("--prefix", default="", help="tiền tố query nếu model yêu cầu, vd. 'query: '")
    ap.add_argument("--laya", action="store_true", help="chạy thêm Laya zero-shot")
    ap.add_argument("--laya-model", default="convaiinnovations/laya")
    ap.add_argument("--laya-subfolder", default="multilingual", help="'' = checkpoint tiếng Anh ở gốc repo")
    args = ap.parse_args()

    Xtr, ytr = load("train.csv")
    Xte, yte = load("test.csv")
    print(f"train={len(Xtr)}  test={len(Xte)}  labels={sorted(set(ytr))}")

    pred, gold = run_keyword(Xte, yte)
    report("1. keyword (orchestrator.classify, 3 nhánh)", Xte, gold, pred)

    report("2. TF-IDF char+word + LogisticRegression (8 nhãn)", Xte, yte, run_tfidf(Xtr, ytr, Xte))

    if args.model:
        pred, ms = run_encoder(args.model, Xtr, ytr, Xte, args.prefix)
        report(f"3. encoder {args.model} + LogisticRegression (8 nhãn, {ms:.1f} ms/câu CPU)", Xte, yte, pred)

    if args.laya:
        pred, ms = run_laya(Xte, args.laya_model, args.laya_subfolder)
        report(f"4. Laya zero-shot {args.laya_model}/{args.laya_subfolder} (8 nhãn, {ms:.1f} ms/câu)", Xte, yte, pred)


if __name__ == "__main__":
    main()
