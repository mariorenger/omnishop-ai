# Tổng quan nghiên cứu Hệ gợi ý tin tức (News Recommendation), 2023–2026

> Mục tiêu: trả lời 3 câu hỏi — (1) có phương pháp nào **mới & tốt hơn** các mô hình transformer "cũ"
> (NAML, NRMS, LSTUR, Fastformer) không; (2) **xu hướng hiện nay** là LLM reranker, graph hay gì;
> (3) có **code sẵn để chạy/tối ưu** trên tập MIND không.
>
> Ngày tổng hợp: 2026-09-03. Các số liệu là *reported trong từng paper* — thiết lập (MINDsmall/large,
> subset user, embedding) khác nhau nên **không phải lúc nào cũng so trực tiếp được**; hãy dùng như
> chỉ dấu tương đối, và tự reproduce khi làm luận văn.

---

## 1. Bối cảnh: các baseline "cũ" và vì sao chúng chững lại

Hầu hết mô hình gợi ý tin tức content-based đều theo **cùng một khung 3 khối**:

```
News Encoder  →  User Encoder  →  Click Predictor (dot product)
(mã hoá tiêu đề/nội dung)   (tổng hợp lịch sử click)   (điểm khớp user–candidate)
```

| Model | Năm | Ý tưởng chính (khối được cải tiến) | AUC ~ trên MIND* |
|---|---|---|---|
| **DKN** | 2018 | Knowledge-aware CNN (entity từ KG) | ~0.64 |
| **NPA** | 2019 | Personalized attention theo user ID | ~0.66 |
| **NAML** | 2019 | Multi-view CNN + additive attention (title/body/category) | ~0.66–0.67 |
| **LSTUR** | 2019 | GRU cho sở thích dài hạn + ngắn hạn | ~0.675 |
| **NRMS** | 2019 | Multi-head self-attention ở cả news & user encoder | ~0.67 |
| **Fastformer** | 2021 | **Additive/linear attention** — transformer O(n) hiệu quả | Tốt nhất trong nhóm transformer, vẫn ~0.67–0.68 |

\* Khoảng AUC tham khảo trên MIND; tùy MINDsmall/large và cấu hình.

**Nhận định quan trọng:**

- **Fastformer không phải một hướng khác** — nó vẫn là news/user encoder trong đúng khung trên,
  chỉ tối ưu **chi phí tính toán** (linear attention thay cho self-attention bậc hai). Trong nhóm
  biến thể transformer, Fastformer đạt tốt nhất và vượt NRMS gốc, nhưng vẫn nằm trong "trần" của
  họ mô hình nhỏ content-based.
- Nhóm này **đã bão hoà**: NRMS và NAML cho kết quả rất giống nhau (độ chồng lấp dự đoán ~0.746),
  AUC trên MINDlarge chững quanh **0.67–0.70**. Muốn nhảy vọt cần đổi *nguồn tín hiệu* (PLM/LLM,
  KG/graph) chứ không chỉ đổi kiểu attention.

---

## 2. Có gì mới & tốt hơn? — 5 nhóm hướng

### 2.1. Encoder mạnh hơn nhưng vẫn "small model" (candidate-aware, đa sở thích)
Vẫn dùng GloVe/CNN nhưng user-encoder tinh vi hơn, hoặc mô hình hoá quan hệ user–candidate.

- **CAUM** (Candidate-Aware User Modeling) — đưa candidate vào lúc mô hình hoá user; AUC ~**0.697**
  trên MIND, thuộc nhóm mạnh nhất của "small model".
- **MINS, MINER, CenNewsRec** — đa kênh/đa sở thích (multi-interest).
- **"Simplifying content-based" (Iana et al., 2023)** — bài học đáng giá: **hàm mục tiêu huấn luyện
  và cách mô hình hoá user quan trọng hơn kiến trúc encoder phức tạp**; additive attention đơn giản
  + training objective tốt là đủ sánh với model phức tạp.

### 2.2. PLM/LLM làm **encoder** (Discriminative LLM — DLLM) — nơi có SOTA benchmark thực sự
Thay embedding tĩnh (GloVe) bằng **BERT/RoBERTa/LLaMA** ⇒ gain **ổn định, đo được ở full-scale**.

- **MANNeR** (Iana et al., *EMNLP Findings 2024*): backbone là **metric learning**, học các
  news-encoder *chuyên biệt theo khía cạnh* (relevance / diversity / sentiment) rồi **kết hợp tuyến
  tính lúc inference** ⇒ đổi hành vi gợi ý **không cần train lại**. Là **SOTA discriminative** trên
  MIND: cải thiện **~13% AUC so với TANR**, dẫn đầu về AUC/nDCG/HitRate/Precision.
- **DIRE** (trong khung ONCE, WSDM 2024): thay content encoder bằng **LLaMA fine-tune** cho hướng
  discriminative.
- Đây là câu trả lời trực tiếp cho "có tốt hơn NRMS/NAML/Fastformer không": **có** — nâng cấp
  encoder lên PLM/LLM là mức cải thiện chắc chắn và đo được nhất.

### 2.3. Generative LLM (GLLM): prompt / reranker / sinh gợi ý trực tiếp
LLM đóng vai **lý luận & sinh**, thay vì chỉ mã hoá.

- **RecPrompt** (Liu et al., *CIKM 2024*): khung **self-tuning prompt** (Recommender + Prompt
  Optimizer + Monitor), tự động tinh chỉnh prompt. Báo cáo **+3.36% AUC, +10.49% MRR, +9.64% nDCG@5,
  +6.20% nDCG@10** so với deep models — kèm **explainability** (đề xuất metric *TopicScore*).
  ⚠️ Thí nghiệm trên **400 user** (subset), không phải full-scale.
- **ONCE / GENRE** (Liu et al., *WSDM 2024*): dùng **GPT-3.5 để augment dữ liệu** (viết lại tiêu đề,
  sinh user-profile/knowledge) rồi train model nhỏ ⇒ dạng "LLM as data enhancer".
- **LLM category description** (2025): dùng LLM sinh mô tả category ⇒ **+5.8% AUC** so với
  NAML/NRMS/NPA. **PNR-LLM**, **Hierarchical LLM** (2025) — cùng hướng.
- **LLM as reranker**: **LLM4Rerank** (*WWW 2025*) — khung auto-rerank đa tiêu chí (accuracy +
  diversity + …); "LLM as explainable re-ranker" — dùng LLM xếp lại top-K từ retriever nhẹ và
  **giải thích** lý do.

### 2.4. Graph / GNN
Khai thác đồ thị user–news–entity để bắt quan hệ bậc cao & giảm cold-start.

- **DIGAT** (dual-graph interaction), **GLoCIM** (Global-view Long **Chain** Interest Modeling, 2024 —
  sở thích theo chuỗi dài toàn cục), graph **multi-view representation learning** (2024).
- Kết hợp **Knowledge Graph + GNN** cho **cold-start** và tín hiệu đa phương thức (multimodal KG).
- Nhận định: **ổn định nhưng đã bớt "nóng"** so với 2020–2022; mạnh nhất khi ghép KG cho cold-start
  hoặc mô hình hoá chuỗi sở thích dài, **không còn là headline trend 2025**.

### 2.5. Frontier 2025–2026: Generative Retrieval + Semantic IDs, Agentic
Hướng mới nhất của RecSys nói chung, đang lan sang news:

- **Semantic IDs**: tokenize item thành chuỗi "ID ngữ nghĩa" để LLM **sinh thẳng item** (generative
  retrieval) — hiệu quả phụ thuộc mạnh vào không gian embedding nền.
- **Generative recommendation surveys 2025–2026**, **MLP distilled generative recommenders** (làm
  generative-rec rẻ đi), **agentic / multi-agent RecSys + RAG**.
- Với **news cụ thể** vẫn còn **sớm**, nhưng là hướng đáng theo dõi nếu luận văn muốn "đón đầu".

---

## 3. Vậy xu hướng thực sự là gì? (LLM reranker hay graph?)

**Trả lời thẳng: LLM là dòng chủ đạo hiện nay, nhưng phải tách rõ 2 vai trò — vì chúng phục vụ
mục tiêu khác nhau và được đánh giá khác nhau.** Đây là điểm hay bị nhầm trong tài liệu:

| Vai trò của LLM | Đại diện | Mạnh ở | Điểm yếu / lưu ý |
|---|---|---|---|
| **(a) LLM-as-encoder / enhancer** (discriminative) | MANNeR, DIRE, ONCE/GENRE | **Accuracy đáng tin cậy ở full-scale** — nơi các số SOTA trên MIND thực sự đến từ | Chi phí embedding lớn; ít "giải thích được" |
| **(b) LLM-as-reasoner / reranker / generator** (generative) | RecPrompt, LLM4Rerank, LLM explainable re-ranker | **Explainability, cold-start, zero/few-shot, controllability** | Thường eval trên **subset nhỏ**; **đắt**; chưa chắc vượt model discriminative tinh chỉnh tốt ở ranking full-scale |

- **Graph/GNN**: là **hướng bổ trợ**, tốt cho cold-start / quan hệ bậc cao — **không phải trend nổi
  bật nhất** năm 2025.
- **LLM reranker**: đang lên (WWW 2025), hợp với **pipeline công nghiệp** *retrieval nhẹ → LLM rerank
  top-K → giải thích*.
- **Góc hiệu năng của Fastformer** đã dịch chuyển: từ "làm transformer nhanh" sang **"làm LLM-rec rẻ"**
  (distillation, adapter/lightweight task-adaptive modules, MLP distilled generative rec).

**Khuyến nghị định vị cho luận văn:** dùng **discriminative PLM-based (MANNeR hoặc NRMS/NAML + BERT)
làm xương sống accuracy**; **thêm một tầng LLM reranker/generative** cho **explainability + cold-start**;
so sánh **chi phí vs. lợi ích**. Đây là câu chuyện "hai lớp" vừa vững benchmark vừa bắt trend.

---

## 4. Code sẵn để chạy & tối ưu trên MIND (phần thực hành)

Xếp theo mức phù hợp cho **nghiên cứu/luận văn** (đã kiểm tra link):

### ⭐ 1. NewsRecLib — *lựa chọn tốt nhất cho nghiên cứu học thuật*
<https://github.com/andreeaiana/newsreclib> · PyTorch-Lightning + Hydra
- **13 mô hình** sẵn: NRMS, NAML, NPA, LSTUR, TANR, DKN, MINER, MINS, **CAUM**, CenNewsRec, **MANNeR**,
  SentiRec, SentiDebias.
- **Hỗ trợ PLM encoder** (RoBERTa/BERT) — bật/tắt qua config.
- MIND **small & large**; metric **accuracy (AUROC/MRR/nDCG@k) + beyond-accuracy (diversity/
  personalization) + fairness** ⇒ rất hợp để làm phần "đánh giá đa chiều" cho luận văn.
- Chạy 1 dòng:
  ```bash
  python newsreclib/train.py experiment=nrms_mindsmall_pretrainedemb_celoss_bertsent
  python newsreclib/train.py experiment=nrms_mindlarge_pretrainedemb_celoss_bertsent
  ```

### 2. Microsoft Recommenders — *baseline sạch, chính thống*
<https://github.com/recommenders-team/recommenders> (trước là microsoft/recommenders)
- Notebook quick-start cho **NRMS, NAML, LSTUR, NPA, DKN** trên MIND (`examples/00_quick_start/*_MIND.ipynb`).
- Tốt để **reproduce baseline** một cách đáng tin, ít phụ thuộc.

### 3. Legommenders — *thư viện content-based có LLM support (WWW/TheWebConf 2025)*
<https://github.com/Jyonn/Legommenders> (và ONCE: <https://github.com/Jyonn/ONCE>)
- Khung content-based **tích hợp PLM & LLM**, chạy GENRE/DIRE/ONCE; hỗ trợ **MIND**.
- Chọn nếu muốn đi hướng **LLM-based** một cách bài bản.

### 4. NewsReX — *mới (2025), tối ưu tốc độ/GPU*
<https://github.com/igor17400/NewsReX>
- Backend **JAX/Flax (JIT+XLA) hoặc PyTorch — đổi bằng 1 flag**; NRMS/NAML/LSTUR/**CROWN**/**PP-Rec**/DIGAT.
- **Optuna HPO**, multi-seed, dashboard trực quan ⇒ hợp khi cần **train nhanh / GPU hạn chế / tối ưu siêu tham số**.
  ```bash
  uv run python src/train.py experiment=mind/nrms framework=jax
  ```

### 5. yusanshi/news-recommendation — *PyTorch đơn giản, dễ đọc/sửa*
<https://github.com/yusanshi/news-recommendation> — NRMS/NAML/LSTUR/DKN/HiFi-Ark/TANR. Tốt để **học code & thử ý tưởng**.

### 6. Hướng LLM prompt/generative (cần API key GPT hoặc LLaMA)
- **RecPrompt**: <https://github.com/Ruixinhua/rec-prompt>
- **ONCE (GENRE/DIRE)**: <https://github.com/Jyonn/ONCE>

### 0. Benchmark tối giản có sẵn trong repo này — *để chạy so sánh nhanh*
[`experiments/newsrec_bench/`](../../experiments/newsrec_bench/) — một khung **một-file-mỗi-tầng**
so **NRMS / NAML / Fastformer** vs **LightGCN (thuần graph)** vs **LLM-encoder** vs **HybridOpt
(biến thể tự tối ưu)** trên **cùng loss + cùng metric** (AUC/MRR/nDCG **và** tốc độ/#params).
Chạy ngay trên synthetic (CPU) hoặc cắm thẳng MINDsmall/large. Dùng để lấy trực giác về đánh đổi
*chất lượng ↔ tốc độ ↔ cold-start* trước khi dựng thí nghiệm đầy đủ bằng NewsRecLib.

> Ghi chú về Fastformer: repo `yusanshi` **không** có Fastformer; NewsRecLib cũng không liệt kê sẵn.
> Fastformer chính thức: <https://github.com/wuch15/Fastformer> (module attention, cần tự ghép vào
> news/user encoder). Trong NewsRecLib có thể thay khối attention để tái tạo tinh thần Fastformer.

---

## 5. Lộ trình thí nghiệm gợi ý cho luận văn

1. **Baseline** — reproduce NRMS/NAML/LSTUR (+ Fastformer nếu muốn) trên **MINDsmall** bằng
   **NewsRecLib** hoặc **Microsoft Recommenders**. Chốt số AUC/MRR/nDCG làm mốc.
2. **Nâng encoder PLM** — bật BERT/RoBERTa trong NewsRecLib ⇒ đo mức gain so với GloVe.
3. **SOTA discriminative** — chạy **CAUM** và **MANNeR** ⇒ so với baseline; MANNeR còn cho phép
   phân tích **trade-off accuracy ↔ diversity ↔ sentiment**.
4. **Tầng LLM** — thêm **LLM reranker/generative** (RecPrompt hoặc ONCE) cho **top-K + giải thích**;
   báo cáo cả **chi phí (thời gian/token)**, không chỉ accuracy.
5. **Đánh giá beyond-accuracy** — diversity, fairness, cold-start (NewsRecLib hỗ trợ sẵn) ⇒ đây là
   **điểm khác biệt học thuật** dễ ghi điểm hơn là chỉ chạy đua AUC.

---

## Nguồn tham khảo

- A Survey on LLM-based News Recommender Systems (arXiv:2502.09797, 2/2025) —
  <https://arxiv.org/abs/2502.09797>
- MANNeR — *Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation*
  (EMNLP Findings 2024) — <https://aclanthology.org/2024.findings-emnlp.558/> · arXiv:2307.16089
- Simplifying Content-Based Neural News Recommendation (Iana et al., 2023) — arXiv:2304.03112
- RecPrompt (CIKM 2024) — arXiv:2312.10463 · code <https://github.com/Ruixinhua/rec-prompt>
- ONCE / GENRE / DIRE (WSDM 2024) — arXiv:2305.06566 · code <https://github.com/Jyonn/ONCE>
- Legommenders (TheWebConf 2025) — <https://github.com/Jyonn/Legommenders>
- LLM4Rerank (ACM Web Conference 2025) — <https://dl.acm.org/doi/10.1145/3696410.3714922>
- News Recommendation with Category Description by an LLM (CEUR Vol-4056) —
  <https://ceur-ws.org/Vol-4056/short2.pdf>
- Enhancing News Recommendation with Hierarchical LLM (arXiv:2504.20452, 2025)
- GLoCIM: Global-view Long Chain Interest Modeling (arXiv:2408.00859, 2024)
- Fastformer: Additive Attention Can Be All You Need (arXiv:2108.09084, 2021) — code
  <https://github.com/wuch15/Fastformer>
- NewsRecLib (EMNLP Demo 2023) — <https://github.com/andreeaiana/newsreclib> · docs
  <https://newsreclib.readthedocs.io>
- Microsoft Recommenders — <https://github.com/recommenders-team/recommenders>
- NewsReX — <https://github.com/igor17400/NewsReX>
- yusanshi/news-recommendation — <https://github.com/yusanshi/news-recommendation>
- MIND dataset — <https://msnews.github.io>
