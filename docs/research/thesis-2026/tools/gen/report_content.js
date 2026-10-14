// Content of the report. Numbers come from out_facts.json (facts.py); prose cites references by key: [@key, @key2].
module.exports = (F) => {
  const f4 = (x) => (x == null ? "—" : x.toFixed(4).replace(".", ","));
  const s4 = (x) => (x == null ? "—" : (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(4).replace(".", ","));
  const ci = (c) => (c ? `[${s4(c[0])}; ${s4(c[1])}]` : "—");
  const pct = (x) => x.toFixed(1).replace(".", ",") + "%";
  const n = (x) => x.toLocaleString("de-DE");
  const S = F.data_small, L = F.data_large;
  const mrow = (r) => [r.label, f4(r.auc), f4(r.mrr), f4(r.ndcg5), f4(r.ndcg10), f4(r.cold_auc), f4(r.warm_auc)];
  const MH = ["Mô hình / cấu hình", "AUC", "MRR", "nDCG@5", "nDCG@10", "AUC lạnh", "AUC ấm"];
  const MW = [0.37, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105];
  const MA = ["l", "r", "r", "r", "r", "r", "r"];

  const title = "NGHIÊN CỨU BIỂU DIỄN NGỮ NGHĨA TỪ MÔ HÌNH NGÔN NGỮ LỚN TRONG HỆ GỢI Ý TIN TỨC: ĐÁNH GIÁ CÓ KIỂM SOÁT CÁC HƯỚNG TIẾP CẬN 2024–2026 TRÊN TẬP DỮ LIỆU MIND";
  const titleEn = "Semantic Representations from Large Language Models for News Recommendation: A Controlled Empirical Study of 2024–2026 Directions on the MIND Benchmark";

  const abstractVi = [
    "Giai đoạn 2024–2026 chứng kiến nhiều hướng đưa mô hình ngôn ngữ lớn (LLM) vào hệ gợi ý: LLM làm bộ mã hoá, LLM sinh tri thức, gợi ý sinh với Semantic ID, suy luận, RAG/GraphRAG và LLM làm bộ xếp hạng. Gợi ý tin tức là phép thử khắc nghiệt cho các hướng này vì phần lớn lượt nhấp rơi vào bài mới: trên MIND, " + pct(S.cold_click_share_test_pct) + " (MIND-small) và " + pct(L.cold_click_share_test_pct) + " (MIND-large) lượt nhấp của tập kiểm thử thuộc về bài không xuất hiện khi huấn luyện. Báo cáo trình bày một đánh giá thực nghiệm có kiểm soát các hướng trên dưới cùng một giao thức: chia theo thời gian (validation là ngày cuối của tập huấn luyện; tập dev chính thức làm tập kiểm thử và chỉ được dùng một lần), khoảng tin cậy 95% ghép cặp theo impression, lát cắt bài lạnh/ấm, đo nhiễu seed, baseline NRMS/NAML/Fastformer huấn luyện lại đúng công thức, và các dự đoán được ghi trước khi chạy.",
    "Trên MIND-small (73.152 impression kiểm thử; khoảng 16 họ kỹ thuật, ≈150 biến thể) không hướng LLM zero-shot nào vượt hẳn baseline huấn luyện đúng: cấu hình tốt nhất (Qwen3-Embedding-0.6B, chuyên mục trong văn bản, tổng hợp lịch sử kiểu late interaction) đạt nDCG@10 = 0,4094 so với 0,4078 của NRMS (khoảng tin cậy của chênh lệch chứa 0). Hai yếu tố có hiệu ứng dương rõ và tái lập được là cách tổng hợp lịch sử (log-sum-exp, +0,0128) và trường dữ liệu đưa vào encoder (chuyên mục, +0,0182 với Qwen3); RAG/GraphRAG ở mức biểu diễn, Semantic ID sinh, giám khảo LLM 1,7B, đa sở thích và LoRA cho hiệu ứng ≤ 0,005 hoặc âm. Trên MIND-large (376.471 impression kiểm thử), validation “ấm” (18,1% lượt nhấp lạnh) làm đảo dấu đánh giá của RAG/GraphRAG và thổi phồng các bộ đọc học được; ghép tuyến tính đạt AUC 0,6863 / nDCG@10 0,4373; bộ xếp hạng GBDT chỉ tăng thêm khi có đặc trưng phơi nhiễm không nhãn (0,7013 / 0,4516) và đã bão hoà; kiểm định chéo theo người dùng thổi phồng ước lượng 0,038 nDCG@10, theo khối thời gian chỉ còn 0,007. Encoder lớn hơn không phải nút thắt: BGE-large (335M) hơn Qwen3-Embedding-0.6B +0,0102 khi làm đầu vào cho cùng bộ đọc, và ghép hai không gian vector tốt hơn từng cái.",
    "Phần chỉ dùng nội dung và lịch sử của hệ thống thấp hơn PLM-NR (AUC 0,7064 theo công bố, đo trên tập test của leaderboard) khoảng 0,02 AUC: khoảng trống nằm ở việc tinh chỉnh encoder end-to-end chứ không ở tầng ghép. Một thí nghiệm tinh chỉnh hai lớp cuối có cổng dừng và nhánh đối chứng khớp cặp (v17) đã được cài đặt, kiểm thử và đang chờ chạy. Báo cáo cũng tổng hợp các hướng LLM-rec 2024–2026 tại các hội nghị hàng đầu và đối chiếu chúng với bằng chứng thực nghiệm thu được.",
  ];
  const keywordsVi = "gợi ý tin tức; mô hình ngôn ngữ lớn; biểu diễn văn bản; MIND; khởi đầu lạnh; late interaction; bộ xếp hạng GBDT; phương pháp đánh giá.";
  const abstractEn = [
    "Between 2024 and 2026, large language models (LLMs) entered recommender systems through many routes: LLM-based encoders, LLM-generated knowledge, generative recommendation with Semantic IDs, reasoning, RAG/GraphRAG and LLM rankers. News recommendation is a hard test for all of them because most clicks go to fresh articles: on MIND, 85.9% (small) and 82.2% (large) of the test clicks are on articles never seen in training. We report a controlled empirical study of these directions under one protocol: a chronological split (validation = last training day, official dev = test, touched once), impression-level paired 95% confidence intervals, cold/warm slices, measured seed noise, NRMS/NAML/Fastformer retrained with their published recipes, and predictions registered before each run.",
    "On MIND-small no zero-shot LLM configuration clearly beats properly trained baselines: the best one (Qwen3-Embedding-0.6B, category in the embedded text, log-sum-exp late interaction) reaches nDCG@10 0.4094 versus 0.4078 for retrained NRMS (interval includes 0). History aggregation (+0.0128) and the input fields (+0.0182 for categories with Qwen3) are the only robust gains; representation-level RAG/GraphRAG, a generative Semantic-ID ranker, a 1.7B LLM judge, multi-interest models and LoRA give ≤ 0.005 or negative effects. On MIND-large a warm validation day (18.1% cold clicks) flips the sign of RAG/GraphRAG and inflates learned readers; a linear fusion reaches AUC 0.6863 / nDCG@10 0.4373 and a gradient-boosted stacker adds value only with label-free exposure features (0.7013 / 0.4516) before saturating; user-fold cross-validation overstates nDCG@10 by 0.038, time-block folds by 0.007. Encoder size is not the bottleneck: BGE-large (335M) beats Qwen3-Embedding-0.6B by 0.0102 under the same reader and concatenating both spaces is best. The content-only system stays about 0.02 AUC below PLM-NR; a gated partial fine-tuning experiment with a matched frozen control is implemented and pending.",
  ];
  const keywordsEn = "news recommendation; large language models; text embeddings; MIND; cold start; late interaction; gradient-boosted stacking; evaluation methodology.";

  // ------------------------------------------------------------------ front matter: abbreviations
  const front = [
    { t: "h1", text: "DANH MỤC TỪ VIẾT TẮT VÀ THUẬT NGỮ" },
    { t: "table", caption: "Từ viết tắt và quy ước dùng trong báo cáo.", widths: [0.22, 0.78], size: 21, header: ["Ký hiệu", "Ý nghĩa"], rows: [
      ["LLM / PLM", "Mô hình ngôn ngữ lớn / mô hình ngôn ngữ tiền huấn luyện (BERT, RoBERTa…)."],
      ["AUC, MRR, nDCG@k", "Diện tích dưới đường ROC; nghịch đảo thứ hạng trung bình; nDCG cắt ở k — tính trên từng impression rồi lấy trung bình (giao thức MIND)."],
      ["Impression", "Một lần hiển thị danh sách ứng viên cho một người dùng; nhãn 1 = được nhấp."],
      ["Bài lạnh / ấm", "Bài được nhấp trong tập đánh giá mà không (lạnh) / có (ấm) xuất hiện trong bất kỳ dòng huấn luyện nào; AUC lạnh/ấm tính trên lát tương ứng."],
      ["Zero-shot", "Không huấn luyện thêm trên dữ liệu nhấp: chỉ dùng vector của encoder đóng băng và một phép tổng hợp cố định."],
      ["Reader", "Bộ đọc lịch sử học được trên bảng vector bài đóng băng (đầu 2 lớp + chú ý theo ứng viên)."],
      ["LSE", "Log-sum-exp: tổng hợp lịch sử kiểu late interaction, nội suy giữa trung bình và cực đại (công thức {E:lse})."],
      ["RAG / GraphRAG", "Sinh tăng cường truy xuất / phiên bản dùng cộng đồng đồ thị; ở đây áp dụng ở mức biểu diễn bài."],
      ["SID", "Semantic ID: chuỗi mã rời rạc lượng tử hoá vector nội dung (RQ-KMeans/RQ-VAE)."],
      ["GBDT / stacker", "Cây tăng cường gradient (XGBoost) học cách kết hợp điểm thành phần và đặc trưng."],
      ["+pop / streaming", "Biến thể đọc popularity online: đếm nhấp của các impression trước đó trong chính ngày kiểm thử — không so được với số công bố offline."],
      ["CI", "Khoảng tin cậy 95% ghép cặp theo impression (chỉ phản ánh nhiễu mẫu kiểm thử, không phản ánh nhiễu huấn luyện)."],
      ["V10, v7 … v17", "Phiên bản bộ thí nghiệm: V10 là bộ đo gốc; v7–v13 chạy trên MIND-small, v14–v17 trên MIND-large (Bảng {T:runs})."],
    ] },
    { t: "pagebreak" },
  ];

  // ------------------------------------------------------------------ body
  const body = [];
  const add = (...b) => body.push(...b);

  // ===== 1
  add({ t: "h1", text: "1. GIỚI THIỆU" });
  add({ t: "h2", text: "1.1. Bối cảnh: mô hình ngôn ngữ trong hệ gợi ý" });
  add({ t: "p", text: "Mô hình ngôn ngữ đã thay đổi cách hệ gợi ý biểu diễn nội dung: từ vector từ tĩnh GloVe [@glove] trong các mô hình tin tức nơ-ron NRMS [@nrms], NAML [@naml] và Fastformer [@fastformer], sang mô hình ngôn ngữ tiền huấn luyện được tinh chỉnh như PLM-NR [@plmnr] và UNBERT [@unbert], và gần đây là các mô hình nhúng xây trên LLM như BGE [@cpack], LLM2Vec [@llm2vec] và Qwen3-Embedding [@qwen3emb]. Các khảo sát [@linsurvey, @fansurvey] phân loại vai trò của LLM theo vị trí trong pipeline (kỹ nghệ đặc trưng, mã hoá, chấm điểm/xếp hạng, điều phối) và theo cách thích nghi (có tinh chỉnh LLM hay không, có giữ mô hình gợi ý truyền thống khi suy luận hay không). Trong 2025–2026 trọng tâm dịch chuyển mạnh sang gợi ý sinh với Semantic ID [@tiger, @onerec], suy luận trong gợi ý [@rearec, @onerecthink] và tác tử [@agentsurvey]." });
  add({ t: "h2", text: "1.2. Đặc thù của bài toán gợi ý tin tức" });
  add({ t: "bullets", items: [
    "**Khởi đầu lạnh phía bài viết chiếm ưu thế.** Tin tức có vòng đời ngắn và bài mới xuất hiện liên tục. Số đo trong báo cáo này: " + pct(S.cold_click_share_test_pct) + " lượt nhấp kiểm thử của MIND-small và " + pct(L.cold_click_share_test_pct) + " của MIND-large rơi vào bài không xuất hiện trong dữ liệu huấn luyện (Mục 3.2). Phương pháp chỉ dựa ID gần như không có tín hiệu cho các lượt nhấp này.",
    "**Bài toán xếp hạng slate cho sẵn.** Mỗi impression là danh sách ứng viên đã được hiển thị; mô hình xếp hạng chúng và được đo bằng AUC, MRR, nDCG@5, nDCG@10 trên từng impression [@mind].",
    "**Văn bản ngắn, không có thân bài.** Tiêu đề trung bình " + String(S.title_words).replace(".", ",") + " từ, abstract " + String(S.abstract_words).replace(".", ",") + " từ; bản phát hành MIND không kèm thân bài (chỉ có URL).",
    "**Dịch chuyển phân phối theo thời gian.** Validation lấy theo ngày cuối của tập huấn luyện “ấm” hơn hẳn tập kiểm thử: trên MIND-large chỉ " + pct(L.cold_click_share_val_pct) + " lượt nhấp validation là bài lạnh (Mục 3.3).",
  ] });
  add({ t: "h2", text: "1.3. Khoảng trống nghiên cứu" });
  add({ t: "p", text: "Ba vấn đề phương pháp làm khó việc kiểm chứng các tuyên bố về LLM trong gợi ý tin tức. **(i) Baseline yếu.** Bộ đo ban đầu (V10) của chính nghiên cứu này cho NRMS nDCG@10 = 0,3685 trên MIND-small, thấp hơn số công bố 0,4121 [@iana] tới 0,04 — đủ để biến một kỹ thuật bình thường thành “vượt trội”. Hiện tượng baseline yếu đã được cảnh báo từ lâu [@dacrema] và tái xuất hiện với hệ dùng LLM [@fioretti]. **(ii) Giao thức không nhất quán.** Số công bố trên MIND-large thường đo trên tập test của leaderboard, trong khi nhiều nghiên cứu báo cáo trên dev; biến thể dùng tín hiệu phổ biến trực tuyến không cùng bài toán với mô hình chỉ dùng nội dung. **(iii) Kết luận trái chiều về kích thước mô hình.** Có nghiên cứu cho thấy LM lớn hơn không chắc tốt hơn trên MIND [@zhao], trong khi các khảo sát quy mô lớn nhấn mạnh tiềm năng của encoder rất lớn [@litcf]. Vì vậy cần một đánh giá có kiểm soát: cùng dữ liệu, cùng giao thức, cùng bộ đo, có khoảng tin cậy và có đo nhiễu huấn luyện." });
  add({ t: "h2", text: "1.4. Câu hỏi nghiên cứu" });
  add({ t: "bullets", items: [
    "**RQ1 —** Biểu diễn văn bản zero-shot từ encoder/LLM hiện đại đứng ở đâu so với các mô hình tin tức nơ-ron được huấn luyện đúng công thức (NRMS, NAML, Fastformer) dưới cùng một giao thức theo thời gian?",
    "**RQ2 —** Nút thắt của hệ dựa trên embedding đóng băng nằm ở bộ mã hoá (kích thước, họ mô hình), ở trường dữ liệu đưa vào, hay ở cách tổng hợp lịch sử đọc?",
    "**RQ3 —** Các hướng LLM-rec 2024–2026 (RAG/GraphRAG, Semantic ID sinh, tower người dùng LLM, LLM sinh tri thức, LLM làm giám khảo, đa sở thích, tinh chỉnh tham số hiệu quả) có giữ được lợi ích trên tin tức khi đánh giá nghiêm ngặt không?",
    "**RQ4 —** Ở quy mô MIND-large với dịch chuyển phân phối mạnh, nên ghép các thành phần thế nào (tuyến tính hay GBDT), và các ước lượng trên validation đáng tin đến đâu?",
    "**RQ5 —** Khoảng cách còn lại so với các mô hình PLM tinh chỉnh end-to-end nằm ở đâu, và có thu hẹp được bằng tinh chỉnh một phần encoder không? (thí nghiệm v17, chờ kết quả)",
  ] });
  add({ t: "h2", text: "1.5. Đóng góp" });
  add({ t: "numbers", items: [
    "**Một khung đánh giá có kiểm soát** cho LLM trong gợi ý tin tức: giao thức theo thời gian, tập kiểm thử chạm một lần, CI ghép cặp theo impression, lát lạnh/ấm, đo nhiễu seed, đối chiếu tái lập với bộ đo gốc, dự đoán ghi trước và kiểm thử chống rò nhãn (" + F.compute.selftests + " kiểm thử tự động).",
    "**Đánh giá có hệ thống ≈16 họ kỹ thuật (≈150 biến thể) thuộc bảy hướng LLM-rec** trên MIND-small, so với NRMS/NAML/Fastformer huấn luyện lại đúng công thức; tách được nguồn gốc khoảng cách giữa bộ đo gốc và số công bố (vector từ huấn luyện sẵn).",
    "**Các khám phá về dữ liệu thực tế:** ưu thế bài lạnh, validation “ấm” đánh lừa việc chọn mô hình (RAG/GraphRAG đổi dấu), vai trò của trường văn bản (abstract, chuyên mục), sức mạnh và giới hạn của popularity.",
    "**Các khám phá về mô hình:** late interaction không tham số ngang/hơn bộ đọc học được trên MIND-small; kích thước encoder không phải nút thắt; bộ xếp hạng GBDT bão hoà; kiểm định chéo theo người dùng thổi phồng ước lượng.",
    "**Một pipeline MIND-large chạy được trên 1 GPU T4 / 32 GB RAM** với đo RAM đỉnh thật theo từng giai đoạn; chẩn đoán và sửa lỗi hết bộ nhớ.",
    "**Định vị trung thực so với văn liệu** và một thí nghiệm có cổng dừng cho bước tiếp theo (v17).",
  ] });
  add({ t: "h2", text: "1.6. Cấu trúc báo cáo" });
  add({ t: "p", text: "Chương 2 tổng quan bảy hướng LLM-rec 2024–2026 theo các công trình ở hội nghị hàng đầu. Chương 3 trình bày dữ liệu và các khám phá về dữ liệu thực tế. Chương 4 mô tả giao thức và các phương pháp đã cài đặt. Chương 5 báo cáo kết quả trên MIND-small và MIND-large. Chương 6 thảo luận, trả lời các câu hỏi nghiên cứu và nêu hạn chế; Chương 7 trình bày hướng phát triển; Chương 8 kết luận." });

  // ===== 2
  add({ t: "h1", text: "2. TỔNG QUAN: CÁC HƯỚNG LLM CHO HỆ GỢI Ý 2024–2026" });
  add({ t: "h2", text: "2.1. Khung phân loại" });
  add({ t: "p", text: "Lin và cs. [@linsurvey] (TOIS 2025) tổ chức các công trình theo hai câu hỏi: LLM nằm ở đâu trong pipeline (kỹ nghệ đặc trưng, mã hoá đặc trưng, chấm điểm/xếp hạng, điều khiển pipeline) và thích nghi thế nào (có tinh chỉnh LLM hay không; có dùng mô hình gợi ý truyền thống khi suy luận hay không). Zhao và cs. [@fansurvey] (TKDE 2024) phân theo mô hình huấn luyện: tiền huấn luyện, tinh chỉnh và prompting. Báo cáo dùng một phân loại thực dụng gồm bảy hướng; mỗi hướng ứng với một hoặc vài giả thuyết được kiểm định ở Chương 5. Bảng tổng hợp (Bảng {T:dirs}) ở cuối chương nối từng hướng với công trình tiêu biểu, thí nghiệm tương ứng và kết quả đo được." });
  add({ t: "h2", text: "2.2. Hướng 1 — LLM làm bộ mã hoá nội dung" });
  add({ t: "p", text: "Mô hình nhúng xây trên LLM dẫn đầu các bộ đo truy hồi văn bản: LLM2Vec [@llm2vec] (COLM 2024) biến LLM decoder-only thành encoder bằng chú ý hai chiều và học tương phản; Qwen3-Embedding [@qwen3emb] (2025, 0,6B–8B tham số) dùng pooling token cuối và chỉ dẫn. Trong gợi ý, LLMEmb [@llmemb] (AAAI 2025) tinh chỉnh tương phản có giám sát rồi thêm tín hiệu cộng tác để sinh embedding item, nhắm bài toán đuôi dài; LLM2Rec [@llm2rec] (KDD 2025) tinh chỉnh LLM bằng tín hiệu cộng tác rồi biến nó thành mô hình nhúng item, cải thiện cả trong và ngoài miền; LEARN [@learn] (AAAI 2025) giữ LLM đóng băng làm bộ mã hoá item và học cầu nối hai tháp sang không gian cộng tác, triển khai trong công nghiệp; PAD [@pad] (SIGIR 2025) tiền huấn luyện riêng, căn chỉnh rồi tách rời biểu diễn văn bản và cộng tác, lợi ích lớn nhất ở item lạnh." });
  add({ t: "p", text: "Riêng với tin tức, Zhao và cs. [@zhao] (ECIR 2025) cho thấy LM lớn hơn không chắc tốt hơn trên MIND, cần chọn siêu tham số tinh chỉnh cẩn thận hơn, nhưng giúp người dùng ít lịch sử; Li và cs. [@litcf] khảo sát lọc cộng tác dựa trên văn bản khi tăng encoder từ 10⁸ lên 10¹¹ tham số. *Liên hệ với báo cáo:* câu hỏi “encoder lớn hơn có giúp không” được kiểm chứng trực tiếp với BGE-small/base/large, Qwen3-Embedding-0.6B và vector nối hai encoder, cả ở chế độ zero-shot lẫn làm đầu vào cho bộ đọc học được (Mục 5.4 và 5.9)." });
  add({ t: "h2", text: "2.3. Hướng 2 — LLM sinh tri thức và làm giàu dữ liệu" });
  add({ t: "p", text: "KAR [@kar] (RecSys 2024) dùng LLM sinh tri thức suy luận về sở thích người dùng và tri thức thực tế về item, mã hoá bằng bộ điều hợp lai rồi đưa vào mô hình CTR; KAR được triển khai trên nền tảng tin tức và âm nhạc của Huawei. ONCE [@once] (WSDM 2024) kết hợp LLM mã nguồn mở làm bộ mã hoá nội dung với LLM đóng để sinh dữ liệu bổ sung bằng prompt, báo cáo cải thiện tương đối tới 19,32%. EmbSum [@embsum] (RecSys 2024) dùng LLM tóm tắt sở thích để giám sát bộ mã hoá–giải mã sinh embedding đa vector cho người dùng và nội dung. Với MIND, Yada và Yamana [@yada] dùng GPT-4 sinh mô tả chuyên mục, tăng tối đa 5,8% AUC cho NAML/NRMS/NPA; PNR-LLM [@pnrllm] (WWW 2025 Companion) làm giàu bài bằng prompt phân cấp. IRLLRec [@irllrec] (SIGIR 2025) căn chỉnh ý định do LLM xây dựng với ý định rút từ tương tác." });
  add({ t: "p", text: "*Liên hệ với báo cáo:* phiên bản rẻ nhất của hướng này — đưa tên chuyên mục/tiểu mục và tên entity vào văn bản được nhúng, không gọi LLM sinh mô tả — được đo với đối chứng cùng encoder; một phiên bản KAR zero-shot (Qwen3-1,7B mở rộng bài và viết hồ sơ độc giả) được đo trên tập con (Mục 5.4–5.5)." });
  add({ t: "h2", text: "2.4. Hướng 3 — Mô hình hoá người dùng bằng LLM và đa sở thích" });
  add({ t: "p", text: "Ý tưởng biểu diễn người dùng bằng nhiều vector sở thích có từ MIND với dynamic routing [@mindmi] (CIKM 2019) và ComiRec [@comirec] (KDD 2020); trong tin tức, MINER [@miner] dùng poly-attention để trích nhiều vector sở thích và CAUM [@caum] đưa ứng viên vào mô hình hoá người dùng. Với LLM, người dùng có thể được biểu diễn trực tiếp bằng một LLM đọc lịch sử dạng văn bản; Yuan và cs. [@masking] (2026) cho thấy kiểu mặt nạ chú ý (nhân quả, hai chiều, lai) ảnh hưởng tới chất lượng embedding người dùng và đề xuất mở dần chú ý tương lai có khởi động bằng gradient. Ở chiều ngược lại, Iana và cs. [@iana] (SIGIR 2023) chỉ ra rằng thay bộ mã hoá người dùng phức tạp bằng tích vô hướng giữa ứng viên và các bài đã nhấp (late fusion) thường cho kết quả tốt hơn." });
  add({ t: "p", text: "*Liên hệ với báo cáo:* đo (i) late interaction không tham số, (ii) đa sở thích zero-shot bằng k-means và đa sở thích học được, (iii) tower Qwen3 tinh chỉnh LoRA với ba kiểu mặt nạ — kèm đo nhiễu seed để biết chênh lệch nào là kết luận (Mục 5.4–5.6)." });
  add({ t: "h2", text: "2.5. Hướng 4 — Gợi ý sinh và Semantic ID" });
  add({ t: "p", text: "TIGER [@tiger] (NeurIPS 2023) lượng tử hoá embedding nội dung bằng RQ-VAE thành Semantic ID và huấn luyện Transformer sinh ID của item kế tiếp; LETTER [@letter] (CIKM 2024) học tokenizer có điều chuẩn ngữ nghĩa, cộng tác và đa dạng; HSTU [@hstu] (ICML 2024) đưa gợi ý về dạng biến đổi chuỗi sinh với quy luật mở rộng theo tính toán. Năm 2025: ActionPiece [@actionpiece] (ICML 2025) token hoá chuỗi hành động theo ngữ cảnh; RPG [@rpg] (KDD 2025) sinh Semantic ID dài song song trong một lượt; OneRec [@onerec, @onerectr] (Kuaishou) thay chuỗi truy xuất–xếp hạng bằng một mô hình sinh duy nhất có căn chỉnh sở thích và đã triển khai; GRID [@grid] (CIKM 2025) cung cấp khung mã nguồn mở và chỉ ra nhiều thành phần kiến trúc bị bỏ qua có ảnh hưởng lớn. Năm 2026, độ tin cậy của các so sánh bị đặt câu hỏi: đánh giá ở mức Semantic ID có thể thổi phồng kết quả ở mức item do va chạm mã [@sidreliab]." });
  add({ t: "p", text: "*Liên hệ với báo cáo:* MIND là bài toán xếp hạng slate cho sẵn với 82–86% lượt nhấp vào bài mới. Báo cáo thử một phiên bản tối giản (RQ-KMeans 3 tầng × 256 mã + Transformer nhân quả dùng làm bộ xếp hạng slate) để kiểm tra xem tín hiệu “mã kế tiếp” có chuyển sang tin tức không (Mục 5.4)." });
  add({ t: "h2", text: "2.6. Hướng 5 — Suy luận trong gợi ý (2025–2026)" });
  add({ t: "p", text: "ReaRec [@rearec] đưa tính toán lúc suy luận vào gợi ý tuần tự bằng suy luận ẩn nhiều bước; OneRec-Think [@onerecthink] (ACL 2026) xen kẽ suy luận văn bản với token item trong mô hình sinh và đã triển khai; SIDReasoner [@sidreasoner] (KDD 2026) khơi gợi suy luận trên Semantic ID bằng căn chỉnh SID–ngôn ngữ. Tại SIGIR 2026, FLR [@flr] phân rã suy luận ẩn thành nhiều nhân tố, VRec [@vrec] thêm bước kiểm chứng trung gian (reason–verify–recommend), và Chen và cs. [@blalign] căn chỉnh LLM theo danh sách bằng suy luận Bayes. *Liên hệ với báo cáo:* hướng này đòi hỏi LLM lớn và chi phí suy luận cao trên mỗi impression; báo cáo chưa thử và xếp vào các hướng mở có điều kiện (Mục 7.3)." });
  add({ t: "h2", text: "2.7. Hướng 6 — RAG, GraphRAG và tác tử" });
  add({ t: "p", text: "GraphRAG [@graphrag] tóm tắt các cộng đồng của đồ thị tri thức để trả lời câu hỏi tổng quát; K-RagRec [@kragrec] (ACL 2025) truy xuất tri thức có cấu trúc từ đồ thị tri thức cho gợi ý dựa LLM, nhấn mạnh rằng RAG thông thường dễ đưa nhiễu vào; khảo sát tác tử [@agentsurvey] (EMNLP Findings 2025) phân loại hệ tác tử thành định hướng gợi ý, tương tác và mô phỏng. *Liên hệ với báo cáo:* RAG được kiểm chứng ở mức biểu diễn (làm giàu vector bài bằng entity Wikidata hoặc k bài gần nhất trong bộ nhớ nhân quả) và GraphRAG-lite (cộng đồng Louvain, tóm tắt cộng đồng bằng Qwen3-1,7B) (Mục 5.4, 5.7)." });
  add({ t: "h2", text: "2.8. Hướng 7 — LLM làm bộ xếp hạng và giám khảo" });
  add({ t: "p", text: "Hou và cs. [@llmrank] (ECIR 2024) cho thấy LLM có thể xếp hạng zero-shot nhưng nhạy với thứ tự trình bày và thiên lệch phổ biến; LLM4Rerank [@llm4rerank] (WWW 2025) dùng LLM xếp lại tự động theo nhiều tiêu chí. *Liên hệ với báo cáo:* đo một giám khảo Qwen3-1,7B (log-odds Yes/No ở token đầu) xếp lại top-10 của mean-pool, có và không gộp với điểm gốc (Mục 5.4)." });
  add({ t: "h2", text: "2.9. Gợi ý tin tức với PLM và bài toán chi phí tinh chỉnh" });
  add({ t: "p", text: "PLM-NR [@plmnr] (SIGIR 2021) thay bộ mã hoá bài của NRMS bằng PLM và chỉ tinh chỉnh hai lớp cuối; UNBERT [@unbert] (IJCAI 2021) khớp người dùng–bài ở mức từ và mức bài; GLORY [@glory] (RecSys 2023) dùng đồ thị nhấp toàn cục. Chi phí mã hoá bài bằng PLM khi huấn luyện là nút thắt chính: SpeedyFeed [@speedyfeed] (KDD 2022) tái sử dụng kết quả mã hoá trung gian và tăng tốc hơn 100 lần; GRAM [@gram] (NAACL 2022) tích luỹ gradient theo item; Tiny-NewsRec [@tinynewsrec] (EMNLP 2022) chưng cất PLM về mô hình nhỏ. Khảo sát về gợi ý tin tức dùng LLM [@newssurvey] (2025) tổng hợp các hướng này; bộ dữ liệu EB-NeRD [@ebnerd] (RecSys Challenge 2024) có thân bài, điều MIND không có. *Liên hệ với báo cáo:* thí nghiệm v17 dùng công thức PLM-NR (hai lớp cuối) và ý tưởng tái sử dụng mã hoá trung gian của SpeedyFeed (Mục 4.9)." });
  add({ t: "h2", text: "2.10. Độ tin cậy của đánh giá" });
  add({ t: "p", text: "Dacrema và cs. [@dacrema] (RecSys 2019) cho thấy nhiều mô hình nơ-ron công bố không tái lập được hoặc thua baseline đơn giản được tinh chỉnh tốt; Fioretti và cs. [@fioretti] (RecSys 2025) tái lập LLMRec và thấy kết quả thua các baseline đã có; Zhang và cs. [@sidreliab] (2026) chỉ ra va chạm Semantic ID làm thổi phồng kết quả; Iana và cs. [@iana] thống nhất giao thức để so sánh công bằng các mô hình tin tức. Các bài học này định hình giao thức của báo cáo (Mục 4.2): baseline huấn luyện đúng công thức, một tập kiểm thử chạm một lần, khoảng tin cậy ghép cặp và nhiễu seed đo thực." });
  add({ t: "h2", text: "2.11. Tổng hợp và vị trí của nghiên cứu" });
  add({ t: "table", id: "dirs", caption: "Bảy hướng LLM-rec 2024–2026, công trình tiêu biểu, thí nghiệm tương ứng trong báo cáo và kết quả trên MIND.", size: 18,
    widths: [0.15, 0.30, 0.27, 0.28], header: ["Hướng", "Công trình tiêu biểu (venue)", "Thí nghiệm trong báo cáo", "Kết quả đo được"], rows: [
      ["1. LLM làm bộ mã hoá", "LLM2Vec (COLM'24); Qwen3-Emb (2025); LLMEmb (AAAI'25); LLM2Rec (KDD'25); LEARN (AAAI'25); PAD (SIGIR'25); Zhao và cs. (ECIR'25)", "Quét encoder; vector nối; reader học được (H10); pooling (H8)", "Tổng hợp lịch sử và trường dữ liệu quan trọng hơn kích thước encoder; BGE-large > Qwen3-0.6B khi làm đầu vào reader"],
      ["2. LLM sinh tri thức", "KAR (RecSys'24); ONCE (WSDM'24); EmbSum (RecSys'24); IRLLRec (SIGIR'25); Yada & Yamana (2024); PNR-LLM (WWW'25 Comp.)", "Chuyên mục/entity trong văn bản (F); KAR zero-shot (H6); lịch sử dạng văn bản (H1)", "Chuyên mục +0,0182 với Qwen3; KAR mở rộng bài +0,0079 (n = 2.500, mất khi có pop); hồ sơ do LLM viết −0,0308"],
      ["3. Người dùng & đa sở thích", "MIND (CIKM'19); ComiRec (KDD'20); MINER (ACL-F'22); CAUM (SIGIR'22); Iana và cs. (SIGIR'23); Yuan và cs. (2026)", "Tower Qwen3 LoRA 3 mặt nạ (H5); k-means & reader K sở thích (H11)", "Khác biệt giữa mặt nạ không tái lập; đa sở thích âm hoặc ≈ 0"],
      ["4. Gợi ý sinh & SID", "TIGER (NeurIPS'23); LETTER (CIKM'24); HSTU (ICML'24); ActionPiece (ICML'25); RPG (KDD'25); OneRec (2025); GRID (CIKM'25)", "RQ-KMeans 3×256; SID-GPT xếp hạng slate; sid_profile (H7)", "SID-GPT ≈ ngẫu nhiên (0,2898 so với 0,2897); profile +0,0035"],
      ["5. Suy luận", "ReaRec (2025); OneRec-Think (ACL'26); SIDReasoner (KDD'26); FLR, VRec, Bayesian list-wise (SIGIR'26)", "Chưa thử (chi phí suy luận)", "—"],
      ["6. RAG / GraphRAG / tác tử", "GraphRAG (2024); K-RagRec (ACL'25); khảo sát tác tử (EMNLP-F'25)", "RAG biểu diễn (H3); GraphRAG-lite, tóm tắt cộng đồng bằng LLM (H4/H4b)", "≤ +0,0043 trên MIND-small; đổi dấu validation → test trên MIND-large"],
      ["7. LLM xếp hạng", "LLMRank (ECIR'24); LLM4Rerank (WWW'25)", "Giám khảo Qwen3-1,7B xếp lại top-10 (H9)", "−0,0112 [−0,0216; −0,0008] (n = 2.000)"],
    ] });

  // ===== 3
  add({ t: "h1", text: "3. DỮ LIỆU VÀ CÁC KHÁM PHÁ VỀ DỮ LIỆU THỰC TẾ" });
  add({ t: "h2", text: "3.1. Tập dữ liệu MIND và cách chia" });
  add({ t: "p", text: "MIND [@mind] được xây từ nhật ký nhấp của Microsoft News trong sáu tuần với khoảng một triệu người dùng. Mỗi bài có chuyên mục, tiểu mục, tiêu đề, abstract, URL và các entity Wikidata của tiêu đề/abstract; mỗi impression có lịch sử nhấp của người dùng và danh sách ứng viên kèm nhãn. Báo cáo dùng giao thức V10: ngày cuối của tập huấn luyện chính thức làm validation, các ngày trước đó làm `train_core`, tập dev chính thức làm tập kiểm thử. Bảng {T:data} tổng hợp các số đo trích từ log của các lần chạy." });
  add({ t: "table", id: "data", caption: "Thống kê dữ liệu (đo từ log các lần chạy; “—” là không đo).", widths: [0.40, 0.30, 0.30], align: ["l", "r", "r"], size: 20,
    header: ["Chỉ tiêu", "MIND-small", "MIND-large"], rows: [
      ["Số bài (sau khi gộp train + dev)", n(S.articles), n(L.articles_merged)],
      ["Impression của tập huấn luyện chính thức", "156.965", n(L.train_impressions)],
      ["  trong đó train_core / validation (ngày cuối)", "126.695 / " + n(S.val_impressions), n(L.train_core) + " / " + n(L.validation)],
      ["Validation thực dùng", n(S.val_impressions), n(L.val_used_v14) + " (v14); " + n(L.val_used_v15) + " (từ v15)"],
      ["Impression kiểm thử (dev chính thức)", n(S.test_impressions), n(L.test)],
      ["Mẫu huấn luyện (click + negative)", "—", n(L.train_samples) + " (dùng 0,6 tr. ở v14; 1,2 tr. từ v15)"],
      ["Chuyên mục / tiểu mục", S.categories + " / " + S.subcategories, "—"],
      ["Độ dài tiêu đề / abstract (từ)", String(S.title_words).replace(".", ",") + " / " + String(S.abstract_words).replace(".", ",") + " (" + String(S.abstract_empty_pct).replace(".", ",") + "% rỗng)", "—"],
      ["Bài có tên entity", String(S.entity_pct).replace(".", ",") + "% (" + String(S.entities_per_article).replace(".", ",") + " entity/bài)", "—"],
      ["Số lượt nhấp lịch sử của người đọc test", "TB " + String(S.test_reader_clicks_mean).replace(".", ",") + "; " + String(S.test_readers_over50_pct).replace(".", ",") + "% có > 50; tối đa " + S.test_reader_clicks_max, "—"],
      ["Lượt nhấp kiểm thử rơi vào bài lạnh", pct(S.cold_click_share_test_pct), pct(L.cold_click_share_test_pct) + " (validation: " + pct(L.cold_click_share_val_pct) + ")"],
      ["Thân bài", "không có", "không có"],
    ] });
  add({ t: "h2", text: "3.2. Khám phá 1: bài lạnh chiếm ưu thế ở tập kiểm thử" });
  add({ t: "p", text: "Bài lạnh được định nghĩa là bài được nhấp trong tập đánh giá nhưng không xuất hiện trong bất kỳ dòng huấn luyện nào. Tỉ lệ lượt nhấp rơi vào bài lạnh là " + pct(S.cold_click_share_test_pct) + " trên MIND-small và " + pct(L.cold_click_share_test_pct) + " trên MIND-large (Hình {H:cold}). Hệ quả: biểu diễn nội dung là bắt buộc; mọi tín hiệu chỉ dựa ID chỉ phục vụ phần nhỏ còn lại. Đồng thời, trên lát ấm các baseline huấn luyện bằng nhấp vượt xa embedding zero-shot (AUC ấm 0,56–0,64 so với 0,44–0,54 trên MIND-small): chúng học thuộc nội dung bài đã thấy. Đáng chú ý, AUC ấm của embedding tiêu đề + abstract đóng băng thấp hơn 0,5 (BGE-small 0,4416): với bài đã xuất hiện, độ giống ngữ nghĩa với lịch sử có xu hướng xếp ngược với hành vi nhấp." });
  add({ t: "fig", id: "cold", file: "fig02_cold_share.png", width: 5.8, caption: "Tỉ lệ lượt nhấp rơi vào bài chưa xuất hiện khi huấn luyện: tập kiểm thử của cả hai bản MIND lạnh hơn hẳn validation của MIND-large." });
  add({ t: "h2", text: "3.3. Khám phá 2: validation “ấm” đánh lừa việc chọn mô hình" });
  add({ t: "p", text: "Trên MIND-large chỉ " + pct(L.cold_click_share_val_pct) + " lượt nhấp validation là bài lạnh so với " + pct(L.cold_click_share_test_pct) + " ở test. Bảng {T:valtest} và Hình {H:valtest} so lợi ích của từng thành phần so với mean-pool đóng băng (`frozen`) trên hai tập (v14): các bộ chấm zero-shot giữ gần nguyên (giảm 0,005–0,009), các bộ đọc học được mất 0,035–0,040, còn RAG kNN và GraphRAG **đổi dấu**. Hệ quả trực tiếp: trọng số ghép khớp trên validation ưu ái các thành phần “ấm” — ở v14, thêm ba reader và lịch sử 100 lượt vào ghép tuyến tính làm test giảm −0,0014 [−0,0017; −0,0011] dù validation tăng (0,4152 → 0,4177). Ngoài ra mọi hàng đều “dễ” hơn trên test (kể cả vector ngẫu nhiên, +0,020), nên chỉ so được chênh lệch so với đối chứng, không so được mức tuyệt đối giữa hai tập." });
  add({ t: "table", id: "valtest", caption: "Lợi ích nDCG@10 so với `frozen` trên validation và trên test (MIND-large, v14).", widths: [0.34, 0.22, 0.22, 0.22], align: ["l", "r", "r", "c"], size: 20,
    header: ["Thành phần", "validation − frozen", "test − frozen", "Đổi dấu?"], rows: F.large_v14_val_test.map((r) => [ "`" + r.row + "`", s4(r.val), s4(r.test), (r.val > 0) !== (r.test > 0) ? "**có**" : "không"]),
    highlight: F.large_v14_val_test.map((r, i) => ((r.val > 0) !== (r.test > 0) ? i : -1)).filter((i) => i >= 0) });
  add({ t: "fig", id: "valtest", file: "fig03_val_vs_test_v14.png", width: 6.0, caption: "Validation “ấm” phóng đại lợi ích của bộ đọc học được và đảo dấu RAG kNN / GraphRAG (MIND-large, v14)." });
  add({ t: "p", text: "Biện pháp áp dụng từ v16: chọn nền và trọng số của phép ghép theo nDCG@10 có trọng số bài lạnh — mỗi impression validation được gán trọng số sao cho lượt nhấp lạnh chiếm 80%, gần với 82,2% của test (Mục 4.7). Lưu ý: tỉ lệ 80% được đặt khi đã biết phân phối lạnh/ấm của ngày test (thông tin về phân phối, không phải nhãn)." });
  add({ t: "h2", text: "3.4. Khám phá 3: popularity rất mạnh nhưng là phản hồi streaming" });
  add({ t: "p", text: "Popularity online của V10 — đếm nhấp của các impression trước đó trong chính ngày kiểm thử — đạt nDCG@10 = 0,4026 trên MIND-small, cao hơn mọi mô hình nơ-ron của bộ đo gốc, và 0,4040 trên MIND-large. Ghép nó với embedding đóng băng (công thức `bge_zs_pop` của V10) đạt 0,4330. Đây là thiết lập hợp lệ cho hệ thống thật nhưng đọc nhãn của chính ngày test, nên không so được với số công bố offline và không nộp được lên leaderboard. Báo cáo vì vậy luôn tách hai khung: **chỉ nội dung** (so được với văn liệu) và **+pop** (streaming). Phụ lục B liệt kê các hàng +pop trên MIND-small." });
  add({ t: "h2", text: "3.5. Khám phá 4: trường văn bản quan trọng hơn tưởng" });
  add({ t: "p", text: "Kiểm kê dữ liệu cho thấy MIND không có thân bài; từ phiên bản đầu, vector bài đã nhúng “tiêu đề. abstract”. Ba loại thông tin chưa từng đưa vào encoder được thử ở v13: chuyên mục/tiểu mục, tên entity và lịch sử dài hơn 50 lượt nhấp. Kết quả (chi tiết ở Bảng {T:textfields}): bỏ abstract làm giảm 0,016–0,032; chuyên mục giúp Qwen3 rõ rệt (+0,0182, validation cùng chiều +0,0129) nhưng không giúp BGE-small (validation ngược dấu) và BGE-large (−0,0030 trên MIND-large); tên entity nhỉnh nhỏ nhưng nhất quán (+0,0033); lịch sử 100/200 lượt ≈ 0 vì chỉ " + String(S.test_readers_over50_pct).replace(".", ",") + "% người đọc test có hơn 50 lượt. Lợi ích của chuyên mục với Qwen3 nằm ở lát ấm (AUC ấm 0,4939 → 0,6180), lát lạnh gần như không đổi (+0,0005 [−0,0007; +0,0017])." });
  add({ t: "h2", text: "3.6. Khám phá 5: quy mô, bộ nhớ và chi phí" });
  add({ t: "p", text: "MIND-large ở cấu hình của báo cáo (vector 1.024 chiều cho 104.151 bài, 1,2 triệu mẫu huấn luyện, bảng đặc trưng " + F.compute.v16b_stack_rows + " cho stacker, " + F.compute.test_rows + " dòng test) cần RAM đỉnh " + String(F.compute.v16b_peak_ram_gb).replace(".", ",") + " GB (đo thật ở v16b; giới hạn container " + F.compute.container_gb + " GB; GPU đỉnh 5,0 GB). Con số “16,2 GB” của v15 chỉ là ảnh chụp RSS cuối mỗi giai đoạn, không phải đỉnh — và chính sai lệch đo lường này dẫn tới sự cố: lần chạy v16 chết sau 1,6 giờ vì phép gather của view kNN cấp phát cùng lúc hai mảng [104.151, 20, d] float32 (" + F.compute.oom_bytes + " ở d = 2.048, vector nối). Tính theo khối ≈ 600 MB cho kết quả giống hệt (sai khác < 10⁻⁶) với đỉnh tạm chỉ +0,46 GB. Hệ quả thực tế: Colab miễn phí (~12,7 GB RAM) không đủ cho MIND-large ở quy mô này; cần ≥ 24 GB." });
  add({ t: "box", title: "Tóm tắt các khám phá về dữ liệu", text: [
    "(1) 82–86% lượt nhấp kiểm thử rơi vào bài lạnh — biểu diễn nội dung là bắt buộc, còn lợi thế của mô hình huấn luyện bằng nhấp nằm chủ yếu ở bài đã thấy.",
    "(2) Validation theo ngày cuối của train ấm hơn test (18,1% so với 82,2% nhấp lạnh): nó phóng đại bộ đọc học được và đảo dấu RAG/GraphRAG; việc chọn mô hình phải tính đến điều này.",
    "(3) Popularity online là tín hiệu rất mạnh nhưng là phản hồi streaming; phải báo cáo tách khung.",
    "(4) Trường dữ liệu (abstract, chuyên mục) tạo khác biệt cỡ 0,02–0,03 — lớn hơn hầu hết các kỹ thuật LLM phức tạp.",
    "(5) Ở MIND-large, RAM là ràng buộc chính; phải đo đỉnh thật trong từng giai đoạn chứ không phải ảnh chụp cuối giai đoạn.",
  ] });

  // ===== 4
  add({ t: "h1", text: "4. PHƯƠNG PHÁP" });
  add({ t: "h2", text: "4.1. Bài toán" });
  add({ t: "p", text: "Mỗi impression gồm người dùng u với lịch sử nhấp H = (h₁, …, h_L) và danh sách ứng viên đã hiển thị C = (c₁, …, c_J), mỗi ứng viên có nhãn y ∈ {0, 1}. Mô hình gán điểm s(u, c); AUC, MRR, nDCG@5 và nDCG@10 được tính trên từng impression rồi lấy trung bình theo đúng bộ đo của MIND [@mind]. Ký hiệu L = |H| là độ dài lịch sử và vₓ ∈ ℝᵈ là vector chuẩn hoá L2 của bài x do encoder sinh ra." });
  add({ t: "h2", text: "4.2. Giao thức đánh giá" });
  add({ t: "bullets", items: [
    "**Chia theo thời gian, kiểm thử một lần.** Mọi lựa chọn (siêu tham số, nhiệt độ τ, epoch, trọng số ghép, cấu hình stacker) chỉ dùng validation; mỗi cấu hình đã chọn chạm tập kiểm thử đúng một lần.",
    "**Khoảng tin cậy ghép cặp.** Với hai hệ A, B trên cùng tập N impression, Δᵢ = mᵢ(A) − mᵢ(B); khoảng tin cậy 95% theo công thức ({E:ci}), sai số chuẩn ước lượng theo cụm impression (các lượt nhấp cùng impression tương quan). Dòng chạy trên tập con luôn được so với đối chứng trên đúng tập con đó. Khi log không in CI ghép cặp giữa hai hàng, báo cáo dùng “CI chặn” (cận ngoài từ hai CI so với cùng đối chứng) và ghi rõ.",
    "**Lát lạnh/ấm** cho AUC; **Bonferroni** theo số họ khi kết luận về các họ kỹ thuật trên MIND-small.",
    "**Nhiễu huấn luyện** đo bằng cùng công thức, khác seed (Mục 5.3); chênh lệch nhỏ hơn mức này giữa hai mô hình đã huấn luyện không được coi là kết luận.",
    "**Đối chiếu tái lập.** Mean-pool BGE-small phải khớp V10 (AUC 0,6241 / nDCG@10 0,3909 — khớp); popularity và `frozen+pop` khớp; rung 0 của thang ablation tái hiện NRMS của V10 (chép trọng số: lệch điểm ≤ 3·10⁻⁸); vector ngẫu nhiên cho AUC 0,50.",
    "**Dự đoán ghi trước khi chạy** cho mỗi phiên bản (Mục 5.12) để tránh tự thuyết phục sau khi thấy số.",
    "**Chống rò nhãn** bằng kiểm thử tự động: xáo nhãn test trong từng impression → điểm validation và test của `stack_llm`/`stack_meta` giống hệt từng bit; điểm out-of-fold của một khối không đổi khi chỉ lật nhãn của khối đó; hai đột biến cố ý (đưa nhãn test vào bảng; đưa phần giữ lại vào phần huấn luyện) đều bị bắt.",
  ] });
  add({ t: "eq", id: "ci", math: "ci", text: "Δ̄ = (1/N) Σᵢ [mᵢ(A) − mᵢ(B)],   CI₉₅ = Δ̄ ± 1,96 · SE_cụm(Δ̄)" });
  add({ t: "h2", text: "4.3. Bộ mã hoá" });
  add({ t: "p", text: "Các encoder đóng băng: BGE-small-en-v1.5 (384 chiều), BGE-base, BGE-large-en-v1.5 (335 triệu tham số, 1.024 chiều; pooling CLS) [@cpack] và Qwen3-Embedding-0.6B (1.024 chiều; pooling token cuối) [@qwen3emb]; vector nối = [v_Qwen/√2 ; v_BGE/√2], nên tích vô hướng bằng trung bình hai tích vô hướng. Văn bản mặc định là “tiêu đề. abstract”, cắt 256 token. Trước khi chạy, encoder được so với Sentence-Transformers trên 64 bài (cosine nhỏ nhất 0,9998 với Qwen3) để bắt lỗi pooling/EOS/tiền tố." });
  add({ t: "h2", text: "4.4. Bộ chấm zero-shot" });
  add({ t: "p", text: "Mean-pool là đối chứng của V10; late interaction log-sum-exp (LSE) [@colbert] nội suy giữa trung bình (τ → ∞) và MaxSim (τ → 0), τ chọn trên validation; top-k lấy trung bình k độ giống lớn nhất; đa sở thích zero-shot gom lịch sử thành K cụm bằng k-means (tâm μₖ, tỉ lệ πₖ) — K = 1 đúng bằng mean-pool, K ≥ L đúng bằng LSE (đã kiểm bằng test):" });
  add({ t: "eq", math: "mean", text: "s_mean(u, c) = ⟨ (1/|H|) Σ_{h∈H} v_h , v_c ⟩" });
  add({ t: "eq", id: "lse", math: "lse", text: "s_LSE(u, c) = τ · [ log Σ_{h∈H} exp(⟨v_h, v_c⟩ / τ) − log |H| ]" });
  add({ t: "eq", math: "km", text: "s_KM(u, c) = τ · log Σ_{k=1..K} π_k · exp(⟨μ_k, v_c⟩ / τ)" });
  add({ t: "p", text: "Biến thể văn bản: `t` (chỉ tiêu đề), `cta` (thêm câu mẫu “Category: c. Subcategory: s.”), `ctae` (thêm tối đa 8 tên entity), `views` (mã hoá tiêu đề/abstract/chuyên mục riêng rồi ghép có trọng số); lịch sử 100/200 lượt nhấp (`hist100`, `hist200`)." });
  add({ t: "h2", text: "4.5. Bộ đọc lịch sử học được (reader)" });
  add({ t: "p", text: "Reader là bản sao của `llmenc_ca` trong V10 (đã kiểm: điểm giống hệt mã V10): đầu g(·) hai lớp Linear → GELU → LayerNorm → Dropout(0,2) → Linear với độ rộng m = 64 (v7–v15) hoặc 128 (v16b) trên bảng vector bài đóng băng, và chú ý theo ứng viên:" });
  add({ t: "eq", math: "reader", text: "α_{c,i} = softmax_i( ⟨W·g(v_c), g(v_{h_i})⟩ / √m ),   s(u, c) = ⟨ Σᵢ α_{c,i} · g(v_{h_i}) , g(v_c) ⟩" });
  add({ t: "p", text: "Các view của bảng vector: `frozen` (tiêu đề + abstract), `text` (văn bản có chuyên mục), `knn_rag` (vector làm giàu bằng k bài gần nhất), `entity_rag`. Biến thể: reader không nhìn ứng viên kiểu Fastformer/NRMS/additive (`um_ff`, `um_nrms`, `um_add`) để cô lập vai trò của mô-đun chú ý; reader K sở thích (`mi1` đối chứng, `mi4`, `mi4d` có phạt bất đồng); reader `prior` chỉ nhìn bài ứng viên. Huấn luyện theo giao thức V10 (4 negative mỗi click, chọn epoch theo validation), 2–3 seed." });
  add({ t: "h2", text: "4.6. Các hướng LLM khác đã cài đặt" });
  add({ t: "table", caption: "Mã giả thuyết và cách cài đặt từng hướng (MIND-small, v7/v13).", widths: [0.07, 0.19, 0.50, 0.24], size: 18,
    header: ["Mã", "Hướng", "Cài đặt trong báo cáo", "Dòng văn liệu"], rows: [
      ["H1", "Lịch sử dạng văn bản", "10 tiêu đề gần nhất ghép thành một truy vấn có chỉ dẫn, nhúng bằng cùng encoder, ± gộp với mean-pool", "Qwen3-Emb. [@qwen3emb]"],
      ["H2", "Thích nghi co-click", "Head dư (tuyến tính/MLP), LoRA/DoRA trong encoder, Matryoshka, hard negative theo impression; chọn epoch theo validation", "LoRA [@lora]; DoRA [@dora]; LLM2Rec [@llm2rec]"],
      ["H3", "RAG mức biểu diễn", "Làm giàu vector bài bằng ngữ cảnh entity Wikidata hoặc k bài cũ gần nhất (bộ nhớ nhân quả, không dùng nhãn)", "K-RagRec [@kragrec]"],
      ["H4/H4b", "GraphRAG-lite", "Cộng đồng Louvain trên đồ thị kNN bài; tóm tắt cộng đồng bằng Qwen3-1,7B thay tâm cụm", "GraphRAG [@graphrag]"],
      ["H5", "Tower người dùng LLM", "Qwen3 + LoRA, InfoNCE với bài được nhấp kế tiếp; mặt nạ nhân quả / hai chiều / mềm (λ: 0→1); n = 30.000 impression", "[@masking]; LLM2Vec [@llm2vec]"],
      ["H6", "KAR zero-shot", "Qwen3-1,7B mở rộng bài (chủ đề, độc giả) và viết hồ sơ độc giả từ 10 tiêu đề; trộn E₀ + α·E_gen; n = 2.500", "KAR [@kar]; ONCE [@once]"],
      ["H7", "Semantic ID", "RQ-KMeans 3 tầng × 256; `sid_profile` (trùng tiền tố mã); SID-GPT 4 tầng xếp hạng slate bằng log P(mã ứng viên | lịch sử), ± PMI", "TIGER [@tiger]; OneRec [@onerectr]"],
      ["H8", "Late interaction", "LSE, top-k, MaxSim, trọng số recency", "ColBERT [@colbert]"],
      ["H9", "LLM giám khảo", "Qwen3-1,7B (tắt thinking), log-odds Yes/No, xếp lại top-10; n = 2.000", "[@llmrank, @llm4rerank]"],
      ["H10", "Reader học được", "Mục 4.5 trên các view frozen / head / LoRA / RAG", "V10; [@fastformer, @nrms]"],
      ["H11", "Đa sở thích", "k-means zero-shot K = 2, 3, 5; reader K query", "[@mindmi, @comirec, @miner]"],
      ["F", "Thêm dữ liệu", "Chuyên mục, entity, views, lịch sử 100/200", "NAML [@naml]; [@yada]"],
      ["R", "Baseline chuẩn", "NRMS/NAML/Fastformer: GloVe-300, 16 head × 16, lr 1e-4, negative bốc lại mỗi epoch, mask padding; refit trên train_core + validation", "[@nrms, @naml, @fastformer]"],
    ] });
  add({ t: "h2", text: "4.7. Ghép tuyến tính (supermodel)" });
  add({ t: "p", text: "Mỗi thành phần xuất điểm thô cho từng ứng viên (reader: trung bình z-score các seed). Điểm được chuẩn hoá z trong từng impression rồi cộng có trọng số; nền (thành phần tốt nhất trên validation) có trọng số 1, các trọng số khác chọn trên lưới {0, …, 4} bằng coordinate ascent ba lượt trên validation; test chỉ chạm một lần mỗi bậc. Thang bậc: `super_base` → `super_zs` (+ bộ chấm zero-shot) → `super_llm` (+ reader) → `super_llm+pop`. Từ v16, nền và trọng số được chọn theo nDCG@10 có trọng số bài lạnh 80%." });
  add({ t: "eq", math: "zfuse", text: "z_k(u, c) = (s_k(u, c) − μ_k) / σ_k  (trong impression),   s_super(u, c) = z_nền + Σ_k w_k · z_k" });
  add({ t: "h2", text: "4.8. Bộ xếp hạng GBDT (stacker)" });
  add({ t: "p", text: "Stacker (XGBoost [@xgboost] trên GPU) học cách tin từng thành phần tuỳ tình huống. Bốn nhóm đặc trưng cho mỗi ứng viên:" });
  add({ t: "bullets", items: [
    "**scores** — z-score trong impression của mọi thành phần đã xuất điểm từ các run (Qwen3, BGE-large, vector nối), kèm giá trị thô của vài bộ chấm.",
    "**ctx** — 11 đặc trưng lịch sử/slate tính từ vector đóng băng: độ giống với lượt nhấp gần nhất và ba lượt gần nhất, bài đã nằm trong lịch sử, tỉ lệ chuyên mục/tiểu mục của ứng viên trong lịch sử, trùng chuyên mục với lượt nhấp gần nhất, độ dài lịch sử, cỡ slate, độ giống trung bình/lớn nhất với các ứng viên khác, tỉ lệ cùng chuyên mục trong slate; v16 thêm 4 đặc trưng trùng entity có trọng số IDF.",
    "**meta** (16 cột) — đếm theo bài trên mọi thứ đã có nhãn trước ngày đầu của tập (số lần hiển thị/nhấp, CTR làm trơn, “chưa từng hiển thị”, “lạnh”, CTR của chuyên mục/tiểu mục) và **độ phơi nhiễm không nhãn** trong log impression: số giờ từ lần đầu bài xuất hiện trong một impression trước đó, số impression trước đã hiển thị bài, số lần hiển thị trong 1/6/24 giờ đã trọn, cùng 4 cột z-score trong impression.",
    "**pop** — popularity online của V10 và CTR theo cửa sổ 1/6/24 giờ đã trọn (đọc nhãn của các giờ trước): chỉ dùng cho biến thể streaming.",
  ] });
  add({ t: "p", text: "Ba biến thể: `stack_llm` = scores + ctx; `stack_meta` = + meta (không đọc nhãn, nhưng đọc danh sách ứng viên trong log impression — thông tin mà mô hình chỉ dùng nội dung không có); `stack_pop` = + pop. Stacker chỉ huấn luyện trên impression validation; điểm test là trung bình z-score dự đoán của các mô hình fold. **v15:** 4 fold theo người dùng, dừng sớm trên 1/10 người dùng, mục tiêu chọn trên fold 0 trong {rank:ndcg, rank:pairwise, binary:logistic} (chọn `binary:logistic`), eta 0,06, độ sâu 7, ≤ 800 cây, dừng sớm 50. **v16:** fold là các khối thời gian liền kề của ngày validation (đệm ≤ 30 phút), dừng sớm trên phần gần khối giữ lại nhất; thước đo chọn là nDCG@10 trên các impression có nhấp vào bài **mới** với phần huấn luyện của stacker; năm cấu hình (`v15`, `clean`, `clean_d5`, `clean_d3`, `noprior_d5`; `clean` bỏ cỡ slate và làm thô các cột gần như định danh bài) được so một lần trên holdout tiến (60% đầu huấn luyện, 10% dừng sớm, ~30% đánh giá) và trộn hai cấu hình tốt nhất; công thức v15 được huấn luyện song song làm đối chứng `stack_meta_v15cfg`." });
  add({ t: "fig", file: "fig01_pipeline.png", width: 6.3, caption: "Kiến trúc hệ thống thí nghiệm: encoder đóng băng → các họ thành phần → ghép tuyến tính hoặc stacker GBDT, đánh giá dưới cùng một giao thức." });
  add({ t: "h2", text: "4.9. Thí nghiệm tinh chỉnh có cổng dừng (v17)" });
  add({ t: "p", text: "Sau v16b (Mục 5.9), khoảng trống còn lại so với văn liệu nằm ở mô hình nội dung được tinh chỉnh end-to-end. v17 kiểm định giả thuyết này với chi phí thấp nhất, theo công thức PLM-NR [@plmnr] (chỉ tinh chỉnh hai lớp cuối) và ý tưởng tái sử dụng mã hoá trung gian của SpeedyFeed [@speedyfeed]:" });
  add({ t: "numbers", items: [
    "**Cache** trạng thái ẩn ở lớp 22 của BGE-large cho toàn bộ 104.151 bài ([104.151, 96, 1.024] float16 ≈ 20 GB trên đĩa cục bộ, đọc bằng pread); kiểm tra lúc chạy: đường cache + hai lớp trên phải tái tạo vector của encoder đầy đủ (cosine ≥ 0,99).",
    "**Huấn luyện** hai lớp trên (fp32 chủ, autocast fp16) cùng reader, gradient phía ứng viên; phía lịch sử đọc một bảng vector được làm mới mỗi 600 bước.",
    "**Hai nhánh khớp cặp** từ cùng điểm xuất phát: mở khoá (`ft_reader`) và đối chứng đóng băng (`ft_base`) — cùng số bước, cùng mẫu, cùng thứ tự, cùng lịch validation. Lý do: thử nghiệm đồ chơi cho thấy riêng việc huấn luyện tiếp reader đã tăng validation (đối chứng đóng băng +0,0215 so với nhánh mở khoá +0,0040); một cổng chỉ so với điểm xuất phát sẽ gán nhầm lợi ích cho việc mở khoá.",
    "**Cổng quyết định** (chỉ dùng validation có trọng số bài lạnh): PASS nếu nhánh mở khoá hơn nhánh đóng băng ≥ +0,006 và trung bình thô không giảm quá 0,001; INCONCLUSIVE nếu ≥ +0,002; còn lại FAIL.",
    "**Dự đoán ghi trước:** PASS ~35%, INCONCLUSIVE ~25%, FAIL ~40%; nếu PASS, `ft_reader` nDCG@10 test khoảng 0,418–0,428 (reader BGE-large đóng băng: 0,4152), tác động lên ghép tuyến tính +0,002…+0,006 và lên `stack_meta` +0,000…+0,003.",
  ] });
  add({ t: "h2", text: "4.10. Môi trường và các phiên bản thí nghiệm" });
  add({ t: "p", text: "Mọi lần chạy là một notebook “Run All” trên Kaggle (Tesla T4 16 GB; mã dùng một GPU; RAM 29–34 GB, giới hạn container 32 GB). Mã được kiểm thử trên CPU với dữ liệu đúng định dạng MIND trước mỗi lần chạy (" + F.compute.selftests + " kiểm thử ở v17). Bảng {T:runs} (Phụ lục A) liệt kê các phiên bản." });

  // ===== 5
  add({ t: "h1", text: "5. KẾT QUẢ THỰC NGHIỆM" });
  add({ t: "h2", text: "5.1. MIND-small: baseline huấn luyện đúng so với LLM zero-shot" });
  add({ t: "table", caption: "MIND-small, chỉ nội dung (73.152 impression kiểm thử). Ba hàng đầu là số V10 báo; hai hàng cuối là số công bố (thứ cấp).", widths: MW, align: MA, size: 18,
    header: MH, rows: F.small_main.map(mrow), highlight: [8, 13],
    note: "Nhiễu huấn luyện của một lần chạy là 0,007–0,014 nDCG@10 (Mục 5.3). “TB 2 seed” = trung bình hai seed. Số công bố: Iana và cs. [@iana], trung bình 5 lần chạy." });
  add({ t: "fig", file: "fig05_small_main.png", width: 6.2, caption: "nDCG@10 trên MIND-small: cấu hình zero-shot tốt nhất ngang NRMS huấn luyện lại; công thức V10 cũ thấp hơn số công bố ~0,04." });
  add({ t: "bullets", items: [
    "**Baseline huấn luyện đúng tốt hơn V10 rõ:** NRMS +0,025 (refit +0,039), Fastformer +0,028, NAML +0,005 nDCG@10. NRMS refit (0,6555 / 0,4078) nằm sát số công bố 0,4121 (kém 0,004, trong nhiễu seed).",
    "**Zero-shot BGE-small mean-pool (0,3909) ≈ baseline một seed** (−0,0028 so với NRMS seed 0; không khác biệt với Fastformer) nhưng thua NRMS refit −0,017.",
    "**Late interaction (LSE, không tham số) nâng lên 0,4037**, hơn NRMS/Fastformer một seed nhưng thua NRMS refit −0,0041 [−0,0061; −0,0021].",
    "**Cấu hình zero-shot tốt nhất — Qwen3-Embedding-0.6B + chuyên mục + LSE — đạt 0,4094:** ngang NRMS refit về AUC (0,6556 so với 0,6555), hơn về MRR/nDCG@5 (+0,006), +0,0016 nDCG@10 với CI chặn [−0,0023; +0,0055] chứa 0, và thua về AUC lạnh (0,6637 so với 0,6700). Phát biểu an toàn: “ngang đến nhỉnh hơn mà không cần huấn luyện”, không phải “vượt”.",
    "Reader Qwen3 trên văn bản có chuyên mục (`um_text`) đạt 0,4104 nhưng hai seed lệch nhau 0,0137 — chưa kết luận.",
  ] });
  add({ t: "box", title: "Trả lời RQ1", text: "Trên MIND-small, embedding LLM/encoder hiện đại dùng zero-shot **ngang** baseline nơ-ron được huấn luyện đúng (không cần huấn luyện, không cần ID), nhỉnh hơn ở bài lạnh với một số cấu hình và thua ở bài ấm. Không có bằng chứng cho phát biểu “LLM embedding vượt NRMS/NAML/Fastformer”; kết luận ngược lại chỉ xuất hiện khi so với baseline yếu của bộ đo cũ." });
  add({ t: "h2", text: "5.2. Nguồn gốc khoảng cách giữa bộ đo gốc và số công bố" });
  add({ t: "table", caption: "Thang ablation NRMS trên MIND-small (mỗi rung thêm một thay đổi; Δ so với rung trước).", widths: [0.52, 0.12, 0.12, 0.24], align: ["l", "r", "r", "r"], size: 19,
    header: ["Rung", "AUC", "nDCG@10", "Δ nDCG@10 [CI 95%]"], rows: F.small_ladder.map((r) => [r.rung, f4(r.auc), f4(r.ndcg10), r.delta == null ? "—" : s4(r.delta) + " " + ci(r.ci)]),
    note: "Rung 2 chưa hội tụ (epoch tốt nhất là epoch 10, validation còn tăng), nên +0,049 của GloVe có phần bị phóng đại; kết luận an toàn là “vector từ huấn luyện sẵn là yếu tố lớn nhất”. Rung cuối so với `nrms_ref`." });
  add({ t: "p", text: "Yếu tố quyết định là **vector từ huấn luyện sẵn (GloVe [@glove])**, không phải tokenizer, mask hay cách bốc negative (sửa chúng không giúp, −0,003). Huấn luyện thêm các ngày sát ngày test (refit) cho +0,014 — tin tức trôi theo thời gian, nên các bài báo huấn luyện trên toàn bộ train có lợi thế này. Bài học phương pháp: một baseline “yếu” vì thiếu một thành phần chuẩn đủ tạo ra khoảng cách 0,04 — lớn hơn mọi hiệu ứng LLM đo được trong báo cáo." });
  add({ t: "h2", text: "5.3. Nhiễu huấn luyện (seed)" });
  add({ t: "table", caption: "Chênh lệch nDCG@10 giữa hai seed của cùng một công thức.", widths: [0.55, 0.15, 0.15, 0.15], align: ["l", "r", "r", "r"], size: 20,
    header: ["Cặp", "Seed A", "Seed B", "Chênh"], rows: F.small_seed_noise.map((r) => [r.pair, f4(r.a), f4(r.b), f4(r.gap)]) });
  add({ t: "p", text: "Nhiễu huấn luyện 0,007–0,014 nDCG@10 lớn hơn nhiều so với độ rộng của CI ghép cặp (±0,002, chỉ phản ánh mẫu kiểm thử) và so với độ lệch thường được trích trong văn liệu (±0,002–0,003). Hệ quả: chênh lệch dưới ~0,007 giữa hai mô hình **đã huấn luyện** một seed không phải kết luận; trung bình ba seed còn nhiễu ~0,004. Các hàng zero-shot không có nhiễu này." });
  add({ t: "h2", text: "5.4. Các hướng LLM trên MIND-small" });
  add({ t: "fig", id: "dirs", file: "fig04_small_directions.png", width: 6.3, caption: "Hiệu ứng của từng hướng so với đối chứng của chính nó (nDCG@10, MIND-small). Vùng xám: dải nhiễu seed ±0,007." });
  add({ t: "p", text: "Hình {H:dirs} tổng hợp hiệu ứng của từng hướng so với đối chứng của chính nó. Ba nhóm rõ rệt:" });
  add({ t: "bullets", items: [
    "**Có hiệu ứng dương rõ, xác định, tái lập được:** late interaction (LSE +0,0128; top-k +0,0107; tái lập trên Qwen3: +0,0127) và chuyên mục trong văn bản với Qwen3 (+0,0182 [+0,0169; +0,0196]). KAR mở rộng bài +0,0079 [+0,0021; +0,0137] nhưng chỉ trên 2.500 impression và mất khác biệt khi có popularity (+0,0036).",
    "**Xấp xỉ 0:** GraphRAG cộng đồng (+0,0043, AUC lạnh −0,0031), Semantic ID profile (+0,0035), LoRA/DoRA co-click (+0,0016…+0,0029, AUC lạnh giảm; head MLP −0,005), tên entity (+0,0033), lịch sử dài (+0,0012).",
    "**Âm:** RAG biểu diễn (−0,0080…−0,0003), lịch sử dạng văn bản (−0,0211…−0,0014), hồ sơ độc giả do LLM viết (−0,0308), giám khảo Qwen3-1,7B (−0,0112 [−0,0216; −0,0008]; gộp với điểm gốc −0,0051, không khác biệt), đa sở thích k-means (−0,0030 so với LSE), và reader học được so với LSE (−0,0080 [−0,0099; −0,0061]).",
    "**SID-GPT dùng làm bộ xếp hạng slate ngang ngẫu nhiên:** nDCG@10 = " + f4(F.small_sid_gpt.sid_gpt) + " so với vector ngẫu nhiên " + f4(F.small_sid_gpt.random_vec) + " — tín hiệu “mã kế tiếp” không chuyển được sang xếp hạng tin tức khi 86% lượt nhấp là bài mới (với cài đặt tối giản: tokenizer không học, mô hình 4 tầng).",
    "**Quét encoder (mean-pool, so với BGE-small):** TF-IDF + LSA −0,0443, BGE-base −0,0185, Qwen3-Embedding-0.6B −0,0125, BGE-large +0,0015 (không khác biệt) — encoder lớn hơn/LLM **không** giúp mean-pool zero-shot.",
    "**Mô-đun chú ý không phải yếu tố quyết định:** reader Fastformer trên vector BGE đóng băng −0,0008 (không khác biệt), reader NRMS −0,0095 so với reader chú ý theo ứng viên.",
  ] });
  add({ t: "box", title: "Trả lời RQ2 và RQ3 (MIND-small)", text: [
    "**RQ2:** nút thắt của hệ embedding đóng băng nằm ở **cách tổng hợp lịch sử** (LSE +0,013, không tham số) và **trường dữ liệu đưa vào encoder** (abstract ~0,02–0,03; chuyên mục +0,018 với Qwen3), không ở kích thước encoder.",
    "**RQ3:** trong các hướng LLM-rec đã thử, chỉ phiên bản rẻ của “LLM làm giàu dữ liệu” (chuyên mục trong văn bản) cho lợi ích rõ; RAG/GraphRAG mức biểu diễn, Semantic ID sinh, giám khảo LLM nhỏ, đa sở thích, LoRA và tower LLM cho hiệu ứng ≤ 0,005, âm, hoặc không tái lập được.",
  ] });
  add({ t: "h2", text: "5.5. Trường văn bản và đa sở thích (v13)" });
  add({ t: "table", id: "textfields", caption: "Biến thể văn bản và lịch sử (MIND-small; đối chứng: LSE cùng encoder với 50 lượt nhấp).", widths: [0.10, 0.34, 0.11, 0.11, 0.11, 0.23], align: ["l", "l", "r", "r", "r", "r"], size: 18,
    header: ["Encoder", "Biến thể", "val", "test", "Δ val", "Δ test [CI 95%]"], rows: F.small_text_fields.map((r) => [r.enc, r.row, f4(r.val), f4(r.test), s4(r.dval), s4(r.dtest) + " " + ci(r.ci)]) });
  add({ t: "p", text: "Với Qwen3, chuyên mục cho lợi ích cùng chiều trên cả validation và test; với BGE, validation ngược dấu ⇒ không có bằng chứng ổn định. Hướng thử tiếp theo được ghi nhận: đối chứng “chuyên mục ngẫu nhiên cùng định dạng” để tách thông tin chủ đề khỏi hiệu ứng định dạng chuỗi (đã cài, chưa chạy)." });
  add({ t: "table", caption: "Đa sở thích: reader một vector so với K = 4 sở thích (nDCG@10 test, trung bình seed; cos = độ giống trung bình giữa các vector sở thích).", widths: [0.12, 0.46, 0.14, 0.12, 0.16], align: ["l", "l", "r", "r", "r"], size: 19,
    header: ["Encoder", "Reader", "nDCG@10", "std", "cos sở thích"], rows: F.small_multi_interest.map((r) => [r.enc, r.row, f4(r.nd), f4(r.std), r.cos || "—"]) });
  add({ t: "p", text: "Đa sở thích zero-shot luôn nằm giữa mean-pool và LSE và đơn điệu theo K (K = 2/3/5: −0,0079/−0,0048/−0,0030 so với LSE) — một kết quả âm sạch vì LSE chính là trường hợp K lớn. Reader K sở thích không phạt cho các vector gần trùng nhau (cos 0,85–0,91: thực chất một sở thích); có phạt thì tách ra (cos 0,34–0,52) nhưng không cải thiện xếp hạng ngoài nhiễu seed." });
  add({ t: "h2", text: "5.6. Tower người dùng LLM và mặt nạ chú ý" });
  add({ t: "table", caption: "Tower Qwen3 + LoRA (n = 30.000 impression); “+meanpool” gộp với mean-pool.", widths: [0.46, 0.27, 0.27], align: ["l", "r", "r"], size: 20,
    header: ["Mặt nạ", "nDCG@10", "+ meanpool"], rows: F.small_tower.map((r) => [r.mode, f4(r.nd), f4(r.nd_mp)]) });
  add({ t: "p", text: "Chênh lệch “mềm − nhân quả” là +0,0020 [+0,0003; +0,0036] ở v7 nhưng −0,0063 [−0,0080; −0,0046] khi chạy lại ở v13 với cùng mã, cùng seed; cùng công thức khác seed lệch tới 0,0124. Kết luận: dữ liệu không chứng minh được mặt nạ có ảnh hưởng trong bài toán này; tower Qwen3 + meanpool chỉ ngang LSE của BGE (−0,0007 [−0,0035; +0,0022]) và thua NRMS refit. Biến thể “mềm” ở đây chỉ là phần lịch trình tuyến tính của Yuan và cs. [@masking]; phần khởi động bằng gradient chưa được cài." });
  add({ t: "h2", text: "5.7. MIND-large: ghép tuyến tính (v14)" });
  add({ t: "table", caption: "MIND-large, v14 (376.471 impression kiểm thử; encoder Qwen3-Embedding-0.6B).", widths: MW, align: MA, size: 18, header: MH, rows: F.large_v14.map(mrow), highlight: [6] });
  add({ t: "p", text: "Thang ghép: `super_base` → `super_zs` +0,0166 [+0,0161; +0,0171] (thêm văn bản có chuyên mục, trọng số 0,5); `super_zs` → `super_llm` −0,0014 (thêm reader làm giảm test dù validation tăng — Mục 3.3); tower LoRA không giúp (−0,0002). Lợi ích của phép ghép đến từ việc kết hợp **một** reader với **một** bộ chấm zero-shot khác loại; các thành phần thêm vào không giúp. Popularity cộng thêm +0,0279 (`super_llm+pop` 0,7048 / 0,4542, streaming)." });
  add({ t: "h2", text: "5.8. MIND-large: bộ xếp hạng GBDT (v15)" });
  add({ t: "table", caption: "MIND-large, v15. `stack_meta` đọc độ phơi nhiễm trong log impression (không nhãn); ba hàng cuối là streaming.", widths: MW, align: MA, size: 18, header: MH, rows: F.large_v15.map(mrow), highlight: [4, 5] });
  add({ t: "table", caption: "So sánh ghép cặp (v15; Δ nDCG@10 [CI 95%], Δ AUC, Δ AUC lạnh).", widths: [0.36, 0.30, 0.17, 0.17], align: ["l", "r", "r", "r"], size: 19,
    header: ["A − B", "Δ nDCG@10 [CI]", "Δ AUC", "Δ AUC lạnh"], rows: F.large_v15_pairs.map((r) => [r.pair, s4(r.d) + " " + ci(r.ci), s4(r.dauc), s4(r.dcold)]) });
  add({ t: "bullets", items: [
    "`stack_llm` **hơn ghép tuyến tính +0,0064 nhưng không ở bài lạnh** (AUC lạnh y hệt: −0,0001): phần tăng nằm ở bài đã thấy (AUC ấm +0,033). Giả thuyết “GBDT sửa được điểm yếu bài lạnh” không được ủng hộ cho biến thể chỉ dùng nội dung.",
    "**Đặc trưng phơi nhiễm không nhãn tạo bước nhảy lớn nhất** (`stack_meta` +0,0176 so với `stack_llm`; AUC lạnh +0,0277), đổi lại AUC ấm giảm 0,035: mô hình đánh đổi bài cũ lấy bài mới — có lợi vì 82% lượt nhấp là bài lạnh.",
    "**BGE-large tốt hơn Qwen3 ở mọi bộ chấm zero-shot trên văn bản thường** (+0,012…+0,015 nDCG@10, Bảng {T:enc}), nhưng chuyên mục trong văn bản chỉ giúp Qwen3 (+0,0186) — với BGE-large là −0,0030.",
    "**Ước lượng out-of-fold của stacker lạc quan** hơn test 0,03–0,04 nDCG@10 (`stack_llm` 0,4664 → 0,4332; `stack_meta` 0,4909 → 0,4508), trong khi ghép tuyến tính có test ≥ validation — giả thuyết: fold theo người dùng cùng một ngày cho phép cây học mức phổ biến trong ngày của từng bài, thông tin không có ở ngày test. Giả thuyết này được kiểm ở v16b.",
  ] });
  add({ t: "table", id: "enc", caption: "Zero-shot trên MIND-large: BGE-large so với Qwen3-Embedding-0.6B (nDCG@10 / AUC test).", widths: [0.34, 0.22, 0.22, 0.22], align: ["l", "r", "r", "r"], size: 19,
    header: ["Bộ chấm", "BGE-large", "Qwen3-0.6B", "BGE − Qwen3"], rows: F.large_v15_encoders.map((r) => [r.row, f4(r.bge[0]) + " / " + f4(r.bge[1]), f4(r.qwen[0]) + " / " + f4(r.qwen[1]), s4(r.bge[0] - r.qwen[0])]) });
  add({ t: "h2", text: "5.9. MIND-large: bộ đọc theo encoder và kiểm định chéo trung thực (v16b)" });
  add({ t: "p", text: "v16b dùng cùng đầu 128 chiều cho cả ba encoder để tách hiệu ứng encoder khỏi độ rộng đầu. Kết quả (Bảng {T:readers}, Hình {H:readers}): thứ tự **vector nối > BGE-large > Qwen3** đúng ở cả hai seed của `um_frozen`; BGE-large hơn Qwen3 +0,0102 [+0,0095; +0,0108], vector nối hơn BGE-large +0,0071 [+0,0066; +0,0076]. Reader BGE-large hơn zero-shot tốt nhất của chính nó +0,0105 — ngược với MIND-small, nơi LSE không tham số hơn reader: nhiều dữ liệu huấn luyện hơn làm reader có ích. Với câu hỏi “cần embedding lớn hơn?”: encoder LLM 0,6B **kém** encoder 335M khi làm đầu vào cho reader, và ghép hai không gian vector tốt hơn mỗi cái — độ đa dạng của biểu diễn có ích hơn kích thước." });
  add({ t: "table", id: "readers", caption: "Reader theo encoder (MIND-large, v16b; nDCG@10 test, trung bình 2 seed, trong ngoặc là từng seed).", widths: [0.22, 0.26, 0.26, 0.26], align: ["l", "r", "r", "r"], size: 19,
    header: ["Reader", "Qwen3-0.6B", "BGE-large", "Nối Qwen3 ⊕ BGE"], rows: ["um_frozen", "um_knn_rag", "um_text"].map((k) => {
      const R = F.large_v16b_readers[k];
      const c = (e) => (R[e] ? `${f4(R[e][0])} (${f4(R[e][1])} / ${f4(R[e][2])})` : "—");
      return ["`" + k + "`", c("qwen3"), c("bge"), c("cat")];
    }) });
  add({ t: "fig", id: "readers", file: "fig07_readers_by_encoder.png", width: 5.9, caption: "Reader trên ba encoder (MIND-large, v16b): vạch đen là khoảng giữa hai seed." });
  add({ t: "table", caption: "Kết quả chính v16b so với v15 (MIND-large; AUC / nDCG@10 / AUC lạnh).", widths: [0.40, 0.12, 0.12, 0.12, 0.24], align: ["l", "r", "r", "r", "r"], size: 19,
    header: ["Hàng", "AUC", "nDCG@10", "AUC lạnh", "v15 (AUC / nDCG@10)"], rows: F.large_v16b.map((r) => [r.label, f4(r.auc), f4(r.ndcg10), f4(r.cold_auc), r.v15 ? f4(r.v15[0]) + " / " + f4(r.v15[1]) : "—"]), highlight: [0, 2] });
  add({ t: "table", caption: "So sánh ghép cặp v16b (Δ nDCG@10 [CI 95%]).", widths: [0.55, 0.45], align: ["l", "r"], size: 20,
    header: ["A − B", "Δ nDCG@10 [CI]"], rows: F.large_v16b_pairs.map((r) => [r.pair, s4(r.d) + " " + ci(r.ci)]) });
  add({ t: "p", text: "**Stacker đã bão hoà.** Thành phần nội dung tốt hơn rõ (`super_llm` +0,0105 so với v15) nhưng gần như không chuyển lên hàng đầu của stacker: `stack_meta` chỉ +0,0008 so với v15, và `stack_llm` không hơn nổi ghép tuyến tính (−0,0004 [−0,0010; +0,0002]). Khi có đặc trưng phơi nhiễm, GBDT quy về cùng mức ~0,45 dù điểm nội dung là 0,4268 hay 0,4373." });
  add({ t: "table", caption: "Kiểm định giả thuyết “stacker v15 học thuộc ngày validation” (nDCG@10).", widths: [0.40, 0.15, 0.15, 0.15, 0.15], align: ["l", "r", "r", "r", "r"], size: 19,
    header: ["Hàng (cách chia fold)", "Validation OOF", "Val. bài mới", "Test", "Test − Val."], rows: F.large_v16b_cv.map((r) => [r.row, f4(r.val), f4(r.val_new), f4(r.test), s4(r.test - r.val)]), highlight: [0] });
  add({ t: "fig", file: "fig08_stacker_cv.png", width: 6.2, caption: "Fold theo người dùng thổi phồng ước lượng 0,038; fold theo khối thời gian và lát “bài mới” cho ước lượng trung thực (lệch 0,007)." });
  add({ t: "p", text: "Kết luận ba phần: (i) giả thuyết “v15 tự đánh lừa mình” **đúng** — ước lượng của công thức v15 lệch 0,038 so với test, còn fold theo khối thời gian chỉ lệch 0,007; (ii) giả thuyết “sửa nó thì test tốt hơn” **sai** — công thức mới thua đối chứng −0,0027 [−0,0031; −0,0023]: cái hỏng là *ước lượng*, không phải mô hình; (iii) lợi ích thực của v16 là một con số validation đáng tin. Bỏ cỡ slate và làm thô các cột gần như định danh bài làm *giảm* điểm holdout (−0,014): cây dùng cỡ slate như ngữ cảnh để đổi cách trọng số các đặc trưng khác (suy đoán, chưa có ablation từng cột)." });
  add({ t: "box", title: "Trả lời RQ4", text: "Ở MIND-large, ghép tuyến tính đơn giản (chọn theo validation có trọng số bài lạnh) đã đạt mức của GBDT khi chỉ dùng nội dung + lịch sử; GBDT chỉ có giá trị khi được cấp đặc trưng mới (phơi nhiễm không nhãn, popularity streaming) và bão hoà ở ~0,45 nDCG@10. Ước lượng validation chỉ đáng tin khi fold theo khối thời gian và được đánh giá trên lát bài mới; fold theo người dùng thổi phồng 0,038." });
  add({ t: "h2", text: "5.10. Phân tích đặc trưng của stacker" });
  add({ t: "fig", file: "fig10_gain_shares.png", width: 6.0, caption: "Tỉ trọng split gain của XGBoost theo nhóm đặc trưng (v15 và v16b)." });
  add({ t: "p", text: "Nhóm điểm thành phần LLM chiếm 0,47–0,66 tổng gain; nhóm ngữ cảnh 0,22–0,40; nhóm meta 0,08–0,21; popularity 0,22–0,24 ở biến thể streaming. Lưu ý khi diễn giải: nhóm ngữ cảnh gồm cả đặc trưng tính từ vector LLM (độ giống với lịch sử, độ giống trong slate) lẫn đặc trưng không liên quan LLM — riêng cỡ slate `ctx_n_cand` chiếm 0,10–0,21. Vì vậy không thể cộng nhóm ngữ cảnh vào “phần đóng góp của LLM”. Các đặc trưng có gain lớn nhất ở v16b: `cat:um_knn_rag` (0,09–0,13), `ctx_n_cand`, `bge:um_frozen`, `ctx_slate_max`, `m_ctr`, và ở biến thể streaming `pop`, `popw_ctr_1h`." });
  add({ t: "h2", text: "5.11. So sánh với kết quả công bố" });
  add({ t: "table", caption: "Số công bố trên MIND-large (thứ cấp, đo trên tập test của leaderboard) và các hàng của báo cáo (đo trên MINDlarge_dev).", widths: [0.44, 0.14, 0.14, 0.14, 0.14], align: ["l", "r", "r", "r", "r"], size: 19,
    header: ["Hệ thống", "AUC", "MRR", "nDCG@5", "nDCG@10"], rows: [
      ...F.published_large.map((p) => [p.label, f4(p.auc), f4(p.mrr), f4(p.ndcg5), f4(p.ndcg10)]),
      ["Báo cáo: `super_llm` v16b (nội dung + lịch sử)", "0,6863", "—", "—", "0,4373"],
      ["Báo cáo: `stack_llm` v15 (nội dung + lịch sử)", "0,6843", "0,3355", "0,3730", "0,4332"],
      ["Báo cáo: `stack_meta` v16b (+ phơi nhiễm log)", "0,7013", "—", "—", "0,4516"],
      ["Báo cáo: `stack_pop` v16b (streaming)", "0,7196", "—", "—", "0,4731"],
    ], highlight: [4],
    note: "Số công bố lấy từ kết quả tìm kiếm (chưa mở PDF): NRMS từ bài báo MIND [@mind], PLM-NR [@plmnr], UNBERT [@unbert]; hàng đầu bảng là ensemble Fastformer + PLM-NR. Log v16b chỉ in AUC, nDCG@10 và AUC lạnh. Tập đo khác nhau (dev so với test của leaderboard): chênh ~0,01 AUC không phải một phát biểu về mô hình." });
  add({ t: "fig", file: "fig09_published.png", width: 6.3, caption: "Vị trí của hệ thống so với số công bố. Gạch chéo: dùng thông tin mà các mô hình công bố không dùng." });
  add({ t: "p", text: "Đọc đúng bảng này: (i) hàng **chỉ dùng nội dung và lịch sử** (`super_llm`, `stack_llm`: AUC 0,684–0,686) nằm trong dải NRMS của văn liệu (AUC 0,670–0,691 tuỳ bài báo) và thấp hơn PLM-NR/UNBERT khoảng 0,02 AUC; về nDCG@10, 0,4373 thấp hơn PLM-NR 0,0065. (ii) `stack_meta` (0,7013 / 0,4516) ngang PLM-NR về AUC và cao hơn về nDCG@10, nhưng đọc độ phơi nhiễm trong log impression — thông tin PLM-NR/UNBERT không dùng — nên **không cùng bài toán**. (iii) `stack_pop` đọc nhãn của các giờ trước trong ngày test nên không nộp được lên leaderboard. Khoảng trống còn lại so với văn liệu nằm ở phía **mô hình nội dung được tinh chỉnh end-to-end**, không ở tầng ghép." });
  add({ t: "h2", text: "5.12. Đối chiếu dự đoán ghi trước với kết quả" });
  add({ t: "table", caption: "Các dự đoán ghi trước khi chạy (chủ quan) và số đo.", widths: [0.50, 0.30, 0.20], size: 19,
    header: ["Dự đoán", "Số đo", "Kết quả"], rows: [
      ["v13: chuyên mục là ứng viên tốt nhất, |Δ| ≲ 0,01", "Qwen3 +0,0182; BGE ≈ 0", "đúng hướng, sai độ lớn"],
      ["v13: k-means nằm giữa mean-pool và LSE, có thể ≥ LSE", "giữa hai mức, luôn < LSE", "đúng nửa đầu"],
      ["v15: `stack_llm` > `super_llm`, Δ trong +0,003…+0,02", "+0,0064", "đúng"],
      ["v15: AUC lạnh của stacker > supermodel", "−0,0001 (`stack_llm`)", "sai"],
      ["v15: `stack_meta` > `stack_llm` (~60%)", "+0,0176", "đúng"],
      ["v16b: `stack_meta` AUC 0,698–0,712; nDCG@10 0,449–0,464", "0,7013; 0,4516", "trong khoảng"],
      ["v16b: `stack_llm` tăng ≥ +0,01 so với v15 (~15–20%)", "+0,0037", "sai"],
      ["v16b: khoảng test − val. của `stack_meta` nhỏ hơn đối chứng (~65%)", "−0,0072 so với −0,0379", "đúng"],
      ["v16b (ngầm định): `stack_meta` > `stack_meta_v15cfg`", "−0,0027", "sai"],
      ["v17: PASS ~35% / INCONCLUSIVE ~25% / FAIL ~40%", "chưa chạy", "—"],
    ] });

  // ===== 6
  add({ t: "h1", text: "6. THẢO LUẬN" });
  add({ t: "h2", text: "6.1. Trả lời các câu hỏi nghiên cứu" });
  add({ t: "bullets", items: [
    "**RQ1.** Embedding hiện đại dùng zero-shot ngang baseline nơ-ron huấn luyện đúng trên MIND-small (0,4094 so với 0,4078; khác biệt không có ý nghĩa), không vượt.",
    "**RQ2.** Nút thắt là cách tổng hợp lịch sử và trường dữ liệu; kích thước encoder không phải nút thắt (MIND-small: BGE-large ≈ BGE-small; MIND-large: BGE-large 335M > Qwen3 0,6B làm đầu vào reader; ghép hai encoder tốt nhất).",
    "**RQ3.** Phần lớn hướng LLM-rec 2024–2026 đã thử không giữ được lợi ích trên tin tức dưới giao thức nghiêm ngặt; ngoại lệ là đưa chuyên mục vào văn bản (với Qwen3) và late interaction. Hai lỗi đánh giá dễ gặp — validation ấm và baseline yếu — đủ tạo ra các “kết quả dương” giả.",
    "**RQ4.** Ghép tuyến tính chọn theo validation có trọng số bài lạnh đã ngang GBDT khi chỉ dùng nội dung; GBDT có giá trị khi có đặc trưng phơi nhiễm không nhãn và bão hoà ~0,45; cần CV theo khối thời gian để có ước lượng trung thực.",
    "**RQ5.** Khoảng cách ~0,02 AUC so với PLM-NR nằm ở mô hình nội dung tinh chỉnh end-to-end; v17 kiểm định điều này với cổng dừng (chờ kết quả).",
  ] });
  add({ t: "h2", text: "6.2. Các khám phá về mô hình" });
  add({ t: "numbers", items: [
    "**Late interaction thắng bộ mã hoá người dùng học được khi dữ liệu ít** (MIND-small: LSE +0,0080 so với reader), nhất quán với Iana và cs. [@iana]; khi dữ liệu nhiều (MIND-large) reader vượt zero-shot +0,01.",
    "**Encoder lớn hơn không tự động tốt hơn** — nhất quán với Zhao và cs. [@zhao]; độ đa dạng của không gian biểu diễn (ghép hai encoder) có giá trị hơn kích thước.",
    "**Encoder phản ứng khác nhau với cùng một thay đổi đầu vào:** chuyên mục giúp Qwen3 (decoder-only, có chỉ dẫn) nhưng không giúp BGE (BERT) — kết luận về “dùng thêm trường dữ liệu” phụ thuộc encoder.",
    "**Lợi ích của mô hình học được nằm chủ yếu ở bài ấm** (AUC ấm), trong khi 82–86% lượt nhấp là bài lạnh — nên lợi ích trên validation (ấm) không chuyển sang test.",
    "**Các tăng cường ở mức biểu diễn (RAG, GraphRAG) làm mờ tính đặc thù chủ đề** của bài mới: AUC lạnh giảm (GraphRAG −0,0181, kNN −0,0147 trên MIND-large).",
    "**Semantic ID sinh tối giản không chuyển sang xếp hạng slate tin tức** (≈ ngẫu nhiên); các kết quả mạnh của hướng này chủ yếu ở bài toán tuần tự với lịch sử dày và cần tokenizer học được [@letter, @grid].",
    "**LLM nhỏ (1,7B) làm giám khảo zero-shot và làm hồ sơ độc giả đều có hại** so với embedding đơn giản.",
    "**Stacker GBDT bão hoà và dễ tự đánh lừa:** chênh lệch ước lượng do cách chia fold (0,038) lớn hơn mọi cải tiến cấu hình (≤ 0,003).",
  ] });
  add({ t: "h2", text: "6.3. Đối chiếu với xu hướng LLM-rec 2025–2026" });
  add({ t: "p", text: "Kết quả của báo cáo không phủ nhận các hướng 2025–2026 mà khoanh vùng điều kiện để chúng có ích trên tin tức. (i) **LLM làm encoder** [@llmemb, @llm2rec, @learn, @pad]: các công trình này đều kèm một bước *thích nghi với hành vi* (tinh chỉnh tương phản, căn chỉnh, tách rời); số đo của báo cáo cho thấy dùng embedding đóng băng thuần chỉ ngang baseline — phù hợp với việc bước thích nghi là cần thiết, và là lý do của v17. (ii) **LLM sinh tri thức** [@kar, @once, @yada]: phiên bản rẻ (chuyên mục trong văn bản) có ích với Qwen3; KAR zero-shot với LLM 1,7B có lợi nhỏ ở mở rộng bài nhưng có hại ở hồ sơ độc giả — chất lượng LLM sinh có thể là yếu tố quyết định. (iii) **Gợi ý sinh/SID** [@tiger, @onerec, @rpg]: trên tin tức với slate cho sẵn và bài mới chiếm ưu thế, cần tokenizer học được và đánh giá ở mức item [@sidreliab] trước khi kết luận. (iv) **Suy luận** [@onerecthink, @flr, @vrec]: chưa thử; với chi phí suy luận trên mỗi impression, hướng hợp lý cho tin tức là dùng suy luận ở tầng xếp lại một danh sách ngắn. (v) **Độ tin cậy đánh giá** [@dacrema, @fioretti]: báo cáo cung cấp thêm bằng chứng rằng baseline yếu và validation ấm đủ tạo ra cải thiện giả." });
  add({ t: "h2", text: "6.4. Hàm ý thực tiễn" });
  add({ t: "bullets", items: [
    "Một hệ gợi ý tin tức rẻ và mạnh: vector bài tính một lần khi xuất bản bằng encoder đóng băng (văn bản có chuyên mục), tổng hợp lịch sử bằng LSE, ghép với reader nhẹ và các đặc trưng phơi nhiễm không nhãn trong một bộ xếp hạng cây. *Độ trễ phục vụ chưa được đo trong báo cáo.*",
    "Chọn mô hình theo thước đo phản ánh phân phối bài lạnh của lúc phục vụ, không theo validation thô.",
    "Báo cáo tách khung “chỉ nội dung” và “streaming”; đo RAM đỉnh thật; dự đoán ghi trước.",
  ] });
  add({ t: "h2", text: "6.5. Hạn chế và mối đe dọa tính hợp lệ" });
  add({ t: "bullets", items: [
    "**Một họ dữ liệu, một ngày kiểm thử:** kết luận có thể không chuyển sang EB-NeRD (có thân bài) hay nền tảng khác.",
    "**Dev thay vì test của leaderboard:** số của báo cáo và số công bố đo trên hai tập khác nhau; số công bố là thứ cấp (chưa đối chiếu PDF).",
    "**Nhiều hàng một seed;** CI ghép cặp không chứa nhiễu huấn luyện (0,007–0,014). XGBoost trên GPU không tất định đến từng bit.",
    "**Không tinh chỉnh siêu tham số** cho bất kỳ mô hình nào ngoài các lưới đã nêu; baseline dùng GloVe-6B, Fastformer được cài lại theo cấu trúc mã chính thức.",
    "**Phạm vi LLM:** LLM sinh/giám khảo chỉ 1,7B; Qwen3-Embedding 4B/8B chưa thử; tower LLM và KAR/giám khảo đo trên tập con (n = 2.000–30.000).",
    "**Lựa chọn có thông tin phân phối test:** mục tiêu 80% bài lạnh của thước đo có trọng số được đặt khi đã biết tỉ lệ 82,2% (thông tin phân phối, không phải nhãn).",
    "`stack_meta` **đọc danh sách ứng viên của log impression** — hợp lệ khi phục vụ thật nhưng không cùng điều kiện thông tin với mô hình chỉ dùng nội dung.",
  ] });

  // ===== 7
  add({ t: "h1", text: "7. HƯỚNG PHÁT TRIỂN" });
  add({ t: "h2", text: "7.1. Thí nghiệm v17 và quy tắc quyết định" });
  add({ t: "p", text: "Notebook v17 đã được cài đặt và kiểm thử trên CPU (bao gồm tính đúng của đường cache, gradient chỉ vào hai lớp trên và reader, hai nhánh cùng số bước, cổng và các lý do từ chối). Chưa đo: tốc độ thật trên T4 và đĩa, số học fp16 trên GPU, và quan trọng nhất là hiệu quả. **Nếu PASS:** mở rộng (nhiều bước và lớp hơn, gradient cả hai phía, đưa vector tinh chỉnh vào vector nối). **Nếu FAIL:** dừng hướng tinh chỉnh, giữ reader đóng băng và viết luận văn trên kết quả v16b. **Nếu INCONCLUSIVE:** thêm seed trước khi mở rộng." });
  add({ t: "h2", text: "7.2. Mở rộng dữ liệu" });
  add({ t: "p", text: "EB-NeRD [@ebnerd] (RecSys Challenge 2024) có thân bài và nhật ký lớn hơn, cho phép kiểm tra (i) hiệu ứng của văn bản dài với encoder LLM, (ii) tính ổn định của các khám phá về dữ liệu (tỉ lệ bài lạnh, validation ấm) trên một nền tảng khác." });
  add({ t: "h2", text: "7.3. Các hướng 2025–2026 đáng thử có điều kiện" });
  add({ t: "bullets", items: [
    "**Mô tả chuyên mục do LLM sinh** [@yada] với đối chứng “chuyên mục ngẫu nhiên cùng định dạng” (đã cài) để tách thông tin chủ đề khỏi hiệu ứng định dạng.",
    "**Thích nghi encoder với hành vi** theo LLM2Rec/LLMEmb [@llm2rec, @llmemb] — tức hướng v17 mở rộng nếu cổng PASS.",
    "**Semantic ID có tokenizer học được** (LETTER, RPG) và đánh giá ở mức item [@letter, @rpg, @sidreliab], dùng làm đặc trưng cho bộ xếp hạng thay vì bộ sinh.",
    "**Suy luận ẩn chi phí thấp** (kiểu ReaRec/FLR) [@rearec, @flr] ở tầng xếp lại danh sách ngắn, đo cả chi phí suy luận.",
  ] });

  // ===== 8
  add({ t: "h1", text: "8. KẾT LUẬN" });
  add({ t: "p", text: "Báo cáo đánh giá có kiểm soát bảy hướng ứng dụng LLM cho hệ gợi ý giai đoạn 2024–2026 trên bài toán gợi ý tin tức MIND, dưới một giao thức theo thời gian với khoảng tin cậy ghép cặp, lát bài lạnh/ấm và nhiễu seed đo thực. Bốn kết luận chính:" });
  add({ t: "numbers", items: [
    "**Dữ liệu quyết định cách đánh giá:** 82–86% lượt nhấp kiểm thử là bài lạnh và validation theo ngày cuối của train ấm hơn hẳn; bỏ qua điều này dẫn tới chọn sai mô hình (RAG/GraphRAG đổi dấu, bộ đọc học được bị thổi phồng).",
    "**Embedding LLM dùng zero-shot ngang — không vượt — baseline nơ-ron huấn luyện đúng;** lợi ích bền vững đến từ cách tổng hợp lịch sử (late interaction) và trường dữ liệu (chuyên mục), không từ kích thước encoder.",
    "**Phần lớn hướng LLM-rec đã thử** (RAG/GraphRAG mức biểu diễn, Semantic ID sinh tối giản, giám khảo LLM nhỏ, đa sở thích, LoRA, tower LLM) không giữ được lợi ích trên tin tức khi đánh giá nghiêm ngặt.",
    "**Ở MIND-large,** ghép tuyến tính chọn đúng cách đạt AUC 0,6863 / nDCG@10 0,4373; bộ xếp hạng GBDT chỉ thêm giá trị nhờ đặc trưng phơi nhiễm không nhãn (0,7013 / 0,4516) và đã bão hoà; khoảng ~0,02 AUC còn lại so với PLM-NR thuộc về tinh chỉnh encoder end-to-end — đối tượng của thí nghiệm v17 có cổng dừng.",
  ] });
  add({ t: "p", text: "Giá trị học thuật chính của báo cáo nằm ở bằng chứng có kiểm soát (kể cả kết quả âm) và ở các khám phá về dữ liệu giải thích vì sao nhiều cải thiện được báo cáo có thể không tồn tại dưới một giao thức nghiêm ngặt." });

  // ------------------------------------------------------------------ appendix
  const appendix = [
    { t: "h1", text: "PHỤ LỤC A. CÁC PHIÊN BẢN THÍ NGHIỆM" },
    { t: "table", id: "runs", caption: "Các phiên bản thí nghiệm, dữ liệu, nội dung và thời gian chạy (Kaggle, Tesla T4).", widths: [0.08, 0.13, 0.49, 0.18, 0.12], size: 18,
      header: ["Bản", "Dữ liệu", "Nội dung chính", "Thời gian / tài nguyên", "Trạng thái"], rows: F.runs.map((r) => [r.ver, r.data, r.what, r.time, r.status]) },
    { t: "h1", text: "PHỤ LỤC B. SỐ LIỆU BỔ SUNG" },
    { t: "table", caption: "MIND-small, khung streaming (+pop; không so được với số công bố).", widths: MW, align: MA, size: 18, header: MH, rows: F.small_pop.map(mrow) },
    { t: "table", caption: "MIND-small, quét encoder (mean-pool zero-shot; Δ nDCG@10 so với BGE-small).", widths: [0.6, 0.4], align: ["l", "r"], size: 20,
      header: ["Encoder", "Δ nDCG@10"], rows: F.small_encoder_sweep.map((r) => [r.enc, s4(r.delta)]) },
    { t: "table", caption: "MIND-large, validation so với test của các hàng v15 (nDCG@10).", widths: [0.4, 0.2, 0.2, 0.2], align: ["l", "r", "r", "r"], size: 20,
      header: ["Hàng", "Validation", "Test", "Test − Val."], rows: F.large_v15_valtest.map((r) => [r.row, f4(r.val), f4(r.test), s4(r.test - r.val)]) },
    { t: "fig", file: "fig06_large_ladder.png", width: 6.0, caption: "Thang kết quả trên MIND-large (nDCG@10; màu thể hiện loại thông tin mà mỗi hàng sử dụng)." },
  ];

  return { title, titleEn, abstractVi, keywordsVi, abstractEn, keywordsEn, front, body, appendix };
};
