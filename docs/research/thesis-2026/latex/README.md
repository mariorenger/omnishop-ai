# Slide LaTeX (bản 2): Gợi ý tin tức sử dụng các kỹ thuật LLM

Bản trình bày lại cùng nội dung với deck Beamer của nhóm, sắp theo đúng trình tự nghiên cứu để một người tự đọc cũng theo được:
đặt bài toán, quy trình 4 bước, kết luận của từng hướng H1–H12, vì sao thành công hay thất bại, rồi ghép thành hệ thống.

| Tệp | Nội dung |
|---|---|
| `Slide_LLMRec_v2.tex` | Mã nguồn Beamer 16:9, tiếng Việt, một file. Mọi hình vẽ bằng TikZ, không cần ảnh ngoài. Mỗi slide có lời thoại gợi ý (`\note{}`). |
| `Slide_LLMRec_v2.pdf` | Bản biên dịch với giao diện dự phòng: 52 trang (49 slide đánh số, cộng bìa, mục lục và trang cảm ơn). |
| `Slide_LLMRec_v2_loi_thoai.pdf` | Cùng nội dung; mỗi trang có slide ở nửa trên và lời thoại ở nửa dưới, dùng để tự luyện thuyết trình. |

## Biên dịch

```bash
pdflatex Slide_LLMRec_v2.tex
pdflatex Slide_LLMRec_v2.tex   # lần 2 để cập nhật số trang
```

- pdfLaTeX với `inputenc` utf8 và `babel` `vietnamese`. Trên Overleaf: Menu → Compiler → pdfLaTeX.
- Nếu cùng thư mục có `beamerthemehust.sty`, file tự dùng theme `hust`. Nếu không có, file dùng giao diện dự phòng (thanh tiêu đề đỏ, chân trang có tên đề tài, tác giả và số trang).
- Muốn ra bản có lời thoại: bỏ dấu `%` ở hai dòng `\usepackage{pgfpages}` và `\setbeameroption{show notes on second screen=bottom}` trong phần khai báo.
- Đã kiểm tra với TeX Live 2023: 0 lỗi, 0 cảnh báo tràn khung (overfull), không slide nào phải thu nhỏ với giao diện dự phòng.
- Các khung nội dung có tùy chọn `shrink`. Nếu theme `hust` có thanh tiêu đề hoặc chân trang cao hơn, Beamer tự thu nhỏ nội dung của khung đó vừa đủ, thay vì để chữ tràn xuống chân trang. Đã thử với một theme giả lập có đầu trang và chân trang cao hơn khoảng 0,6 cm: 26 khung được thu nhỏ, nhiều nhất khoảng 10%, và không khung nào tràn.

## Cấu trúc

1. Bài toán và dữ liệu: bài toán, MIND và các chỉ số, năm câu hỏi nghiên cứu.
2. Cách tiếp cận: quy trình 4 bước (dựng baseline, sàng lọc trên MIND-small, xác nhận trên MIND-large, ghép hệ thống), bảng nối văn liệu với giả thuyết, mô hình cơ sở, giao thức đánh giá, hệ thống cuối.
3. Bức tranh tổng thể: ba cột thành công / có điều kiện / không hiệu quả, bảng tra cứu từng hướng, **quy mô dữ liệu mà mỗi hướng LLM đã chạy** (đủ / một phần / tập con), biểu đồ độ lớn tác động, bốn lý do thất bại.
4. Chẩn đoán: RQ1 (thang ablation NRMS; NRMS, NAML, Fastformer huấn luyện lại so với embedding cố định), RQ2 (quét bộ mã hóa).
5. Các hướng thành công: H8 (kèm slide so LSE với attention học được trên cùng vector: kiểu NRMS, Fastformer, 4 truy vấn, theo ứng viên), H10, cùng H3 và F (có điều kiện).
6. Các hướng không thành công: H2 và H12, H5, H11, nhóm sinh hoặc nén bằng LLM (H1, H4, H4b, H6, H7, H9), các hướng chưa đo (SID-News, T).
7. Từ thành phần đến hệ thống: validation lệch, validation có trọng số theo tỷ lệ bài mới, hợp nhất tuyến tính, GBDT, rò rỉ trong ngày và CV theo khối thời gian.
8. Kết quả và so sánh: offline, online, phân rã mức tăng, split gain, so với số công bố, triển khai, giới hạn, kết luận.

Phụ lục: thuật ngữ, câu hỏi có thể gặp (2 trang), các lần chạy và nguồn số liệu, tài liệu tham khảo.

## Kết luận từng hướng

Tiêu chí "thành công": hơn đối chứng mạnh nhất trong cùng điều kiện (không chỉ hơn mean pooling), khoảng tin cậy không chứa 0 và được giữ trong hệ thống cuối.

| Kết luận | Hướng |
|---|---|
| Thành công | H8 (tương tác muộn: LSE, top-k), H10 (reader theo ứng viên: hơn LSE trên MIND-large và với Qwen3) |
| Có điều kiện | F (chuyên mục trong văn bản: giúp Qwen3, không giúp BGE), H3 (kNN: chỉ có ích khi đi qua reader) |
| Không hiệu quả | H1, H2, H4, H4b, H5, H6, H7, H9, H11, H12 (H2, H5, H6, H9, H12 **chưa chạy đủ dữ liệu**: kết luận chỉ đúng trong ngân sách đã thử) |
| Chưa đo | T (cổng cold/warm kiểu PAD), SID-News |

## Nguồn số liệu

- MIND-small: `experiments/recsys_v11/RESULTS_v7.md`, `RESULTS_v13.md`. MIND-large: `RESULTS_v14_large.md`, `RESULTS_v15_large.md`, `RESULTS_v16b_large.md`; các số ghi v18, v19: `RESULTS_v18a_large.md`, `RESULTS_v18b_large.md`, `RESULTS_v19a_large.md`, `RESULTS_v19b_large.md`.
- Mọi bảng MIND-large của hệ thống cuối lấy từ một lần chạy là v16b. Dòng đánh dấu (*) lấy từ v17.
- Một số con số lấy từ bản nháp của nhóm (log chạy của nhóm) và chưa có trong các file `RESULTS` của repo. Cần tự đối chiếu trước khi thuyết trình:
  - bảng H12 của v17 (`ft_base`, `ft_reader`) và dòng `super_llm*` (0.4378);
  - cột MRR, nDCG@5 và Val của các dòng v16b trong bảng kết quả offline và online;
  - `pool_recency`, `pool_maxsim`;
  - khoảng tin cậy của ba bước đầu trong biểu đồ phân rã (LSE, chuyên mục, reader);
  - H4b, con số 41 thành phần và 99 đặc trưng của GBDT, thời gian mã hóa 4–10 ms mỗi bài.

## Các chỗ khác với bản nháp

- Kết luận theo đối chứng mạnh nhất. H2 và H11 chỉ hơn mean pooling, còn H11 thấp hơn LSE, nên chuyển sang "không hiệu quả". H10 giữ "thành công", nhưng mức tăng ghi là +0.0105 so với LSE cùng encoder. Con số +0.0473 trong bản nháp trùng với mức chênh giữa `super_base` (0.4260) và mean pooling Qwen3 (0.3787), tức là gộp cả đổi encoder, kNN và trung bình hai seed. T và SID-News chưa có lần chạy đo nên ghi "chưa đo".
- Khoảng tin cậy là CI ghép cặp theo từng impression, xấp xỉ chuẩn (±1,96·SE), không phải bootstrap.
- Hội nghị của tài liệu tham khảo: EmbSum là RecSys'24, LETTER là CIKM'24, PAD là SIGIR'25, PLM-NR là SIGIR 2021. MRR công bố của PLM-NR là 0.3539, của UNBERT là 0.3568.
- Bỏ số Fastformer 0.4230 và NAML 0.4160 trên MIND-large vì không tìm được nguồn công bố. Số NAML 0.3882 và Fastformer 0.3895 trên MIND-small của bản nháp không có trong các file `RESULTS`; deck dùng số trong log: NAML 0.4006, Fastformer 0.3926 (v7) / 0.3927 (v13, 2 seed), Fastformer refit 0.4033, NRMS 0.3937 / 0.4005 (2 seed), NRMS refit 0.4078.
- So sánh LSE với attention học được trên cùng vector (reader kiểu NRMS, Fastformer, 4 truy vấn, theo ứng viên) giờ có slide riêng ngay sau H8, đủ BGE và Qwen3 trên cả hai quy mô. Bản trước chỉ có cột BGE-small trong slide H10. Với Qwen3 trên MIND-small, reader theo ứng viên đã hơn LSE (+0.0084, cả hai seed); với BGE-small thì chưa. Reader kiểu NAML (một lớp additive attention) trên vector cố định chưa chạy.
- Bỏ câu "riêng việc chọn bằng validation có trọng số đã tăng +0.0214", vì phép so sánh này lấy số từ hai lần chạy khác thành phần nên không tách được tác dụng của trọng số.
- Phân rã mức tăng dùng trọn lần chạy v16b. Kết quả là khoảng 80% đến từ embedding và reader, 20% từ exposure, với bước GBDT + exposure là +0.0143 [+0.0137, +0.0149]. Bản nháp ghi 81% / 19% vì ghép `super_llm` của v17 với `stack_meta` của v16b. Hợp nhất tuyến tính chỉ dùng nội dung và lịch sử được ghi là 0.4373 (`super_zs`, v16b).
- Phần cứng: mã chỉ dùng 1 GPU T4; RAM đỉnh đo được là 21,2 GB (v16b).
- Nhiễu seed đo trên test: 0.007–0.014 nDCG@10 trên MIND-small, 0.001–0.007 với reader trên MIND-large.
- Sửa vài số cho khớp log: `pool_km5` trên MIND-small là 0.4007; H2 tăng +0.0016 đến +0.0029; con số +0.0653 → +0.0302 của v14 là của reader + kNN (reader thường là +0.0596 → +0.0192). Thêm ghi chú rằng bậc 2 của thang NRMS chưa hội tụ.
- Để biên dịch được: `babel` dùng `vietnamese` thay cho `vietnam`; bỏ chữ `LLMREC` thừa sau `\title`.

## Cập nhật 2026-10-19 (sau các lần chạy v18–v19)

Sửa ít, giữ nguyên mạch trình bày và các con số chính của hệ thống (0.4516 offline, 0.4731 online; chờ v19c đo nhiễu của stacker trước khi đổi).
- **Thêm một slide** ngay sau bảng tra cứu: mỗi hướng LLM đã chạy trên bao nhiêu dữ liệu huấn luyện và bao nhiêu impression test, gắn nhãn đủ / một phần / tập con. Mục đích: không kết luận "LLM kém" cho các hướng chưa chạy đủ (H2, H5, H12 mới dùng một phần dữ liệu huấn luyện; H6, H9 mới đo trên 2.000–2.500 impression, trên nền mean pooling).
- Dấu $^{\dagger}$ (chưa chạy đủ dữ liệu) ở slide tổng thể và bảng tra cứu.
- H5: thêm lần chạy v19b (3.5% dữ liệu, chấm toàn bộ test): 0.4002, hơn LSE +0.0085, kém reader −0.0048, không thêm gì cho hệ thống; tiêu đề slide đổi theo.
- H1, H6, H9: thêm số đo trên MIND-large (v19a, v19b) và kích thước tập con.
- RQ2: thêm Qwen3-Embedding-4B (v18a): làm đầu vào reader chỉ ngang 0.6B, kém BGE-large.
- Không gian biểu diễn: reader huấn luyện trên đủ 2.71 triệu mẫu (v18b, v19a): 0.4316 (+0.0091); vào hệ thống chỉ +0.0006 đến +0.0016.
- So với PLM-NR, giới hạn, kết luận RQ2 / RQ5, hướng tiếp theo, một câu hỏi phụ lục và bảng các lần chạy: cập nhật theo các số trên.
