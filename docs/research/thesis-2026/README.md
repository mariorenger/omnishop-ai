# Báo cáo và slide: LLM cho gợi ý tin tức trên MIND (10/2026)

| File | Nội dung |
|---|---|
| `Bao_Cao_Nghien_Cuu_LLM_News_Recommendation.docx` / `.pdf` | Báo cáo nghiên cứu 42 trang (tiếng Việt): 8 chương, 2 phụ lục, 62 tài liệu tham khảo. |
| `Slide_Bao_Cao_LLM_News_Recommendation.pptx` / `.pdf` | 33 slide, sửa thẳng trên file slide nháp của học viên (giữ nền, khung, màu, font). Nội dung tiếng Anh, ghi chú thuyết trình tiếng Việt. |
| `tools/` | Mã dựng lại cả hai file từ số đo. |

## Nguồn số liệu

* Mọi con số đo được lấy từ `experiments/recsys_v11/RESULTS_v7.md`, `RESULTS_v13.md`, `RESULTS_v14_large.md`, `RESULTS_v15_large.md`,
  `RESULTS_v16_partial.md`, `RESULTS_v16b_large.md`. Các dự đoán ghi trước và thiết kế v17 lấy từ `experiments/recsys_v11/README.md`.
  Tất cả được gom vào `tools/gen/facts.py` (báo cáo, biểu đồ và slide đều đọc từ đây).
* v17 (tinh chỉnh hai lớp cuối BGE-large, có cổng dừng) **chưa chạy**: chỉ có thiết kế và dự đoán.
* Số công bố (NRMS, PLM-NR, UNBERT trên MIND-large; NRMS/NAML của Iana và cs. trên MIND-small) lấy từ nguồn thứ cấp và đo trên tập test khác
  (leaderboard so với `MINDlarge_dev`), nên chênh lệch cỡ 0,01 AUC không phải là kết luận về mô hình.
* Venue của tài liệu tham khảo được xác minh bằng tìm kiếm web (10/2026); PDF gốc chưa được mở trong môi trường dựng file.
* Độ trễ phục vụ và chi phí hạ tầng **chưa được đo** nên không có con số nào về chúng.

## Dựng lại

Cần Python (`python-pptx`, `matplotlib`, `pymupdf`), Node với `docx` 9.11 và LibreOffice (để ra PDF và đọc số trang mục lục).

```bash
cd docs/research/thesis-2026/tools
python3 gen/facts.py out_facts.json
python3 gen/figures.py out/fig
node gen/build_report.js out/report.docx                    # lượt 1: chưa có số trang mục lục
soffice --headless --convert-to pdf --outdir out out/report.docx
python3 gen/toc_pages.py out/report.pdf out/report.toc.json out/pages.json
node gen/build_report.js out/report.docx out/pages.json      # lượt 2; lặp hai bước trên đến khi pages.json không đổi
python3 gen/figures_slide.py out/fig_slide_en
python3 gen/edit_deck.py <slide nháp gốc>.pptx out/slides.pptx out/fig_slide_en
```

`edit_deck.py` cần file slide nháp gốc (không lưu trong repo): nó giữ bìa và slide cảm ơn, nhân bản khung slide nội dung của bản nháp
cho từng slide mới, rồi xoá các slide nội dung cũ.
