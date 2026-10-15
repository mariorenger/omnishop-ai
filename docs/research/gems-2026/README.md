# Slide trình bày bài báo GEMS (Beamer, LaTeX)

**GEMS: Tìm kiếm memetic với giải mã hình học cho giao hàng khẩn cấp bằng xe và UAV**. Nhóm 15, giảng viên hướng dẫn: TS. Đỗ Tuấn Anh.

| Tệp | Nội dung |
|---|---|
| `Slide_GEMS.tex` | Mã nguồn Beamer, khổ 16:9, tiếng Việt. Toàn bộ nằm trong một file, không dùng hình ngoài: mọi hình, biểu đồ và sơ đồ đều vẽ bằng TikZ/pgfplots. |
| `Slide_GEMS.pdf` | Bản đã biên dịch, 122 trang. |

## Biên dịch

```bash
xelatex Slide_GEMS.tex
xelatex Slide_GEMS.tex   # lần 2 để cập nhật liên kết và số trang
```

- Phải dùng **XeLaTeX** (fontspec, font TeX Gyre có đủ dấu tiếng Việt). Trên Overleaf: Menu → Compiler → XeLaTeX.
- Đã kiểm tra với TeX Live 2023. Gói cần có: beamer, fontspec, TeX Gyre (heros, termes, cursor), amsmath, booktabs, tabularx, multirow, tikz, pgfplots, algorithm/algpseudocode, appendixnumberbeamer, microtype.
- Hai lần chạy cho kết quả 0 lỗi và 0 cảnh báo tràn khung (overfull).

## Cấu trúc

**Phần chính** (56 trang, 47 slide nội dung được đánh số, cộng trang bìa, 7 trang chuyển mục và trang cảm ơn):

1. Giới thiệu: bối cảnh, ba khoảng trống, hai đóng góp, tóm tắt kết quả
2. Các nghiên cứu liên quan (2.1–2.4) và bảng định vị GEMS
3. Mô hình bài toán: Hình 1, các phương trình (1)–(2), giả thiết, NP-khó
4. Phương pháp GEMS: Hình 2, các phương trình (3)–(7), Thuật toán 1–2, độ phức tạp
5. Thiết kế thực nghiệm: Bảng 1, đối chứng, phương trình (8), thống kê
6. Kết quả: Bảng 2–3, Hình 3–4, hình KTC và bảng cơ chế sàng lọc, phạm vi áp dụng
7. Kết luận
8. Tài liệu tham khảo: đủ 23 mục như trong bài

**Phụ lục** (66 trang, đánh số riêng): trang danh mục có liên kết, rồi đủ Phụ lục A–T theo đúng ký tự của bài, gồm mọi bảng, hình và phương trình của từng phụ lục. Các phụ lục dài được tách thành nhiều slide.

## Nguyên tắc

- Mọi số liệu, bảng và hình đều lấy nguyên từ bản thảo, không thêm số đo mới. Đã đối chiếu tự động: mọi con số trong slide đều xuất hiện trong bản thảo. Các trường hợp không khớp chỉ là số mục và tọa độ vẽ.
- Ngoài các hình của bài, slide có thêm vài sơ đồ quy trình vẽ lại từ mô tả trong bài: chia ngân sách thời gian, vòng sàng lọc, quản lý quần thể. Hai hình chỉ mang tính minh họa và được ghi rõ là "minh họa" trên slide: ba loại điểm ứng viên ở Mục 4.3 và ví dụ LOX 8 khách ở Phụ lục D. Hai hình này không dùng dữ liệu thực nghiệm.
- Hình KTC và bảng cơ chế sàng lọc thuộc Phụ lục Q của bài được đặt ở Mục 6.5. Slide Phụ lục Q có liên kết tới hai trang đó.
