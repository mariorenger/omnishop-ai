# RAG & xử lý tài liệu (ingestion)

## Luồng xử lý bất đồng bộ (async)

Khi tenant tải tệp lên (`POST /api/knowledge/upload`):

1. API **lưu tệp thô** vào `file_asset` và tạo `document` trạng thái `queued`,
   rồi trả về **ngay lập tức**. Không trích xuất trong request.
2. Một job `ingest_document` được đẩy vào hàng đợi Valkey.
3. **Worker** (tiến trình riêng, `app/worker.py`) lấy job và chạy toàn bộ phần
   nặng: `extract → chunk → embed`, cập nhật trạng thái
   `queued → processing → ready | error`.

Văn bản (dán tay) đi đường `pending → embed_document → ready` (chunk ngay vì nhỏ).

### Vì sao tách async
Trích xuất PDF/OCR/Office tốn CPU và có thể lâu. Nếu làm trong request, nhiều
tenant upload cùng lúc sẽ chiếm hết worker của web và gây timeout. Tách sang
worker giúp **API luôn phản hồi nhanh**, người dùng thấy trạng thái tiến triển.

## Nhiều tenant cùng upload thì sao?

- Hàng đợi là một Valkey list, worker dùng `BRPOP` (FIFO). Mỗi job mang theo
  `organization_id`, mọi thao tác DB đều chạy qua `tenant_tx(org_id)` nên **cô
  lập tenant** được giữ nguyên (RLS).
- **Mở rộng ngang:** chạy **N worker** cùng lúc — tất cả `BRPOP` trên cùng một
  list, Redis giao mỗi job cho đúng một worker → xử lý song song, không trùng.
  Đây là cách scale khi tải tăng (chỉ cần tăng số tiến trình/worker).
- **Chống nghẽn một tenant:** vì mỗi tài liệu là một job riêng, tenant tải 100
  tệp không "khoá" tenant khác quá một job; với nhiều worker, các tenant được
  phục vụ xen kẽ. (Nâng cao sau: hàng đợi theo trọng số/round-robin theo tenant.)
- **Lỗi không làm chết vòng lặp:** worker bắt exception từng job, ghi
  `document.status='error'` + thông điệp lỗi để tenant xem và bấm **Xử lý lại**;
  vòng lặp worker vẫn tiếp tục.

## Định dạng tệp hỗ trợ

Văn bản (`.txt .md .csv .tsv .json .html`), tài liệu (`.pdf .docx .pptx .xlsx`),
hình ảnh (`.png .jpg .webp .gif .bmp .tiff`). PDF scan và ảnh đi qua
**OCRProvider** (Tesseract mặc định, có thể đổi sang VLM). Tối đa 25MB/tệp.

## Tenant quản lý tài liệu

Trên trang **Kiến thức**, mỗi tài liệu hiển thị **trạng thái**, số **ký tự** đã
trích xuất, số **đoạn (chunk)**. Bấm vào một tài liệu để:
- Xem **văn bản đã trích xuất** thực tế (kiểm tra chất lượng OCR/parse).
- Xem lỗi nếu có.
- **Xử lý lại** (re-extract + re-embed) hoặc **Xoá**.
Đổi tên **Kho kiến thức** ngay tại tiêu đề.

## Truy hồi (retrieval)

Câu hỏi của khách được nhúng vector rồi tìm `chunk` gần nhất (pgvector), lọc
theo `shop_id` và `bot_id` (tài liệu gán cho một trợ lý hoặc dùng chung). Kết
hợp với sản phẩm liên quan, đưa vào ngữ cảnh cho LLM — LLM không truy vấn DB.
