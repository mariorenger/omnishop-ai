# Tự phản biện & rà soát trước khi đóng gói

Rà toàn hệ thống: cái gì đã quản lý được, cái gì vừa bổ sung, và giới hạn còn
lại (nói thẳng, không tô hồng).

## Đã kiểm tra & còn thiếu → đã bổ sung ở đợt này

| Hạng mục | Trước | Sau |
|---|---|---|
| Kiểm thử tự động | chỉ 1 script isolation | **30 test pytest** cho từng nhóm chức năng, chạy xanh |
| Admin làm chủ **giá gói** | seed cứng trong DB | **Sửa trực tiếp trên UI** (tên, giá, token, vượt mức, PAYG, trần tin nhắn) |
| Admin làm chủ **đơn giá token (COGS)** | chỉ biến môi trường | **Sửa trên UI**, có cache + fallback env |
| RAG | vector-only | **hybrid** (vector + từ khoá, RRF) |
| Trích xuất file khi tải | đồng bộ trong request | **bất đồng bộ ở worker**, không nghẽn khi nhiều tenant |
| Tenant xem file | chỉ trạng thái thô | xem **text trích xuất**, ký tự, đoạn, lỗi, xử lý lại, xoá |
| Branding | cứng | **logo + tên** đổi bằng config (admin) |

## Ai làm chủ config nào (đã xác minh bằng test)

- **Admin nền tảng** (một lần, mọi tenant thấy): LLM/embedding/OCR mặc định &
  chính sách, **cổng thanh toán nhận tiền**, Facebook App, **giá gói & hạn mức
  token**, **đơn giá token**, thương hiệu (logo/tên). Test chặn tenant gọi các
  API này (401/403).
- **Tenant/shop** (của riêng họ): khoá AI riêng (BYOK), OCR, kênh
  (FB/IG/Telegram/Zalo/WhatsApp/Shopee/TikTok), bot + prompt + avatar, sản phẩm,
  kiến thức + tên kho, thành viên, chọn gói. Cô lập bằng RLS (test cross-tenant).

## Nhóm test (tất cả xanh — 30 test)

- Unit (không cần DB): chữ ký VNPay/MoMo + VietQR URL; parse webhook
  Telegram/WhatsApp/Zalo + chữ ký Meta; trích xuất txt/csv/json/html + chunk;
  RRF fusion.
- API/tích hợp (Postgres + Redis): auth, **cô lập RLS**, tải file → ingest →
  xem/ xoá/ đổi tên kho, **RAG hybrid** (mã SKU khớp từ khoá, câu hỏi tự nhiên
  khớp ngữ nghĩa), gói & hạn mức token theo chế độ, branding công khai/riêng,
  admin sửa giá gói & đơn giá.

## Chạy tự động 100%

- `docker compose up` chạy: Postgres(+pgvector), Redis, **api** (tự chạy
  migration 002→009 lúc khởi động), **worker** (xử lý ingest/embedding), **web**.
- Không có bước thủ công. Lỗi được **cô lập thành trạng thái** (tài liệu `error`
  + "Xử lý lại"; kênh `degraded` + "Kiểm tra"; thanh toán trả về lỗi rõ ràng),
  vòng lặp worker không chết vì một job hỏng.
- Chạy test: `pytest` (bỏ qua nhóm cần DB nếu DB không bật).

## Giới hạn còn lại (thành thật)

1. **TikTok Shop / Shopee**: code đúng chuẩn field + ký, nhưng **live cần app
   được duyệt** + shop uỷ quyền — không thể bỏ qua bước duyệt của họ.
2. **Làm mới token** (Zalo ~25h, Shopee 4h): hiện lưu token do tenant cấp; chưa
   có cron tự refresh. Nên thêm worker refresh định kỳ khi lên production.
3. **Xuất hoá đơn overage/PAYG tự động**: đã đo token và tính chi phí theo
   tenant & theo khách; chưa tự sinh hoá đơn cuối kỳ đẩy sang cổng — là bước
   nối tiếp rõ ràng.
4. **Quy mô worker**: mặc định một worker (FIFO). Tải cao thì chạy N worker
   (đã hỗ trợ, cùng BRPOP một hàng đợi) — xem `docs/rag-and-ingestion.md`.
5. **"Không bao giờ lỗi" tuyệt đối** là không khả thi với phụ thuộc ngoài (Meta/
   ngân hàng/khoá sai). Mục tiêu đã đạt: lỗi hiển thị thành trạng thái để tự xử
   lý, không phải đọc log/debug.

## UI

Rà các màn: đăng nhập, tổng quan, hộp thư, sản phẩm (bảng dày), kiến thức (+chi
tiết), kênh, thành viên, thanh toán (gói/token/khách), cài đặt, quản trị
(branding/gói/đơn giá/thanh toán/Facebook/LLM), trợ lý (chat full trái). Font
Plus Jakarta Sans tự host, gradient pastel, tương phản cao, tooltip (i) cho chú
thích. Không còn màn "bo trong div bé".
