# Production Readiness Review — chạy thật multi-tenant

Đánh giá trung thực: cái gì đã sẵn sàng để chạy thật, cái gì còn thiếu. Cập nhật
2026-08-24. Trạng thái: ✅ xong · ◑ có nhưng cần hoàn thiện · ❌ chưa có.

## 1. Đa người thuê (multi-tenant) — đã thật chưa?

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Mô hình tenant (User/Org/Membership/Shop/Channel) | ✅ | Mọi bảng dữ liệu có `organization_id`. |
| Cô lập dữ liệu 3 lớp (app + **Postgres RLS** + vector metadata) | ✅ | Có test `scripts/test_isolation.py` chứng minh không đọc/ghi chéo tenant. |
| Phân quyền theo vai trò (owner/admin/agent/viewer) | ✅ | Enforce server-side; billing chỉ owner. |
| Audit log hành động quan trọng | ✅ | Bảng `audit_log`. |
| Mã hoá bí mật at-rest (OAuth token, API key) | ◑ | Đã mã hoá bằng khoá app (`APP_SECRET`). **Nên chuyển sang KMS/OpenBao** cho production và **đổi `APP_SECRET` mặc định**. |
| Cô lập nâng cao cho khách enterprise (schema/DB riêng) | ❌ | Đã có đường nâng cấp trong ADR-008; chưa cài. |

**Kết luận:** cô lập tenant đã là *thật* và có test. Điểm phải làm trước khi mở bán:
đổi `APP_SECRET`, chuyển khoá bí mật sang KMS.

## 2. Luồng "chạy thật" & các tuỳ chọn để chọn — đã đủ chưa?

| Tuỳ chọn tenant tự chọn | Trạng thái |
|---|---|
| Nhà cung cấp LLM (Anthropic / OpenAI / Gemini / vLLM-local) + **chọn model từ danh sách** + Test | ✅ |
| OCR (Tesseract / VLM / tắt) | ✅ |
| Embedding (platform-admin chọn; cố định toàn nền tảng) | ✅ (theo thiết kế) |
| Gói dịch vụ + hạn mức (free/starter/growth) | ✅ enforce quota server-side |
| Kênh kết nối theo gói | ✅ danh sách `channels_allowed` gate theo gói |
| RAG có API + tham số (top_k, ngưỡng, nguồn, sinh câu trả lời) | ✅ `POST /api/rag/query` |

**Kết luận:** luồng chọn cấu hình để chạy thật đã đủ. RAG/OCR ở mức cơ bản nhưng
**API và tham số đã đầy đủ** để cải thiện sau mà không phá vỡ hợp đồng API.

## 3. Thanh toán → gói/limit → kết nối kênh — đã thông chưa?

| Bước | Trạng thái | Ghi chú |
|---|---|---|
| Đăng ký → workspace → gói free | ✅ | |
| Nâng cấp gói (checkout → xác nhận → kích hoạt) | ◑ | Luồng **thật nhưng cổng demo/manual**. Cần cắm **Stripe** (thẻ) và **VNPay/MoMo** (VN) qua `PaymentProvider` — đã có khe cắm. |
| Ra gói/limit sau thanh toán | ✅ | Entitlement + quota áp dụng ngay. |
| Hoá đơn | ✅ | Bảng `invoice`/`payment`, UI liệt kê. |
| Kết nối kênh (UI nhập thông tin, lưu mã hoá, gate theo gói) | ✅ | Trang **Kênh kết nối** + form theo từng loại. |
| **Facebook Messenger / Instagram** gửi–nhận thật | ◑ | Code Graph API **thật** (verify token, gửi tin, webhook + chữ ký). Cần: **một Facebook App của bạn** đã qua **Meta App Review** (quyền `pages_messaging`), đặt `META_APP_SECRET`/`META_VERIFY_TOKEN`, và URL webhook công khai. |
| **TikTok Shop / Shopee** | ◑ | Khung + lưu thông tin đã có; **gửi–nhận cần phê duyệt đối tác** (partner-gated) rồi cài adapter. |

**Kết luận:** pay → gói/limit → kết nối kênh đã **thông về mặt sản phẩm**. Messenger/IG
đã có code thật, chỉ cần bạn cung cấp Facebook App đã duyệt. TikTok/Shopee cần
tài khoản đối tác.

## 4. Hạ tầng vận hành — còn thiếu để lên production

| Hạng mục | Trạng thái | Việc cần làm |
|---|---|---|
| Công cụ migration DB | ◑ | Đang chạy `*.sql` idempotent lúc khởi động. Nên dùng Alembic cho migration có version. |
| Cổng thanh toán thật + webhook | ❌ | Stripe/VNPay/MoMo adapter + xác thực webhook. |
| Meta App Review + webhook công khai | ❌ | Đăng ký app, xin quyền, cấu hình webhook. |
| Rate limiting / chống lạm dụng | ❌ | Giới hạn theo tenant/endpoint; chặn spam widget. |
| Quản lý bí mật (KMS/Vault) | ❌ | Thay khoá app bằng KMS/OpenBao. |
| Sao lưu & khôi phục DB | ❌ | Lịch backup + thử restore. |
| Giám sát & cảnh báo (traces/metrics/logs) | ◑ | Có `correlation_id`; chưa gắn OpenTelemetry/Prometheus/Grafana/Langfuse. |
| HTTPS/TLS, domain, CDN cho `web` | ❌ | Reverse proxy TLS trước `web`. |
| Hàng đợi bền (Temporal) cho sync dài | ◑ | Đang dùng Valkey queue; đủ cho MVP. |
| CI/CD + test tự động | ◑ | Có test chạy tay; cần đưa vào CI. |
| Chính sách lưu trữ & xoá dữ liệu (PII/GDPR) | ❌ | Xoá org/khách/hội thoại/kiến thức theo yêu cầu. |
| Email giao dịch (mời thành viên, hoá đơn) | ❌ | Cắm Resend/Postmark/SES qua `EmailProvider`. |

## 5. Việc nên làm trước khi mở bán (thứ tự đề xuất)

1. Đổi `APP_SECRET`, đưa bí mật vào KMS; bật HTTPS trước `web`.
2. Cắm **một** cổng thanh toán thật (Stripe cho thẻ quốc tế **hoặc** VNPay/MoMo cho VN).
3. Hoàn tất **Meta App Review** + webhook để Messenger/IG chạy khách thật.
4. Rate limiting + sao lưu DB + giám sát cơ bản (Langfuse cho chi phí LLM).
5. Email giao dịch + chính sách xoá dữ liệu.

> Tóm lại: lõi multi-tenant, phân quyền, cấu hình AI, gói/limit và khung kênh đã
> chạy thật và có kiểm thử. Ba việc mang tính "bật công tắc bên ngoài" còn lại —
> **cổng thanh toán thật, Meta App Review, và hạ tầng vận hành (secret/backup/
> giám sát)** — cần tài khoản/hạ tầng của bạn, không phải viết lại kiến trúc.
