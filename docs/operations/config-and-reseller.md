# Cấu hình 2 tầng & mô hình bán lại (reseller / white-label)

Tài liệu này trả lời: *"Tôi bán nền tảng cho một bên khác, bên đó chạy phần
admin và bán dịch vụ lại cho các shop — kiến trúc có đáp ứng được không?"*

## Hai tầng cấu hình

Nền tảng tách bạch rõ **cấu hình cấp nền tảng (admin)** và **cấu hình cấp
khách hàng (tenant/shop)**. Ai chạm được gì do RBAC quyết định (kiểm tra
phía server, không phải chỉ ẩn nút trên UI).

| Tầng | Ai cấu hình | Nội dung | Endpoint |
|---|---|---|---|
| **Nền tảng (admin)** | Chủ vận hành / bên mua nền tảng | LLM mặc định + model, embedding, cổng thanh toán (Stripe/VNPay/MoMo…), Facebook App (Meta), chính sách cho phép tenant tự cấu hình AI/OCR, danh sách tenant | `/api/admin/settings`, `/api/admin/settings/payment`, `/api/admin/settings/meta`, `/api/admin/tenants` |
| **Khách hàng (tenant/shop)** | Từng shop tự làm | Key AI riêng (Gemini/OpenAI…), OCR, kênh kết nối (FB/IG/Web), trợ lý (bot) + prompt, sản phẩm, kiến thức, thành viên, gói cước | `/api/settings/*`, `/api/channels`, `/api/bots`, `/api/products`, `/api/billing/*` |

Đã smoke-test: mọi endpoint ở cả hai tầng trả về `200` với đúng vai trò.

## Mô hình bán lại — khuyến nghị: **một triển khai cho mỗi bên mua (white-label)**

Cách chuẩn và ít rủi ro nhất cho tình huống của bạn:

1. Bạn (nhà cung cấp) bàn giao **toàn bộ stack** cho bên mua (họ tự host, hoặc
   bạn host hộ một instance riêng).
2. Bên mua trở thành **platform admin** của instance đó: họ đặt LLM mặc định,
   cổng thanh toán **của họ**, Facebook App **của họ**, và chính sách nền tảng.
3. Các shop của bên mua **tự đăng ký** làm tenant, tự cấu hình kênh/bot/sản phẩm
   và **tự dán key AI của mình** (vì `allow_tenant_llm` mặc định bật) — không cần
   admin can thiệp từng shop.

Mô hình này chạy được **ngay hôm nay, không cần sửa code**: mỗi bên mua là một
"platform admin" độc lập, dữ liệu tách hoàn toàn (mỗi instance một database),
thương hiệu tách hoàn toàn. Cô lập tenant trong một instance đã được bảo vệ 3
lớp (scoping ứng dụng + Postgres RLS + lọc metadata vector — xem ADR-008).

### Nếu sau này muốn "một nền tảng dùng chung, nhiều đại lý"

Tức là một database duy nhất, nhiều đại lý (reseller) là một tầng **ở giữa**
platform và shop, có gộp doanh thu / thương hiệu con riêng cho từng đại lý —
đây là một **tầng phân cấp mới** chưa được mô hình hoá. Nó cần:

- Bảng `reseller` (partner) + khoá `reseller_id` trên `organization`.
- Vai trò `reseller_admin` (giữa `platform_admin` và `owner`), có RLS riêng.
- Branding theo reseller, và gộp hoá đơn/COGS theo reseller.

Đây là một **milestone riêng**, nên làm khi thực sự có nhu cầu marketplace nhiều
đại lý. Với mục tiêu "bán cho một bên, bên đó bán lại cho shop", cách
white-label một-triển-khai-mỗi-bên ở trên là phù hợp và gọn hơn.

## Điều kiện để một shop chạy thật, không cần admin can thiệp

- Người triển khai đặt `META_APP_ID` / `META_APP_SECRET` / `OAUTH_REDIRECT_BASE`
  **một lần** (hoặc nhập App ID/Secret trong *Quản trị → Facebook App*).
- `allow_tenant_llm` để **bật** (mặc định) → shop tự dán key Gemini/OpenAI.
- Sau đó shop tự: bấm *Kết nối Facebook* → chọn Page → tạo bot → chạy thật.
