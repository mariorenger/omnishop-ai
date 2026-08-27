# Tài liệu tích hợp — thanh toán & kênh kết nối

Hướng dẫn để **admin nền tảng** tự cấu hình cổng nhận tiền, và **người dùng
(shop)** tự kết nối kênh của họ. Mỗi mục theo đúng tài liệu chính hãng: cần đăng
ký gì, điền trường nào, lấy ở đâu, đặt webhook về đâu, và điều kiện go-live.

Ký hiệu: 🟢 chạy thật ngay · 🟡 cần hợp đồng/duyệt đối tác mới go-live (code đã sẵn).

---

## A. Thanh toán (admin cấu hình — *Quản trị → Cấu hình thanh toán*)

### 🟢 VietQR — QR chuyển khoản
- **Đăng ký:** không cần hợp đồng cổng. Chỉ cần một tài khoản ngân hàng nhận tiền.
  Muốn dùng template riêng: đăng ký tại `my.vietqr.io`.
- **Điền:** Mã ngân hàng (BIN, ví dụ Vietcombank `970436`, Techcombank `970407`,
  MBBank `970422`, ACB `970416`), Số tài khoản, Tên chủ tài khoản, Mẫu QR
  (`compact2` mặc định).
- **Cơ chế:** hệ thống dựng link ảnh QR chuẩn EMVCo/NAPAS:
  `https://img.vietqr.io/image/{BIN}-{STK}-{template}.png?amount=&addInfo=&accountName=`.
  Khách quét bằng app ngân hàng bất kỳ, nội dung CK là mã hoá đơn.
- **Đối soát:** thủ công (chủ shop bấm xác nhận) hoặc nối webhook biến động số dư
  của ngân hàng (ngoài phạm vi module). Tài liệu: <https://www.vietqr.io/en/danh-sach-api/link-tao-ma-nhanh/>

### 🟡 VNPay
- **Đăng ký:** hợp đồng merchant với VNPay → nhận **TMN Code** + **Hash Secret**
  (email sandbox/production).
- **Điền:** TMN Code, Hash Secret (bí mật, mã hoá khi lưu), Pay URL
  (sandbox `https://sandbox.vnpayment.vn/paymentv2/vpcpay.html`, go-live
  `https://pay.vnpay.vn/vpcpay.html`), Return URL.
- **Cơ chế:** tạo URL thanh toán `vnp_Version=2.1.0`, tham số sắp xếp theo khoá,
  ký **HMAC-SHA512** ra `vnp_SecureHash`. Khách trả tiền → VNPay gọi:
  - **Return URL** (trình duyệt): `GET /api/billing/return/vnpay` → xác minh chữ ký,
    kích hoạt gói, chuyển về app.
  - **IPN URL** (server-to-server, nguồn chân lý): `POST /api/billing/ipn/vnpay` →
    xác minh + kích hoạt. Khai báo IPN URL trong cổng merchant VNPay.
  Điều kiện hợp lệ: chữ ký khớp **và** `vnp_ResponseCode=00`.
  Tài liệu: <https://sandbox.vnpayment.vn/apis/docs/thanh-toan-pay/pay.html>

### 🟡 MoMo (AIO v2)
- **Đăng ký:** hợp đồng → **Partner Code**, **Access Key**, **Secret Key**.
- **Điền:** Partner Code, Access Key, Secret Key (bí mật), Endpoint
  (sandbox `https://test-payment.momo.vn/v2/gateway/api/create`), Redirect URL, IPN URL.
- **Cơ chế:** POST create-payment với `requestType=captureWallet`, chữ ký
  **HMAC-SHA256** trên chuỗi *alphabetical*
  `accessKey=..&amount=..&extraData=..&ipnUrl=..&orderId=..&orderInfo=..&partnerCode=..&redirectUrl=..&requestId=..&requestType=..`.
  Nhận `payUrl` → chuyển hướng khách. Xác nhận qua **IPN**
  `POST /api/billing/ipn/momo` (kiểm chữ ký + `resultCode=0`).
  Tài liệu: <https://developers.momo.vn/v3/docs/payment/api/payment-api/init>

### 🟢 Stripe (thẻ quốc tế)
- Secret key + (tuỳ chọn) Webhook secret. Tạo Checkout Session thật; kích hoạt qua
  webhook `checkout.session.completed` tại `POST /api/billing/webhook/stripe`.

> **Lưu ý số tiền:** các cổng VN tính bằng **VND** (số nguyên). Đặt giá gói theo VND
> khi dùng VietQR/VNPay/MoMo.

---

## B. Kênh kết nối (shop tự cấu hình — *Kênh kết nối → Kết nối kênh*)

Mỗi form hiển thị đúng trường cần điền + gợi ý (i) chỗ lấy giá trị + link tài liệu.
Webhook nền tảng cần đặt `OAUTH_REDIRECT_BASE` là URL công khai https.

### 🟢 Telegram
- **Lấy khoá:** chat `@BotFather` → `/newbot` → **Bot Token** (`123456789:AA...`).
- **Webhook:** tự đăng ký khi kết nối (`setWebhook` →
  `/api/channels/webhook/telegram/{public_key}`).
- **Gửi/nhận:** Bot API `sendMessage` / update `message`. Live ngay, không cần duyệt.
  Tài liệu: <https://core.telegram.org/bots/api>

### 🟢 Facebook Messenger / Instagram
- **Lấy khoá:** Facebook App (quyền `pages_messaging`, IG thêm
  `instagram_manage_messages`) → **Page ID** + **Page Access Token**. Có nút *Kết nối
  Facebook* (OAuth) để lấy tự động.
- **Webhook:** chung Meta `/api/channels/webhook/meta` (GET verify bằng
  `META_VERIFY_TOKEN`, POST ký `X-Hub-Signature-256` bằng `META_APP_SECRET`).
  Cần Meta App Review để gửi ngoài 24h/khách chưa nhắn trước.
  Tài liệu: <https://developers.facebook.com/docs/messenger-platform/get-started>

### 🟢 Zalo OA
- **Lấy khoá:** OA liên kết một Zalo App (developers.zalo.me). **OA ID** +
  **OA Access Token** (OAuth v4 `oauth.zaloapp.com/v4/oa/access_token`). App Secret
  (tuỳ chọn) để xác thực webhook `X-ZEvent-Signature`.
- **Lưu ý:** access token hết hạn ~25 giờ → cần refresh token định kỳ (khuyến nghị
  cron làm mới; hiện lưu token do người dùng cấp).
- **Webhook:** `POST /api/channels/webhook/zalo` (sự kiện `user_send_text`).
- **Gửi:** `POST openapi.zalo.me/v3.0/oa/message/cs`.
  Tài liệu: <https://developers.zalo.me/docs/official-account/bat-dau>

### 🟢 WhatsApp Cloud
- **Lấy khoá:** WhatsApp Business Account (Meta) → **Phone Number ID** (ID nội bộ, KHÔNG
  phải số) + **Access Token** (System User dài hạn; token 24h khi test).
- **Webhook:** chung endpoint Meta; body `object=whatsapp_business_account`, định tuyến
  theo `phone_number_id`. Gửi: `POST graph.facebook.com/v21.0/{phone_number_id}/messages`.
  Tài liệu: <https://developers.facebook.com/docs/whatsapp/cloud-api/get-started>

### 🟡 TikTok Shop
- **Lấy khoá:** App trên TikTok Shop Partner Center → **App Key**, **App Secret**;
  shop uỷ quyền (OAuth) → **Access Token** + **Shop Cipher** (+ Shop ID).
- **Ký:** mọi request HMAC-SHA256 bằng App Secret (app_key, timestamp, path, body).
  Token cần refresh. Go-live phụ thuộc duyệt app + phân quyền messaging.
  Tài liệu: <https://partner.tiktokshop.com/docv2/page/authorization-overview-202407>

### 🟡 Shopee
- **Lấy khoá:** App trên Shopee Open Platform → **Partner ID**, **Partner Key**;
  shop uỷ quyền → **Access Token** (hết hạn **4 giờ**) + **Shop ID**.
- **Ký:** HMAC-SHA256 base string `partner_id + path + timestamp + access_token + shop_id`.
  Chat: `/api/v2/sellerchat/send_message`. Go-live phụ thuộc duyệt app.
  Tài liệu: <https://open.shopee.com/documents>

---

### Hiện trạng code
| Nhóm | Gửi/nhận thật | Xác thực chữ ký | Go-live cần |
|---|---|---|---|
| VietQR | ✅ (QR) | — (đối soát ngân hàng) | chỉ số TK |
| VNPay | ✅ redirect | ✅ HMAC-SHA512 (return+IPN) | hợp đồng merchant |
| MoMo | ✅ create+redirect | ✅ HMAC-SHA256 (IPN) | hợp đồng merchant |
| Stripe | ✅ | ✅ webhook | tài khoản Stripe |
| Telegram | ✅ | (token trong URL) | không |
| Messenger/IG | ✅ | ✅ X-Hub-Signature-256 | Meta App Review |
| Zalo OA | ✅ | (X-ZEvent tuỳ chọn) | OA + app duyệt |
| WhatsApp | ✅ | ✅ (chung Meta) | WABA |
| TikTok Shop | lưu cấu hình | ký khi bật | app + shop duyệt |
| Shopee | lưu cấu hình | ký khi bật | app + shop duyệt |
