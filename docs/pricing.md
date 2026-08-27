# Định giá & quản lý token

Ba kiểu gói linh hoạt, chọn theo việc **ai trả tiền cho mô hình AI**.

| Gói | code | Kiểu AI | Giá | Token đi kèm | Ghi chú |
|---|---|---|---|---|---|
| Free | `free` | Tự nhập khoá (BYOK) | $0 | — | 200 tin nhắn/tháng, dùng khoá AI của bạn |
| **Thuê bot** | `starter` | Tự nhập khoá (BYOK) | $19/tháng | — | 20.000 tin nhắn/tháng; **giá vừa phải** vì bạn tự trả token bằng khoá OpenAI/Gemini/Claude |
| **Trọn gói AI** | `growth` | Nền tảng cấp (managed) | $99/tháng | 5.000.000 token | **giá cao** vì đã gồm chi phí mô hình; vượt hạn mức $0.02/1k token |
| **Trả theo dùng** | `payg` | Nền tảng cấp (managed) | $0 cố định | — | Trả **$0.03/1k token** thực dùng, không cam kết tháng |

Ý tưởng đúng như yêu cầu: **luồng trọn gói** (user không cần nhập khoá LLM) →
giá cao; **chỉ thuê bot** (user tự nhập khoá OpenAI/Gemini/Claude) → giá vừa
phải; cộng thêm **pay-as-you-go** theo token.

## Cơ chế (entitlements trong `plan.entitlements`)

- `llm_mode`: `byok` (tenant tự nhập khoá — nền tảng không trả token) hoặc
  `managed` (nền tảng cấp mô hình, giá gồm token).
- `billing_mode`: `subscription` hoặc `payg`.
- `ai_tokens_month`: token đi kèm/tháng (gói managed thuê bao).
- `overage_per_1k`: giá vượt hạn mức ($/1k token); `0` = chặn cứng khi hết.
- `payg_per_1k`: giá mỗi 1k token (gói PAYG).
- `ai_messages_month`: trần tin nhắn (fair-use, gói BYOK); `0` = không giới hạn.

## Quản lý token

**Theo tenant** (`GET /api/subscription` → `quota`, kiểm tra server trước mỗi
lượt AI managed):
- *managed thuê bao*: cho chạy đến khi hết `ai_tokens_month`; nếu cho vượt
  (`overage_per_1k>0`) thì tiếp tục và tính thêm, nếu không thì **tạm dừng trả
  lời tự động** (chuyển nhân viên).
- *payg*: không bao giờ chặn — mọi token được đo và tính tiền.
- *byok*: chỉ áp trần **tin nhắn** (tenant tự trả token bằng khoá của họ).

**Theo khách hàng cuối** (`GET /api/usage/by-customer`): mỗi `usage_event` gắn
`customer_ref`, tổng hợp token/tin nhắn/chi phí theo từng khách trong tháng —
hiển thị ở trang Thanh toán để theo dõi và quản lý mức dùng.

Mọi số liệu token/chi phí lấy từ `usage_event` (đo ở orchestrator sau mỗi lượt
trả lời), đơn giá cấu hình qua `COST_INPUT_PER_M` / `COST_OUTPUT_PER_M`.
