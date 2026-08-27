# Đăng nhập & phân quyền

## Đăng nhập (OAuth + mật khẩu)

- **Email + mật khẩu**: đăng ký/đăng nhập cục bộ (mật khẩu băm PBKDF2).
- **Đăng nhập Google (OAuth 2.0)**: nút "Tiếp tục với Google" hiện khi đã cấu
  hình. Lần đầu đăng nhập tự tạo tài khoản + một workspace. Cấu hình Client
  ID/Secret ở env (`GOOGLE_CLIENT_*`) hoặc **Quản trị → Đăng nhập Google**.
  Redirect URI đăng ký ở Google Cloud: `<OAUTH_REDIRECT_BASE>/api/auth/google/callback`.
- Tài khoản đăng nhập Google không cần mật khẩu (cột `password_hash` cho phép NULL).

## Tài khoản admin được cấp thế nào?

Câu hỏi "admin biết tài khoản/mật khẩu ở đâu": người triển khai đặt
`BOOTSTRAP_ADMIN_EMAIL` + `BOOTSTRAP_ADMIN_PASSWORD` trong env. Khi khởi động,
nếu tài khoản đó chưa có, hệ thống **tạo và gán quyền admin**; nếu đã có thì
đảm bảo quyền admin. Đó chính là thông tin đăng nhập admin.

Ngoài ra, các email trong `PLATFORM_ADMIN_EMAILS` khi đăng nhập Google lần đầu
sẽ **tự động thành admin**.

## Hai tầng vai trò

### Vai trò cấp NỀN TẢNG (control plane) — cột `app_user.platform_role`
| Vai trò | Quyền |
|---|---|
| **admin** | Toàn quyền: mọi cấu hình (LLM/OCR/thanh toán/Facebook/Google/branding/**gói & giá & đơn giá token**), quản lý khách hàng, cấp quyền nhân sự |
| **manager** (Quản lý) | **Chỉ đọc**: xem thống kê toàn nền tảng, danh sách khách hàng, **xuất báo cáo CSV**. KHÔNG sửa cấu hình, KHÔNG sửa khách hàng, KHÔNG cấp quyền |
| *(none)* | Người dùng tenant bình thường |

Admin cấp/thu hồi quyền nhân sự ở **Quản trị → Nhân sự vận hành** (theo email;
người đó phải đã có tài khoản). Thực thi phía server:
`require_platform_admin` (ghi) vs `require_platform_reader` (đọc: admin | manager).

### Vai trò cấp TENANT (trong một tổ chức) — `membership.role`
`owner > admin > agent > viewer` — quản lý cửa hàng/bot/kênh/sản phẩm của chính
tổ chức đó (RLS cô lập giữa các tổ chức).

## Xuất báo cáo (admin + manager)
- `GET /api/admin/reports/tenants.csv` — mỗi khách hàng: gói, cửa hàng, tin nhắn,
  token, chi phí tháng này.
- `GET /api/admin/reports/usage.csv` — sử dụng theo ngày, 30 ngày gần nhất.

Đã có test kiểm chứng: manager đọc/xuất được, mọi thao tác ghi trả về 403; tenant
thường không thấy control plane.
