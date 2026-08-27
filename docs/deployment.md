# Triển khai & sao lưu (gọn)

Đủ an toàn cho giai đoạn đầu — không over-engineer.

## Chạy
`docker compose up -d` → Postgres(pgvector) + Redis + api + worker + web.
API tự chạy migration `002→011` và bootstrap admin khi khởi động.

## Checklist env tối thiểu (bắt buộc trước khi phục vụ thật)
1. **`APP_SECRET`** — chuỗi ngẫu nhiên ≥ 32 ký tự (mọi secret & token mã hoá bằng
   nó). Sinh nhanh: `openssl rand -hex 32`. API sẽ **cảnh báo lúc khởi động** nếu
   còn để mặc định/quá ngắn.
2. **Mật khẩu Postgres** — đổi `omni/omni_app` mặc định trong compose; **không mở
   port 5432 ra internet**.
3. **`BOOTSTRAP_ADMIN_EMAIL` / `BOOTSTRAP_ADMIN_PASSWORD`** — tài khoản admin đăng
   nhập lần đầu.
4. **`OAUTH_REDIRECT_BASE`** — URL https công khai; đứng sau reverse proxy
   (nginx/Caddy/ALB) terminate **HTTPS**.

Tuỳ chọn khi cần: `GOOGLE_CLIENT_*`, `META_APP_*`, khoá cổng thanh toán — đều có
thể nhập trong UI Quản trị.

> Chưa cần: đổi crypto sang KMS/Fernet, khoá CORS (web same-origin + Bearer token,
> không dùng cookie). Để dành khi hệ thống lớn/nhạy cảm hơn.

## Sao lưu & phục hồi (pg_dump — an toàn, portable)

Dùng `pg_dump`, **không** copy raw thư mục data khi DB đang chạy.

Sao lưu:
```
docker compose exec -T db pg_dump -U omni omnishop | gzip > backup_$(date +%F).sql.gz
```

Phục hồi:
```
gunzip -c backup_YYYY-MM-DD.sql.gz | docker compose exec -T db psql -U omni omnishop
```

Tự động hằng ngày: đặt lệnh sao lưu trên vào cron của máy chủ, đẩy file ra nơi lưu
trữ ngoài (S3/ổ khác). Dữ liệu vẫn bền nhờ named volume `pgdata` qua restart; chỉ
mất nếu chạy `docker compose down -v`.

## Nâng quy mô
Tải cao thì chạy nhiều `worker` (cùng BRPOP một hàng đợi Redis). Xem
`docs/rag-and-ingestion.md`.
