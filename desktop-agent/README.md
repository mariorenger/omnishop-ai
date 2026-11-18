# Desktop Agent

Agent desktop tối giản, đóng gói **một file exe** (Go, UI nhúng sẵn bằng `go:embed`, không dependency ngoài).
Không cấu hình LLM (`base_url` trống) → chạy chế độ **offline if/else** (lệnh: ls, đọc, tìm, tạo, sửa, chạy; gõ `help`).
Bấm exe → mở trình duyệt tới `http://127.0.0.1:7878` → chat, chọn project, agent đọc/sửa file và chạy lệnh.

## Tính năng
- Chọn thư mục project (có file browser trong UI), xem cây file, bấm để xem nội dung.
- Tools cho LLM: `list_dir`, `read_file`, `write_file`, `edit_file`, `search_files`, `run_command`.
- Ghi file / sửa file / chạy lệnh **phải được duyệt** trong UI (hoặc bật "Tự duyệt").
- Sandbox đường dẫn: không đọc/ghi ra ngoài thư mục project (kể cả qua symlink).
- Gọi LLM qua API chuẩn OpenAI (`/chat/completions` + tool calling) → trỏ được vào LLM local/nội bộ.

## Cấu hình
`config.json` cạnh exe (xem `config.example.json`), hoặc biến môi trường `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `AGENT_PORT`.
Cờ: `--project <dir>`, `--no-browser`. Log: `%AppData%\desktop-agent\agent.log`.

## Build
```bash
./build.sh          # → dist/desktop-agent.exe + dist/config.json
go run . --project .   # chạy thử trên máy dev
```

## Bảo mật
- Chỉ bind `127.0.0.1`; mọi `/api` cần token ngẫu nhiên mỗi lần chạy + kiểm tra Host.
- API key nằm trong `config.json`, không nhúng vào exe. Với môi trường công ty nên dùng token theo user.
- `run_command` chạy shell với quyền user hiện tại; giữ duyệt thủ công nếu chưa tin LLM.

## Gắn RAG của bạn
Thêm tool mới trong `tools.go` (`toolDefs` + `runTool`), ví dụ `rag_search` gọi API RAG nội bộ.
