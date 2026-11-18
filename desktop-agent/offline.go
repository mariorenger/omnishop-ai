package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const offlineHelp = `Mình là **bot if/else** (chế độ offline, không dùng model). Các lệnh mình hiểu:

- **ls** [thư mục] – liệt kê file (vd: ls src)
- **tree** [thư mục] – xem cây thư mục
- **đọc** <file> – xem nội dung file (vd: đọc README.md)
- **tìm** <từ khóa> – tìm cả **tên tệp** lẫn **nội dung** trong project (vd: tìm hoadon). Chỉ theo tên: **tìm tên** *.pdf · chỉ nội dung: **tìm nội dung** TODO
- **tạo** <file> : <nội dung> – tạo hoặc ghi đè file
- **thêm** <file> : <dòng> – thêm một dòng vào cuối file
- **sửa** <file> : <cũ> => <mới> – thay chữ trong file
- **đổi tên** <cũ> => <mới> – đổi tên / di chuyển
- **xóa** <file> – xóa file hoặc thư mục rỗng
- **chạy** <lệnh> – chạy lệnh terminal (vd: chạy dir). Tự dừng sau 60 giây; muốn lâu hơn: **chạy -t 300 <lệnh>**
- **google** <từ khóa> / **tin tức** <chủ đề> – tìm trên web (cần internet). **đọc** <https://…> – đọc nội dung một trang web
- **project** – cho biết thư mục đang làm việc
- **task** – xem các tác vụ định kỳ (monitor; gõ tắt job cũng được). Tạo: **task thêm** <tên> | <lịch> | <lệnh> [báo nếu có|thiếu <chữ>] [timeout <giây>]
  – lịch: *mỗi 30s*, *mỗi 5 phút*, *mỗi giờ*, *hàng ngày 08:30*. Ví dụ: task thêm ping | mỗi 1 phút | ping -n 1 google.com báo nếu thiếu Reply
  – quản lý: **task chạy** <id> · **task log** <id> · **task dừng** <id> · **task bật** <id> · **task xóa** <id>

Các lệnh làm thay đổi dữ liệu (tạo, thêm, sửa, đổi tên, xóa, chạy) sẽ hỏi bạn duyệt trước.
Cũng gõ được tiếng Anh: ls, tree, read, find, write, append, edit, mv, rm, run.`

func (s *Server) runOffline(ctx context.Context, msg string, emit emitFn) {
	s.mu.Lock()
	root := s.project
	s.mu.Unlock()
	s.hint = func(t string) { emit("assistant", t) }

	text := strings.TrimSpace(msg)
	text = strings.TrimLeft(text, "/!")
	fields := strings.Fields(text)
	if len(fields) == 0 {
		emit("assistant", "Bạn chưa nhập gì. Gõ help để xem lệnh.")
		return
	}
	verb := strings.ToLower(fields[0])
	rest := strings.TrimSpace(text[len(fields[0]):])
	// Bỏ từ đệm ("xem file x", "ls thư mục x", "xóa tệp x") — nhưng CHỈ với những lệnh mà
	// đối số đầu là một đường dẫn, để không cắt nhầm tên thư mục như "thư mục mới/a.txt" khi tạo.
	isDir := false
	switch verb {
	case "ls", "dir", "list", "liệt", "tree", "cây", "xem", "mở", "open", "xóa", "xoá", "rm", "del", "delete":
		low := strings.ToLower(rest)
		for _, p := range []string{"thư mục", "folder", "directory", "dir"} {
			if strings.HasPrefix(low, p+" ") || low == p {
				isDir, rest = true, strings.TrimSpace(rest[len(p):])
				break
			}
		}
	}
	switch verb {
	case "đọc", "read", "cat", "type", "xem", "mở", "open", "tạo", "write", "ghi", "new", "thêm", "append", "add",
		"sửa", "edit", "thay", "replace", "xóa", "xoá", "rm", "del", "delete":
		low := strings.ToLower(rest)
		for _, p := range []string{"tập tin", "file", "tệp"} {
			if strings.HasPrefix(low, p+" ") {
				rest = strings.TrimSpace(rest[len(p):])
				break
			}
		}
	}

	exec1 := func(name string, args map[string]any) string {
		b, _ := json.Marshal(args)
		tc := ToolCall{ID: fmt.Sprintf("off_%d", time.Now().UnixNano()), Type: "function"}
		tc.Function.Name, tc.Function.Arguments = name, string(b)
		res := s.execTool(ctx, root, tc, emit)
		s.mu.Lock()
		s.history = append(s.history, Msg{Role: "tool", ToolCallID: tc.ID, Content: res})
		s.mu.Unlock()
		return res
	}
	call := func(name string, args map[string]any, showOutput bool) {
		res := exec1(name, args)
		switch {
		case strings.HasPrefix(res, "error:"):
			emit("assistant", "⚠️ Không làm được: "+strings.TrimSpace(strings.TrimPrefix(res, "error:")))
		case strings.Contains(res, "[hết giờ sau"):
			emit("assistant", "⏱️ Lệnh chạy quá lâu nên đã bị dừng. Muốn cho chạy lâu hơn: `chạy -t 300 <lệnh>` (300 giây) hoặc chỉnh `command_timeout_sec` trong config.json.\n\n```\n"+clip(strings.TrimRight(res, "\n"), 6000)+"\n```")
		case showOutput:
			emit("assistant", "```\n"+clip(strings.TrimRight(res, "\n"), 6000)+"\n```")
		default:
			emit("assistant", "✅ "+res)
		}
	}

	switch {
	case verb == "google" || verb == "web" || verb == "news" || verb == "tin":
		q := rest
		if verb == "tin" { // "tin tức <chủ đề>"
			q = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(strings.ToLower(rest), "tức"), " "))
			if q == "" {
				s.hint("Tin gì? Ví dụ: tin tức giá vàng hôm nay")
				return
			}
			q = "tin tức " + q
		}
		if strings.TrimSpace(q) == "" {
			s.hint("Tìm gì? Ví dụ: google giá vàng hôm nay · tin tức bóng đá")
			return
		}
		call("web_search", map[string]any{"query": q}, true)
	case (verb == "đọc" || verb == "mở" || verb == "open" || verb == "read" || verb == "fetch" || verb == "tải") && (strings.HasPrefix(strings.ToLower(rest), "http://") || strings.HasPrefix(strings.ToLower(rest), "https://")):
		call("fetch_url", map[string]any{"url": rest}, true)
	case verb == "task" || verb == "tasks" || verb == "job" || verb == "jobs" || verb == "monitor":
		s.offlineTask(rest, call)
	case verb == "project" || verb == "pwd" || verb == "cwd":
		emit("assistant", "📁 Đang làm việc trong: `"+root+"`")
	case verb == "tree" || verb == "cây":
		if rest == "" {
			rest = "."
		}
		call("tree", map[string]any{"path": rest}, true)
	case verb == "thêm" || verb == "append" || verb == "add":
		path, content, ok := strings.Cut(rest, ":")
		if !ok || strings.TrimSpace(path) == "" {
			emit("assistant", "Cú pháp: thêm <file> : <dòng cần thêm>")
			return
		}
		call("append_file", map[string]any{"path": strings.TrimSpace(path), "content": strings.TrimPrefix(content, " ")}, false)
	case verb == "xóa" || verb == "xoá" || verb == "delete" || verb == "rm" || verb == "del":
		if rest == "" {
			emit("assistant", "Xóa gì? Ví dụ: xóa b.txt")
			return
		}
		call("delete_file", map[string]any{"path": rest}, false)
	case verb == "mv" || verb == "move" || verb == "rename" || verb == "đổi":
		rest = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(rest, "tên"), "name"))
		from, to, ok := strings.Cut(rest, "=>")
		if !ok || strings.TrimSpace(from) == "" || strings.TrimSpace(to) == "" {
			emit("assistant", "Cú pháp: đổi tên <cũ> => <mới>")
			return
		}
		call("move_file", map[string]any{"from": strings.TrimSpace(from), "to": strings.TrimSpace(to)}, false)
	case verb == "help" || verb == "?" || verb == "giúp" || verb == "trợ" || verb == "huongdan":
		emit("assistant", offlineHelp)
	case verb == "hi" || verb == "hello" || verb == "chào" || verb == "xin":
		emit("assistant", "Chào bạn! Mình là bot offline chạy bằng if/else. Gõ help để xem mình làm được gì.")
	case verb == "ls" || verb == "dir" || verb == "list" || verb == "liệt" || (isDir && (verb == "xem" || verb == "mở")):
		if verb == "liệt" {
			rest = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(strings.ToLower(rest), "kê"), " "))
		}
		if rest == "" {
			rest = "."
		}
		call("list_dir", map[string]any{"path": rest}, true)
	case verb == "đọc" || verb == "read" || verb == "cat" || verb == "type" || verb == "xem" || verb == "mở" || verb == "open":
		if rest == "" {
			emit("assistant", "Đọc file nào? Ví dụ: đọc README.md")
			return
		}
		// typo / no diacritics: if the name does not exist but one file is clearly the closest, open it
		if p, err := resolve(root, rest); err == nil {
			if _, e := os.Stat(p); isNotExist(e) {
				if m := suggestFiles(ctx, root, rest, 5); len(m) > 0 && m[0].Score >= 0.85 && !strings.HasSuffix(m[0].Path, "/") &&
					(len(m) == 1 || m[1].Score <= m[0].Score-0.1) {
					emit("assistant", fmt.Sprintf("Không thấy %q – mở tệp gần giống nhất: `%s`", rest, m[0].Path))
					rest = m[0].Path
				}
			}
		}
		call("read_file", map[string]any{"path": rest}, true)
	case verb == "tìm" || verb == "find" || verb == "search" || verb == "grep":
		if rest == "" {
			emit("assistant", "Tìm gì? Ví dụ: tìm TODO")
			return
		}
		mode, q := "both", rest
		low := strings.ToLower(rest)
		switch low { // chỉ gõ từ khóa chế độ mà quên nội dung cần tìm
		case "tên", "ten", "file", "tệp", "name", "nội dung", "noi dung", "chữ", "content", "text":
			emit("assistant", "Tìm gì? Ví dụ: tìm tên báo cáo · tìm nội dung password")
			return
		}
		for _, p := range []string{"tên", "ten", "file", "tệp", "name"} {
			if strings.HasPrefix(low, p+" ") {
				mode, q = "name", strings.TrimSpace(rest[len(p):])
				break
			}
		}
		for _, p := range []string{"nội dung", "noi dung", "chữ", "trong", "content", "text"} {
			if strings.HasPrefix(low, p+" ") {
				mode, q = "content", strings.TrimSpace(rest[len(p):])
				break
			}
		}
		if q == "" {
			emit("assistant", "Tìm gì? Ví dụ: tìm TODO · tìm tên báo cáo · tìm nội dung password")
			return
		}
		section := func(title, res string) string {
			switch {
			case strings.HasPrefix(res, "error: bad pattern"):
				return title + "\nbỏ qua (mẫu không phải regex hợp lệ)"
			case strings.HasPrefix(res, "error:"):
				return title + "\n⚠️ " + strings.TrimSpace(strings.TrimPrefix(res, "error:"))
			case res == "no matches":
				return title + "\nkhông có kết quả"
			}
			return title + "\n```\n" + clip(strings.TrimRight(res, "\n"), 6000) + "\n```"
		}
		var parts []string
		if mode != "content" {
			parts = append(parts, section("**📄 Tên tệp khớp**", exec1("find_files", map[string]any{"pattern": q})))
		}
		if mode != "name" {
			parts = append(parts, section("**🔎 Nội dung khớp**", exec1("search_files", map[string]any{"pattern": q})))
		}
		emit("assistant", strings.Join(parts, "\n\n"))
	case verb == "tạo" || verb == "write" || verb == "ghi" || verb == "new":
		path, content, ok := strings.Cut(rest, ":")
		if !ok || strings.TrimSpace(path) == "" {
			emit("assistant", "Cú pháp: tạo <file> : <nội dung>")
			return
		}
		call("write_file", map[string]any{"path": strings.TrimSpace(path), "content": strings.TrimPrefix(content, " ")}, false)
	case verb == "sửa" || verb == "edit" || verb == "thay" || verb == "replace":
		path, body, ok := strings.Cut(rest, ":")
		oldS, newS, ok2 := strings.Cut(body, "=>")
		if !ok || !ok2 || strings.TrimSpace(path) == "" {
			emit("assistant", "Cú pháp: sửa <file> : <chữ cũ> => <chữ mới>")
			return
		}
		call("edit_file", map[string]any{"path": strings.TrimSpace(path), "old_string": strings.TrimSpace(oldS), "new_string": strings.TrimSpace(newS)}, false)
	case verb == "chạy" || verb == "run" || verb == "cmd" || verb == "exec":
		if rest == "" {
			emit("assistant", "Chạy lệnh gì? Ví dụ: chạy dir")
			return
		}
		args := map[string]any{"command": rest}
		// optional: chạy -t 120 <lệnh>   (giới hạn 120 giây)
		if f := strings.Fields(rest); len(f) >= 3 && (f[0] == "-t" || f[0] == "--timeout") {
			if n, err := strconv.Atoi(strings.TrimSuffix(f[1], "s")); err == nil && n > 0 {
				idx := strings.Index(rest, f[1]) + len(f[1])
				args = map[string]any{"command": strings.TrimSpace(rest[idx:]), "timeout_sec": n}
			}
		}
		call("run_command", args, true)
	default:
		emit("assistant", fmt.Sprintf("Mình chưa hiểu %q. Gõ help để xem các lệnh mình biết.", msg))
	}
}

var (
	reAlertHas  = regexp.MustCompile(`(?i)\s+(?:báo nếu có|alert if contains)\s+(.+?)\s*$`)
	reAlertMiss = regexp.MustCompile(`(?i)\s+(?:báo nếu thiếu|báo nếu không có|alert if missing)\s+(.+?)\s*$`)
	reTaskTO    = regexp.MustCompile(`(?i)\s+timeout\s+(\d+)\s*$`)
)

func (s *Server) offlineTask(rest string, call func(string, map[string]any, bool)) {
	f := strings.Fields(rest)
	sub := ""
	arg := ""
	if len(f) > 0 {
		sub = strings.ToLower(f[0])
		arg = strings.TrimSpace(rest[len(f[0]):])
	}
	// "tạm dừng" is two words
	if sub == "tạm" && strings.HasPrefix(strings.ToLower(arg), "dừng") {
		sub, arg = "dừng", strings.TrimSpace(arg[len("dừng"):])
	}
	switch sub {
	case "", "ds", "list", "danh", "sách", "ls":
		call("list_tasks", map[string]any{}, true)
	case "thêm", "tạo", "add", "new":
		parts := strings.SplitN(arg, "|", 3)
		if len(parts) < 3 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[2]) == "" {
			s.hint("Cú pháp: task thêm <tên> | <lịch> | <lệnh> [báo nếu có|thiếu <chữ>] [timeout <giây>]\nVí dụ: task thêm kiểm tra log | mỗi 5 phút | findstr ERROR app.log báo nếu có ERROR")
			return
		}
		cmd := strings.TrimSpace(parts[2])
		args := map[string]any{"name": strings.TrimSpace(parts[0]), "schedule": strings.TrimSpace(parts[1])}
		if m := reTaskTO.FindStringSubmatch(cmd); m != nil {
			n, _ := strconv.Atoi(m[1])
			args["timeout_sec"] = n
			cmd = strings.TrimSpace(cmd[:len(cmd)-len(m[0])])
		}
		if m := reAlertHas.FindStringSubmatch(cmd); m != nil {
			args["alert_contains"] = strings.Trim(m[1], `"'`)
			cmd = strings.TrimSpace(cmd[:len(cmd)-len(m[0])])
		} else if m := reAlertMiss.FindStringSubmatch(cmd); m != nil {
			args["alert_not_contains"] = strings.Trim(m[1], `"'`)
			cmd = strings.TrimSpace(cmd[:len(cmd)-len(m[0])])
		}
		args["command"] = cmd
		call("create_task", args, false)
	case "chạy", "run":
		call("run_task", map[string]any{"id": arg}, true)
	case "xóa", "xoá", "del", "delete", "rm":
		call("delete_task", map[string]any{"id": arg}, false)
	case "dừng", "dung", "pause", "stop", "tắt":
		call("toggle_task", map[string]any{"id": arg, "enabled": false}, false)
	case "bật", "bat", "resume", "start", "on":
		call("toggle_task", map[string]any{"id": arg, "enabled": true}, false)
	case "log", "xem", "show", "chi":
		call("list_tasks", map[string]any{"id": arg}, true)
	default:
		s.hint("Lệnh task: task · task thêm … · task chạy <id> · task log <id> · task dừng <id> · task bật <id> · task xóa <id>")
	}
}
