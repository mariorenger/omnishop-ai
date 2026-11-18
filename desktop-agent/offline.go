package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const offlineHelp = `Mình là **bot if/else** (chế độ offline, không dùng model). Các lệnh mình hiểu:

- **ls** [thư mục] – liệt kê file (vd: ls src)
- **tree** [thư mục] – xem cây thư mục
- **đọc** <file> – xem nội dung file (vd: đọc README.md)
- **tìm** <regex> – tìm trong toàn bộ project (vd: tìm TODO)
- **tạo** <file> : <nội dung> – tạo hoặc ghi đè file
- **thêm** <file> : <dòng> – thêm một dòng vào cuối file
- **sửa** <file> : <cũ> => <mới> – thay chữ trong file
- **đổi tên** <cũ> => <mới> – đổi tên / di chuyển
- **xóa** <file> – xóa file hoặc thư mục rỗng
- **chạy** <lệnh> – chạy lệnh terminal (vd: chạy dir)
- **project** – cho biết thư mục đang làm việc

Các lệnh làm thay đổi dữ liệu (tạo, thêm, sửa, đổi tên, xóa, chạy) sẽ hỏi bạn duyệt trước.
Cũng gõ được tiếng Anh: ls, tree, read, find, write, append, edit, mv, rm, run.`

func (s *Server) runOffline(ctx context.Context, msg string, emit emitFn) {
	s.mu.Lock()
	root := s.project
	s.mu.Unlock()

	text := strings.TrimSpace(msg)
	text = strings.TrimLeft(text, "/!")
	fields := strings.Fields(text)
	if len(fields) == 0 {
		emit("assistant", "Bạn chưa nhập gì. Gõ help để xem lệnh.")
		return
	}
	verb := strings.ToLower(fields[0])
	rest := strings.TrimSpace(text[len(fields[0]):])
	// drop filler words: "xem file x", "đọc tệp x", "xem thư mục x"
	low := strings.ToLower(rest)
	isDir := false
	for _, p := range []string{"thư mục", "folder", "dir", "directory"} {
		if strings.HasPrefix(low, p+" ") || low == p {
			isDir, rest = true, strings.TrimSpace(rest[len(p):])
			break
		}
	}
	for _, p := range []string{"file", "tệp", "tập tin"} {
		if strings.HasPrefix(strings.ToLower(rest), p+" ") {
			rest = strings.TrimSpace(rest[len(p):])
			break
		}
	}

	call := func(name string, args map[string]any, showOutput bool) {
		b, _ := json.Marshal(args)
		tc := ToolCall{ID: fmt.Sprintf("off_%d", time.Now().UnixNano()), Type: "function"}
		tc.Function.Name, tc.Function.Arguments = name, string(b)
		res := s.execTool(ctx, root, tc, emit)
		s.mu.Lock()
		s.history = append(s.history, Msg{Role: "tool", ToolCallID: tc.ID, Content: res})
		s.mu.Unlock()
		switch {
		case strings.HasPrefix(res, "error:"):
			emit("assistant", "⚠️ Không làm được: "+strings.TrimSpace(strings.TrimPrefix(res, "error:")))
		case showOutput:
			emit("assistant", "```\n"+clip(strings.TrimRight(res, "\n"), 6000)+"\n```")
		default:
			emit("assistant", "✅ "+res)
		}
	}

	switch {
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
		call("read_file", map[string]any{"path": rest}, true)
	case verb == "tìm" || verb == "find" || verb == "search" || verb == "grep":
		if rest == "" {
			emit("assistant", "Tìm gì? Ví dụ: tìm TODO")
			return
		}
		call("search_files", map[string]any{"pattern": rest}, true)
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
		call("run_command", map[string]any{"command": rest}, true)
	default:
		emit("assistant", fmt.Sprintf("Mình chưa hiểu %q. Gõ help để xem các lệnh mình biết.", msg))
	}
}
