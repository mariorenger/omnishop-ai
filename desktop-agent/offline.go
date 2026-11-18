package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

const offlineHelp = `Mình là bot if/else (chế độ offline, không dùng model). Các lệnh:

• ls [thư mục]                      – liệt kê file          (vd: ls src)
• đọc <file>                        – xem nội dung file     (vd: đọc README.md)
• tìm <regex>                       – tìm trong project     (vd: tìm TODO)
• tạo <file> : <nội dung>           – tạo/ghi file (cần duyệt)
• sửa <file> : <cũ> => <mới>        – thay chữ trong file (cần duyệt)
• chạy <lệnh>                       – chạy lệnh terminal   (cần duyệt, vd: chạy dir)

Có thể gõ tiếng Anh: ls, read, find, write, edit, run.`

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

	call := func(name string, args map[string]any) {
		b, _ := json.Marshal(args)
		tc := ToolCall{ID: fmt.Sprintf("off_%d", len(s.history)), Type: "function"}
		tc.Function.Name, tc.Function.Arguments = name, string(b)
		res := s.execTool(ctx, root, tc, emit)
		s.mu.Lock()
		s.history = append(s.history, Msg{Role: "tool", ToolCallID: tc.ID, Content: res})
		s.mu.Unlock()
		if strings.HasPrefix(res, "error:") {
			emit("assistant", "Không làm được: "+strings.TrimPrefix(res, "error: "))
		} else {
			emit("assistant", "Xong.")
		}
	}

	switch {
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
		call("list_dir", map[string]any{"path": rest})
	case verb == "đọc" || verb == "read" || verb == "cat" || verb == "type" || verb == "xem" || verb == "mở" || verb == "open":
		if rest == "" {
			emit("assistant", "Đọc file nào? Ví dụ: đọc README.md")
			return
		}
		call("read_file", map[string]any{"path": rest})
	case verb == "tìm" || verb == "find" || verb == "search" || verb == "grep":
		if rest == "" {
			emit("assistant", "Tìm gì? Ví dụ: tìm TODO")
			return
		}
		call("search_files", map[string]any{"pattern": rest})
	case verb == "tạo" || verb == "write" || verb == "ghi" || verb == "new":
		path, content, ok := strings.Cut(rest, ":")
		if !ok || strings.TrimSpace(path) == "" {
			emit("assistant", "Cú pháp: tạo <file> : <nội dung>")
			return
		}
		call("write_file", map[string]any{"path": strings.TrimSpace(path), "content": strings.TrimPrefix(content, " ")})
	case verb == "sửa" || verb == "edit" || verb == "thay" || verb == "replace":
		path, body, ok := strings.Cut(rest, ":")
		oldS, newS, ok2 := strings.Cut(body, "=>")
		if !ok || !ok2 || strings.TrimSpace(path) == "" {
			emit("assistant", "Cú pháp: sửa <file> : <chữ cũ> => <chữ mới>")
			return
		}
		call("edit_file", map[string]any{"path": strings.TrimSpace(path), "old_string": strings.TrimSpace(oldS), "new_string": strings.TrimSpace(newS)})
	case verb == "chạy" || verb == "run" || verb == "cmd" || verb == "exec":
		if rest == "" {
			emit("assistant", "Chạy lệnh gì? Ví dụ: chạy dir")
			return
		}
		call("run_command", map[string]any{"command": rest})
	default:
		emit("assistant", fmt.Sprintf("Mình chưa hiểu %q. Gõ help để xem các lệnh mình biết.", msg))
	}
}
