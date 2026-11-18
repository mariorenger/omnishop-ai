package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type bot struct {
	s *Server
	t *testing.T
	d string
}

func newBot(t *testing.T) *bot {
	d := newProj(t)
	s := &Server{cfg: Config{AutoApprove: true, CmdTimeout: 60}, pending: map[string]chan bool{}, project: d}
	return &bot{s: s, t: t, d: d}
}

// say sends one message and returns everything the bot said (assistant events joined).
func (b *bot) say(msg string) string {
	var said []string
	b.s.runOffline(context.Background(), msg, func(ev string, data any) {
		if ev == "assistant" || ev == "error" {
			said = append(said, fmt.Sprint(data))
		}
	})
	return strings.Join(said, "\n")
}
func (b *bot) expect(msg string, subs ...string) {
	b.t.Helper()
	out := b.say(msg)
	for _, s := range subs {
		if !strings.Contains(out, s) {
			b.t.Errorf("%q\n  want %q\n  got  %q", msg, s, out)
		}
	}
}
func (b *bot) file(p string) string { x, _ := os.ReadFile(filepath.Join(b.d, p)); return string(x) }

func TestOfflineCommands(t *testing.T) {
	b := newBot(t)

	// --- chào / help / project / không hiểu / rỗng
	b.expect("chào", "Chào bạn")
	b.expect("Hello", "Chào bạn")
	b.expect("help", "ls", "tìm", "chạy -t")
	b.expect("?", "ls")
	b.expect("/help", "ls")
	b.expect("   ", "Bạn chưa nhập gì")
	b.expect("bla bla", "chưa hiểu")
	b.expect("project", filepath.Base(b.d))

	// --- ls / dir / liệt kê / xem thư mục (hoa thường, tiếng Anh/Việt)
	b.expect("ls", "a.txt", "sub/")
	b.expect("LS", "a.txt")
	b.expect("dir", "a.txt")
	b.expect("ls sub", "b.py", "deep/")
	b.expect("ls ./sub", "b.py")
	b.expect("ls thư mục sub", "b.py")
	b.expect("xem thư mục sub", "b.py")
	b.expect("liệt kê", "a.txt")
	b.expect("liệt kê sub", "b.py")
	b.expect("ls nope", "Không làm được")
	b.expect("tree", "└── ", "sub/")
	b.expect("cây", "sub/")

	// --- đọc
	b.expect("đọc a.txt", "hello TODO world")
	b.expect("Đọc a.txt", "hello TODO world")
	b.expect("read a.txt", "hello TODO world")
	b.expect("cat a.txt", "hello")
	b.expect("xem a.txt", "hello")
	b.expect("xem file a.txt", "hello")
	b.expect("đọc file sub/b.py", "print")
	b.expect("mở sub/b.py", "print")
	b.expect("đọc", "Đọc file nào")
	b.expect("đọc nope.txt", "Không làm được")
	b.expect("đọc ../../etc/passwd", "outside the project")
	b.expect("đọc /etc/passwd", "outside the project")

	// --- tìm
	b.expect("tìm TODO", "a.txt:1:")
	b.expect("find TODO", "a.txt:1:")
	b.expect("tìm zzz", "không có kết quả")
	b.expect("tìm", "Tìm gì")
	// tìm theo TÊN và theo NỘI DUNG
	b.expect("tìm b.py", "Tên tệp khớp", "sub/b.py", "Nội dung khớp", "không có kết quả")
	b.expect("tìm B.PY", "sub/b.py") // không phân biệt hoa/thường
	b.expect("tìm tên sub", "sub/")
	b.expect("tìm tên *.py", "sub/b.py") // glob
	b.expect("tìm file a.txt", "a.txt")
	b.expect("tìm tên TODO", "không có kết quả") // TODO chỉ có trong nội dung, không phải tên
	b.expect("tìm nội dung TODO", "a.txt:1:")
	b.expect("tìm chữ TODO", "a.txt:1:")
	b.expect("tìm TODO", "Nội dung khớp", "a.txt:1:", "Tên tệp khớp", "không có kết quả")
	b.expect("tìm *.py", "sub/b.py", "bỏ qua") // glob hợp lệ cho tên, không phải regex cho nội dung
	b.expect("tìm tên", "Tìm gì")
	b.expect("tìm (", "bỏ qua")

	// --- tạo (có dấu, khoảng trắng, thư mục lồng, ghi đè, nội dung có dấu ':')
	b.expect("tạo moi.txt : nội dung mới", "wrote")
	if b.file("moi.txt") != "nội dung mới" {
		t.Errorf("tạo: %q", b.file("moi.txt"))
	}
	b.expect("tạo thư mục mới/tệp con.txt : ok", "wrote")
	if b.file("thư mục mới/tệp con.txt") != "ok" {
		t.Errorf("tạo (dấu/khoảng trắng): %q", b.file("thư mục mới/tệp con.txt"))
	}
	b.expect("đọc tệp con.txt", "Không làm được") // phải có đường dẫn đầy đủ
	b.expect("đọc thư mục mới/tệp con.txt", "ok")
	b.expect("tạo t.txt : giờ: 10:30", "wrote")
	if b.file("t.txt") != "giờ: 10:30" {
		t.Errorf("nội dung có ':' bị cắt: %q", b.file("t.txt"))
	}
	b.expect("tạo moi.txt : ghi đè", "wrote")
	if b.file("moi.txt") != "ghi đè" {
		t.Error("ghi đè fail")
	}
	b.expect("write w.txt : hi", "wrote")
	b.expect("tạo", "Cú pháp")
	b.expect("tạo chi_ten.txt", "Cú pháp")
	b.expect("tạo ../ngoai.txt : x", "outside")

	// --- thêm
	b.expect("thêm moi.txt : dòng 2", "appended")
	if b.file("moi.txt") != "ghi đè\ndòng 2\n" {
		t.Errorf("thêm: %q", b.file("moi.txt"))
	}
	b.expect("append moi.txt : dòng 3", "appended")
	b.expect("thêm moi.txt", "Cú pháp")

	// --- sửa
	b.expect("sửa a.txt : world => agent", "edited")
	if !strings.Contains(b.file("a.txt"), "hello TODO agent") {
		t.Errorf("sửa: %q", b.file("a.txt"))
	}
	b.expect("sửa a.txt : world => x", "found 0 times")
	b.expect("sửa a.txt : hello TODO => xin chào", "edited")
	b.expect("edit a.txt : xin chào=>hello", "edited") // không có khoảng trắng quanh =>
	b.expect("thay a.txt : hello => hi", "edited")
	b.expect("sửa a.txt", "Cú pháp")
	b.expect("sửa a.txt : chi_cu", "Cú pháp")

	// --- đổi tên / di chuyển
	b.expect("đổi tên moi.txt => doi.txt", "moved")
	if b.file("doi.txt") == "" || b.file("moi.txt") != "" {
		t.Error("đổi tên fail")
	}
	b.expect("mv doi.txt => sub/doi2.txt", "moved")
	b.expect("rename sub/doi2.txt => doi3.txt", "moved")
	b.expect("đổi tên doi3.txt => a.txt", "already exists")
	b.expect("đổi tên nope => z", "Không làm được")
	b.expect("đổi tên a.txt", "Cú pháp")

	// --- xóa
	b.expect("xóa doi3.txt", "deleted")
	b.expect("xoá t.txt", "deleted")
	b.expect("rm w.txt", "deleted")
	b.expect("xóa sub", "Không làm được") // thư mục không rỗng
	b.expect("xóa sub/deep", "deleted")
	b.expect("xóa nope", "Không làm được")
	b.expect("xóa", "Xóa gì")
	b.expect("xóa .", "root")

	// --- chạy
	b.expect("chạy echo xin chao", "xin chao")
	b.expect("run echo hello", "hello")
	b.expect(`chạy echo "a b"`, "a b")
	b.expect("chạy", "Chạy lệnh gì")
	b.expect("chạy exit 2", "exit status 2")
	if runtime.GOOS != "windows" {
		b.expect("chạy ls", "a.txt")
		b.expect("chạy -t 1 sleep 5", "⏱️", "hết giờ sau 1s")
		b.expect("chạy -t 3 echo nhanh", "nhanh")
		b.expect("chạy sleep 10 &", "tiến trình nền")
	}
}

func TestOfflineDenied(t *testing.T) {
	b := newBot(t)
	b.s.cfg.AutoApprove = false
	for _, m := range []string{"tạo deny.txt : x", "xóa a.txt", "chạy echo hi", "sửa a.txt : hello => x"} {
		var said []string
		ctx := context.Background()
		stop := denyAll(b.s) // người dùng bấm "Từ chối"
		b.s.runOffline(ctx, m, func(ev string, data any) {
			if ev == "assistant" {
				said = append(said, fmt.Sprint(data))
			}
		})
		stop()
		if !strings.Contains(strings.Join(said, ""), "denied") {
			t.Errorf("%q should be denied, got %v", m, said)
		}
	}
	if b.file("deny.txt") != "" || b.file("a.txt") == "" || strings.Contains(b.file("a.txt"), "x") {
		t.Error("denied actions changed files")
	}
}
