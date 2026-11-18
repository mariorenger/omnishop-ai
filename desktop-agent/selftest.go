package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// `desktop-agent --selftest` runs three real multi-step goals against the LLM configured in
// config.json (open a file by an approximate name, search+write a report, write+run code) and
// prints PASS/FAIL with a readable transcript. It is how you check a REAL model drives the
// agent loop correctly on your own machine.

type stScenario struct {
	name  string
	setup func(dir, py string) (goal string, err error)
	check func(dir, answer string) (ok bool, why string)
}

func findPython() string {
	for _, c := range []string{"python3", "python", "py"} {
		if p, err := exec.LookPath(c); err == nil {
			// Windows Store stub "python.exe" opens the Store; make sure it really runs
			if out, err := exec.Command(p, "-c", "print(42)").Output(); err == nil && strings.TrimSpace(string(out)) == "42" {
				return c
			}
		}
	}
	return ""
}

func write(dir, rel, content string) error {
	p := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	return os.WriteFile(p, []byte(content), 0o644)
}

func stScenarios() []stScenario {
	return []stScenario{
		{
			name: "1. Mở tệp theo tên gần đúng rồi trả lời",
			setup: func(dir, py string) (string, error) {
				if err := write(dir, "họp/Biên bản họp tuần.txt", "Cuộc họp tuần diễn ra ngày 15/03/2026 tại phòng B2.\nNgười ghi biên bản: Lan.\n"); err != nil {
					return "", err
				}
				if err := write(dir, "họp/Biên bản họp quý.txt", "Cuộc họp quý diễn ra ngày 01/04/2026 tại phòng A1.\n"); err != nil {
					return "", err
				}
				return "Mở giúp tôi file bien ban hop tuan (tôi nhớ không chính xác tên) và cho biết cuộc họp đó diễn ra ngày nào, ở đâu.", nil
			},
			check: func(dir, ans string) (bool, string) {
				if !strings.Contains(ans, "15/03/2026") {
					return false, "câu trả lời không có ngày 15/03/2026 (đọc nhầm tệp hoặc bịa)"
				}
				if strings.Contains(ans, "01/04/2026") {
					return false, "lẫn với biên bản họp quý"
				}
				return true, ""
			},
		},
		{
			name: "2. Tìm trong nhiều tệp rồi ghi báo cáo",
			setup: func(dir, py string) (string, error) {
				for f, c := range map[string]string{
					"a.py": "# TODO: sửa lỗi A\nprint(1)\n# TODO: viết test\n",
					"b.py": "x = 2  # TODO: tối ưu\n",
					"c.py": "print('xong')\n",
				} {
					if err := write(dir, f, c); err != nil {
						return "", err
					}
				}
				return "Tìm các dòng TODO trong các file .py của dự án, rồi tạo file bao_cao.md liệt kê mỗi file có bao nhiêu TODO và tổng cộng bao nhiêu. Sau đó cho tôi biết tổng.", nil
			},
			check: func(dir, ans string) (bool, string) {
				b, err := os.ReadFile(filepath.Join(dir, "bao_cao.md"))
				if err != nil {
					return false, "chưa tạo bao_cao.md"
				}
				s := string(b)
				if !strings.Contains(s, "a.py") || !strings.Contains(s, "b.py") || !strings.Contains(s, "3") {
					return false, "bao_cao.md thiếu a.py/b.py hoặc tổng 3: " + clip(s, 150)
				}
				if !strings.Contains(ans, "3") {
					return false, "câu trả lời không nêu tổng 3"
				}
				return true, ""
			},
		},
		{
			name: "3. Viết code rồi chạy thật, báo kết quả",
			setup: func(dir, py string) (string, error) {
				if py == "" {
					return "", fmt.Errorf("SKIP: máy chưa có Python")
				}
				if err := write(dir, "data.txt", "10\n20\n\n30\n"); err != nil {
					return "", err
				}
				return fmt.Sprintf("File data.txt có mỗi dòng một số (có cả dòng trống). Hãy viết script Python tên tong.py tính tổng các số, rồi chạy nó bằng lệnh `%s tong.py` và cho tôi biết kết quả.", py), nil
			},
			check: func(dir, ans string) (bool, string) {
				if _, err := os.Stat(filepath.Join(dir, "tong.py")); err != nil {
					return false, "chưa tạo tong.py"
				}
				if !strings.Contains(ans, "60") {
					return false, "câu trả lời không có kết quả 60 (không chạy thật hoặc bịa)"
				}
				return true, ""
			},
		},
	}
}

// runSelfTest returns the number of failed scenarios (0 = all good).
func runSelfTest(cfg Config, w io.Writer) int {
	if strings.TrimSpace(cfg.BaseURL) == "" {
		fmt.Fprintln(w, "Chưa cấu hình LLM: hãy điền base_url (và model) trong config.json rồi chạy lại --selftest.")
		return 2
	}
	defaultCmdTimeout = cfg.CmdTimeout
	readChunkChars = min(8000, max(1500, cfg.ContextTokens/5))
	cfg.AutoApprove = true
	py := findPython()
	fmt.Fprintf(w, "== Tự kiểm tra agent với LLM thật ==\nmodel=%s  base_url=%s  tool_mode=%s  context=%d\n\n", cfg.Model, cfg.BaseURL, cfg.ToolMode, cfg.ContextTokens)
	failed, passed, skipped := 0, 0, 0
	for _, sc := range stScenarios() {
		dir, _ := os.MkdirTemp("", "agent-selftest-")
		goal, err := sc.setup(dir, py)
		fmt.Fprintf(w, "--- %s\n", sc.name)
		if err != nil {
			fmt.Fprintf(w, "   bỏ qua: %v\n\n", err)
			skipped++
			os.RemoveAll(dir)
			continue
		}
		fmt.Fprintf(w, "   Yêu cầu: %s\n", goal)
		s := &Server{cfg: cfg, pending: map[string]chan bool{}, project: dir}
		var answer strings.Builder
		var errs []string
		steps := 0
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		s.runAgent(ctx, goal, func(ev string, data any) {
			switch ev {
			case "tool_call":
				m := data.(map[string]any)
				steps++
				fmt.Fprintf(w, "   → bước %d: %s %s\n", steps, m["name"], clip(strings.ReplaceAll(fmt.Sprint(m["args"]), "\n", " "), 140))
			case "tool_result":
				fmt.Fprintf(w, "     ← %s\n", clip(strings.ReplaceAll(fmt.Sprint(data.(map[string]any)["result"]), "\n", " ⏎ "), 140))
			case "assistant":
				answer.WriteString(fmt.Sprint(data) + "\n")
				fmt.Fprintf(w, "   💬 %s\n", clip(strings.ReplaceAll(fmt.Sprint(data), "\n", " "), 200))
			case "error":
				errs = append(errs, fmt.Sprint(data))
				fmt.Fprintf(w, "   ⚠ %s\n", fmt.Sprint(data))
			}
		})
		cancel()
		ok, why := sc.check(dir, answer.String())
		if len(errs) > 0 && ok {
			ok, why = false, "agent báo lỗi: "+errs[0]
		}
		if steps == 0 && ok {
			ok, why = false, "model không gọi công cụ nào"
		}
		if ok {
			passed++
			fmt.Fprintf(w, "   ✅ ĐẠT (%d bước, %.0fs)\n\n", steps, time.Since(start).Seconds())
		} else {
			failed++
			fmt.Fprintf(w, "   ❌ KHÔNG ĐẠT (%d bước, %.0fs): %s\n\n", steps, time.Since(start).Seconds(), why)
		}
		os.RemoveAll(dir)
	}
	fmt.Fprintf(w, "== KẾT QUẢ: %d đạt, %d không đạt, %d bỏ qua ==\n", passed, failed, skipped)
	if failed > 0 {
		fmt.Fprintln(w, "Gợi ý: model quá nhỏ thường (a) không viết đúng khối ```tool → thử model lớn hơn hoặc tool_mode \"native\"/\"auto\"; (b) trả lời trước khi chạy công cụ; (c) bịa kết quả. Xem lại bản ghi từng bước ở trên.")
		return failed
	}
	return 0
}
