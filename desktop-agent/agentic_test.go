package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// A *reactive* fake model: unlike the scripted fakes in agent_test.go it decides each next step
// from the REAL tool output it receives (like an actual LLM would), so these tests prove the loop
// keeps going, feeds results back, recovers from failures and stops when the goal is reached.

var (
	reResult = regexp.MustCompile("(?s)^Kết quả của `(\\w+)`:\\n(.*?)\\n\\(Tiếp tục")
	reDate   = regexp.MustCompile(`\d{2}/\d{2}/\d{4}`)
	rePhong  = regexp.MustCompile(`phòng \w+`)
	reNum    = regexp.MustCompile(`\d+`)
)

func goalOf(req map[string]any) string {
	ms := req["messages"].([]any)
	for i := len(ms) - 1; i >= 0; i-- {
		m := ms[i].(map[string]any)
		c, _ := m["content"].(string)
		if m["role"] == "user" && !strings.HasPrefix(c, "Kết quả của") && !strings.HasPrefix(c, "Khối tool") {
			return c
		}
	}
	return ""
}

func tool(name string, args map[string]any) (int, any) {
	b, _ := jsonMarshal(args)
	return 200, chat("```tool\n{\"name\":\"" + name + "\",\"args\":" + string(b) + "}\n```")
}

// solver returns the model policy. buggyFirst makes the first script it writes wrong.
func solver(buggyFirst bool) func(n int, req map[string]any) (int, any) {
	return func(n int, req map[string]any) (int, any) {
		goal := goalOf(req)
		last := lastUserText(req)
		m := reResult.FindStringSubmatch(last)
		name, out := "", ""
		if m != nil {
			name, out = m[1], m[2]
		}
		switch {
		case strings.Contains(goal, "bien ban hop tuan"): // ---- scenario 1
			switch name {
			case "":
				return tool("find_files", map[string]any{"pattern": "bien ban hop tuan"})
			case "find_files":
				for _, l := range strings.Split(out, "\n") {
					if strings.Contains(l, "tuần") && strings.HasSuffix(l, ".txt") {
						return tool("read_file", map[string]any{"path": strings.TrimSpace(l)})
					}
				}
				return 200, chat("Không tìm thấy tệp.")
			case "read_file":
				return 200, chat(fmt.Sprintf("Cuộc họp diễn ra ngày %s, %s.", reDate.FindString(out), rePhong.FindString(out)))
			}
		case strings.Contains(goal, "TODO"): // ---- scenario 2
			switch name {
			case "":
				return tool("search_files", map[string]any{"pattern": "TODO"})
			case "search_files":
				counts := map[string]int{}
				total := 0
				for _, l := range strings.Split(out, "\n") {
					if f, _, ok := strings.Cut(l, ":"); ok && strings.HasSuffix(f, ".py") {
						counts[f]++
						total++
					}
				}
				var sb strings.Builder
				for _, f := range []string{"a.py", "b.py", "c.py"} {
					if counts[f] > 0 {
						fmt.Fprintf(&sb, "- %s: %d\n", f, counts[f])
					}
				}
				fmt.Fprintf(&sb, "Tổng: %d\n", total)
				return tool("write_file", map[string]any{"path": "bao_cao.md", "content": sb.String()})
			case "write_file":
				return 200, chat("Đã tạo bao_cao.md. Tổng cộng có 3 TODO.")
			}
		case strings.Contains(goal, "tong.py"): // ---- scenario 3 (write code, run it, fix it if it fails)
			py := regexp.MustCompile("`(\\w+) tong.py`").FindStringSubmatch(goal)[1]
			good := "nums = [int(l) for l in open('data.txt') if l.strip()]\nprint(sum(nums))\n"
			bad := "print(sum(int(l) for l in open('data.txt')))\n" // crashes on the blank line
			switch name {
			case "":
				code := good
				if buggyFirst {
					code = bad
				}
				return tool("write_file", map[string]any{"path": "tong.py", "content": code})
			case "write_file":
				return tool("run_command", map[string]any{"command": py + " tong.py"})
			case "run_command":
				if strings.Contains(out, "Error") || strings.Contains(out, "exit status") { // saw a failure → fix and retry
					return tool("write_file", map[string]any{"path": "tong.py", "content": good})
				}
				nums := reNum.FindAllString(out, -1)
				return 200, chat("Kết quả tổng là " + nums[len(nums)-1])
			}
		}
		return 200, chat("Xin chào, mình có thể giúp gì?")
	}
}

func jsonMarshal(v any) ([]byte, error) { return jsonMarshalImpl(v) }

func TestSelfTestHarnessWithReactiveModel(t *testing.T) {
	if findPython() == "" {
		t.Skip("python not available")
	}
	f := newFakeLLM(t, solver(false))
	var out bytes.Buffer
	cfg := Config{BaseURL: f.srv.URL, Model: "fake", ToolMode: "text", ContextTokens: 16000, MaxOutputTokens: 1500, CmdTimeout: 60}
	if code := runSelfTest(cfg, &out); code != 0 {
		t.Fatalf("exit %d\n%s", code, out.String())
	}
	s := out.String()
	for _, w := range []string{"3 đạt, 0 không đạt", "find_files", "read_file", "search_files", "write_file", "run_command", "ĐẠT"} {
		wantHas(t, s, w)
	}
	if strings.Contains(s, "KHÔNG ĐẠT") {
		t.Errorf("%s", s)
	}
}

func TestSelfTestDetectsAModelThatCannotDoTheWork(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		return 200, chat("Xin chào! Mình sẽ cố gắng giúp bạn.")
	})
	var out bytes.Buffer
	cfg := Config{BaseURL: f.srv.URL, Model: "weak", ToolMode: "text", ContextTokens: 16000, MaxOutputTokens: 1500, CmdTimeout: 60}
	code := runSelfTest(cfg, &out)
	if code < 2 {
		t.Fatalf("a model that never uses tools must fail the self-test (exit %d)\n%s", code, out.String())
	}
	wantHas(t, out.String(), "KHÔNG ĐẠT")
	wantHas(t, out.String(), "Gợi ý")
	// and a model that hallucinates a result without running anything must also fail
	f2 := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		return 200, chat("Tổng là 60, cuộc họp ngày 15/03/2026, tổng 3 TODO.")
	})
	cfg.BaseURL = f2.srv.URL
	out.Reset()
	if code := runSelfTest(cfg, &out); code == 0 {
		t.Fatalf("hallucination without tool use must not pass:\n%s", out.String())
	}
	wantHas(t, out.String(), "không gọi công cụ nào")
	// no LLM configured
	out.Reset()
	if code := runSelfTest(Config{}, &out); code != 2 {
		t.Errorf("no base_url → exit 2, got %d", code)
	}
}

// ---- stand-alone multi-step goals, checked against the real filesystem
func TestAgentMultiStepUntilGoalReached(t *testing.T) {
	if findPython() == "" {
		t.Skip("python not available")
	}
	py := findPython()
	d := t.TempDir()
	mustWrite(t, filepath.Join(d, "data.txt"), "10\n20\n\n30\n")
	f := newFakeLLM(t, solver(true)) // first script is BUGGY: the loop must notice the traceback and fix it
	s := &Server{cfg: Config{BaseURL: f.srv.URL, Model: "m", ToolMode: "text", ContextTokens: 16000, MaxOutputTokens: 1500, AutoApprove: true, CmdTimeout: 60},
		pending: map[string]chan bool{}, project: d}
	r := s.drive(fmt.Sprintf("Viết script Python tong.py tính tổng các số trong data.txt rồi chạy bằng lệnh `%s tong.py` và cho biết kết quả.", py))

	if got := strings.Join(r.calls, ","); got != "write_file,run_command,write_file,run_command" {
		t.Fatalf("expected write → run (fails) → fix → run (works), got %s\nerrs=%v said=%v", got, r.errs, r.said)
	}
	if !strings.Contains(r.results[1], "ValueError") {
		t.Errorf("the first run must really fail with a traceback, got %q", r.results[1])
	}
	wantHas(t, r.results[3], "60")
	if len(r.errs) != 0 || len(r.said) == 0 || !strings.Contains(r.said[len(r.said)-1], "60") {
		t.Fatalf("final answer must report the real result: %+v", r)
	}
	if b, _ := os.ReadFile(filepath.Join(d, "tong.py")); !strings.Contains(string(b), "if l.strip()") {
		t.Errorf("script was not fixed on disk: %q", b)
	}
	// every request carried the REAL output of the previous step (not a canned one)
	var fedBack int
	for _, req := range f.reqs[1:] {
		if strings.HasPrefix(lastUserText(req), "Kết quả của") {
			fedBack++
		}
	}
	if fedBack != len(f.reqs)-1 {
		t.Errorf("%d of %d follow-up requests carried a tool result", fedBack, len(f.reqs)-1)
	}
}

func TestAgentExploresThenActs(t *testing.T) {
	d := t.TempDir()
	must(t, write(d, "họp/Biên bản họp tuần.txt", "Cuộc họp tuần diễn ra ngày 15/03/2026 tại phòng B2.\n"))
	must(t, write(d, "họp/Biên bản họp quý.txt", "Cuộc họp quý diễn ra ngày 01/04/2026 tại phòng A1.\n"))
	f := newFakeLLM(t, solver(false))
	s := &Server{cfg: Config{BaseURL: f.srv.URL, Model: "m", ToolMode: "text", ContextTokens: 16000, MaxOutputTokens: 1500, AutoApprove: true, CmdTimeout: 60},
		pending: map[string]chan bool{}, project: d}
	r := s.drive("Mở file bien ban hop tuan giúp tôi, cuộc họp ngày nào?")
	if strings.Join(r.calls, ",") != "find_files,read_file" {
		t.Fatalf("calls: %v errs=%v", r.calls, r.errs)
	}
	last := r.said[len(r.said)-1]
	wantHas(t, last, "15/03/2026")
	wantHas(t, last, "phòng B2")
	if strings.Contains(last, "01/04/2026") {
		t.Error("picked the wrong file")
	}
}
