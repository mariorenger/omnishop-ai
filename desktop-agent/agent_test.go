package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// fakeLLM is an OpenAI-compatible /chat/completions server driven by a script.
type fakeLLM struct {
	t     *testing.T
	mu    sync.Mutex
	reqs  []map[string]any
	reply func(n int, req map[string]any) (status int, body any)
	srv   *httptest.Server
}

func newFakeLLM(t *testing.T, reply func(n int, req map[string]any) (int, any)) *fakeLLM {
	f := &fakeLLM{t: t, reply: reply}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		f.mu.Lock()
		f.reqs = append(f.reqs, req)
		n := len(f.reqs)
		f.mu.Unlock()
		st, body := f.reply(n, req)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(st)
		if s, ok := body.(string); ok {
			fmt.Fprint(w, s)
			return
		}
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func chat(content string, calls ...map[string]any) map[string]any {
	m := map[string]any{"role": "assistant", "content": content}
	if len(calls) > 0 {
		m["tool_calls"] = calls
	}
	return map[string]any{"choices": []any{map[string]any{"message": m}}}
}
func nativeCall(id, name, args string) map[string]any {
	return map[string]any{"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": args}}
}
func toolBlock(name, args string) string {
	return "Mình sẽ làm.\n```tool\n{\"name\":\"" + name + "\",\"args\":" + args + "}\n```"
}

type agentRun struct {
	said, errs, calls []string
	results           []string
}

func (s *Server) drive(msg string) agentRun {
	var r agentRun
	s.runAgent(context.Background(), msg, func(ev string, data any) {
		switch ev {
		case "assistant":
			r.said = append(r.said, fmt.Sprint(data))
		case "error":
			r.errs = append(r.errs, fmt.Sprint(data))
		case "tool_call":
			r.calls = append(r.calls, data.(map[string]any)["name"].(string))
		case "tool_result":
			r.results = append(r.results, data.(map[string]any)["result"].(string))
		}
	})
	return r
}

func newLLMServer(t *testing.T, f *fakeLLM, mode string) (*Server, string) {
	d := newProj(t)
	s := &Server{cfg: Config{BaseURL: f.srv.URL, Model: "m", ToolMode: mode, ContextTokens: 16000, MaxOutputTokens: 1500, AutoApprove: true, CmdTimeout: 60},
		pending: map[string]chan bool{}, project: d}
	return s, d
}

func lastUserText(req map[string]any) string {
	ms := req["messages"].([]any)
	return ms[len(ms)-1].(map[string]any)["content"].(string)
}

// ---- text protocol (works with any chat model, no function calling needed)
func TestAgentTextProtocolRunsRealCommand(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		if _, has := req["tools"]; has {
			t.Error("text mode must not send native tools")
		}
		switch n {
		case 1:
			return 200, chat(toolBlock("write_file", `{"path":"hello.py","content":"print(6*7)"}`))
		case 2:
			return 200, chat(toolBlock("run_command", `{"command":"echo THAT-IS-REAL"}`))
		default:
			if !strings.Contains(lastUserText(req), "THAT-IS-REAL") {
				t.Errorf("model must receive the real command output, got %q", lastUserText(req))
			}
			return 200, chat("Đã chạy xong, kết quả là THAT-IS-REAL.")
		}
	})
	s, d := newLLMServer(t, f, "text")
	r := s.drive("viết và chạy file python in 6*7")
	if strings.Join(r.calls, ",") != "write_file,run_command" {
		t.Fatalf("calls: %v errs=%v", r.calls, r.errs)
	}
	if b, _ := os.ReadFile(filepath.Join(d, "hello.py")); string(b) != "print(6*7)" {
		t.Errorf("file not written: %q", b)
	}
	wantHas(t, strings.Join(r.results, "\n"), "THAT-IS-REAL")
	wantHas(t, strings.Join(r.said, "\n"), "Đã chạy xong")
	// the system prompt lists the tools and the protocol
	sys := f.reqs[0]["messages"].([]any)[0].(map[string]any)["content"].(string)
	for _, w := range []string{"```tool", "run_command", "read_file", "create_task", "offset"} {
		wantHas(t, sys, w)
	}
}

func TestAgentToolSyntaxVariants(t *testing.T) {
	for name, text := range map[string]string{
		"fenced":      toolBlock("run_command", `{"command":"echo V"}`),
		"json fence":  "```json\n{\"name\":\"run_command\",\"arguments\":{\"command\":\"echo V\"}}\n```",
		"flat args":   "```tool\n{\"name\":\"run_command\",\"command\":\"echo V\"}\n```",
		"alias bash":  "```tool\n{\"name\":\"bash\",\"args\":{\"cmd\":\"echo V\"}}\n```",
		"string args": "```tool\n{\"name\":\"run_command\",\"arguments\":\"{\\\"command\\\":\\\"echo V\\\"}\"}\n```",
		"xml":         "<tool_call>{\"name\":\"run_command\",\"arguments\":{\"command\":\"echo V\"}}</tool_call>",
		"bare json":   "Tôi sẽ chạy: {\"name\": \"run_command\", \"args\": {\"command\": \"echo V\"}}",
		"nested fn":   "```tool\n{\"function\":{\"name\":\"run_command\",\"arguments\":{\"command\":\"echo V\"}}}\n```",
	} {
		t.Run(name, func(t *testing.T) {
			f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
				if n == 1 {
					return 200, chat(text)
				}
				return 200, chat("xong")
			})
			s, _ := newLLMServer(t, f, "text")
			r := s.drive("chạy thử")
			if len(r.calls) != 1 || r.calls[0] != "run_command" || !strings.Contains(strings.Join(r.results, ""), "V") {
				t.Fatalf("not executed: calls=%v results=%v errs=%v", r.calls, r.results, r.errs)
			}
		})
	}
	// plain answers that merely mention JSON are NOT executed
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		return 200, chat(`Ví dụ cấu hình: {"name": "demo", "port": 80}`)
	})
	s, _ := newLLMServer(t, f, "text")
	if r := s.drive("cho ví dụ json"); len(r.calls) != 0 || len(r.said) != 1 {
		t.Errorf("non-tool JSON must be treated as an answer: %+v", r)
	}
}

func TestAgentBrokenBlockGetsOneRetry(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		if n == 1 {
			return 200, chat("```tool\n{name: run_command, args: oops\n```")
		}
		if n == 2 {
			if !strings.Contains(lastUserText(req), "không phải JSON") {
				t.Errorf("model should be told the block was invalid: %q", lastUserText(req))
			}
			return 200, chat(toolBlock("run_command", `{"command":"echo FIXED"}`))
		}
		return 200, chat("ok FIXED")
	})
	s, _ := newLLMServer(t, f, "text")
	r := s.drive("x")
	if len(r.calls) != 1 || !strings.Contains(strings.Join(r.said, " "), "ok FIXED") {
		t.Fatalf("%+v", r)
	}
}

// ---- native function calling
func TestAgentNativeToolCalls(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		if _, has := req["tools"]; !has {
			t.Error("native mode must send tools")
		}
		if n == 1 {
			return 200, chat("", nativeCall("c1", "run_command", `{"command":"echo NATIVE"}`))
		}
		ms := req["messages"].([]any)
		last := ms[len(ms)-1].(map[string]any)
		if last["role"] != "tool" || last["tool_call_id"] != "c1" || !strings.Contains(last["content"].(string), "NATIVE") {
			t.Errorf("tool result not fed back correctly: %+v", last)
		}
		return 200, chat("xong NATIVE")
	})
	s, _ := newLLMServer(t, f, "native")
	r := s.drive("chạy")
	if len(r.calls) != 1 || !strings.Contains(strings.Join(r.said, ""), "xong NATIVE") {
		t.Fatalf("%+v", r)
	}
}

func TestAgentAutoFallsBackToText(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		if _, has := req["tools"]; has {
			return 400, `{"error":{"message":"this model does not support tools"}}`
		}
		if n == 2 {
			return 200, chat(toolBlock("run_command", `{"command":"echo FALLBACK"}`))
		}
		return 200, chat("xong")
	})
	s, _ := newLLMServer(t, f, "auto")
	r := s.drive("chạy")
	if len(r.calls) != 1 || len(r.errs) != 0 {
		t.Fatalf("auto mode should recover: %+v", r)
	}
	if !s.forceText {
		t.Error("should remember to use text mode")
	}
}

func TestAgentDefaultsToTextMode(t *testing.T) {
	s := &Server{cfg: Config{}}
	if s.toolMode() != "text" {
		t.Error("default must be the text protocol (works with every model)")
	}
}

// ---- safety rails
func TestAgentLoopGuardAndStepLimit(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		return 200, chat(toolBlock("run_command", `{"command":"echo same"}`))
	})
	s, _ := newLLMServer(t, f, "text")
	r := s.drive("x")
	if len(r.errs) != 1 || !strings.Contains(r.errs[0], "lặp") || len(r.calls) > 4 {
		t.Fatalf("repeating the same call must stop quickly: %+v", r)
	}
	n := 0
	f2 := newFakeLLM(t, func(_ int, req map[string]any) (int, any) {
		n++
		return 200, chat(toolBlock("run_command", fmt.Sprintf(`{"command":"echo step%d"}`, n)))
	})
	s2, _ := newLLMServer(t, f2, "text")
	r = s2.drive("x")
	if len(r.calls) != maxAgentSteps || len(r.errs) != 1 || !strings.Contains(r.errs[0], "quá nhiều bước") {
		t.Fatalf("step limit: calls=%d errs=%v", len(r.calls), r.errs)
	}
}

func TestAgentDenyIsReportedToModel(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		if n == 1 {
			return 200, chat(toolBlock("write_file", `{"path":"no.txt","content":"x"}`))
		}
		if !strings.Contains(lastUserText(req), "denied") {
			t.Errorf("model must learn the user said no: %q", lastUserText(req))
		}
		return 200, chat("Ok, mình không ghi.")
	})
	s, d := newLLMServer(t, f, "text")
	s.cfg.AutoApprove = false
	defer denyAll(s)()
	r := s.drive("ghi file")
	if _, err := os.Stat(filepath.Join(d, "no.txt")); err == nil {
		t.Error("denied write happened")
	}
	if len(r.errs) != 0 {
		t.Errorf("%+v", r)
	}
}

func TestAgentServerErrorsAreShown(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) { return 500, `{"error":"boom"}` })
	s, _ := newLLMServer(t, f, "text")
	if r := s.drive("x"); len(r.errs) != 1 || !strings.Contains(r.errs[0], "500") {
		t.Errorf("%+v", r)
	}
	f2 := newFakeLLM(t, func(n int, req map[string]any) (int, any) { return 200, `not json at all` })
	s2, _ := newLLMServer(t, f2, "text")
	if r := s2.drive("x"); len(r.errs) != 1 {
		t.Errorf("%+v", r)
	}
}

// ---- the 16k limit
func reqTokens(req map[string]any) int {
	n := 0
	for _, m := range req["messages"].([]any) {
		n += estTokens(m.(map[string]any)["content"].(string)) + 6
	}
	return n
}

func TestAgentStaysWithin16kAcrossLongConversation(t *testing.T) {
	big := strings.Repeat("Dòng dữ liệu rất dài với tiếng Việt có dấu để chiếm nhiều token. ", 400) // ~26k chars
	maxSeen := 0
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		if tk := reqTokens(req); tk > maxSeen {
			maxSeen = tk
		}
		if req["max_tokens"] != float64(1500) {
			t.Errorf("max_tokens not sent: %v", req["max_tokens"])
		}
		return 200, chat("ok " + big[:3000]) // long answers too
	})
	s, _ := newLLMServer(t, f, "text")
	d := s.project
	mustWrite(t, filepath.Join(d, "huge.txt"), big+big+big)
	for i := 0; i < 25; i++ {
		s.drive(fmt.Sprintf("câu hỏi số %d: %s", i, big[:2000]))
	}
	budget := s.inputBudget()
	if maxSeen > budget {
		t.Fatalf("request used ~%d tokens, budget %d (context 16000)", maxSeen, budget)
	}
	if len(s.history) > 130 {
		t.Errorf("history not compacted: %d", len(s.history))
	}
	// the newest user message must always be sent
	last := f.reqs[len(f.reqs)-1]["messages"].([]any)
	wantHas(t, last[len(last)-1].(map[string]any)["content"].(string), "câu hỏi số 24")
}

func TestAgentHugeToolOutputIsClipped(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		if n == 1 {
			return 200, chat(toolBlock("run_command", `{"command":"echo go"}`))
		}
		if tk := reqTokens(req); tk > 16000-1500 {
			t.Errorf("too big: %d", tk)
		}
		return 200, chat("xong")
	})
	s, d := newLLMServer(t, f, "text")
	mustWrite(t, filepath.Join(d, "huge.txt"), strings.Repeat("x", 2_000_000))
	old := readChunkChars
	readChunkChars = 3200
	defer func() { readChunkChars = old }()
	f.reply = func(n int, req map[string]any) (int, any) {
		if n == 1 {
			return 200, chat(toolBlock("read_file", `{"path":"huge.txt"}`))
		}
		last := lastUserText(req)
		if len(last) > 4500 {
			t.Errorf("tool result sent to the model is %d chars, must be clipped", len(last))
		}
		wantHas(t, last, "offset=")
		return 200, chat("đã đọc phần đầu")
	}
	if r := s.drive("đọc file lớn"); len(r.errs) != 0 {
		t.Fatalf("%+v", r)
	}
}

func TestAgentRetriesOnContextOverflow(t *testing.T) {
	var sizes []int
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		sizes = append(sizes, reqTokens(req))
		if n == 1 { // the server is stricter than we assumed
			return 400, `{"error":{"message":"This model's maximum context length is 8192 tokens, however you requested 9000 tokens"}}`
		}
		return 200, chat("ok sau khi cắt")
	})
	s, _ := newLLMServer(t, f, "text")
	for i := 0; i < 6; i++ {
		s.history = append(s.history, Msg{Role: "user", Content: strings.Repeat("lịch sử ", 800)}, Msg{Role: "assistant", Content: strings.Repeat("trả lời ", 800)})
	}
	r := s.drive("câu mới")
	if len(r.errs) != 0 || len(sizes) != 2 || sizes[1] >= sizes[0] {
		t.Fatalf("should retry with a smaller prompt: sizes=%v %+v", sizes, r)
	}
}

func TestFitContext(t *testing.T) {
	var h []Msg
	for i := 0; i < 50; i++ {
		h = append(h, Msg{Role: "user", Content: fmt.Sprintf("hỏi %d ", i) + strings.Repeat("a", 3000)},
			Msg{Role: "assistant", Content: "gọi"}, Msg{Role: "user", Meta: "result", Content: strings.Repeat("kết quả ", 2000)},
			Msg{Role: "assistant", Content: fmt.Sprintf("đáp %d", i)})
	}
	out := fitContext("SYS", h, 6000)
	if msgsTokens(out) > 6000 {
		t.Errorf("over budget: %d", msgsTokens(out))
	}
	if out[0].Role != "system" || out[len(out)-1].Content != "đáp 49" {
		t.Errorf("must keep system first and newest last: %q", out[len(out)-1].Content)
	}
	if !startsTurn(out[1]) {
		t.Errorf("must not start mid-turn: %+v", out[1])
	}
	// a single enormous newest turn is shrunk, not dropped
	huge := []Msg{{Role: "user", Content: "làm đi"}, {Role: "assistant", Content: "gọi"},
		{Role: "tool", ToolCallID: "1", Content: strings.Repeat("z", 400000)}}
	out = fitContext("SYS", huge, 3000)
	if msgsTokens(out) > 3000 || out[1].Content != "làm đi" {
		t.Errorf("huge turn: tokens=%d", msgsTokens(out))
	}
	// native mode: never leave a tool message without its assistant tool_calls
	var nh []Msg
	for i := 0; i < 20; i++ {
		tc := ToolCall{ID: fmt.Sprint(i), Type: "function"}
		tc.Function.Name, tc.Function.Arguments = "run_command", `{"command":"x"}`
		nh = append(nh, Msg{Role: "user", Content: fmt.Sprint("q", i)}, Msg{Role: "assistant", ToolCalls: []ToolCall{tc}},
			Msg{Role: "tool", ToolCallID: fmt.Sprint(i), Content: strings.Repeat("r", 4000)}, Msg{Role: "assistant", Content: "a"})
	}
	out = fitContext("SYS", nh, 2500)
	for i, m := range out {
		if m.Role == "tool" && (i == 0 || out[i-1].Role != "assistant" || len(out[i-1].ToolCalls) == 0) {
			t.Errorf("orphan tool message at %d", i)
		}
	}
}

func TestEstTokensIsConservative(t *testing.T) {
	vi := "Hôm nay trời đẹp, chúng tôi đi dạo ở công viên và uống cà phê cùng nhau."
	if estTokens(vi) < len([]rune(vi))/3 {
		t.Errorf("underestimates Vietnamese: %d for %d chars", estTokens(vi), len([]rune(vi)))
	}
	if estTokens("") != 0 {
		t.Error("empty")
	}
}

// The model asks for a slightly wrong name; the tool answers with "did you mean", the model retries.
func TestAgentRecoversFromWrongFileName(t *testing.T) {
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) {
		switch n {
		case 1:
			return 200, chat(toolBlock("read_file", `{"path":"abd_notes.txt"}`)) // typo of docs/abc_notes.txt
		case 2:
			last := lastUserText(req)
			if !strings.Contains(last, "docs/abc_notes.txt") || !strings.Contains(last, "gần giống") {
				t.Errorf("model must be shown the closest names, got %q", last)
			}
			return 200, chat(toolBlock("read_file", `{"path":"docs/abc_notes.txt"}`))
		default:
			wantHas(t, lastUserText(req), "hello notes")
			return 200, chat("Nội dung tệp là: hello notes")
		}
	})
	s, d := newLLMServer(t, f, "text")
	must(t, os.MkdirAll(filepath.Join(d, "docs"), 0o755))
	mustWrite(t, filepath.Join(d, "docs", "abc_notes.txt"), "hello notes")
	r := s.drive("đọc giúp tôi file abd_notes")
	if len(r.errs) != 0 || len(r.calls) != 2 || !strings.Contains(strings.Join(r.said, " "), "hello notes") {
		t.Fatalf("%+v", r)
	}
	// the system prompt tells the model what to do when unsure about a name
	sys := f.reqs[0]["messages"].([]any)[0].(map[string]any)["content"].(string)
	for _, w := range []string{"find_files", "gần giống", "search_files"} {
		wantHas(t, sys, w)
	}
}
