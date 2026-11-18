package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---- helpers: an OpenAI-style SSE server driven by a function that writes chunks

type sseW struct {
	w http.ResponseWriter
	f http.Flusher
}

func (x sseW) data(v any) {
	b, _ := json.Marshal(v)
	fmt.Fprintf(x.w, "data: %s\n\n", b)
	x.f.Flush()
}
func (x sseW) raw(s string) { fmt.Fprint(x.w, s); x.f.Flush() }
func (x sseW) text(t string) {
	x.data(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": t}}}})
}
func (x sseW) done() { x.raw("data: [DONE]\n\n") }

func newSSE(t *testing.T, h func(n int, req map[string]any, w sseW)) (*httptest.Server, *[]map[string]any) {
	var mu sync.Mutex
	var reqs []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		mu.Lock()
		reqs = append(reqs, req)
		n := len(reqs)
		mu.Unlock()
		rw.Header().Set("Content-Type", "text/event-stream")
		rw.Header().Set("Cache-Control", "no-cache")
		h(n, req, sseW{rw, rw.(http.Flusher)})
	}))
	t.Cleanup(srv.Close)
	return srv, &reqs
}

func streamServer(t *testing.T, url string, mode string) (*Server, string) {
	d := newProj(t)
	return &Server{cfg: Config{BaseURL: url, Model: "m", ToolMode: mode, ContextTokens: 16000, MaxOutputTokens: 1500, AutoApprove: true, CmdTimeout: 60, Stream: true},
		pending: map[string]chan bool{}, project: d}, d
}

type rawEv struct{ ev, data string }

func (s *Server) driveRaw(ctx context.Context, msg string) []rawEv {
	var out []rawEv
	s.runAgent(ctx, msg, func(ev string, data any) {
		switch v := data.(type) {
		case string:
			out = append(out, rawEv{ev, v})
		case map[string]any:
			b, _ := json.Marshal(v)
			out = append(out, rawEv{ev, string(b)})
		default:
			out = append(out, rawEv{ev, fmt.Sprint(data)})
		}
	})
	return out
}

func deltasOf(evs []rawEv) string {
	var sb strings.Builder
	for _, e := range evs {
		if e.ev == "delta" {
			sb.WriteString(e.data)
		}
	}
	return sb.String()
}
func lastOf(evs []rawEv, ev string) string {
	for i := len(evs) - 1; i >= 0; i-- {
		if evs[i].ev == ev {
			return evs[i].data
		}
	}
	return ""
}
func countOf(evs []rawEv, ev string) int {
	n := 0
	for _, e := range evs {
		if e.ev == ev {
			n++
		}
	}
	return n
}

// ---- parser

func TestParseSSE(t *testing.T) {
	in := ": keep-alive\n\n" +
		`data: {"choices":[{"delta":{"role":"assistant"}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"content":"Xin "}}]}` + "\n\n" +
		"event: ping\n\n" +
		`data: {"choices":[{"delta":{"content":"chào "}}]}` + "\n\n" +
		"data: this is not json\n\n" +
		`data: {"choices":[{"delta":{"content":[{"type":"text","text":"thế giới"}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"run_command","arguments":""}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"run_command","arguments":"{\"comm"}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"echo hi\"}"}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}}]}}]}` + "\n\n" +
		`data: {"choices":[],"usage":{"total_tokens":9}}` + "\n\n" +
		"data: [DONE]\n\n" +
		`data: {"choices":[{"delta":{"content":"AFTER DONE"}}]}` + "\n\n"
	var got []string
	m, err := parseSSE(strings.NewReader(in), func(d string) { got = append(got, d) }, nil)
	if err != nil {
		t.Fatal(err)
	}
	if m.Content != "Xin chào thế giới" || strings.Join(got, "|") != "Xin |chào |thế giới" {
		t.Errorf("content %q deltas %v", m.Content, got)
	}
	if len(m.ToolCalls) != 2 || m.ToolCalls[0].ID != "c1" || m.ToolCalls[0].Function.Name != "run_command" ||
		m.ToolCalls[0].Function.Arguments != `{"command":"echo hi"}` || m.ToolCalls[1].Function.Name != "read_file" {
		t.Errorf("tool calls merged wrongly: %+v", m.ToolCalls)
	}
	if strings.Contains(m.Content, "AFTER") {
		t.Error("must stop at [DONE]")
	}
	// legacy completion-style chunks
	m, _ = parseSSE(strings.NewReader(`data: {"choices":[{"text":"cũ"}]}`+"\n\ndata: [DONE]\n"), nil, nil)
	if m.Content != "cũ" {
		t.Errorf("legacy text: %q", m.Content)
	}
	// error chunk keeps what was received
	m, err = parseSSE(strings.NewReader(`data: {"choices":[{"delta":{"content":"được một nửa"}}]}`+"\n\n"+`data: {"error":{"message":"overloaded"}}`+"\n\n"), nil, nil)
	if err == nil || !strings.Contains(err.Error(), "overloaded") || m.Content != "được một nửa" {
		t.Errorf("error chunk: %v %q", err, m.Content)
	}
	// a huge single line must not break the scanner
	big := strings.Repeat("a", 500_000)
	m, err = parseSSE(strings.NewReader(`data: {"choices":[{"delta":{"content":"`+big+`"}}]}`+"\n\n"), nil, nil)
	if err != nil || len(m.Content) != len(big) {
		t.Errorf("big line: %v %d", err, len(m.Content))
	}
}

// ---- the filter that keeps raw tool blocks off the screen

func TestStreamFilter(t *testing.T) {
	feedAll := func(chunks ...string) (emitted string, full string) {
		var sb strings.Builder
		f := &streamFilter{emit: func(d string) { sb.WriteString(d) }}
		var all strings.Builder
		for _, c := range chunks {
			all.WriteString(c)
			f.feed(c)
		}
		return sb.String(), all.String()
	}
	// a tool block is never shown, however the text is split into chunks
	text := "Mình sẽ xem thư mục.\n```tool\n{\"name\":\"list_dir\",\"args\":{\"path\":\".\"}}\n```"
	for size := 1; size <= 9; size++ {
		var chunks []string
		for i := 0; i < len(text); {
			j := i + size
			for j < len(text) && j > i && !isRuneStart(text[j]) {
				j++
			}
			if j > len(text) {
				j = len(text)
			}
			chunks = append(chunks, text[i:j])
			i = j
		}
		em, _ := feedAll(chunks...)
		if em != "Mình sẽ xem thư mục.\n" {
			t.Errorf("chunk size %d: emitted %q", size, em)
		}
	}
	for _, blk := range []string{"<tool_call>{\"name\":\"x\"}</tool_call>", `{"name":"run_command","args":{}}`, "```json\n{\"name\":\"x\"}\n```"} {
		em, _ := feedAll("Xong. ", blk)
		if strings.Contains(em, "name") || strings.Contains(em, "tool_call") || strings.Contains(em, "```") {
			t.Errorf("leaked %q from %q", em, blk)
		}
	}
	// plain text: everything is forwarded (only a short tail may be held back)
	plain := "Tổng số dòng là 60; dùng `ls` để xem, ví dụ {a} và ``` không phải tool."
	em, full := feedAll(plain[:10], plain[10:25], plain[25:])
	if !strings.HasPrefix(full, em) || len(full)-len(em) > 11 {
		t.Errorf("plain text held back too much: %q of %q", em, full)
	}
	// emitted text is always a prefix of the full text (monotonic, no duplication)
	var sb strings.Builder
	f := &streamFilter{emit: func(d string) { sb.WriteString(d) }}
	acc := ""
	for _, c := range []string{"a`", "``", "tool", "\n{", "x"} {
		acc += c
		f.feed(c)
		if !strings.HasPrefix(acc, sb.String()) {
			t.Fatalf("emitted %q is not a prefix of %q", sb.String(), acc)
		}
	}
}

func isRuneStart(b byte) bool { return b&0xC0 != 0x80 }

// ---- end to end through the agent loop

func TestAgentStreamsWhileModelIsStillWriting(t *testing.T) {
	release := make(chan struct{})
	srv, reqs := newSSE(t, func(n int, req map[string]any, w sseW) {
		if req["stream"] != true {
			t.Error("request must ask for stream:true")
		}
		w.text("Xin ")
		w.text("chào, ")
		<-release // the model "keeps thinking": nothing else is sent until the test says so
		w.text("mình giúp được gì?")
		w.done()
	})
	s, _ := streamServer(t, srv.URL, "text")
	firstDelta := make(chan struct{}, 1)
	var mu sync.Mutex
	var evs []rawEv
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		s.runAgent(context.Background(), "chào", func(ev string, data any) {
			mu.Lock()
			evs = append(evs, rawEv{ev, fmt.Sprint(data)})
			mu.Unlock()
			if ev == "delta" {
				select {
				case firstDelta <- struct{}{}:
				default:
				}
			}
		})
	}()
	select {
	case <-firstDelta: // arrived although the server has not finished → really streamed, not buffered
	case <-time.After(3 * time.Second):
		t.Fatal("no text arrived while the model was still writing: the response is buffered, not streamed")
	}
	mu.Lock()
	if countOf(evs, "assistant") != 0 {
		t.Error("the final message must come AFTER the deltas")
	}
	mu.Unlock()
	close(release)
	<-finished
	if got := deltasOf(evs); got != "Xin chào, mình giúp được gì?" {
		t.Errorf("deltas joined: %q", got)
	}
	if lastOf(evs, "assistant") != "Xin chào, mình giúp được gì?" {
		t.Errorf("final assistant: %q", lastOf(evs, "assistant"))
	}
	if len(*reqs) != 1 {
		t.Errorf("one LLM call expected, got %d", len(*reqs))
	}
	if evs[len(evs)-1].ev != "assistant" {
		t.Errorf("order: %+v", evs)
	}
}

func TestAgentStreamHidesToolBlockAndRunsTool(t *testing.T) {
	block := "Mình sẽ chạy lệnh.\n```tool\n{\"name\":\"run_command\",\"args\":{\"command\":\"echo STREAM-OK\"}}\n```"
	srv, _ := newSSE(t, func(n int, req map[string]any, w sseW) {
		if n == 1 {
			for i := 0; i < len(block); { // 4 bytes at a time, never splitting a UTF-8 character
				j := min(i+4, len(block))
				for j < len(block) && !isRuneStart(block[j]) {
					j++
				}
				w.text(block[i:j])
				i = j
			}
		} else {
			w.text("Đã chạy xong: STREAM-OK")
		}
		w.done()
	})
	s, _ := streamServer(t, srv.URL, "text")
	evs := s.driveRaw(context.Background(), "chạy thử")
	shown := deltasOf(evs)
	for _, bad := range []string{"```", "tool", `"name"`, "run_command"} {
		if strings.Contains(shown, bad) {
			t.Errorf("raw tool block leaked to the screen (%q) in %q", bad, shown)
		}
	}
	if !strings.HasPrefix(shown, "Mình sẽ chạy lệnh") {
		t.Errorf("text before the block must stream: %q", shown)
	}
	var callNames []string
	var assistants []string
	for _, e := range evs {
		if e.ev == "tool_call" {
			callNames = append(callNames, "tool")
		}
		if e.ev == "assistant" {
			assistants = append(assistants, e.data)
		}
	}
	if len(callNames) != 1 || !strings.Contains(texts2(evs), "STREAM-OK") {
		t.Fatalf("tool must still run: %+v", evs)
	}
	if len(assistants) != 2 || assistants[0] != "Mình sẽ chạy lệnh." || !strings.Contains(assistants[1], "STREAM-OK") {
		t.Errorf("final messages: %q", assistants)
	}
	// ordering: first bubble is closed (assistant) BEFORE the tool card appears
	idxA, idxT := -1, -1
	for i, e := range evs {
		if e.ev == "assistant" && idxA < 0 {
			idxA = i
		}
		if e.ev == "tool_call" && idxT < 0 {
			idxT = i
		}
	}
	if !(idxA >= 0 && idxA < idxT) {
		t.Errorf("assistant must precede tool_call: %d %d", idxA, idxT)
	}
}

func texts2(evs []rawEv) string {
	var sb strings.Builder
	for _, e := range evs {
		sb.WriteString(e.data + "\n")
	}
	return sb.String()
}

func TestAgentStreamNativeToolCalls(t *testing.T) {
	srv, _ := newSSE(t, func(n int, req map[string]any, w sseW) {
		if n == 1 {
			w.text("Đang chạy.")
			frag := func(idx int, id, name, args string) {
				tc := map[string]any{"index": idx, "function": map[string]any{"arguments": args}}
				if id != "" {
					tc["id"] = id
					tc["type"] = "function"
					tc["function"].(map[string]any)["name"] = name
				}
				w.data(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{tc}}}}})
			}
			frag(0, "c1", "run_command", "")
			frag(0, "", "", `{"comma`)
			frag(0, "", "", `nd":"echo NATIVE-STREAM"}`)
		} else {
			ms := req["messages"].([]any)
			last := ms[len(ms)-1].(map[string]any)
			if last["role"] != "tool" || !strings.Contains(last["content"].(string), "NATIVE-STREAM") {
				t.Errorf("tool result not fed back: %+v", last)
			}
			w.text("Xong NATIVE-STREAM")
		}
		w.done()
	})
	s, _ := streamServer(t, srv.URL, "native")
	evs := s.driveRaw(context.Background(), "chạy")
	if !strings.Contains(texts2(evs), "NATIVE-STREAM") || countOf(evs, "tool_call") != 1 || countOf(evs, "error") != 0 {
		t.Fatalf("%+v", evs)
	}
}

func TestAgentStreamFallsBackWhenServerIgnoresStream(t *testing.T) {
	// plain JSON answer (server ignores "stream")
	f := newFakeLLM(t, func(n int, req map[string]any) (int, any) { return 200, chat("Trả lời một lần.") })
	s, _ := streamServer(t, f.srv.URL, "text")
	evs := s.driveRaw(context.Background(), "x")
	if countOf(evs, "delta") != 0 || lastOf(evs, "assistant") != "Trả lời một lần." || countOf(evs, "error") != 0 {
		t.Errorf("%+v", evs)
	}
	// SSE body with a wrong Content-Type
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, "data: "+`{"choices":[{"delta":{"content":"Lạc "}}]}`+"\n\n"+"data: "+`{"choices":[{"delta":{"content":"header"}}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	s2, _ := streamServer(t, srv.URL, "text")
	evs = s2.driveRaw(context.Background(), "x")
	if lastOf(evs, "assistant") != "Lạc header" {
		t.Errorf("SSE with wrong content-type: %+v", evs)
	}
}

func TestAgentStreamDisabledByConfig(t *testing.T) {
	srv, reqs := newSSE(t, func(n int, req map[string]any, w sseW) {
		w.raw(`{"choices":[{"message":{"role":"assistant","content":"không stream"}}]}`)
	})
	s, _ := streamServer(t, srv.URL, "text")
	s.cfg.Stream = false
	evs := s.driveRaw(context.Background(), "x")
	if _, has := (*reqs)[0]["stream"]; has {
		t.Error("stream must not be requested when disabled")
	}
	if countOf(evs, "delta") != 0 || lastOf(evs, "assistant") != "không stream" {
		t.Errorf("%+v", evs)
	}
	if !loadConfigDefaults().Stream {
		t.Error("streaming must be ON by default")
	}
}

func loadConfigDefaults() Config { t := Config{Stream: true}; return loadConfigWith(t) }

func TestAgentStreamMidStreamError(t *testing.T) {
	srv, _ := newSSE(t, func(n int, req map[string]any, w sseW) {
		w.text("Mình đang trả lời dở ")
		w.data(map[string]any{"error": map[string]any{"message": "model overloaded"}})
	})
	s, _ := streamServer(t, srv.URL, "text")
	evs := s.driveRaw(context.Background(), "x")
	if lastOf(evs, "assistant") != "Mình đang trả lời dở" || !strings.Contains(lastOf(evs, "error"), "overloaded") {
		t.Errorf("partial text must be kept and the reason shown: %+v", evs)
	}
}

func TestAgentStreamStopKeepsPartialText(t *testing.T) {
	started := make(chan struct{})
	srv, _ := newSSE(t, func(n int, req map[string]any, w sseW) {
		w.text("Đang viết dở…")
		close(started)
		<-time.After(20 * time.Second) // the model never finishes
	})
	s, _ := streamServer(t, srv.URL, "text")
	ctx, cancel := context.WithCancel(context.Background())
	go func() { <-started; time.Sleep(150 * time.Millisecond); cancel() }()
	st := time.Now()
	evs := s.driveRaw(ctx, "x")
	if time.Since(st) > 4*time.Second {
		t.Fatalf("Stop must interrupt a streaming answer immediately, took %v", time.Since(st))
	}
	if got := lastOf(evs, "assistant"); got != "Đang viết dở…\n\n(đã dừng)" {
		t.Errorf("partial text with a stop marker expected, got %q", got)
	}
	if countOf(evs, "error") != 0 {
		t.Errorf("Stop is not an error: %+v", evs)
	}
}

func TestAgentStreamSilenceTimeout(t *testing.T) {
	old := llmIdleTimeout
	llmIdleTimeout = 400 * time.Millisecond
	defer func() { llmIdleTimeout = old }()
	srv, _ := newSSE(t, func(n int, req map[string]any, w sseW) {
		w.text("bắt đầu")
		time.Sleep(3 * time.Second) // goes silent
	})
	s, _ := streamServer(t, srv.URL, "text")
	st := time.Now()
	evs := s.driveRaw(context.Background(), "x")
	if time.Since(st) > 2*time.Second || !strings.Contains(lastOf(evs, "error"), "không phản hồi") {
		t.Errorf("a silent stream must time out with a clear message: %v %+v", time.Since(st), evs)
	}
	// a slow but steady stream is NOT cut off (the timer resets on every chunk)
	srv2, _ := newSSE(t, func(n int, req map[string]any, w sseW) {
		for i := 0; i < 5; i++ {
			w.text("x")
			time.Sleep(200 * time.Millisecond)
		}
		w.done()
	})
	s2, _ := streamServer(t, srv2.URL, "text")
	if evs = s2.driveRaw(context.Background(), "x"); lastOf(evs, "assistant") != "xxxxx" {
		t.Errorf("slow steady stream was cut: %+v", evs)
	}
}

func TestAgentStreamRetriesContextOverflowBeforeAnyText(t *testing.T) {
	srv, _ := newSSE(t, func(n int, req map[string]any, w sseW) {
		if n == 1 {
			w.raw("") // headers are already 200 in SSE; emulate a proper HTTP error instead below
		}
		w.text("ok")
		w.done()
	})
	_ = srv
	// real HTTP 400 before streaming starts → context overflow retry still works
	calls := 0
	srv2 := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			rw.WriteHeader(400)
			fmt.Fprint(rw, `{"error":{"message":"maximum context length exceeded"}}`)
			return
		}
		rw.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(rw, "data: "+`{"choices":[{"delta":{"content":"đã thử lại"}}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer srv2.Close()
	s, _ := streamServer(t, srv2.URL, "text")
	evs := s.driveRaw(context.Background(), "x")
	if calls != 2 || lastOf(evs, "assistant") != "đã thử lại" {
		t.Errorf("calls=%d %+v", calls, evs)
	}
}

// ---- through HTTP: deltas reach the browser, but the saved conversation only has final messages

func TestHTTPChatStreamsAndStoresOnlyFinalText(t *testing.T) {
	srv, _ := newSSE(t, func(n int, req map[string]any, w sseW) {
		for _, p := range []string{"Một ", "hai ", "ba"} {
			w.text(p)
			time.Sleep(20 * time.Millisecond)
		}
		w.done()
	})
	d := mkProject(t, map[string]string{"a.txt": "x"})
	a := newTestApp(t, "")
	a.app.cfg.BaseURL, a.app.cfg.Model, a.app.cfg.Stream = srv.URL, "m", true
	pid, sid := a.openProject(d)
	_, evs := a.chat(pid, sid, "đếm", nil)
	var deltas []string
	var finals []string
	for _, e := range evs {
		var s string
		_ = json.Unmarshal(e.Data, &s)
		if e.Event == "delta" {
			deltas = append(deltas, s)
		}
		if e.Event == "assistant" {
			finals = append(finals, s)
		}
	}
	if strings.Join(deltas, "") != "Một hai ba" || len(deltas) < 2 || len(finals) != 1 || finals[0] != "Một hai ba" {
		t.Fatalf("browser must get deltas then one final: deltas=%q finals=%q", deltas, finals)
	}
	log := a.session(pid, sid)["log"].([]any)
	asst := 0
	for _, e := range log {
		m := e.(map[string]any)
		if m["t"] == "assistant" {
			asst++
			if m["text"] != "Một hai ba" {
				t.Errorf("saved text: %v", m["text"])
			}
		}
	}
	if asst != 1 {
		t.Errorf("saved log must hold ONE assistant entry (not one per delta): %d", asst)
	}
}
