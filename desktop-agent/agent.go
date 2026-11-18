package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type Msg struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Meta       string     `json:"-"` // "result": synthetic tool-result message (text protocol)
}

// No overall http.Client timeout: a streamed answer from a slow local model can legitimately take
// minutes. Instead callLLM watches for silence (llmIdleTimeout while streaming, llmTotalTimeout otherwise).
var (
	llmClient       = &http.Client{}
	llmIdleTimeout  = 120 * time.Second
	llmTotalTimeout = 5 * time.Minute
)

type llmError struct {
	Status int
	Body   string
}

func (e *llmError) Error() string { return fmt.Sprintf("LLM HTTP %d: %s", e.Status, clip(e.Body, 400)) }

func (s *Server) toolMode() string {
	switch strings.ToLower(s.cfg.ToolMode) {
	case "native", "auto":
		if s.forceText {
			return "text"
		}
		return strings.ToLower(s.cfg.ToolMode)
	}
	return "text"
}

// callLLM asks the model once. With streaming on (and onDelta != nil) the text is passed to onDelta
// piece by piece as it arrives. If the server ignores "stream" and answers with plain JSON, that
// works too. On an interrupted stream the text received so far is returned together with the error.
func (s *Server) callLLM(ctx context.Context, msgs []Msg, native bool, onDelta func(string)) (Msg, error) {
	stream := s.cfg.Stream && onDelta != nil
	req := map[string]any{"model": s.cfg.Model, "messages": msgs, "temperature": 0.2}
	if s.cfg.MaxOutputTokens > 0 {
		req["max_tokens"] = s.cfg.MaxOutputTokens
	}
	if native {
		req["tools"] = toolDefs
	}
	if stream {
		req["stream"] = true
	}
	body, _ := json.Marshal(req)
	url := strings.TrimRight(s.cfg.BaseURL, "/") + "/chat/completions"

	rctx, rcancel := context.WithCancel(ctx)
	defer rcancel()
	limit := llmTotalTimeout
	if stream {
		limit = llmIdleTimeout
	}
	idle := time.AfterFunc(limit, rcancel)
	defer idle.Stop()
	silent := func(err error) error { // our watchdog fired (not the user pressing Stop)
		if err != nil && ctx.Err() == nil && rctx.Err() != nil {
			return fmt.Errorf("LLM không phản hồi trong %v", limit)
		}
		return err
	}

	hreq, _ := http.NewRequestWithContext(rctx, "POST", url, bytes.NewReader(body))
	hreq.Header.Set("Content-Type", "application/json")
	if stream {
		hreq.Header.Set("Accept", "text/event-stream")
	}
	if s.cfg.APIKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	}
	resp, err := llmClient.Do(hreq)
	if err != nil {
		return Msg{}, silent(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return Msg{}, &llmError{resp.StatusCode, string(raw)}
	}
	if stream {
		idle.Reset(limit)
		if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "event-stream") {
			m, err := parseSSE(resp.Body, onDelta, func() { idle.Reset(limit) })
			if err != nil {
				return m, silent(err)
			}
			return finishMsg(m), nil
		}
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return Msg{}, silent(err)
	}
	if stream && bytes.HasPrefix(bytes.TrimSpace(raw), []byte("data:")) { // SSE without the proper header
		m, err := parseSSE(bytes.NewReader(raw), onDelta, nil)
		if err != nil {
			return m, err
		}
		return finishMsg(m), nil
	}
	m, err := parseCompletionJSON(raw)
	if err != nil {
		return Msg{}, err
	}
	return finishMsg(m), nil
}

func finishMsg(m Msg) Msg {
	for i := range m.ToolCalls {
		if m.ToolCalls[i].Type == "" {
			m.ToolCalls[i].Type = "function"
		}
		if m.ToolCalls[i].ID == "" {
			m.ToolCalls[i].ID = fmt.Sprintf("call_%d", i)
		}
	}
	return m
}

func partsText(v any) string {
	switch c := v.(type) {
	case string:
		return c
	case []any: // some servers return content parts
		var sb strings.Builder
		for _, p := range c {
			if pm, ok := p.(map[string]any); ok {
				if t, ok := pm["text"].(string); ok {
					sb.WriteString(t)
				}
			}
		}
		return sb.String()
	}
	return ""
}

func parseCompletionJSON(raw []byte) (Msg, error) {
	var r struct {
		Choices []struct {
			Message struct {
				Content   any        `json:"content"`
				ToolCalls []ToolCall `json:"tool_calls"`
			} `json:"message"`
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &r); err != nil || len(r.Choices) == 0 {
		return Msg{}, fmt.Errorf("phản hồi LLM không hợp lệ: %s", clip(string(raw), 300))
	}
	m := Msg{Role: "assistant", ToolCalls: r.Choices[0].Message.ToolCalls, Content: partsText(r.Choices[0].Message.Content)}
	if m.Content == "" {
		m.Content = r.Choices[0].Text
	}
	return m, nil
}

// parseSSE reads an OpenAI-style event stream ("data: {...}" lines, "data: [DONE]").
// Text deltas go to onDelta immediately; tool-call fragments are merged by index.
func parseSSE(r io.Reader, onDelta func(string), tick func()) (Msg, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	var content strings.Builder
	calls := map[int]*ToolCall{}
	var order []int
	build := func() Msg {
		m := Msg{Role: "assistant", Content: content.String()}
		for _, i := range order {
			m.ToolCalls = append(m.ToolCalls, *calls[i])
		}
		return m
	}
	for sc.Scan() {
		if tick != nil {
			tick()
		}
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") { // comments (": keep-alive"), "event:", blank lines
			continue
		}
		payload := strings.TrimSpace(line[5:])
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			break
		}
		var ch struct {
			Error   any `json:"error"`
			Choices []struct {
				Delta struct {
					Content   any `json:"content"`
					ToolCalls []struct {
						Index    *int   `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				Text string `json:"text"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(payload), &ch) != nil {
			continue // tolerate junk lines
		}
		if ch.Error != nil {
			return build(), fmt.Errorf("LLM báo lỗi giữa chừng: %v", ch.Error)
		}
		for _, c := range ch.Choices {
			d := partsText(c.Delta.Content)
			if d == "" {
				d = c.Text
			}
			if d != "" {
				content.WriteString(d)
				if onDelta != nil {
					onDelta(d)
				}
			}
			for _, tc := range c.Delta.ToolCalls {
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}
				t := calls[idx]
				if t == nil {
					t = &ToolCall{Type: "function"}
					calls[idx] = t
					order = append(order, idx)
				}
				if tc.ID != "" && t.ID == "" {
					t.ID = tc.ID
				}
				if tc.Function.Name != "" && t.Function.Name == "" { // some servers repeat the name in every fragment
					t.Function.Name = tc.Function.Name
				}
				t.Function.Arguments += tc.Function.Arguments
			}
		}
	}
	if err := sc.Err(); err != nil {
		return build(), err
	}
	return build(), nil
}

// streamFilter forwards text deltas to the UI but stops at the first sign of a tool block, so the
// raw ```tool {...} never flashes on screen. It also holds back a tail that could still turn into
// such a marker. (The final "assistant" event always carries the complete clean text, so a wrongly
// suppressed code block is corrected at the end.)
type streamFilter struct {
	buf     strings.Builder
	emitted int
	stopped bool
	emit    func(string)
}

var toolMarkers = []string{"```tool", "```json", "<tool_call>", `{"name"`, `{ "name"`, `{"tool"`, `{ "tool"`}

func (f *streamFilter) feed(d string) {
	if f.stopped || d == "" {
		return
	}
	f.buf.WriteString(d)
	s := f.buf.String()
	cut := -1
	for _, m := range toolMarkers {
		if i := strings.Index(s, m); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	end := len(s)
	if cut >= 0 {
		end, f.stopped = cut, true
	} else {
		for k := min(len(s), 11); k > 0; k-- { // longest marker is 11 bytes
			tail := s[len(s)-k:]
			held := false
			for _, m := range toolMarkers {
				if strings.HasPrefix(m, tail) {
					held = true
					break
				}
			}
			if held {
				end = len(s) - k
				break
			}
		}
	}
	if end > f.emitted {
		f.emit(s[f.emitted:end])
		f.emitted = end
	}
}

// ------------------------------------------------------------------ prompts

func toolSignature(def map[string]any) string {
	f := def["function"].(map[string]any)
	p := f["parameters"].(map[string]any)
	props, _ := p["properties"].(props)
	if props == nil {
		if mp, ok := p["properties"].(map[string]any); ok {
			props = mp
		}
	}
	req := map[string]bool{}
	if rs, ok := p["required"].([]string); ok {
		for _, r := range rs {
			req[r] = true
		}
	}
	var names []string
	for k := range props {
		n := k
		if !req[k] {
			n += "?"
		}
		names = append(names, n)
	}
	sortStrings(names)
	return fmt.Sprintf("%s(%s)", f["name"], strings.Join(names, ", "))
}

func (s *Server) systemPrompt(native bool) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Bạn là trợ lý trên máy %s của người dùng. Thư mục dự án: %s\n", osName(), s.project)
	sb.WriteString("Bạn làm việc bằng công cụ thật: đọc/ghi/sửa tệp, chạy lệnh terminal, đọc PDF/Word/Excel, tạo task định kỳ. Không bịa kết quả; luôn dùng công cụ rồi mới kết luận.\n")
	if !native {
		sb.WriteString("Để dùng công cụ, trả lời ĐÚNG MỘT khối sau rồi dừng hẳn (không viết thêm gì sau khối):\n```tool\n{\"name\":\"TÊN_CÔNG_CỤ\",\"args\":{\"tham_số\":\"giá trị\"}}\n```\n")
		sb.WriteString("Hệ thống chạy thật và gửi kết quả lại cho bạn. Xem kết quả rồi gọi công cụ tiếp hoặc trả lời. Mỗi lượt chỉ một công cụ. Khi xong việc, trả lời người dùng bằng văn bản thường (không có khối tool).\n")
		sb.WriteString("Công cụ (dấu ? = tùy chọn):\n")
		for _, d := range toolDefs {
			sb.WriteString("- " + toolSignature(d) + "\n")
		}
		sb.WriteString("Ví dụ chạy lệnh:\n```tool\n{\"name\":\"run_command\",\"args\":{\"command\":\"dir\"}}\n```\n")
	}
	sb.WriteString("Quy tắc: dùng đường dẫn tương đối. Muốn chạy mã: write_file ra tệp rồi run_command (vd python script.py). Việc lặp lại hàng ngày/định kỳ: create_task (schedule như \"every 5m\" hoặc \"daily 08:30\"). Tệp dài đọc theo trang bằng offset. Nếu tên tệp người dùng nói không chắc/gõ gần đúng/không dấu: dùng find_files hoặc list_dir/tree để tìm đúng tên TRƯỚC khi read_file; nếu read_file báo không thấy thì dùng mục \"gần giống\" trong kết quả rồi thử lại. Tìm chữ bên trong nhiều tệp (kể cả PDF/Word/Excel) dùng search_files; tìm tệp theo tên dùng find_files. Trả lời ngắn gọn, cùng ngôn ngữ người dùng.")
	return sb.String()
}

// ------------------------------------------------------------------ text-protocol parsing

var (
	reFence   = regexp.MustCompile("(?s)```[a-zA-Z_]*\\s*\n?(\\{.*?\\})\\s*```")
	reXMLCall = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)
	toolAlias = map[string]string{
		"bash": "run_command", "shell": "run_command", "terminal": "run_command", "cmd": "run_command", "execute": "run_command",
		"run": "run_command", "exec": "run_command", "powershell": "run_command", "command": "run_command",
		"read": "read_file", "cat": "read_file", "open_file": "read_file", "read_document": "read_file", "open": "read_file",
		"write": "write_file", "create_file": "write_file", "save_file": "write_file",
		"ls": "list_dir", "list": "list_dir", "dir": "list_dir", "list_files": "list_dir",
		"grep": "search_files", "search": "search_files", "find": "find_files",
	}
)

func knownTool(name string) bool {
	for _, d := range toolDefs {
		if d["function"].(map[string]any)["name"] == name {
			return true
		}
	}
	return false
}

// extractJSONObjects finds balanced {...} objects in free text.
func extractJSONObjects(s string) []string {
	var out []string
	for i := 0; i < len(s); i++ {
		if s[i] != '{' {
			continue
		}
		depth, inStr, esc := 0, false, false
		for j := i; j < len(s); j++ {
			c := s[j]
			switch {
			case inStr:
				if esc {
					esc = false
				} else if c == '\\' {
					esc = true
				} else if c == '"' {
					inStr = false
				}
			case c == '"':
				inStr = true
			case c == '{':
				depth++
			case c == '}':
				depth--
				if depth == 0 {
					out = append(out, s[i:j+1])
					i = j
					j = len(s)
				}
			}
		}
	}
	return out
}

// toToolCall interprets one JSON object as a tool call (lenient about shapes).
func toToolCall(js string) (string, string, bool) {
	var m map[string]any
	if json.Unmarshal([]byte(js), &m) != nil {
		return "", "", false
	}
	if fn, ok := m["function"].(map[string]any); ok { // OpenAI style nested
		m = fn
	}
	name := ""
	for _, k := range []string{"name", "tool", "tool_name", "function", "action"} {
		if v, ok := m[k].(string); ok && v != "" {
			name = v
			break
		}
	}
	if name == "" {
		return "", "", false
	}
	if al, ok := toolAlias[strings.ToLower(name)]; ok && !knownTool(name) {
		name = al
	}
	if !knownTool(name) {
		return "", "", false
	}
	var args map[string]any
	for _, k := range []string{"args", "arguments", "parameters", "input", "params"} {
		switch v := m[k].(type) {
		case map[string]any:
			args = v
		case string:
			_ = json.Unmarshal([]byte(v), &args)
		}
		if args != nil {
			break
		}
	}
	if args == nil { // flat shape: {"name":"run_command","command":"dir"}
		args = map[string]any{}
		for k, v := range m {
			switch k {
			case "name", "tool", "tool_name", "function", "action":
			default:
				args[k] = v
			}
		}
	}
	b, _ := json.Marshal(args)
	return name, string(b), true
}

// parseToolText finds the first valid tool call in model text. Returns the text with the block removed.
func parseToolText(text string) (name, args, rest string, ok bool) {
	var cands []string
	for _, re := range []*regexp.Regexp{reFence, reXMLCall} {
		for _, m := range re.FindAllStringSubmatch(text, -1) {
			cands = append(cands, m[1])
		}
	}
	cands = append(cands, extractJSONObjects(text)...)
	for _, c := range cands {
		if n, a, k := toToolCall(c); k {
			rest = strings.Replace(text, c, "", 1)
			rest = regexp.MustCompile("(?s)```[a-zA-Z_]*\\s*```").ReplaceAllString(rest, "")
			rest = strings.ReplaceAll(strings.ReplaceAll(rest, "<tool_call>", ""), "</tool_call>", "")
			return n, a, strings.TrimSpace(rest), true
		}
	}
	return "", "", text, false
}

func looksLikeBrokenTool(text string) bool {
	return strings.Contains(text, "```tool") || strings.Contains(text, "<tool_call>")
}

// ------------------------------------------------------------------ the loop

const maxAgentSteps = 15

func (s *Server) toolResultChars() int { return readChunkChars + 400 }

func (s *Server) inputBudget() int {
	b := s.cfg.ContextTokens - s.cfg.MaxOutputTokens - 300
	if b < 1500 {
		b = 1500
	}
	return b
}

// runAgent executes the tool-calling loop. Works with models that do NOT support native
// function calling (default "text" protocol) and with OpenAI-style tool_calls.
func (s *Server) runAgent(ctx context.Context, userMsg string, emit emitFn) {
	if strings.TrimSpace(s.cfg.BaseURL) == "" {
		s.runOffline(ctx, userMsg, emit)
		return
	}
	s.mu.Lock()
	s.history = append(s.history, Msg{Role: "user", Content: userMsg})
	s.history = compactHistory(s.history, 120)
	root := s.project
	s.mu.Unlock()

	var lastSig string
	repeat, nudges := 0, 0
	scale := 1.0
	for step := 0; step < maxAgentSteps; step++ {
		mode := s.toolMode()
		native := mode != "text"
		s.mu.Lock()
		msgs := fitContext(s.systemPrompt(native), s.history, int(float64(s.inputBudget())*scale))
		s.mu.Unlock()

		sf := &streamFilter{emit: func(d string) { emit("delta", d) }}
		streamed := false
		m, err := s.callLLM(ctx, msgs, native, func(d string) { streamed = true; sf.feed(d) })
		if err != nil {
			if ctx.Err() != nil { // the user pressed Stop: keep what was already written, no error banner
				partial := strings.TrimSpace(m.Content)
				if _, _, rest, ok := parseToolText(partial); ok {
					partial = strings.TrimSpace(rest)
				}
				if partial != "" {
					emit("assistant", partial+"\n\n(đã dừng)")
				} else if streamed {
					emit("assistant", "")
				}
				return
			}
			if streamed { // close the half-written bubble before showing why it stopped
				partial := strings.TrimSpace(m.Content)
				if _, _, rest, ok := parseToolText(partial); ok {
					partial = strings.TrimSpace(rest)
				}
				emit("assistant", partial)
			}
			if le, ok := err.(*llmError); ok {
				low := strings.ToLower(le.Body)
				switch {
				case native && mode == "auto" && (le.Status == 400 || le.Status == 404 || le.Status == 422 || le.Status == 501):
					s.forceText = true // this server has no function calling → use the text protocol
					step--
					continue
				case scale > 0.4 && le.Status >= 400 && le.Status < 500 && (strings.Contains(low, "context") || strings.Contains(low, "token") || strings.Contains(low, "too long") || strings.Contains(low, "maximum")):
					scale *= 0.6 // context overflow → retry with a tighter budget
					step--
					continue
				}
			}
			emit("error", err.Error())
			return
		}
		scale = 1.0

		calls := m.ToolCalls
		text := m.Content
		if len(calls) == 0 { // model answered in plain text – maybe it contains a tool block
			if n, a, rest, ok := parseToolText(text); ok {
				calls = []ToolCall{{ID: fmt.Sprintf("t%d_%d", time.Now().UnixNano()%1e9, step), Type: "function"}}
				calls[0].Function.Name, calls[0].Function.Arguments = n, a
				text = rest
				m = Msg{Role: "assistant", Content: m.Content} // keep raw text in history (text protocol)
				if native {
					m.ToolCalls = calls
					m.Content = text
				}
			}
		}
		s.mu.Lock()
		s.history = append(s.history, m)
		s.mu.Unlock()
		if strings.TrimSpace(text) != "" || streamed { // (empty text just closes the live bubble)
			emit("assistant", text)
		}
		if len(calls) == 0 {
			if looksLikeBrokenTool(m.Content) && nudges < 2 { // malformed block → ask once more
				nudges++
				s.mu.Lock()
				s.history = append(s.history, Msg{Role: "user", Meta: "result", Content: "Khối tool không phải JSON hợp lệ. Gửi lại đúng định dạng: {\"name\":\"...\",\"args\":{...}} trong khối ```tool."})
				s.mu.Unlock()
				continue
			}
			return
		}
		tc := calls[0] // one tool per step in text mode; native may return several
		if !native {
			calls = calls[:1]
		}
		for _, tc = range calls {
			sig := tc.Function.Name + "|" + tc.Function.Arguments
			if sig == lastSig {
				repeat++
			} else {
				lastSig, repeat = sig, 0
			}
			if repeat >= 2 {
				emit("error", "Bot đang lặp lại cùng một lệnh nhiều lần nên đã dừng. Hãy hỏi lại rõ hơn.")
				return
			}
			res := s.execTool(ctx, root, tc, emit)
			fed := clipRunes(res, s.toolResultChars())
			s.mu.Lock()
			if native {
				s.history = append(s.history, Msg{Role: "tool", ToolCallID: tc.ID, Content: fed})
			} else {
				s.history = append(s.history, Msg{Role: "user", Meta: "result",
					Content: fmt.Sprintf("Kết quả của `%s`:\n%s\n(Tiếp tục. Nếu đã đủ thông tin thì trả lời người dùng, không dùng thêm công cụ.)", tc.Function.Name, fed)})
			}
			s.mu.Unlock()
			if ctx.Err() != nil {
				return
			}
		}
	}
	emit("error", "Đã dừng: quá nhiều bước. Hãy chia nhỏ yêu cầu.")
}

// execTool emits events, asks for approval when needed, runs the tool.
func (s *Server) execTool(ctx context.Context, root string, tc ToolCall, emit emitFn) string {
	name, args := tc.Function.Name, tc.Function.Arguments
	ev := map[string]any{"id": tc.ID, "name": name, "args": args}
	if name == "run_command" {
		ev["timeout"] = commandTimeoutOf(args)
	}
	emit("tool_call", ev)
	result := ""
	if needsApproval(name) && !s.cfg.AutoApprove && !s.isAutoApprove() {
		if !s.askApproval(ctx, tc.ID, name, args, emit) {
			result = "error: the user denied this action"
		}
	}
	if result == "" {
		result = runTool(ctx, root, name, args)
	}
	emit("tool_result", map[string]any{"id": tc.ID, "result": clip(result, 8000)})
	return result
}

type emitFn func(ev string, data any)

func (s *Server) askApproval(ctx context.Context, id, name, args string, emit emitFn) bool {
	ch := make(chan bool, 1)
	s.mu.Lock()
	s.pending[id] = ch
	s.mu.Unlock()
	defer func() { s.mu.Lock(); delete(s.pending, id); s.mu.Unlock() }()
	s.mu.Lock()
	if s.pendInfo == nil {
		s.pendInfo = map[string]map[string]any{}
	}
	s.pendInfo[id] = map[string]any{"id": id, "name": name, "args": args}
	s.mu.Unlock()
	defer func() { s.mu.Lock(); delete(s.pendInfo, id); s.mu.Unlock() }()
	emit("approval", map[string]any{"id": id, "name": name, "args": args})
	select {
	case ok := <-ch:
		return ok
	case <-ctx.Done():
		return false
	case <-time.After(10 * time.Minute):
		return false
	}
}
