package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime"
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
}

var llmClient = &http.Client{Timeout: 5 * time.Minute}

func (s *Server) callLLM(ctx context.Context, msgs []Msg) (Msg, error) {
	body, _ := json.Marshal(map[string]any{
		"model": s.cfg.Model, "messages": msgs, "tools": toolDefs,
	})
	url := strings.TrimRight(s.cfg.BaseURL, "/") + "/chat/completions"
	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if s.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	}
	resp, err := llmClient.Do(req)
	if err != nil {
		return Msg{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode != 200 {
		return Msg{}, fmt.Errorf("LLM HTTP %d: %s", resp.StatusCode, clip(string(raw), 500))
	}
	var r struct {
		Choices []struct {
			Message struct {
				Content   *string    `json:"content"`
				ToolCalls []ToolCall `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &r); err != nil || len(r.Choices) == 0 {
		return Msg{}, fmt.Errorf("bad LLM response: %s", clip(string(raw), 300))
	}
	m := Msg{Role: "assistant", ToolCalls: r.Choices[0].Message.ToolCalls}
	if c := r.Choices[0].Message.Content; c != nil {
		m.Content = *c
	}
	for i := range m.ToolCalls {
		if m.ToolCalls[i].Type == "" {
			m.ToolCalls[i].Type = "function"
		}
		if m.ToolCalls[i].ID == "" {
			m.ToolCalls[i].ID = fmt.Sprintf("call_%d", i)
		}
	}
	return m, nil
}

func (s *Server) systemPrompt() string {
	return fmt.Sprintf(`You are a desktop coding/file assistant running on %s.
The user's project directory is: %s
Use the tools to inspect and modify files inside the project and to run shell commands there.
Prefer list_dir/search_files/read_file before editing. Use edit_file for small changes, write_file for new files.
Use relative paths. Keep answers concise. Reply in the user's language.`, runtime.GOOS, s.project)
}

type emitFn func(ev string, data any)

// runAgent executes the tool-calling loop. History is mutated under s.mu by the caller.
func (s *Server) runAgent(ctx context.Context, userMsg string, emit emitFn) {
	s.mu.Lock()
	s.history = append(s.history, Msg{Role: "user", Content: userMsg})
	root := s.project
	s.mu.Unlock()

	for step := 0; step < 25; step++ {
		s.mu.Lock()
		msgs := append([]Msg{{Role: "system", Content: s.systemPrompt()}}, s.history...)
		s.mu.Unlock()

		m, err := s.callLLM(ctx, msgs)
		if err != nil {
			emit("error", err.Error())
			return
		}
		s.mu.Lock()
		s.history = append(s.history, m)
		s.mu.Unlock()
		if strings.TrimSpace(m.Content) != "" {
			emit("assistant", m.Content)
		}
		if len(m.ToolCalls) == 0 {
			return
		}
		for _, tc := range m.ToolCalls {
			name, args := tc.Function.Name, tc.Function.Arguments
			emit("tool_call", map[string]any{"id": tc.ID, "name": name, "args": args})
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
			s.mu.Lock()
			s.history = append(s.history, Msg{Role: "tool", ToolCallID: tc.ID, Content: result})
			s.mu.Unlock()
			if ctx.Err() != nil {
				return
			}
		}
	}
	emit("error", "stopped: too many steps")
}

func (s *Server) askApproval(ctx context.Context, id, name, args string, emit emitFn) bool {
	ch := make(chan bool, 1)
	s.mu.Lock()
	s.pending[id] = ch
	s.mu.Unlock()
	defer func() { s.mu.Lock(); delete(s.pending, id); s.mu.Unlock() }()
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
