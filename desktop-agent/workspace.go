package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Workspace = the projects you opened, each with several chat sessions. Saved to workspace.json
// so everything is still there after a restart.

const (
	defaultTitle   = "Cuộc trò chuyện mới"
	maxLogEntries  = 300
	maxPersistMsgs = 60
	maxSessions    = 60
)

type LogEntry struct {
	T      string `json:"t"` // user | assistant | tool | error
	Text   string `json:"text,omitempty"`
	ID     string `json:"id,omitempty"`
	Name   string `json:"name,omitempty"`
	Args   string `json:"args,omitempty"`
	Result string `json:"result,omitempty"`
	Done   bool   `json:"done,omitempty"`
}

// pMsg is Msg for the disk (keeps Meta, which Msg hides from the LLM request).
type pMsg struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Meta       string     `json:"meta,omitempty"`
}

type Session struct {
	ID      string     `json:"id"`
	Title   string     `json:"title"`
	Created time.Time  `json:"created"`
	Updated time.Time  `json:"updated"`
	Log     []LogEntry `json:"log"`
	History []pMsg     `json:"history"`
}

type Project struct {
	ID       string     `json:"id"`
	Path     string     `json:"path"`
	Name     string     `json:"name"`
	Sessions []*Session `json:"sessions"`
}

type Workspace struct {
	mu            sync.Mutex
	path          string
	Projects      []*Project `json:"projects"`
	ActiveProject string     `json:"active_project"`
	ActiveSession string     `json:"active_session"`
}

func newID() string {
	b := make([]byte, 4)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func loadWorkspace(path string) *Workspace {
	w := &Workspace{path: path}
	if path == "" {
		return w
	}
	if b, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(b, w)
	}
	return w
}

// saveLocked must be called with w.mu held.
func (w *Workspace) saveLocked() {
	if w.path == "" {
		return
	}
	b, err := json.Marshal(w)
	if err != nil {
		return
	}
	_ = os.MkdirAll(filepath.Dir(w.path), 0o755)
	tmp := w.path + ".tmp"
	if os.WriteFile(tmp, b, 0o644) == nil {
		_ = os.Rename(tmp, w.path)
	}
}

func samePath(a, b string) bool {
	a, b = filepath.Clean(a), filepath.Clean(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

func newSession() *Session {
	now := time.Now()
	return &Session{ID: newID(), Title: defaultTitle, Created: now, Updated: now}
}

func (s *Session) empty() bool { return len(s.Log) == 0 }

// findLocked returns the project and (optionally) the session; sid may be "".
func (w *Workspace) findLocked(pid, sid string) (*Project, *Session) {
	for _, p := range w.Projects {
		if p.ID != pid {
			continue
		}
		for _, s := range p.Sessions {
			if s.ID == sid {
				return p, s
			}
		}
		return p, nil
	}
	return nil, nil
}

// OpenProject adds a project (or returns the existing one) and makes it active.
func (w *Workspace) OpenProject(path string) (*Project, *Session, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, nil, err
	}
	st, err := os.Stat(abs)
	if err != nil || !st.IsDir() {
		return nil, nil, fmt.Errorf("không phải thư mục: %s", path)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	var p *Project
	for _, x := range w.Projects {
		if samePath(x.Path, abs) {
			p = x
			break
		}
	}
	if p == nil {
		p = &Project{ID: newID(), Path: abs, Name: filepath.Base(abs)}
		if p.Name == "" || p.Name == string(filepath.Separator) || p.Name == "." {
			p.Name = abs
		}
		w.Projects = append(w.Projects, p)
	}
	var s *Session
	if n := len(p.Sessions); n > 0 {
		s = p.Sessions[0] // most recent first
		for _, x := range p.Sessions {
			if x.empty() {
				s = x
				break
			}
		}
	} else {
		s = newSession()
		p.Sessions = append(p.Sessions, s)
	}
	w.ActiveProject, w.ActiveSession = p.ID, s.ID
	w.saveLocked()
	return p, s, nil
}

// NewSession reuses an existing blank session instead of piling up empty ones.
func (w *Workspace) NewSession(pid string) (*Session, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	p, _ := w.findLocked(pid, "")
	if p == nil {
		return nil, fmt.Errorf("không có project")
	}
	for _, s := range p.Sessions {
		if s.empty() {
			w.ActiveProject, w.ActiveSession = p.ID, s.ID
			w.saveLocked()
			return s, nil
		}
	}
	if len(p.Sessions) >= maxSessions {
		return nil, fmt.Errorf("tối đa %d phiên mỗi project, hãy xóa bớt", maxSessions)
	}
	s := newSession()
	p.Sessions = append([]*Session{s}, p.Sessions...)
	w.ActiveProject, w.ActiveSession = p.ID, s.ID
	w.saveLocked()
	return s, nil
}

func (w *Workspace) pickActiveLocked() {
	for _, p := range w.Projects {
		if p.ID == w.ActiveProject {
			for _, s := range p.Sessions {
				if s.ID == w.ActiveSession {
					return
				}
			}
			if len(p.Sessions) > 0 {
				w.ActiveSession = p.Sessions[0].ID
				return
			}
		}
	}
	w.ActiveProject, w.ActiveSession = "", ""
	if len(w.Projects) > 0 {
		p := w.Projects[0]
		w.ActiveProject = p.ID
		if len(p.Sessions) == 0 {
			p.Sessions = append(p.Sessions, newSession())
		}
		w.ActiveSession = p.Sessions[0].ID
	}
}

func (w *Workspace) DeleteSession(pid, sid string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	p, s := w.findLocked(pid, sid)
	if p == nil || s == nil {
		return fmt.Errorf("không có phiên chat")
	}
	for i, x := range p.Sessions {
		if x == s {
			p.Sessions = append(p.Sessions[:i], p.Sessions[i+1:]...)
			break
		}
	}
	if len(p.Sessions) == 0 {
		p.Sessions = append(p.Sessions, newSession())
	}
	w.pickActiveLocked()
	w.saveLocked()
	return nil
}

func (w *Workspace) RemoveProject(pid string) (sessionIDs []string, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for i, p := range w.Projects {
		if p.ID == pid {
			for _, s := range p.Sessions {
				sessionIDs = append(sessionIDs, s.ID)
			}
			w.Projects = append(w.Projects[:i], w.Projects[i+1:]...)
			w.pickActiveLocked()
			w.saveLocked()
			return sessionIDs, nil
		}
	}
	return nil, fmt.Errorf("không có project")
}

func (w *Workspace) Rename(pid, sid, title string) error {
	title = strings.TrimSpace(title)
	if title == "" {
		return fmt.Errorf("tên trống")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	_, s := w.findLocked(pid, sid)
	if s == nil {
		return fmt.Errorf("không có phiên chat")
	}
	s.Title = clipRunes(title, 80)
	w.saveLocked()
	return nil
}

func (w *Workspace) SetActive(pid, sid string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if p, s := w.findLocked(pid, sid); p != nil && s != nil {
		w.ActiveProject, w.ActiveSession = pid, sid
		w.saveLocked()
	}
}

// ---- history <-> disk

func toPersist(h []Msg) []pMsg {
	h = compactHistory(h, maxPersistMsgs)
	out := make([]pMsg, 0, len(h))
	for _, m := range h {
		out = append(out, pMsg{Role: m.Role, Content: clipRunes(m.Content, 4000), ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID, Meta: m.Meta})
	}
	return out
}

func fromPersist(p []pMsg) []Msg {
	out := make([]Msg, 0, len(p))
	for _, m := range p {
		out = append(out, Msg{Role: m.Role, Content: m.Content, ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID, Meta: m.Meta})
	}
	return out
}

// record stores what the UI needs to redraw a conversation later.
func (w *Workspace) record(s *Session, ev string, data any) {
	w.mu.Lock()
	defer w.mu.Unlock()
	switch ev {
	case "assistant":
		s.Log = append(s.Log, LogEntry{T: "assistant", Text: clipRunes(fmt.Sprint(data), 8000)})
	case "error":
		s.Log = append(s.Log, LogEntry{T: "error", Text: clipRunes(fmt.Sprint(data), 2000)})
	case "tool_call":
		if m, ok := data.(map[string]any); ok {
			s.Log = append(s.Log, LogEntry{T: "tool", ID: fmt.Sprint(m["id"]), Name: fmt.Sprint(m["name"]), Args: clipRunes(fmt.Sprint(m["args"]), 2000)})
		}
	case "tool_result":
		if m, ok := data.(map[string]any); ok {
			id := fmt.Sprint(m["id"])
			for i := len(s.Log) - 1; i >= 0; i-- {
				if s.Log[i].T == "tool" && s.Log[i].ID == id {
					s.Log[i].Result, s.Log[i].Done = clipRunes(fmt.Sprint(m["result"]), 2000), true
					break
				}
			}
		}
	}
	if len(s.Log) > maxLogEntries {
		s.Log = append([]LogEntry(nil), s.Log[len(s.Log)-(maxLogEntries-50):]...)
	}
}
