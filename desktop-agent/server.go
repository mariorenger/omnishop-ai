package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

func (s *Server) routes(mux *http.ServeMux) {
	api := func(path string, h http.HandlerFunc) { mux.HandleFunc(path, s.guard(h)) }
	api("/api/state", s.handleState)
	api("/api/project", s.handleProject)
	api("/api/browse", s.handleBrowse)
	api("/api/tree", s.handleTree)
	api("/api/file", s.handleFile)
	api("/api/chat", s.handleChat)
	api("/api/approve", s.handleApprove)
	api("/api/reset", s.handleReset)
	api("/api/cancel", s.handleCancel)
	api("/api/auto", s.handleAuto)
}

// guard: token + Host check (blocks other web pages / DNS rebinding from talking to the agent).
func (s *Server) guard(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		hostOK := r.Host == s.addr || r.Host == "localhost:"+s.addr[strings.LastIndex(s.addr, ":")+1:]
		if !hostOK || subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Token")), []byte(s.token)) != 1 {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		h(w, r)
	}
}

func jsonOut(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) handleState(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	jsonOut(w, map[string]any{"project": s.project, "model": s.cfg.Model, "base_url": s.cfg.BaseURL,
		"auto": s.auto || s.cfg.AutoApprove, "os": runtime.GOOS})
}

func (s *Server) handleProject(w http.ResponseWriter, r *http.Request) {
	var b struct{ Path string }
	_ = json.NewDecoder(r.Body).Decode(&b)
	abs, err := filepath.Abs(b.Path)
	if st, e2 := os.Stat(abs); err != nil || e2 != nil || !st.IsDir() {
		http.Error(w, "not a directory", 400)
		return
	}
	s.mu.Lock()
	s.project = abs
	s.history = nil
	s.mu.Unlock()
	jsonOut(w, map[string]any{"project": abs})
}

type entry struct {
	Name string `json:"name"`
	Dir  bool   `json:"dir"`
}

func listEntries(dir string, dirsOnly bool) ([]entry, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []entry
	for _, e := range ents {
		if e.Name() == ".git" || (dirsOnly && !e.IsDir()) {
			continue
		}
		out = append(out, entry{e.Name(), e.IsDir()})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Dir != out[j].Dir {
			return out[i].Dir
		}
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})
	return out, nil
}

// handleBrowse lets the UI pick a project folder anywhere (folders only).
func (s *Server) handleBrowse(w http.ResponseWriter, r *http.Request) {
	p := r.URL.Query().Get("path")
	if p == "" {
		p, _ = os.UserHomeDir()
	}
	p = filepath.Clean(p)
	ents, err := listEntries(p, true)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	var drives []string
	if runtime.GOOS == "windows" {
		for c := 'A'; c <= 'Z'; c++ {
			if _, err := os.Stat(string(c) + ":\\"); err == nil {
				drives = append(drives, string(c)+":\\")
			}
		}
	}
	parent := filepath.Dir(p)
	if parent == p {
		parent = ""
	}
	jsonOut(w, map[string]any{"path": p, "parent": parent, "dirs": ents, "drives": drives})
}

func (s *Server) handleTree(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	root := s.project
	s.mu.Unlock()
	p, err := resolve(root, r.URL.Query().Get("path"))
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	ents, err := listEntries(p, false)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	jsonOut(w, map[string]any{"entries": ents})
}

func (s *Server) handleFile(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	root := s.project
	s.mu.Unlock()
	out := runTool(r.Context(), root, "read_file", fmt.Sprintf(`{"path":%q}`, r.URL.Query().Get("path")))
	jsonOut(w, map[string]any{"content": out})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var b struct{ Message string }
	_ = json.NewDecoder(r.Body).Decode(&b)
	s.mu.Lock()
	if s.project == "" {
		s.mu.Unlock()
		http.Error(w, "select a project first", 400)
		return
	}
	if s.busy {
		s.mu.Unlock()
		http.Error(w, "agent is busy", 409)
		return
	}
	s.busy = true
	s.mu.Unlock()
	defer func() { s.mu.Lock(); s.busy = false; s.mu.Unlock() }()

	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "no streaming", 500)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	emit := func(ev string, data any) {
		j, _ := json.Marshal(map[string]any{"event": ev, "data": data})
		fmt.Fprintf(w, "data: %s\n\n", j)
		fl.Flush()
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	s.mu.Lock()
	s.cancel = cancel
	s.mu.Unlock()
	s.runAgent(ctx, b.Message, emit)
	emit("done", nil)
}

func (s *Server) handleApprove(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ID     string
		Allow  bool
		Always bool
	}
	_ = json.NewDecoder(r.Body).Decode(&b)
	s.mu.Lock()
	ch := s.pending[b.ID]
	if b.Always && b.Allow {
		s.auto = true
	}
	s.mu.Unlock()
	if ch != nil {
		ch <- b.Allow
	}
	jsonOut(w, map[string]bool{"ok": ch != nil})
}

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.history = nil
	s.mu.Unlock()
	jsonOut(w, map[string]bool{"ok": true})
}

func (s *Server) handleAuto(w http.ResponseWriter, r *http.Request) {
	var b struct{ On bool }
	_ = json.NewDecoder(r.Body).Decode(&b)
	s.mu.Lock()
	s.auto = b.On
	s.mu.Unlock()
	jsonOut(w, map[string]bool{"auto": b.On})
}

// handleCancel stops the running turn immediately (kills the running command, frees the busy flag).
func (s *Server) handleCancel(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	c := s.cancel
	s.mu.Unlock()
	if c != nil {
		c()
	}
	jsonOut(w, map[string]bool{"ok": c != nil})
}
