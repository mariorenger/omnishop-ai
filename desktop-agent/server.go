package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// App = the HTTP application: auth, workspace (projects → sessions) and one runner per session.
type App struct {
	cfg     Config
	token   string
	addr    string
	mu      sync.Mutex // guards runners, auto
	runners map[string]*Server
	auto    bool
	ws      *Workspace
}

func (a *App) routes(mux *http.ServeMux) {
	api := func(path string, h http.HandlerFunc) { mux.HandleFunc(path, a.guard(h)) }
	api("/api/state", a.handleState)
	api("/api/workspace", a.handleWorkspace)
	api("/api/workspace/active", a.handleActive)
	api("/api/project/open", a.handleProjectOpen)
	api("/api/project/remove", a.handleProjectRemove)
	api("/api/session", a.handleSession)
	api("/api/session/new", a.handleSessionNew)
	api("/api/session/delete", a.handleSessionDelete)
	api("/api/session/rename", a.handleSessionRename)
	api("/api/browse", a.handleBrowse)
	api("/api/tree", a.handleTree)
	api("/api/file", a.handleFile)
	api("/api/raw", a.handleRaw)
	api("/api/open", a.handleOpenExternal)
	api("/api/chat", a.handleChat)
	api("/api/approve", a.handleApprove)
	api("/api/cancel", a.handleCancel)
	api("/api/auto", a.handleAuto)
	api("/api/tasks", a.handleTasks)
	api("/api/task", a.handleTaskAction)
}

// guard: token + Host check (blocks other web pages / DNS rebinding from talking to the agent).
func (a *App) guard(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		hostOK := r.Host == a.addr || r.Host == "localhost:"+a.addr[strings.LastIndex(a.addr, ":")+1:]
		if !hostOK || subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Token")), []byte(a.token)) != 1 {
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

func readJSON(r *http.Request, v any) {
	_ = json.NewDecoder(http.MaxBytesReader(nil, r.Body, 4<<20)).Decode(v)
}

func (a *App) handleState(w http.ResponseWriter, r *http.Request) {
	a.mu.Lock()
	auto := a.auto || a.cfg.AutoApprove
	a.mu.Unlock()
	jsonOut(w, map[string]any{"model": a.cfg.Model, "base_url": a.cfg.BaseURL, "auto": auto, "os": runtime.GOOS})
}

// runner returns (creating if needed) the runner of a session.
func (a *App) runner(p *Project, s *Session) *Server {
	a.mu.Lock()
	defer a.mu.Unlock()
	if r, ok := a.runners[s.ID]; ok {
		return r
	}
	a.ws.mu.Lock()
	hist := fromPersist(s.History)
	a.ws.mu.Unlock()
	r := &Server{cfg: a.cfg, project: p.Path, history: hist, pending: map[string]chan bool{}, auto: a.auto}
	a.runners[s.ID] = r
	return r
}

func (a *App) dropRunner(sid string) {
	a.mu.Lock()
	r := a.runners[sid]
	delete(a.runners, sid)
	a.mu.Unlock()
	if r != nil {
		r.mu.Lock()
		c := r.cancel
		r.mu.Unlock()
		if c != nil {
			c()
		}
	}
}

// ---------------------------------------------------------------- workspace

type sessionView struct {
	ID      string    `json:"id"`
	Title   string    `json:"title"`
	Updated time.Time `json:"updated"`
	Busy    bool      `json:"busy"`
	Empty   bool      `json:"empty"`
}
type projectView struct {
	ID       string        `json:"id"`
	Name     string        `json:"name"`
	Path     string        `json:"path"`
	Missing  bool          `json:"missing"`
	Sessions []sessionView `json:"sessions"`
}

func (a *App) busy(sid string) bool {
	a.mu.Lock()
	r := a.runners[sid]
	a.mu.Unlock()
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.busy
}

func (a *App) handleWorkspace(w http.ResponseWriter, r *http.Request) {
	a.ws.mu.Lock()
	out := map[string]any{"active_project": a.ws.ActiveProject, "active_session": a.ws.ActiveSession}
	var ps []projectView
	var sids []string
	for _, p := range a.ws.Projects {
		pv := projectView{ID: p.ID, Name: p.Name, Path: p.Path}
		if st, err := os.Stat(p.Path); err != nil || !st.IsDir() {
			pv.Missing = true
		}
		for _, s := range p.Sessions {
			pv.Sessions = append(pv.Sessions, sessionView{ID: s.ID, Title: s.Title, Updated: s.Updated, Empty: s.empty()})
			sids = append(sids, s.ID)
		}
		ps = append(ps, pv)
	}
	a.ws.mu.Unlock()
	busy := map[string]bool{}
	for _, id := range sids {
		busy[id] = a.busy(id)
	}
	for i := range ps {
		for j := range ps[i].Sessions {
			ps[i].Sessions[j].Busy = busy[ps[i].Sessions[j].ID]
		}
	}
	if ps == nil {
		ps = []projectView{}
	}
	out["projects"] = ps
	jsonOut(w, out)
}

func (a *App) handleActive(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ProjectID string `json:"project_id"`
		SessionID string `json:"session_id"`
	}
	readJSON(r, &b)
	a.ws.SetActive(b.ProjectID, b.SessionID)
	jsonOut(w, map[string]bool{"ok": true})
}

func (a *App) handleProjectOpen(w http.ResponseWriter, r *http.Request) {
	var b struct {
		Path string `json:"path"`
	}
	readJSON(r, &b)
	p, s, err := a.ws.OpenProject(b.Path)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	jsonOut(w, map[string]string{"project_id": p.ID, "session_id": s.ID})
}

func (a *App) handleProjectRemove(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ID string `json:"id"`
	}
	readJSON(r, &b)
	sids, err := a.ws.RemoveProject(b.ID)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	for _, sid := range sids {
		a.dropRunner(sid)
	}
	jsonOut(w, map[string]bool{"ok": true})
}

func (a *App) handleSessionNew(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ProjectID string `json:"project_id"`
	}
	readJSON(r, &b)
	s, err := a.ws.NewSession(b.ProjectID)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	jsonOut(w, map[string]string{"session_id": s.ID})
}

func (a *App) handleSessionDelete(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ProjectID string `json:"project_id"`
		ID        string `json:"id"`
	}
	readJSON(r, &b)
	if err := a.ws.DeleteSession(b.ProjectID, b.ID); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	a.dropRunner(b.ID)
	jsonOut(w, map[string]bool{"ok": true})
}

func (a *App) handleSessionRename(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ProjectID string `json:"project_id"`
		ID        string `json:"id"`
		Title     string `json:"title"`
	}
	readJSON(r, &b)
	if err := a.ws.Rename(b.ProjectID, b.ID, b.Title); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	jsonOut(w, map[string]bool{"ok": true})
}

// handleSession returns the saved conversation so the UI can redraw it, plus approvals still waiting.
func (a *App) handleSession(w http.ResponseWriter, r *http.Request) {
	pid, sid := r.URL.Query().Get("project_id"), r.URL.Query().Get("id")
	a.ws.mu.Lock()
	p, s := a.ws.findLocked(pid, sid)
	if p == nil || s == nil {
		a.ws.mu.Unlock()
		http.Error(w, "không có phiên chat", 404)
		return
	}
	log := append([]LogEntry{}, s.Log...)
	title := s.Title
	a.ws.mu.Unlock()
	pending := []map[string]any{}
	a.mu.Lock()
	run := a.runners[sid]
	a.mu.Unlock()
	busy := false
	if run != nil {
		run.mu.Lock()
		busy = run.busy
		for _, v := range run.pendInfo {
			pending = append(pending, v)
		}
		run.mu.Unlock()
	}
	jsonOut(w, map[string]any{"id": sid, "title": title, "log": log, "pending": pending, "busy": busy})
}

// ---------------------------------------------------------------- files

type entry struct {
	Name string `json:"name"`
	Dir  bool   `json:"dir"`
}

func listEntries(dir string, dirsOnly bool) ([]entry, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	out := []entry{} // never nil: the UI iterates over it (JSON null would crash it)
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
func (a *App) handleBrowse(w http.ResponseWriter, r *http.Request) {
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

func (a *App) projectRoot(r *http.Request) (string, error) {
	a.ws.mu.Lock()
	defer a.ws.mu.Unlock()
	p, _ := a.ws.findLocked(r.URL.Query().Get("project_id"), "")
	if p == nil {
		return "", fmt.Errorf("chưa chọn project")
	}
	return p.Path, nil
}

func (a *App) handleTree(w http.ResponseWriter, r *http.Request) {
	root, err := a.projectRoot(r)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
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

const maxViewBytes = 2 << 20

// handleFile returns the WHOLE text of a file for the viewer (not the paged chunk the LLM gets).
func (a *App) handleFile(w http.ResponseWriter, r *http.Request) {
	root, err := a.projectRoot(r)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	p, err := resolve(root, r.URL.Query().Get("path"))
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	st, err := os.Stat(p)
	if err != nil || st.IsDir() {
		http.Error(w, "không đọc được tệp", 400)
		return
	}
	var text string
	kind := "text"
	if isDocExt(p) {
		kind = "doc"
		text, err = readDocument(p)
		if err != nil {
			http.Error(w, err.Error(), 422)
			return
		}
	} else {
		f, err := os.Open(p)
		if err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		defer f.Close()
		buf := make([]byte, maxViewBytes+1)
		n, _ := f.Read(buf)
		buf = buf[:n]
		if bytesHasNUL(buf) {
			http.Error(w, "tệp nhị phân – không xem được dạng chữ", 415)
			return
		}
		text = string(buf)
	}
	trunc := false
	if len(text) > maxViewBytes {
		text, trunc = clipRunes(text, maxViewBytes), true
	}
	jsonOut(w, map[string]any{"content": text, "kind": kind, "truncated": trunc, "size": st.Size()})
}

func bytesHasNUL(b []byte) bool {
	for i := 0; i < len(b) && i < 8000; i++ {
		if b[i] == 0 {
			return true
		}
	}
	return false
}

// files that may be shown by the browser itself (the real PDF, images). SVG is excluded on purpose (scripts).
var rawTypes = map[string]string{
	".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
	".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".ico": "image/x-icon",
}

func (a *App) handleRaw(w http.ResponseWriter, r *http.Request) {
	root, err := a.projectRoot(r)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	p, err := resolve(root, r.URL.Query().Get("path"))
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	ct, ok := rawTypes[strings.ToLower(filepath.Ext(p))]
	if !ok {
		http.Error(w, "loại tệp không được xem trực tiếp", 415)
		return
	}
	f, err := os.Open(p)
	if err != nil {
		http.Error(w, err.Error(), 404)
		return
	}
	defer f.Close()
	st, _ := f.Stat()
	if st.IsDir() || st.Size() > 200<<20 {
		http.Error(w, "không xem được tệp này", 400)
		return
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Cache-Control", "no-store")
	http.ServeContent(w, r, filepath.Base(p), st.ModTime(), f)
}

var openAllowed = map[string]bool{".pdf": true, ".docx": true, ".xlsx": true, ".pptx": true, ".txt": true, ".csv": true, ".md": true,
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".webp": true, ".bmp": true, ".json": true, ".log": true}

// handleOpenExternal opens a document with the OS default app (Word, Excel…). Only document/image
// types: never executables or scripts.
func (a *App) handleOpenExternal(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ProjectID string `json:"project_id"`
		Path      string `json:"path"`
	}
	readJSON(r, &b)
	a.ws.mu.Lock()
	p, _ := a.ws.findLocked(b.ProjectID, "")
	a.ws.mu.Unlock()
	if p == nil {
		http.Error(w, "chưa chọn project", 400)
		return
	}
	full, err := resolve(p.Path, b.Path)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if !openAllowed[strings.ToLower(filepath.Ext(full))] {
		http.Error(w, "chỉ mở được tài liệu/ảnh bằng ứng dụng mặc định", 415)
		return
	}
	if st, err := os.Stat(full); err != nil || st.IsDir() {
		http.Error(w, "không có tệp", 404)
		return
	}
	var c *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		c = exec.Command("rundll32", "url.dll,FileProtocolHandler", full)
	case "darwin":
		c = exec.Command("open", full)
	default:
		c = exec.Command("xdg-open", full)
	}
	if err := c.Start(); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	go c.Wait()
	jsonOut(w, map[string]bool{"ok": true})
}

// ---------------------------------------------------------------- chat

func (a *App) handleChat(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ProjectID string `json:"project_id"`
		SessionID string `json:"session_id"`
		Message   string `json:"message"`
	}
	readJSON(r, &b)
	a.ws.mu.Lock()
	p, s := a.ws.findLocked(b.ProjectID, b.SessionID)
	a.ws.mu.Unlock()
	if p == nil || s == nil {
		http.Error(w, "chưa chọn project/phiên chat", 400)
		return
	}
	if st, err := os.Stat(p.Path); err != nil || !st.IsDir() {
		http.Error(w, "thư mục project không còn tồn tại: "+p.Path, 400)
		return
	}
	run := a.runner(p, s)
	run.mu.Lock()
	if run.busy {
		run.mu.Unlock()
		http.Error(w, "agent is busy", 409)
		return
	}
	run.busy = true
	run.mu.Unlock()
	defer func() { run.mu.Lock(); run.busy = false; run.mu.Unlock() }()

	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "no streaming", 500)
		return
	}
	// record the user's message + a title taken from the first one
	a.ws.mu.Lock()
	s.Log = append(s.Log, LogEntry{T: "user", Text: clipRunes(b.Message, 8000)})
	if s.Title == defaultTitle {
		s.Title = clipRunes(strings.Join(strings.Fields(b.Message), " "), 48)
	}
	s.Updated = time.Now()
	a.ws.saveLocked()
	a.ws.mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	emit := func(ev string, data any) {
		a.ws.record(s, ev, data)
		j, _ := json.Marshal(map[string]any{"event": ev, "data": data})
		fmt.Fprintf(w, "data: %s\n\n", j)
		fl.Flush()
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	run.mu.Lock()
	run.cancel = cancel
	run.mu.Unlock()
	run.runAgent(ctx, b.Message, emit)

	run.mu.Lock()
	hist := append([]Msg(nil), run.history...)
	run.mu.Unlock()
	a.ws.mu.Lock()
	s.History = toPersist(hist)
	s.Updated = time.Now()
	a.ws.saveLocked()
	a.ws.mu.Unlock()
	emit("done", nil)
}

func (a *App) handleApprove(w http.ResponseWriter, r *http.Request) {
	var b struct {
		ID     string `json:"id"`
		Allow  bool   `json:"allow"`
		Always bool   `json:"always"`
	}
	readJSON(r, &b)
	if b.Always && b.Allow {
		a.setAuto(true)
	}
	a.mu.Lock()
	runs := make([]*Server, 0, len(a.runners))
	for _, x := range a.runners {
		runs = append(runs, x)
	}
	a.mu.Unlock()
	found := false
	for _, x := range runs {
		x.mu.Lock()
		ch := x.pending[b.ID]
		x.mu.Unlock()
		if ch != nil {
			select {
			case ch <- b.Allow:
			default:
			}
			found = true
		}
	}
	jsonOut(w, map[string]bool{"ok": found})
}

func (a *App) setAuto(on bool) {
	a.mu.Lock()
	a.auto = on
	runs := make([]*Server, 0, len(a.runners))
	for _, x := range a.runners {
		runs = append(runs, x)
	}
	a.mu.Unlock()
	for _, x := range runs {
		x.mu.Lock()
		x.auto = on
		x.mu.Unlock()
	}
}

func (a *App) handleAuto(w http.ResponseWriter, r *http.Request) {
	var b struct {
		On bool `json:"on"`
	}
	readJSON(r, &b)
	a.setAuto(b.On)
	jsonOut(w, map[string]bool{"auto": b.On})
}

// handleCancel stops the running turn of one session (kills its command, frees its busy flag).
func (a *App) handleCancel(w http.ResponseWriter, r *http.Request) {
	var b struct {
		SessionID string `json:"session_id"`
	}
	readJSON(r, &b)
	a.mu.Lock()
	run := a.runners[b.SessionID]
	a.mu.Unlock()
	var c context.CancelFunc
	if run != nil {
		run.mu.Lock()
		c = run.cancel
		run.mu.Unlock()
	}
	if c != nil {
		c()
	}
	jsonOut(w, map[string]bool{"ok": c != nil})
}

// ---------------------------------------------------------------- scheduled tasks

func (a *App) handleTasks(w http.ResponseWriter, r *http.Request) {
	if sched == nil {
		jsonOut(w, map[string]any{"tasks": []any{}, "alerts": []any{}, "seq": 0})
		return
	}
	tasks, alerts, seq := sched.Snapshot()
	jsonOut(w, map[string]any{"tasks": tasks, "alerts": alerts, "seq": seq})
}

// handleTaskAction: buttons in the Tasks panel. Creating tasks is NOT possible here
// (it needs the approval card in chat); running / pausing / deleting is.
func (a *App) handleTaskAction(w http.ResponseWriter, r *http.Request) {
	var b struct {
		Action  string `json:"action"`
		ID      string `json:"id"`
		Enabled bool   `json:"enabled"`
	}
	readJSON(r, &b)
	if sched == nil {
		http.Error(w, "scheduler not running", 500)
		return
	}
	var err error
	switch b.Action {
	case "run":
		_, _, err = sched.RunNow(r.Context(), b.ID)
	case "delete":
		err = sched.Delete(b.ID)
	case "toggle":
		_, err = sched.Toggle(b.ID, b.Enabled)
	default:
		err = fmt.Errorf("unknown action")
	}
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	jsonOut(w, map[string]bool{"ok": true})
}

var _ = mime.TypeByExtension
