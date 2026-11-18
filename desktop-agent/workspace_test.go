package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type testApp struct {
	t      *testing.T
	app    *App
	srv    *httptest.Server
	wsPath string
}

func newTestApp(t *testing.T, wsPath string) *testApp {
	t.Helper()
	if wsPath == "" {
		wsPath = filepath.Join(t.TempDir(), "workspace.json")
	}
	app := &App{cfg: Config{CmdTimeout: 60, ContextTokens: 16000, MaxOutputTokens: 1500}, token: "tok-1234567890abcdef",
		runners: map[string]*Server{}, ws: loadWorkspace(wsPath)}
	mux := http.NewServeMux()
	app.routes(mux)
	srv := httptest.NewServer(mux)
	app.addr = strings.TrimPrefix(srv.URL, "http://")
	t.Cleanup(srv.Close)
	return &testApp{t: t, app: app, srv: srv, wsPath: wsPath}
}

func (a *testApp) req(method, path string, body any) (int, []byte) {
	a.t.Helper()
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	r, _ := http.NewRequest(method, a.srv.URL+path, rd)
	r.Header.Set("X-Token", a.app.token)
	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		a.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, b
}

func (a *testApp) post(path string, body any, out any) int {
	a.t.Helper()
	st, b := a.req("POST", path, body)
	if out != nil && st == 200 {
		_ = json.Unmarshal(b, out)
	}
	return st
}

func (a *testApp) openProject(dir string) (pid, sid string) {
	a.t.Helper()
	var o map[string]string
	if st := a.post("/api/project/open", map[string]string{"path": dir}, &o); st != 200 {
		a.t.Fatalf("open project: %d", st)
	}
	return o["project_id"], o["session_id"]
}

type sseEvent struct {
	Event string          `json:"event"`
	Data  json.RawMessage `json:"data"`
}

// chat sends a message and collects all SSE events until the stream ends. onEvent may react (e.g. approve).
func (a *testApp) chat(pid, sid, msg string, onEvent func(sseEvent)) (status int, events []sseEvent) {
	a.t.Helper()
	b, _ := json.Marshal(map[string]string{"project_id": pid, "session_id": sid, "message": msg})
	r, _ := http.NewRequest("POST", a.srv.URL+"/api/chat", bytes.NewReader(b))
	r.Header.Set("X-Token", a.app.token)
	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		a.t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		io.Copy(io.Discard, resp.Body)
		return resp.StatusCode, nil
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 8<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var e sseEvent
		if json.Unmarshal([]byte(line[6:]), &e) == nil {
			events = append(events, e)
			if onEvent != nil {
				onEvent(e)
			}
		}
	}
	return 200, events
}

func texts(evs []sseEvent) string {
	var sb strings.Builder
	for _, e := range evs {
		sb.Write(e.Data)
		sb.WriteByte('\n')
	}
	return sb.String()
}

func (a *testApp) workspace() map[string]any {
	_, b := a.req("GET", "/api/workspace", nil)
	var o map[string]any
	_ = json.Unmarshal(b, &o)
	return o
}

func (a *testApp) session(pid, sid string) map[string]any {
	_, b := a.req("GET", "/api/session?project_id="+pid+"&id="+sid, nil)
	var o map[string]any
	_ = json.Unmarshal(b, &o)
	return o
}

func mkProject(t *testing.T, files map[string]string) string {
	d := t.TempDir()
	for p, c := range files {
		must(t, write(d, p, c))
	}
	return d
}

func TestAppGuard(t *testing.T) {
	a := newTestApp(t, "")
	r, _ := http.NewRequest("GET", a.srv.URL+"/api/state", nil)
	if resp, _ := http.DefaultClient.Do(r); resp.StatusCode != 403 {
		t.Errorf("no token must be 403, got %d", resp.StatusCode)
	}
	r.Header.Set("X-Token", "wrong")
	if resp, _ := http.DefaultClient.Do(r); resp.StatusCode != 403 {
		t.Errorf("wrong token must be 403")
	}
	r.Header.Set("X-Token", a.app.token)
	r.Host = "evil.example.com"
	if resp, _ := http.DefaultClient.Do(r); resp.StatusCode != 403 {
		t.Errorf("foreign Host must be 403 (DNS rebinding)")
	}
	if st, _ := a.req("GET", "/api/state", nil); st != 200 {
		t.Errorf("valid request: %d", st)
	}
	for _, p := range []string{"/api/workspace", "/api/tree?project_id=x", "/api/file?project_id=x", "/api/raw?project_id=x", "/api/session?id=x", "/api/tasks"} {
		resp, _ := http.Get(a.srv.URL + p)
		if resp.StatusCode != 403 {
			t.Errorf("%s reachable without token: %d", p, resp.StatusCode)
		}
	}
}

func TestProjectsAndSessions(t *testing.T) {
	a := newTestApp(t, "")
	pa := mkProject(t, map[string]string{"a.txt": "A"})
	pb := mkProject(t, map[string]string{"b.txt": "B"})
	pidA, sidA := a.openProject(pa)
	pidA2, sidA2 := a.openProject(pa) // same folder again → same project, no duplicate
	if pidA != pidA2 || sidA != sidA2 {
		t.Errorf("reopening must reuse the project: %s/%s vs %s/%s", pidA, sidA, pidA2, sidA2)
	}
	if st := a.post("/api/project/open", map[string]string{"path": "/does/not/exist"}, nil); st != 400 {
		t.Errorf("bad path: %d", st)
	}
	pidB, sidB := a.openProject(pb)
	ws := a.workspace()
	if len(ws["projects"].([]any)) != 2 || ws["active_project"] != pidB || ws["active_session"] != sidB {
		t.Fatalf("workspace: %+v", ws)
	}
	// isolation: each session works in ITS OWN project
	_, evA := a.chat(pidA, sidA, "ls", nil)
	_, evB := a.chat(pidB, sidB, "ls", nil)
	if !strings.Contains(texts(evA), "a.txt") || strings.Contains(texts(evA), "b.txt") {
		t.Errorf("project A leaked: %s", texts(evA))
	}
	if !strings.Contains(texts(evB), "b.txt") || strings.Contains(texts(evB), "a.txt") {
		t.Errorf("project B leaked: %s", texts(evB))
	}
	// several sessions per project; a blank session is reused instead of piling up
	var n1, n2 map[string]string
	a.post("/api/session/new", map[string]string{"project_id": pidA}, &n1)
	a.post("/api/session/new", map[string]string{"project_id": pidA}, &n2)
	if n1["session_id"] == "" || n1["session_id"] != n2["session_id"] || n1["session_id"] == sidA {
		t.Errorf("blank session must be reused: %v %v", n1, n2)
	}
	// history of the two sessions is separate
	a.chat(pidA, n1["session_id"], "đọc a.txt", nil)
	s1 := a.session(pidA, sidA)
	s2 := a.session(pidA, n1["session_id"])
	if s1["title"] != "ls" || s2["title"] != "đọc a.txt" {
		t.Errorf("titles come from the first message: %v / %v", s1["title"], s2["title"])
	}
	if strings.Contains(jsonStr(s1["log"]), "đọc a.txt") || strings.Contains(jsonStr(s2["log"]), `"ls"`) {
		t.Errorf("sessions share a log")
	}
	// rename / delete
	if st := a.post("/api/session/rename", map[string]string{"project_id": pidA, "id": sidA, "title": "  Kiểm kê  "}, nil); st != 200 {
		t.Fatal(st)
	}
	if a.session(pidA, sidA)["title"] != "Kiểm kê" {
		t.Error("rename")
	}
	if st := a.post("/api/session/rename", map[string]string{"project_id": pidA, "id": sidA, "title": " "}, nil); st != 400 {
		t.Error("empty title must be rejected")
	}
	a.post("/api/session/delete", map[string]string{"project_id": pidA, "id": sidA}, nil)
	if st, _ := a.req("GET", "/api/session?project_id="+pidA+"&id="+sidA, nil); st != 404 {
		t.Errorf("deleted session still readable: %d", st)
	}
	a.app.mu.Lock()
	_, still := a.app.runners[sidA]
	a.app.mu.Unlock()
	if still {
		t.Error("runner of a deleted session must be dropped")
	}
	// deleting the last session leaves one blank one
	a.post("/api/session/delete", map[string]string{"project_id": pidA, "id": n1["session_id"]}, nil)
	ws = a.workspace()
	for _, p := range ws["projects"].([]any) {
		pm := p.(map[string]any)
		if pm["id"] == pidA && len(pm["sessions"].([]any)) != 1 {
			t.Errorf("a project always keeps one session: %+v", pm["sessions"])
		}
	}
	// remove a project
	a.post("/api/project/remove", map[string]string{"id": pidB}, nil)
	ws = a.workspace()
	if len(ws["projects"].([]any)) != 1 || ws["active_project"] != pidA {
		t.Errorf("remove project: %+v", ws)
	}
	if st := a.post("/api/project/remove", map[string]string{"id": pidB}, nil); st != 400 {
		t.Error("removing twice must fail")
	}
}

func jsonStr(v any) string { b, _ := json.Marshal(v); return string(b) }

func TestWorkspacePersistsAcrossRestart(t *testing.T) {
	dir := mkProject(t, map[string]string{"a.txt": "xin chào"})
	a := newTestApp(t, "")
	pid, sid := a.openProject(dir)
	a.chat(pid, sid, "đọc a.txt", nil)
	// simulate an LLM tool-result message with Meta so we can check it survives
	a.app.mu.Lock()
	run := a.app.runners[sid]
	a.app.mu.Unlock()
	run.mu.Lock()
	run.history = append(run.history, Msg{Role: "user", Meta: "result", Content: "Kết quả của `x`:\nok"})
	run.mu.Unlock()
	a.app.ws.mu.Lock()
	a.app.ws.Projects[0].Sessions[0].History = toPersist(run.history)
	a.app.ws.saveLocked()
	a.app.ws.mu.Unlock()

	b := newTestApp(t, a.wsPath) // "restart"
	ws := b.workspace()
	if len(ws["projects"].([]any)) != 1 {
		t.Fatalf("projects lost: %+v", ws)
	}
	got := b.session(pid, sid)
	if got["title"] != "đọc a.txt" || !strings.Contains(jsonStr(got["log"]), "xin chào") {
		t.Fatalf("conversation lost: %+v", got)
	}
	r := b.app.runner(b.app.ws.Projects[0], b.app.ws.Projects[0].Sessions[0])
	var haveMeta bool
	for _, m := range r.history {
		if m.Meta == "result" {
			haveMeta = true
		}
	}
	if !haveMeta || len(r.history) < 2 { // (offline mode keeps no LLM turns; the Meta marker is what must survive)
		t.Errorf("LLM history (with Meta) must be restored so the model keeps its context: %+v", r.history)
	}
	// the restored session keeps working
	if st, ev := b.chat(pid, sid, "ls", nil); st != 200 || !strings.Contains(texts(ev), "a.txt") {
		t.Errorf("chat after restart: %d %s", st, texts(ev))
	}
	// corrupt file → starts empty instead of crashing
	os.WriteFile(a.wsPath, []byte("{not json"), 0o644)
	if w := loadWorkspace(a.wsPath); len(w.Projects) != 0 {
		t.Error("corrupt workspace must load empty")
	}
}

func TestSessionsRunConcurrentlyAndBusyIsPerSession(t *testing.T) {
	d := mkProject(t, map[string]string{"a.txt": "x"})
	a := newTestApp(t, "")
	a.app.auto, a.app.cfg.AutoApprove = true, true
	pid, s1 := a.openProject(d)
	var n map[string]string
	a.chat(pid, s1, "ls", nil) // make s1 non-blank so New creates a second one
	a.post("/api/session/new", map[string]string{"project_id": pid}, &n)
	s2 := n["session_id"]
	if s2 == s1 {
		t.Fatal("need two sessions")
	}
	cmd := "chạy sleep 1.5"
	var wg sync.WaitGroup
	start := time.Now()
	for _, sid := range []string{s1, s2} {
		wg.Add(1)
		go func(sid string) { defer wg.Done(); a.chat(pid, sid, cmd, nil) }(sid)
	}
	time.Sleep(500 * time.Millisecond)
	// while both run: same session → 409, workspace reports both busy
	if st, _ := a.chat(pid, s1, "ls", nil); st != 409 {
		t.Errorf("second message in a busy session must be 409, got %d", st)
	}
	busy := 0
	for _, p := range a.workspace()["projects"].([]any) {
		for _, s := range p.(map[string]any)["sessions"].([]any) {
			if s.(map[string]any)["busy"] == true {
				busy++
			}
		}
	}
	if busy != 2 {
		t.Errorf("want 2 busy sessions, got %d", busy)
	}
	wg.Wait()
	if el := time.Since(start); el > 2900*time.Millisecond {
		t.Errorf("sessions did not run in parallel: %v", el)
	}
	if st, ev := a.chat(pid, s1, "ls", nil); st != 200 || !strings.Contains(texts(ev), "a.txt") {
		t.Errorf("session usable again after finishing: %d", st)
	}
}

func TestCancelIsPerSession(t *testing.T) {
	d := mkProject(t, map[string]string{"a.txt": "x"})
	a := newTestApp(t, "")
	a.app.auto, a.app.cfg.AutoApprove = true, true
	pid, s1 := a.openProject(d)
	a.chat(pid, s1, "ls", nil)
	var n map[string]string
	a.post("/api/session/new", map[string]string{"project_id": pid}, &n)
	s2 := n["session_id"]
	done1, done2 := make(chan time.Duration, 1), make(chan time.Duration, 1)
	go func() { st := time.Now(); a.chat(pid, s1, "chạy sleep 30", nil); done1 <- time.Since(st) }()
	go func() { st := time.Now(); a.chat(pid, s2, "chạy sleep 2", nil); done2 <- time.Since(st) }()
	time.Sleep(700 * time.Millisecond)
	a.post("/api/cancel", map[string]string{"session_id": s1}, nil)
	select {
	case el := <-done1:
		if el > 4*time.Second {
			t.Errorf("cancel too slow: %v", el)
		}
	case <-time.After(6 * time.Second):
		t.Fatal("cancelled session never returned")
	}
	if el := <-done2; el < 1800*time.Millisecond {
		t.Errorf("cancelling session 1 must NOT stop session 2 (finished after %v)", el)
	}
	if st, _ := a.chat(pid, s1, "ls", nil); st != 200 {
		t.Errorf("cancelled session must accept new messages, got %d", st)
	}
}

func TestApprovalFlowAcrossHTTP(t *testing.T) {
	d := mkProject(t, map[string]string{"a.txt": "x"})
	a := newTestApp(t, "")
	pid, sid := a.openProject(d)
	// user clicks "Cho phép"
	approve := func(e sseEvent) {
		if e.Event == "approval" {
			var m map[string]any
			json.Unmarshal(e.Data, &m)
			a.post("/api/approve", map[string]any{"id": m["id"], "allow": true}, nil)
		}
	}
	_, ev := a.chat(pid, sid, "tạo moi.txt : xin chào", approve)
	if b, _ := os.ReadFile(filepath.Join(d, "moi.txt")); string(b) != "xin chào" {
		t.Errorf("approved write did not happen: %q / %s", b, texts(ev))
	}
	// user clicks "Từ chối"
	deny := func(e sseEvent) {
		if e.Event == "approval" {
			var m map[string]any
			json.Unmarshal(e.Data, &m)
			a.post("/api/approve", map[string]any{"id": m["id"], "allow": false}, nil)
		}
	}
	a.chat(pid, sid, "xóa a.txt", deny)
	if _, err := os.Stat(filepath.Join(d, "a.txt")); err != nil {
		t.Error("denied delete happened")
	}
	// a pending approval is visible when the user comes back to the session
	gotPending := make(chan int, 1)
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		a.chat(pid, sid, "xóa a.txt", func(e sseEvent) {
			if e.Event == "approval" {
				s := a.session(pid, sid)
				gotPending <- len(s["pending"].([]any))
				var m map[string]any
				json.Unmarshal(e.Data, &m)
				a.post("/api/approve", map[string]any{"id": m["id"], "allow": false}, nil)
			}
		})
	}()
	select {
	case n := <-gotPending:
		if n != 1 {
			t.Errorf("pending approval must be listed in /api/session, got %d", n)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no approval event")
	}
	<-finished
}

func TestMissingProjectFolder(t *testing.T) {
	d := mkProject(t, map[string]string{"a.txt": "x"})
	a := newTestApp(t, "")
	pid, sid := a.openProject(d)
	os.RemoveAll(d)
	ws := a.workspace()
	if ws["projects"].([]any)[0].(map[string]any)["missing"] != true {
		t.Error("missing folder must be flagged")
	}
	if st, _ := a.chat(pid, sid, "ls", nil); st != 400 {
		t.Errorf("chat in a missing folder: %d", st)
	}
}

func TestFileViewerEndpoints(t *testing.T) {
	d := mkProject(t, map[string]string{"long.txt": strings.Repeat("dòng dài tiếng Việt\n", 3000), "x.exe": "MZ", "n.txt": "ok"})
	copyFile(t, "testdata/vi.pdf", filepath.Join(d, "vi.pdf"))
	copyFile(t, "testdata/hd.docx", filepath.Join(d, "hd.docx"))
	must(t, os.WriteFile(filepath.Join(d, "bin.dat"), []byte{0, 1, 2}, 0o644))
	a := newTestApp(t, "")
	pid, _ := a.openProject(d)
	q := "?project_id=" + pid + "&path="

	// the viewer gets the WHOLE text, not the LLM's paged chunk
	st, b := a.req("GET", "/api/file"+q+"long.txt", nil)
	var f map[string]any
	json.Unmarshal(b, &f)
	if st != 200 || len(f["content"].(string)) < 50000 || strings.Contains(f["content"].(string), "offset=") {
		t.Errorf("viewer must not be paged: status %d len %d", st, len(f["content"].(string)))
	}
	st, b = a.req("GET", "/api/file"+q+"hd.docx", nil)
	if st != 200 || !strings.Contains(string(b), "Hợp đồng dịch vụ") {
		t.Errorf("docx text: %d %s", st, b)
	}
	if st, _ = a.req("GET", "/api/file"+q+"bin.dat", nil); st != 415 {
		t.Errorf("binary → 415, got %d", st)
	}
	if st, _ = a.req("GET", "/api/file"+q+"../../etc/passwd", nil); st != 400 {
		t.Errorf("escape → 400, got %d", st)
	}

	// the REAL pdf is served untouched (so the browser's PDF viewer shows it correctly)
	r, _ := http.NewRequest("GET", a.srv.URL+"/api/raw"+q+"vi.pdf", nil)
	r.Header.Set("X-Token", a.app.token)
	resp, _ := http.DefaultClient.Do(r)
	got, _ := io.ReadAll(resp.Body)
	want, _ := os.ReadFile("testdata/vi.pdf")
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "application/pdf" || !bytes.Equal(got, want) {
		t.Errorf("raw pdf: %d %s equal=%v", resp.StatusCode, resp.Header.Get("Content-Type"), bytes.Equal(got, want))
	}
	if resp.Header.Get("X-Content-Type-Options") != "nosniff" {
		t.Error("nosniff header missing")
	}
	// range requests work (big PDFs load progressively)
	r.Header.Set("Range", "bytes=0-9")
	resp, _ = http.DefaultClient.Do(r)
	if resp.StatusCode != 206 {
		t.Errorf("range → 206, got %d", resp.StatusCode)
	}
	for _, p := range []string{"x.exe", "n.txt", "../../etc/passwd", "hd.docx"} {
		if st, _ = a.req("GET", "/api/raw"+q+p, nil); st != 415 && st != 400 {
			t.Errorf("raw %q must be refused, got %d", p, st)
		}
	}
	// open with default app: only documents/images
	if st := a.post("/api/open", map[string]string{"project_id": pid, "path": "x.exe"}, nil); st != 415 {
		t.Errorf("must never launch an exe, got %d", st)
	}
	if st := a.post("/api/open", map[string]string{"project_id": pid, "path": "../../etc/passwd"}, nil); st != 400 {
		t.Errorf("escape: %d", st)
	}
	if st := a.post("/api/open", map[string]string{"project_id": "nope", "path": "vi.pdf"}, nil); st != 400 {
		t.Errorf("unknown project: %d", st)
	}
	if st, _ := a.req("GET", "/api/tree?project_id="+pid+"&path=.", nil); st != 200 {
		t.Errorf("tree: %d", st)
	}
	if st, _ := a.req("GET", "/api/tree?project_id=zzz&path=.", nil); st != 400 {
		t.Errorf("tree of unknown project: %d", st)
	}
}

func TestEmptyListsAreArraysNotNull(t *testing.T) {
	d := mkProject(t, map[string]string{"x.txt": "1"}) // has files but NO sub-folders
	empty := t.TempDir()
	a := newTestApp(t, "")
	pid, _ := a.openProject(empty)
	_, b := a.req("GET", "/api/browse?path="+d, nil)
	if strings.Contains(string(b), `"dirs":null`) {
		t.Errorf("dirs must be [] not null: %s", b)
	}
	_, b = a.req("GET", "/api/tree?project_id="+pid+"&path=.", nil)
	if strings.Contains(string(b), `"entries":null`) {
		t.Errorf("entries must be [] not null: %s", b)
	}
	_, b = a.req("GET", "/api/workspace", nil)
	if strings.Contains(string(b), "null") {
		t.Errorf("workspace JSON contains null: %s", b)
	}
	_, b = a.req("GET", "/api/tasks", nil)
	if strings.Contains(string(b), `"tasks":null`) {
		t.Errorf("tasks must be []: %s", b)
	}
}
