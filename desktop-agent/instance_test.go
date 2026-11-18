package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// point the app-data folder at a temp dir (Linux: XDG_CONFIG_HOME, Windows: APPDATA)
func isolateAppData(t *testing.T) string {
	d := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", d)
	t.Setenv("APPDATA", d)
	t.Setenv("HOME", d)
	return d
}

func TestSingleInstanceDiscovery(t *testing.T) {
	isolateAppData(t)
	if _, ok := runningInstance(); ok {
		t.Fatal("nothing is running yet")
	}
	a := newTestApp(t, "")
	info := instanceInfo{PID: os.Getpid(), Addr: a.app.addr, Token: a.app.token}
	writeInstance(info)

	got, ok := runningInstance()
	if !ok || got.Addr != a.app.addr || got.URL() != "http://"+a.app.addr+"/?t="+a.app.token {
		t.Fatalf("a live instance must be found: %+v %v", got, ok)
	}
	if runtime.GOOS != "windows" {
		if st, err := os.Stat(instancePath()); err != nil || st.Mode().Perm() != 0o600 {
			t.Errorf("instance.json holds the token: must be 0600, got %v %v", st, err)
		}
	}
	// wrong token → not "ours"
	writeInstance(instanceInfo{PID: 1, Addr: a.app.addr, Token: "not-the-token-0000000000"})
	if _, ok := runningInstance(); ok {
		t.Error("a wrong token must not count as a running instance")
	}
	// corrupt / incomplete files are ignored, never crash
	for _, junk := range []string{"{not json", `{"addr":""}`, ""} {
		os.WriteFile(instancePath(), []byte(junk), 0o600)
		if _, ok := runningInstance(); ok {
			t.Errorf("junk %q accepted", junk)
		}
	}
	// stale file left by a crash: the address answers nothing → start normally
	writeInstance(instanceInfo{PID: 99999, Addr: "127.0.0.1:1", Token: "abcdefabcdefabcdef"})
	st := time.Now()
	if _, ok := runningInstance(); ok || time.Since(st) > 3*time.Second {
		t.Errorf("stale instance must be ignored quickly: %v", time.Since(st))
	}
	// server stopped after the file was written → also stale
	writeInstance(info)
	a.srv.Close()
	if _, ok := runningInstance(); ok {
		t.Error("a stopped server is not a running instance")
	}
}

func TestRemoveInstanceOnlyRemovesOwnFile(t *testing.T) {
	isolateAppData(t)
	writeInstance(instanceInfo{PID: 1, Addr: "127.0.0.1:7878", Token: "tttttttttttttttttt"})
	removeInstance("127.0.0.1:9999") // some other process' address: must keep the file
	if _, err := os.Stat(instancePath()); err != nil {
		t.Fatal("removed a newer instance's file")
	}
	removeInstance("127.0.0.1:7878")
	if _, err := os.Stat(instancePath()); err == nil {
		t.Error("own file must be removed on exit")
	}
	removeInstance("x") // missing file: no panic
}

func TestSecondLaunchOpensProjectInRunningInstance(t *testing.T) {
	isolateAppData(t)
	a := newTestApp(t, "")
	first := mkProject(t, map[string]string{"a.txt": "1"})
	second := mkProject(t, map[string]string{"b.txt": "2"})
	pid1, _ := a.openProject(first)
	writeInstance(instanceInfo{PID: os.Getpid(), Addr: a.app.addr, Token: a.app.token})

	i, ok := runningInstance()
	if !ok {
		t.Fatal("not found")
	}
	if err := askRunningToOpen(i, second); err != nil {
		t.Fatal(err)
	}
	ws := a.workspace()
	projects := ws["projects"].([]any)
	if len(projects) != 2 || ws["active_project"] == pid1 {
		t.Fatalf("--project of a 2nd launch must open + activate the folder in the running copy: %+v", ws)
	}
	if err := askRunningToOpen(i, "/does/not/exist"); err == nil {
		t.Error("bad folder must report an error")
	}
	// the same folder again does not duplicate it
	askRunningToOpen(i, second)
	if n := len(a.workspace()["projects"].([]any)); n != 2 {
		t.Errorf("duplicate project: %d", n)
	}
}

func TestAutoExitRules(t *testing.T) {
	isolateAppData(t)
	a := newTestApp(t, "")
	app := a.app
	now := time.Now()
	idle := 3 * time.Minute

	app.lastSeen.Store(now.Add(-time.Minute).UnixNano())
	if app.shouldAutoExit(now, idle) {
		t.Error("a tab talked to us a minute ago: keep running")
	}
	app.lastSeen.Store(now.Add(-4 * time.Minute).UnixNano())
	if !app.shouldAutoExit(now, idle) {
		t.Error("no tab for 4 minutes and nothing running: must exit")
	}
	if app.shouldAutoExit(now, 0) {
		t.Error("auto exit disabled (0) must never exit")
	}
	// any authorised request (the page polls every few seconds) counts as "a tab is open"
	a.req("GET", "/api/state", nil)
	if app.shouldAutoExit(time.Now(), idle) {
		t.Error("an API call must refresh lastSeen")
	}
	app.lastSeen.Store(now.Add(-4 * time.Minute).UnixNano())

	// a running chat turn blocks the exit
	d := mkProject(t, map[string]string{"a.txt": "x"})
	app.auto, app.cfg.AutoApprove = true, true
	pid, sid := a.openProject(d)
	app.lastSeen.Store(now.Add(-4 * time.Minute).UnixNano())
	go a.chat(pid, sid, "chạy sleep 2", nil)
	time.Sleep(500 * time.Millisecond)
	app.lastSeen.Store(now.Add(-4 * time.Minute).UnixNano())
	if app.shouldAutoExit(now, idle) {
		t.Error("must not exit while a chat turn is running")
	}
	time.Sleep(2500 * time.Millisecond)
	app.lastSeen.Store(now.Add(-4 * time.Minute).UnixNano())
	if !app.shouldAutoExit(now, idle) {
		t.Error("after the turn finished it may exit again")
	}

	// an enabled scheduled task blocks the exit (monitoring must keep running); a paused one does not
	old := sched
	sched = newScheduler("", "")
	defer func() { sched = old }()
	tk, _ := sched.Add(TaskSpec{Name: "mon", Schedule: "every 1m", Command: "echo x"})
	if app.shouldAutoExit(now, idle) {
		t.Error("must not exit while a task is enabled")
	}
	if app.enabledTasks() != 1 {
		t.Error("enabledTasks")
	}
	sched.Toggle(tk.ID, false)
	if !app.shouldAutoExit(now, idle) || app.enabledTasks() != 0 {
		t.Error("a paused task must not keep the program alive")
	}
}

func TestAutoExitConfigDefaults(t *testing.T) {
	isolateAppData(t)
	t.Chdir(t.TempDir())
	if c := loadConfig(); c.AutoExitSeconds != 180 {
		t.Errorf("default auto exit: %d", c.AutoExitSeconds)
	}
	os.WriteFile("config.json", []byte(`{"auto_exit_seconds": 0}`), 0o644)
	if c := loadConfig(); c.AutoExitSeconds != 0 {
		t.Errorf("0 disables: %d", c.AutoExitSeconds)
	}
	os.WriteFile("config.json", []byte(`{"auto_exit_seconds": -5}`), 0o644)
	if c := loadConfig(); c.AutoExitSeconds != 0 {
		t.Errorf("negative = disabled: %d", c.AutoExitSeconds)
	}
	_ = filepath.Join
	_ = strings.TrimSpace
}
