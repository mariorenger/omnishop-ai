package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Single instance. Starting the program a second time must NOT start another copy: two copies would
// share workspace.json and both run the scheduled tasks (every task would run twice). Instead the
// second launch finds the running one through instance.json (address + token, written 0600 in the
// user's own config folder), opens a browser tab on it and exits.

type instanceInfo struct {
	PID   int    `json:"pid"`
	Addr  string `json:"addr"`
	Token string `json:"token"`
}

func (i instanceInfo) URL() string { return fmt.Sprintf("http://%s/?t=%s", i.Addr, i.Token) }

func appDataDir() string {
	d, err := os.UserConfigDir()
	if err != nil {
		return ""
	}
	return filepath.Join(d, "desktop-agent")
}

func instancePath() string {
	if d := appDataDir(); d != "" {
		return filepath.Join(d, "instance.json")
	}
	return ""
}

func writeInstance(i instanceInfo) {
	p := instancePath()
	if p == "" {
		return
	}
	_ = os.MkdirAll(filepath.Dir(p), 0o755)
	b, _ := json.Marshal(i)
	tmp := p + ".tmp"
	if os.WriteFile(tmp, b, 0o600) == nil {
		_ = os.Rename(tmp, p)
	}
}

// removeInstance deletes the file only if it still describes THIS process (never a newer instance).
func removeInstance(addr string) {
	p := instancePath()
	if p == "" {
		return
	}
	if b, err := os.ReadFile(p); err == nil {
		var i instanceInfo
		if json.Unmarshal(b, &i) == nil && i.Addr == addr {
			_ = os.Remove(p)
		}
	}
}

// runningInstance returns the live instance, if any. A file left behind by a crash / kill is ignored
// because nothing answers at its address (or the token does not match).
func runningInstance() (instanceInfo, bool) {
	p := instancePath()
	if p == "" {
		return instanceInfo{}, false
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return instanceInfo{}, false
	}
	var i instanceInfo
	if json.Unmarshal(b, &i) != nil || i.Addr == "" || i.Token == "" {
		return instanceInfo{}, false
	}
	req, _ := http.NewRequest("GET", "http://"+i.Addr+"/api/state", nil)
	req.Header.Set("X-Token", i.Token)
	resp, err := (&http.Client{Timeout: 1500 * time.Millisecond}).Do(req)
	if err != nil {
		return instanceInfo{}, false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return i, resp.StatusCode == 200
}

// askRunningToOpen tells the running instance to open (and activate) a project folder.
func askRunningToOpen(i instanceInfo, path string) error {
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	post := func(p string, body any, out any) error {
		b, _ := json.Marshal(body)
		req, _ := http.NewRequest("POST", "http://"+i.Addr+p, bytes.NewReader(b))
		req.Header.Set("X-Token", i.Token)
		req.Header.Set("Content-Type", "application/json")
		resp, err := (&http.Client{Timeout: 3 * time.Second}).Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			m, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
			return fmt.Errorf("%s: %s", p, m)
		}
		if out != nil {
			return json.NewDecoder(resp.Body).Decode(out)
		}
		return nil
	}
	var ids struct {
		ProjectID string `json:"project_id"`
		SessionID string `json:"session_id"`
	}
	if err := post("/api/project/open", map[string]string{"path": abs}, &ids); err != nil {
		return err
	}
	return post("/api/workspace/active", map[string]string{"project_id": ids.ProjectID, "session_id": ids.SessionID}, nil)
}

// ---------------------------------------------------------------- auto exit

// shouldAutoExit: the program ends itself when no browser tab has talked to it for `idle`, but NEVER
// while a chat turn is running or any scheduled task is enabled (monitoring has to keep working).
func (a *App) shouldAutoExit(now time.Time, idle time.Duration) bool {
	if idle <= 0 || now.Sub(time.Unix(0, a.lastSeen.Load())) < idle {
		return false
	}
	a.mu.Lock()
	ids := make([]string, 0, len(a.runners))
	for id := range a.runners {
		ids = append(ids, id)
	}
	a.mu.Unlock()
	for _, id := range ids {
		if a.busy(id) {
			return false
		}
	}
	if sched != nil {
		tasks, _, _ := sched.Snapshot()
		for _, t := range tasks {
			if t.Enabled {
				return false
			}
		}
	}
	return true
}

func (a *App) enabledTasks() int {
	n := 0
	if sched != nil {
		tasks, _, _ := sched.Snapshot()
		for _, t := range tasks {
			if t.Enabled {
				n++
			}
		}
	}
	return n
}
