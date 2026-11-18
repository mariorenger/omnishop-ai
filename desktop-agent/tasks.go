package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ------------------------------------------------------------------ schedule

type schedule struct {
	Daily  bool
	Every  time.Duration
	HH, MM int
}

const minInterval = 10 * time.Second

var (
	reDaily = regexp.MustCompile(`^(?:daily|everyday|hàng ngày|hằng ngày|mỗi ngày|hang ngay|moi ngay)\s*(?:lúc|at|luc)?\s*(\d{1,2})\s*(?:[:h]\s*(\d{1,2})?)?\s*(?:giờ)?$`)
	reEvery = regexp.MustCompile(`^(?:every|mỗi|moi)?\s*(\d*)\s*(s|sec|secs|seconds?|giây|giay|m|min|mins|minutes?|phút|phut|h|hr|hrs|hours?|giờ|gio|d|days?|ngày|ngay)$`)
)

// parseSchedule understands: "every 30s", "mỗi 5 phút", "30m", "mỗi giờ", "daily 08:30", "hàng ngày 8h30".
func parseSchedule(s string) (schedule, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	if m := reDaily.FindStringSubmatch(s); m != nil {
		hh, _ := strconv.Atoi(m[1])
		mm := 0
		if m[2] != "" {
			mm, _ = strconv.Atoi(m[2])
		}
		if hh > 23 || mm > 59 {
			return schedule{}, fmt.Errorf("giờ không hợp lệ: %q", s)
		}
		return schedule{Daily: true, HH: hh, MM: mm}, nil
	}
	if m := reEvery.FindStringSubmatch(s); m != nil {
		n := 1
		if m[1] != "" {
			n, _ = strconv.Atoi(m[1])
		}
		if n <= 0 {
			return schedule{}, fmt.Errorf("chu kỳ phải > 0")
		}
		var unit time.Duration
		switch m[2][0] {
		case 's', 'g': // s, sec, giây, giay
			unit = time.Second
			if m[2][0] == 'g' && (strings.HasPrefix(m[2], "gi") && (m[2] == "giờ" || m[2] == "gio")) {
				unit = time.Hour
			}
		case 'm', 'p':
			unit = time.Minute
		case 'h':
			unit = time.Hour
		case 'd', 'n':
			unit = 24 * time.Hour
		}
		d := time.Duration(n) * unit
		if d < minInterval {
			return schedule{}, fmt.Errorf("chu kỳ tối thiểu là %v", minInterval)
		}
		return schedule{Every: d}, nil
	}
	return schedule{}, fmt.Errorf("không hiểu lịch %q. Ví dụ: \"mỗi 30s\", \"mỗi 5 phút\", \"mỗi giờ\", \"hàng ngày 08:30\"", s)
}

func (sc schedule) next(after time.Time) time.Time {
	if sc.Daily {
		t := time.Date(after.Year(), after.Month(), after.Day(), sc.HH, sc.MM, 0, 0, after.Location())
		if !t.After(after) {
			t = t.AddDate(0, 0, 1)
		}
		return t
	}
	return after.Add(sc.Every)
}

// ------------------------------------------------------------------ model

type Run struct {
	At     time.Time `json:"at"`
	Status string    `json:"status"` // ok | alert | error | timeout
	Exit   int       `json:"exit"`
	Reason string    `json:"reason,omitempty"`
	Output string    `json:"output"`
	DurMs  int64     `json:"dur_ms"`
}

type Task struct {
	ID               string    `json:"id"`
	Name             string    `json:"name"`
	Schedule         string    `json:"schedule"`
	Command          string    `json:"command"`
	Dir              string    `json:"dir"`
	TimeoutSec       int       `json:"timeout_sec"`
	AlertOnFail      bool      `json:"alert_on_fail"`
	AlertContains    string    `json:"alert_contains,omitempty"`
	AlertNotContains string    `json:"alert_not_contains,omitempty"`
	Enabled          bool      `json:"enabled"`
	CreatedAt        time.Time `json:"created_at"`
	NextRun          time.Time `json:"next_run"`
	History          []Run     `json:"history"`
	Running          bool      `json:"-"`
	sc               schedule
}

type Alert struct {
	Seq      int       `json:"seq"`
	At       time.Time `json:"at"`
	TaskID   string    `json:"task_id"`
	TaskName string    `json:"task_name"`
	Kind     string    `json:"kind"` // alert | recovered
	Message  string    `json:"message"`
}

type Scheduler struct {
	mu      sync.Mutex
	path    string
	tasks   []*Task
	nextID  int
	alerts  []Alert
	seq     int
	webhook string
	now     func() time.Time
	stop    chan struct{}
	wg      sync.WaitGroup
}

const (
	maxTasks    = 30
	maxHistory  = 20
	maxTaskOut  = 1500
	maxAlertLog = 50
)

var sched *Scheduler

func newScheduler(path, webhook string) *Scheduler {
	s := &Scheduler{path: path, webhook: webhook, now: time.Now, stop: make(chan struct{}), nextID: 1}
	s.load()
	return s
}

func (s *Scheduler) load() {
	if s.path == "" {
		return
	}
	b, err := os.ReadFile(s.path)
	if err != nil { // data saved by an older version under the previous name
		b, err = os.ReadFile(filepath.Join(filepath.Dir(s.path), "jobs.json"))
		if err != nil {
			return
		}
	}
	var tasks []*Task
	if json.Unmarshal(b, &tasks) != nil {
		return
	}
	now := s.now()
	for _, j := range tasks {
		sc, err := parseSchedule(j.Schedule)
		if err != nil {
			continue
		}
		j.sc = sc
		// don't replay what was missed while the app was closed
		if !j.NextRun.After(now) {
			j.NextRun = sc.next(now)
		}
		if n, err := strconv.Atoi(j.ID); err == nil && n >= s.nextID {
			s.nextID = n + 1
		}
		s.tasks = append(s.tasks, j)
	}
}

// saveLocked must be called with s.mu held.
func (s *Scheduler) saveLocked() {
	if s.path == "" {
		return
	}
	b, _ := json.MarshalIndent(s.tasks, "", "  ")
	_ = os.MkdirAll(filepath.Dir(s.path), 0o755)
	tmp := s.path + ".tmp"
	if os.WriteFile(tmp, b, 0o644) == nil {
		_ = os.Rename(tmp, s.path)
	}
}

func (s *Scheduler) Start() {
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		t := time.NewTicker(time.Second)
		defer t.Stop()
		for {
			select {
			case <-s.stop:
				return
			case <-t.C:
				s.tick()
			}
		}
	}()
}

func (s *Scheduler) Stop() { close(s.stop); s.wg.Wait() }

func (s *Scheduler) tick() {
	now := s.now()
	var due []*Task
	s.mu.Lock()
	for _, j := range s.tasks {
		if j.Enabled && !j.Running && !j.NextRun.After(now) {
			j.Running = true
			j.NextRun = j.sc.next(now)
			due = append(due, j)
		}
	}
	s.mu.Unlock()
	for _, j := range due {
		j := j
		s.wg.Add(1)
		go func() { defer s.wg.Done(); s.execute(context.Background(), j) }()
	}
}

// execute runs a task once, records history, raises alerts on state changes.
func (s *Scheduler) execute(ctx context.Context, j *Task) Run {
	s.mu.Lock()
	j.Running = true
	cmd, dir, tsec := j.Command, j.Dir, j.TimeoutSec
	s.mu.Unlock()
	if dir == "" {
		dir, _ = os.Getwd()
	}
	start := s.now()
	r := runShell(ctx, dir, cmd, effectiveTimeout(tsec))
	out := strings.TrimSpace(r.Out)
	run := Run{At: start, Exit: r.Exit, Output: clip(out, maxTaskOut), DurMs: time.Since(start).Milliseconds(), Status: "ok"}

	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case r.TimedOut:
		run.Status, run.Reason = "timeout", fmt.Sprintf("quá %ds, đã dừng lệnh", effectiveTimeout(j.TimeoutSec))
	case j.AlertOnFail && (r.Err != nil || r.Exit != 0):
		run.Status, run.Reason = "alert", fmt.Sprintf("lệnh lỗi (exit %d)", r.Exit)
	}
	if run.Status == "ok" {
		if j.AlertContains != "" && strings.Contains(strings.ToLower(out), strings.ToLower(j.AlertContains)) {
			run.Status, run.Reason = "alert", fmt.Sprintf("output có chứa %q", j.AlertContains)
		} else if j.AlertNotContains != "" && !strings.Contains(strings.ToLower(out), strings.ToLower(j.AlertNotContains)) {
			run.Status, run.Reason = "alert", fmt.Sprintf("output không có %q", j.AlertNotContains)
		}
	}
	prevBad := false
	if n := len(j.History); n > 0 {
		prevBad = j.History[n-1].Status != "ok"
	}
	bad := run.Status != "ok"
	j.History = append(j.History, run)
	if len(j.History) > maxHistory {
		j.History = j.History[len(j.History)-maxHistory:]
	}
	j.Running = false
	switch {
	case bad && !prevBad: // only on transition, to avoid spamming every run
		s.raiseLocked(j, "alert", fmt.Sprintf("⚠️ Task %q: %s\n%s", j.Name, run.Reason, clip(out, 300)))
	case !bad && prevBad:
		s.raiseLocked(j, "recovered", fmt.Sprintf("✅ Task %q đã hoạt động bình thường trở lại", j.Name))
	}
	s.saveLocked()
	return run
}

func (s *Scheduler) raiseLocked(j *Task, kind, msg string) {
	s.seq++
	a := Alert{Seq: s.seq, At: s.now(), TaskID: j.ID, TaskName: j.Name, Kind: kind, Message: msg}
	s.alerts = append(s.alerts, a)
	if len(s.alerts) > maxAlertLog {
		s.alerts = s.alerts[len(s.alerts)-maxAlertLog:]
	}
	if s.webhook != "" {
		go postWebhook(s.webhook, msg)
	}
}

func postWebhook(url, text string) {
	b, _ := json.Marshal(map[string]string{"text": text, "content": text})
	c := &http.Client{Timeout: 10 * time.Second}
	if resp, err := c.Post(url, "application/json", bytes.NewReader(b)); err == nil {
		resp.Body.Close()
	}
}

// ------------------------------------------------------------------ management

type TaskSpec struct {
	Name, Schedule, Command, Dir    string
	TimeoutSec                      int
	AlertOnFail                     bool
	AlertContains, AlertNotContains string
}

func (s *Scheduler) Add(sp TaskSpec) (*Task, error) {
	sp.Name = strings.TrimSpace(sp.Name)
	if sp.Name == "" {
		return nil, fmt.Errorf("thiếu tên task")
	}
	if strings.TrimSpace(sp.Command) == "" {
		return nil, fmt.Errorf("thiếu lệnh/mã cần chạy")
	}
	sc, err := parseSchedule(sp.Schedule)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.tasks) >= maxTasks {
		return nil, fmt.Errorf("tối đa %d task", maxTasks)
	}
	for _, o := range s.tasks {
		if strings.EqualFold(o.Name, sp.Name) {
			return nil, fmt.Errorf("đã có task tên %q", sp.Name)
		}
	}
	j := &Task{ID: strconv.Itoa(s.nextID), Name: sp.Name, Schedule: strings.TrimSpace(sp.Schedule), Command: sp.Command, Dir: sp.Dir,
		TimeoutSec: sp.TimeoutSec, AlertOnFail: sp.AlertOnFail, AlertContains: sp.AlertContains, AlertNotContains: sp.AlertNotContains,
		Enabled: true, CreatedAt: s.now(), sc: sc}
	s.nextID++
	j.NextRun = sc.next(s.now())
	s.tasks = append(s.tasks, j)
	s.saveLocked()
	return j, nil
}

// find by id, exact name, or unique name substring.
func (s *Scheduler) findLocked(key string) (*Task, error) {
	key = strings.TrimSpace(key)
	if key == "" {
		return nil, fmt.Errorf("thiếu id hoặc tên task")
	}
	for _, j := range s.tasks {
		if j.ID == key || strings.EqualFold(j.Name, key) {
			return j, nil
		}
	}
	var m []*Task
	for _, j := range s.tasks {
		if strings.Contains(strings.ToLower(j.Name), strings.ToLower(key)) {
			m = append(m, j)
		}
	}
	if len(m) == 1 {
		return m[0], nil
	}
	if len(m) > 1 {
		return nil, fmt.Errorf("%q khớp %d task, hãy dùng id", key, len(m))
	}
	return nil, fmt.Errorf("không có task %q", key)
}

func (s *Scheduler) Delete(key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, err := s.findLocked(key)
	if err != nil {
		return err
	}
	for i, o := range s.tasks {
		if o == j {
			s.tasks = append(s.tasks[:i], s.tasks[i+1:]...)
			break
		}
	}
	s.saveLocked()
	return nil
}

func (s *Scheduler) Toggle(key string, on bool) (*Task, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, err := s.findLocked(key)
	if err != nil {
		return nil, err
	}
	j.Enabled = on
	if on {
		j.NextRun = j.sc.next(s.now())
	}
	s.saveLocked()
	return j, nil
}

// RunNow executes a task immediately and waits for it.
func (s *Scheduler) RunNow(ctx context.Context, key string) (Run, *Task, error) {
	s.mu.Lock()
	j, err := s.findLocked(key)
	if err == nil && j.Running {
		err = fmt.Errorf("task %q đang chạy", j.Name)
	}
	s.mu.Unlock()
	if err != nil {
		return Run{}, nil, err
	}
	return s.execute(ctx, j), j, nil
}

// TaskView is the JSON-friendly snapshot for the UI.
type TaskView struct {
	ID               string    `json:"id"`
	Name             string    `json:"name"`
	Schedule         string    `json:"schedule"`
	Command          string    `json:"command"`
	Enabled          bool      `json:"enabled"`
	Running          bool      `json:"running"`
	NextRun          time.Time `json:"next_run"`
	Last             *Run      `json:"last"`
	History          []Run     `json:"history"`
	AlertOnFail      bool      `json:"alert_on_fail"`
	AlertContains    string    `json:"alert_contains"`
	AlertNotContains string    `json:"alert_not_contains"`
	TimeoutSec       int       `json:"timeout_sec"`
}

func (s *Scheduler) Snapshot() ([]TaskView, []Alert, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]TaskView, 0, len(s.tasks))
	for _, j := range s.tasks {
		v := TaskView{ID: j.ID, Name: j.Name, Schedule: j.Schedule, Command: j.Command, Enabled: j.Enabled, Running: j.Running,
			NextRun: j.NextRun, History: append([]Run(nil), j.History...), AlertOnFail: j.AlertOnFail,
			AlertContains: j.AlertContains, AlertNotContains: j.AlertNotContains, TimeoutSec: j.TimeoutSec}
		if n := len(j.History); n > 0 {
			l := j.History[n-1]
			v.Last = &l
		}
		out = append(out, v)
	}
	sort.Slice(out, func(a, b int) bool { x, _ := strconv.Atoi(out[a].ID); y, _ := strconv.Atoi(out[b].ID); return x < y })
	return out, append([]Alert(nil), s.alerts...), s.seq
}

// ------------------------------------------------------------------ script tasks

// scriptCommand saves code to a file and returns the command that runs it.
func scriptCommand(dir, id, lang, code string) (string, error) {
	lang = strings.ToLower(strings.TrimSpace(lang))
	var ext, cmd string
	switch lang {
	case "python", "py", "":
		ext = ".py"
		py := "python"
		if runtime.GOOS != "windows" {
			py = "python3"
		}
		cmd = py + ` "%s"`
	case "powershell", "ps1", "pwsh":
		ext = ".ps1"
		cmd = `powershell -NoProfile -ExecutionPolicy Bypass -File "%s"`
	case "bat", "cmd", "batch":
		ext = ".bat"
		cmd = `"%s"`
	case "sh", "bash":
		ext = ".sh"
		cmd = `sh "%s"`
	default:
		return "", fmt.Errorf("ngôn ngữ %q chưa hỗ trợ (python, powershell, bat, sh)", lang)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	p := filepath.Join(dir, id+ext)
	if err := os.WriteFile(p, []byte(code), 0o755); err != nil {
		return "", err
	}
	return fmt.Sprintf(cmd, p), nil
}

func tasksDir() string {
	d, err := os.UserConfigDir()
	if err != nil {
		d = os.TempDir()
	}
	return filepath.Join(d, "desktop-agent", "tasks")
}

// ------------------------------------------------------------------ tool entry

func fmtTime(t time.Time) string { return t.Local().Format("02/01 15:04:05") }

func (j TaskView) line() string {
	st := "chưa chạy"
	if j.Last != nil {
		st = map[string]string{"ok": "✅ ok", "alert": "⚠️ báo động", "timeout": "⏱️ hết giờ", "error": "❌ lỗi"}[j.Last.Status] + " lúc " + fmtTime(j.Last.At)
	}
	en := ""
	if !j.Enabled {
		en = " [TẠM DỪNG]"
	}
	if j.Running {
		en += " [đang chạy]"
	}
	return fmt.Sprintf("#%s %s%s — %s — lần sau %s — %s", j.ID, j.Name, en, j.Schedule, fmtTime(j.NextRun), st)
}

func taskTool(ctx context.Context, root, name string, a targs) string {
	if sched == nil {
		return "error: bộ lập lịch chưa khởi động"
	}
	switch name {
	case "create_task":
		cmd := a.str("command")
		code := a.str("script")
		if code == "" {
			if v, ok := a["code"]; ok {
				code, _ = v.(string)
			}
		}
		s := sched
		s.mu.Lock()
		id := strconv.Itoa(s.nextID)
		s.mu.Unlock()
		if code != "" {
			var err error
			cmd, err = scriptCommand(tasksDir(), id, a.str("language"), code)
			if err != nil {
				return "error: " + err.Error()
			}
		}
		j, err := s.Add(TaskSpec{Name: a.str("name"), Schedule: a.str("schedule"), Command: cmd, Dir: root,
			TimeoutSec: a.num("timeout_sec"), AlertOnFail: a.boolean("alert_on_fail", true),
			AlertContains: a.str("alert_contains"), AlertNotContains: a.str("alert_not_contains")})
		if err != nil {
			return "error: " + err.Error()
		}
		return fmt.Sprintf("đã tạo task #%s %q, lịch %q, lần chạy đầu: %s", j.ID, j.Name, j.Schedule, fmtTime(j.NextRun))
	case "list_tasks":
		tasks, _, _ := sched.Snapshot()
		key := a.str("id")
		if key == "" {
			key = a.str("name")
		}
		if key != "" {
			for _, j := range tasks {
				if j.ID == key || strings.EqualFold(j.Name, key) {
					var sb strings.Builder
					sb.WriteString(j.line() + "\nlệnh: " + j.Command + "\n")
					for i := len(j.History) - 1; i >= 0 && i >= len(j.History)-5; i-- {
						r := j.History[i]
						fmt.Fprintf(&sb, "- %s [%s] exit=%d %s\n  %s\n", fmtTime(r.At), r.Status, r.Exit, r.Reason, strings.ReplaceAll(clip(r.Output, 300), "\n", "\n  "))
					}
					return sb.String()
				}
			}
			return "error: không có task " + key
		}
		if len(tasks) == 0 {
			return "(chưa có task nào)"
		}
		var lines []string
		for _, j := range tasks {
			lines = append(lines, j.line())
		}
		return strings.Join(lines, "\n")
	case "run_task":
		run, j, err := sched.RunNow(ctx, a.str("id")+a.str("name"))
		if err != nil {
			return "error: " + err.Error()
		}
		return fmt.Sprintf("task %q chạy xong: %s exit=%d (%dms) %s\n%s", j.Name, run.Status, run.Exit, run.DurMs, run.Reason, run.Output)
	case "delete_task":
		if err := sched.Delete(a.str("id") + a.str("name")); err != nil {
			return "error: " + err.Error()
		}
		return "đã xóa task"
	case "toggle_task":
		j, err := sched.Toggle(a.str("id")+a.str("name"), a.boolean("enabled", true))
		if err != nil {
			return "error: " + err.Error()
		}
		if j.Enabled {
			return fmt.Sprintf("task %q đã bật, lần sau %s", j.Name, fmtTime(j.NextRun))
		}
		return fmt.Sprintf("task %q đã tạm dừng", j.Name)
	}
	return "error: unknown task tool"
}
