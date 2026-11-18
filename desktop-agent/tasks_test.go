package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestParseSchedule(t *testing.T) {
	ok := map[string]schedule{
		"every 30s": {Every: 30 * time.Second}, "mỗi 30s": {Every: 30 * time.Second}, "30s": {Every: 30 * time.Second},
		"mỗi 5 phút": {Every: 5 * time.Minute}, "every 5m": {Every: 5 * time.Minute}, "5m": {Every: 5 * time.Minute},
		"mỗi giờ": {Every: time.Hour}, "every 2h": {Every: 2 * time.Hour}, "mỗi 2 giờ": {Every: 2 * time.Hour},
		"mỗi ngày": {Every: 24 * time.Hour}, "every 1d": {Every: 24 * time.Hour}, "mỗi 90 giây": {Every: 90 * time.Second},
		"daily 08:30": {Daily: true, HH: 8, MM: 30}, "hàng ngày 8h30": {Daily: true, HH: 8, MM: 30},
		"mỗi ngày lúc 7h": {Daily: true, HH: 7}, "Hàng ngày 18:05": {Daily: true, HH: 18, MM: 5}, "daily 9": {Daily: true, HH: 9},
	}
	for in, want := range ok {
		got, err := parseSchedule(in)
		if err != nil || got != want {
			t.Errorf("%q => %+v, %v; want %+v", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "bla", "every", "every 5x", "mỗi 5s", "daily 25:00", "daily 8:99", "0m", "mỗi 0 phút"} {
		if _, err := parseSchedule(bad); err == nil {
			t.Errorf("%q should be rejected", bad)
		}
	}
	// next(): interval and daily
	now := time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)
	if n := (schedule{Every: time.Minute}).next(now); !n.Equal(now.Add(time.Minute)) {
		t.Error("interval next")
	}
	if n := (schedule{Daily: true, HH: 8, MM: 30}).next(now); n.Day() != 11 || n.Hour() != 8 || n.Minute() != 30 {
		t.Errorf("daily after time passed should be tomorrow: %v", n)
	}
	if n := (schedule{Daily: true, HH: 18}).next(now); n.Day() != 10 || n.Hour() != 18 {
		t.Errorf("daily before time should be today: %v", n)
	}
}

func newSched(t *testing.T) *Scheduler {
	t.Helper()
	s := newScheduler(filepath.Join(t.TempDir(), "tasks.json"), "")
	old := sched
	sched = s
	t.Cleanup(func() { sched = old })
	return s
}

func echoOK() string   { return "echo hello-ok" }
func exitFail() string { return "exit 3" }

func TestTaskAlertsAndRecovery(t *testing.T) {
	s := newSched(t)
	dir := t.TempDir()
	mustWrite(t, dir+"/state.txt", "up")
	catState := "cat state.txt"
	if runtime.GOOS == "windows" {
		catState = "type state.txt"
	}
	j, err := s.Add(TaskSpec{Name: "web", Schedule: "every 1m", Command: catState, Dir: dir, AlertOnFail: true, AlertContains: "DOWN"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if r := s.execute(ctx, j); r.Status != "ok" || !strings.Contains(r.Output, "up") {
		t.Fatalf("first run: %+v", r)
	}
	mustWrite(t, dir+"/state.txt", "service DOWN")
	if r := s.execute(ctx, j); r.Status != "alert" || !strings.Contains(r.Reason, "DOWN") {
		t.Fatalf("should alert: %+v", r)
	}
	s.execute(ctx, j) // still down: must NOT alert again (no spam)
	s.execute(ctx, j)
	_, alerts, _ := s.Snapshot()
	if len(alerts) != 1 || alerts[0].Kind != "alert" {
		t.Fatalf("want exactly 1 alert while still failing, got %+v", alerts)
	}
	mustWrite(t, dir+"/state.txt", "up again")
	s.execute(ctx, j)
	_, alerts, seq := s.Snapshot()
	if len(alerts) != 2 || alerts[1].Kind != "recovered" || seq != 2 {
		t.Fatalf("want alert + recovered, got %+v", alerts)
	}
	// history is capped
	for i := 0; i < maxHistory+5; i++ {
		s.execute(ctx, j)
	}
	if v, _, _ := s.Snapshot(); len(v[0].History) != maxHistory {
		t.Errorf("history not capped: %d", len(v[0].History))
	}
}

func TestTaskRules(t *testing.T) {
	s := newSched(t)
	dir := t.TempDir()
	ctx := context.Background()
	fail, _ := s.Add(TaskSpec{Name: "fail", Schedule: "every 1m", Command: exitFail(), Dir: dir, AlertOnFail: true})
	if r := s.execute(ctx, fail); r.Status != "alert" || r.Exit != 3 {
		t.Errorf("non-zero exit: %+v", r)
	}
	quiet, _ := s.Add(TaskSpec{Name: "quiet", Schedule: "every 1m", Command: exitFail(), Dir: dir, AlertOnFail: false})
	if r := s.execute(ctx, quiet); r.Status != "ok" {
		t.Errorf("alert_on_fail=false must not alert: %+v", r)
	}
	miss, _ := s.Add(TaskSpec{Name: "miss", Schedule: "every 1m", Command: echoOK(), Dir: dir, AlertOnFail: true, AlertNotContains: "Reply"})
	if r := s.execute(ctx, miss); r.Status != "alert" || !strings.Contains(r.Reason, "không có") {
		t.Errorf("alert_not_contains: %+v", r)
	}
	has, _ := s.Add(TaskSpec{Name: "has", Schedule: "every 1m", Command: echoOK(), Dir: dir, AlertOnFail: true, AlertContains: "HELLO"}) // case-insensitive
	if r := s.execute(ctx, has); r.Status != "alert" {
		t.Errorf("alert_contains must be case-insensitive: %+v", r)
	}
	if runtime.GOOS != "windows" {
		to, _ := s.Add(TaskSpec{Name: "slow", Schedule: "every 1m", Command: "sleep 20", Dir: dir, TimeoutSec: 1, AlertOnFail: true})
		st := time.Now()
		r := s.execute(ctx, to)
		if r.Status != "timeout" || time.Since(st) > 6*time.Second {
			t.Errorf("timeout task: %+v after %v", r, time.Since(st))
		}
	}
}

func TestTaskValidationAndManagement(t *testing.T) {
	s := newSched(t)
	for _, sp := range []TaskSpec{
		{Name: "", Schedule: "every 1m", Command: "x"}, {Name: "a", Schedule: "every 1m", Command: ""},
		{Name: "a", Schedule: "never", Command: "x"}, {Name: "a", Schedule: "mỗi 5s", Command: "x"},
	} {
		if _, err := s.Add(sp); err == nil {
			t.Errorf("should reject %+v", sp)
		}
	}
	a, _ := s.Add(TaskSpec{Name: "Kiểm tra API", Schedule: "every 1m", Command: echoOK()})
	if _, err := s.Add(TaskSpec{Name: "kiểm tra api", Schedule: "every 1m", Command: "x"}); err == nil {
		t.Error("duplicate name (case-insensitive) must be rejected")
	}
	b, _ := s.Add(TaskSpec{Name: "Kiểm tra DB", Schedule: "daily 08:00", Command: echoOK()})
	if a.ID != "1" || b.ID != "2" {
		t.Errorf("ids: %s %s", a.ID, b.ID)
	}
	if _, err := s.Toggle("kiểm tra", true); err == nil {
		t.Error("ambiguous name must error")
	}
	if j, err := s.Toggle("api", false); err != nil || j.Enabled {
		t.Errorf("toggle by unique substring: %v", err)
	}
	if err := s.Delete("2"); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete("2"); err == nil {
		t.Error("deleting twice must error")
	}
	for i := 0; i < maxTasks; i++ {
		s.Add(TaskSpec{Name: "j" + itoa(i), Schedule: "every 1m", Command: "x"})
	}
	if _, err := s.Add(TaskSpec{Name: "overflow", Schedule: "every 1m", Command: "x"}); err == nil {
		t.Error("task cap not enforced")
	}
}

func TestTaskPersistenceAndNoReplay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tasks.json")
	s := newScheduler(path, "")
	j, _ := s.Add(TaskSpec{Name: "p", Schedule: "every 1m", Command: echoOK(), AlertOnFail: true})
	s.execute(context.Background(), j)
	// pretend the app was closed for a long time
	s.mu.Lock()
	j.NextRun = time.Now().Add(-48 * time.Hour)
	s.saveLocked()
	s.mu.Unlock()

	s2 := newScheduler(path, "")
	v, _, _ := s2.Snapshot()
	if len(v) != 1 || v[0].Name != "p" || len(v[0].History) != 1 {
		t.Fatalf("reload lost data: %+v", v)
	}
	if !v[0].NextRun.After(time.Now()) {
		t.Error("missed runs must not be replayed after restart")
	}
	if s2.nextID != 2 {
		t.Errorf("nextID %d", s2.nextID)
	}
	b, _ := os.ReadFile(path)
	var raw []map[string]any
	if json.Unmarshal(b, &raw) != nil {
		t.Error("tasks.json is not valid JSON")
	}
}

func TestSchedulerTickRunsDueTasks(t *testing.T) {
	s := newSched(t)
	fake := time.Now()
	s.now = func() time.Time { return fake }
	j, _ := s.Add(TaskSpec{Name: "t", Schedule: "every 1m", Command: echoOK(), AlertOnFail: true})
	s.tick()
	time.Sleep(300 * time.Millisecond)
	if v, _, _ := s.Snapshot(); len(v[0].History) != 0 {
		t.Fatal("must not run before it is due")
	}
	fake = fake.Add(61 * time.Second)
	s.tick()
	s.wg.Wait()
	v, _, _ := s.Snapshot()
	if len(v[0].History) != 1 || v[0].History[0].Status != "ok" {
		t.Fatalf("due task did not run: %+v", v[0].History)
	}
	s.tick() // not due again yet
	s.wg.Wait()
	if v, _, _ := s.Snapshot(); len(v[0].History) != 1 {
		t.Fatal("task ran twice for one period")
	}
	// paused task never runs
	s.Toggle(j.ID, false)
	fake = fake.Add(10 * time.Minute)
	s.tick()
	s.wg.Wait()
	if v, _, _ := s.Snapshot(); len(v[0].History) != 1 {
		t.Fatal("paused task ran")
	}
}

func TestTaskTools(t *testing.T) {
	newSched(t)
	d := newProj(t)
	out := run(d, "create_task", `{"name":"ls","schedule":"every 1m","command":"echo xin chao","alert_contains":"LỖI"}`)
	wantHas(t, out, "đã tạo task #1")
	wantHas(t, run(d, "create_task", `{"name":"ls","schedule":"every 1m","command":"x"}`), "đã có task")
	wantHas(t, run(d, "create_task", `{"name":"bad","schedule":"khi nào rảnh","command":"x"}`), "không hiểu lịch")
	wantHas(t, run(d, "list_tasks", `{}`), "#1 ls")
	wantHas(t, run(d, "run_task", `{"id":"1"}`), "xin chao")
	wantHas(t, run(d, "list_tasks", `{"id":"ls"}`), "lệnh: echo xin chao")
	wantHas(t, run(d, "toggle_task", `{"id":"1","enabled":"false"}`), "tạm dừng") // string bool from a weak model
	wantHas(t, run(d, "toggle_task", `{"id":"1","enabled":true}`), "đã bật")
	wantHas(t, run(d, "delete_task", `{"id":"1"}`), "đã xóa")
	wantHas(t, run(d, "list_tasks", `{}`), "chưa có task")
	wantHas(t, run(d, "run_task", `{"id":"9"}`), "error:")
	// weak-model argument shapes: aliases + numeric strings
	wantHas(t, run(d, "create_task", `{"name":"w","every":"mỗi 5 phút","cmd":"echo a","timeout":"30"}`), "đã tạo task")
	if needsApproval("create_task") == false || needsApproval("delete_task") == false {
		t.Error("creating/deleting tasks must require approval")
	}
}

func TestScriptTask(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip()
	}
	s := newSched(t)
	_ = s
	d := newProj(t)
	cmd, err := scriptCommand(t.TempDir(), "7", "sh", "echo from-script-$((1+2))\n")
	must(t, err)
	r := runShell(context.Background(), d, cmd, 10)
	wantHas(t, r.Out, "from-script-3")
	if _, err := scriptCommand(t.TempDir(), "8", "cobol", "x"); err == nil {
		t.Error("unknown language must error")
	}
	// via the tool, with python if available
	if _, err := os.Stat("/usr/bin/python3"); err == nil {
		out := run(d, "create_task", `{"name":"py","schedule":"daily 08:30","language":"python","script":"print(6*7)"}`)
		wantHas(t, out, "đã tạo task")
		wantHas(t, run(d, "run_task", `{"id":"py"}`), "42")
	}
}

func TestOfflineTaskCommands(t *testing.T) {
	newSched(t)
	b := newBot(t)
	b.expect("task", "chưa có task")
	b.expect("job", "chưa có task") // gõ tắt cũ vẫn dùng được
	b.expect("jobs", "chưa có task")
	b.expect("task thêm hello | mỗi 1 phút | echo xin chao báo nếu thiếu chao", "đã tạo task #1")
	b.expect("task thêm | x", "Cú pháp")
	b.expect("task thêm a | bla | echo", "không hiểu lịch")
	b.expect("task", "#1 hello", "mỗi 1 phút")
	b.expect("task chạy hello", "xin chao")
	b.expect("task log 1", "lệnh: echo xin chao", "[ok]")
	b.expect("task dừng 1", "tạm dừng")
	b.expect("task tạm dừng 1", "tạm dừng")
	b.expect("task bật hello", "đã bật")
	b.expect("task thêm báo cáo | hàng ngày 08:30 | echo done timeout 20", "đã tạo task #2")
	b.expect("tasks", "#2 báo cáo", "hàng ngày 08:30")
	b.expect("task xóa 2", "đã xóa")
	b.expect("task xóa hello", "đã xóa")
	b.expect("task linh tinh", "Lệnh task")
	// command containing pipes/quotes is preserved
	b.expect(`task thêm p | mỗi 30 giây | echo "a | b"`, "đã tạo task")
	v, _, _ := sched.Snapshot()
	if v[0].Command != `echo "a | b"` {
		t.Errorf("command mangled: %q", v[0].Command)
	}
}

func TestLegacyJobsFileIsStillLoaded(t *testing.T) {
	dir := t.TempDir()
	old := `[{"id":"5","name":"cũ","schedule":"every 1m","command":"echo x","enabled":true,"alert_on_fail":true,"next_run":"2000-01-01T00:00:00Z","history":[]}]`
	mustWrite(t, dir+"/jobs.json", old)
	s := newScheduler(dir+"/tasks.json", "")
	v, _, _ := s.Snapshot()
	if len(v) != 1 || v[0].Name != "cũ" || s.nextID != 6 {
		t.Fatalf("legacy jobs.json not migrated: %+v", v)
	}
	if _, err := s.Add(TaskSpec{Name: "mới", Schedule: "every 1m", Command: "x"}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir + "/tasks.json"); err != nil {
		t.Error("must save under the new name")
	}
}
