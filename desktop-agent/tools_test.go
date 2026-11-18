package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func newProj(t *testing.T) string {
	t.Helper()
	d := t.TempDir()
	must(t, os.WriteFile(filepath.Join(d, "a.txt"), []byte("hello TODO world\nline2\n"), 0o644))
	must(t, os.MkdirAll(filepath.Join(d, "sub", "deep"), 0o755))
	must(t, os.WriteFile(filepath.Join(d, "sub", "b.py"), []byte("print('x')\n"), 0o644))
	return d
}
func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func run(root, name, args string) string {
	return runTool(context.Background(), root, name, args)
}
func wantHas(t *testing.T, got, sub string) {
	t.Helper()
	if !strings.Contains(got, sub) {
		t.Fatalf("want %q in %q", sub, got)
	}
}

func TestSandbox(t *testing.T) {
	d := newProj(t)
	for _, p := range []string{"../x", "../../etc/passwd", "sub/../../x", "/etc/passwd", `..\x`} {
		if runtime.GOOS != "windows" && strings.Contains(p, `\`) {
			continue
		}
		if out := run(d, "read_file", `{"path":`+q(p)+`}`); !strings.HasPrefix(out, "error:") {
			t.Errorf("escape %q allowed: %s", p, out)
		}
	}
	// symlink pointing outside must be blocked
	outside := t.TempDir()
	must(t, os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("s"), 0o644))
	if err := os.Symlink(outside, filepath.Join(d, "link")); err == nil {
		if out := run(d, "read_file", `{"path":"link/secret.txt"}`); !strings.HasPrefix(out, "error:") {
			t.Errorf("symlink escape allowed: %s", out)
		}
		if out := run(d, "write_file", `{"path":"link/new.txt","content":"x"}`); !strings.HasPrefix(out, "error:") {
			t.Errorf("symlink write escape allowed: %s", out)
		}
	}
	if run("", "list_dir", `{"path":"."}`) != "error: no project selected" {
		t.Error("no project must error")
	}
}
func q(s string) string {
	return `"` + strings.ReplaceAll(strings.ReplaceAll(s, `\`, `\\`), `"`, `\"`) + `"`
}

func TestListReadTreeSearch(t *testing.T) {
	d := newProj(t)
	wantHas(t, run(d, "list_dir", `{"path":"."}`), "a.txt")
	wantHas(t, run(d, "list_dir", `{"path":"."}`), "sub/")
	wantHas(t, run(d, "list_dir", `{"path":"sub/deep"}`), "empty")
	wantHas(t, run(d, "list_dir", `{"path":"nope"}`), "error:")
	wantHas(t, run(d, "read_file", `{"path":"a.txt"}`), "hello TODO")
	wantHas(t, run(d, "read_file", `{"path":"nope.txt"}`), "error:")
	wantHas(t, run(d, "read_file", `{"path":"sub"}`), "error:") // directory
	wantHas(t, run(d, "tree", `{"path":"."}`), "└── ")
	wantHas(t, run(d, "search_files", `{"pattern":"TODO"}`), "a.txt:1:")
	wantHas(t, run(d, "search_files", `{"pattern":"zzzz"}`), "no matches")
	wantHas(t, run(d, "search_files", `{"pattern":"("}`), "bad pattern")
	must(t, os.WriteFile(filepath.Join(d, "bin.dat"), []byte{0, 1, 2, 3}, 0o644))
	wantHas(t, run(d, "read_file", `{"path":"bin.dat"}`), "binary")
}

func TestWriteEditAppendMoveDelete(t *testing.T) {
	d := newProj(t)
	wantHas(t, run(d, "write_file", `{"path":"x/y/new.txt","content":"xin chào"}`), "wrote")
	b, _ := os.ReadFile(filepath.Join(d, "x/y/new.txt"))
	if string(b) != "xin chào" {
		t.Fatalf("content %q", b)
	}
	wantHas(t, run(d, "write_file", `{"path":"tên file có dấu.txt","content":"ok"}`), "wrote")
	wantHas(t, run(d, "read_file", `{"path":"tên file có dấu.txt"}`), "ok")

	wantHas(t, run(d, "edit_file", `{"path":"a.txt","old_string":"world","new_string":"agent"}`), "edited")
	wantHas(t, run(d, "read_file", `{"path":"a.txt"}`), "hello TODO agent")
	wantHas(t, run(d, "edit_file", `{"path":"a.txt","old_string":"zzz","new_string":"q"}`), "found 0 times")
	must(t, os.WriteFile(filepath.Join(d, "dup.txt"), []byte("aa aa"), 0o644))
	wantHas(t, run(d, "edit_file", `{"path":"dup.txt","old_string":"aa","new_string":"b"}`), "found 2 times")
	wantHas(t, run(d, "edit_file", `{"path":"a.txt","old_string":"","new_string":"b"}`), "empty")

	wantHas(t, run(d, "append_file", `{"path":"a.txt","content":"dòng mới"}`), "appended")
	wantHas(t, run(d, "read_file", `{"path":"a.txt"}`), "dòng mới")
	wantHas(t, run(d, "append_file", `{"path":"fresh.txt","content":"1"}`), "appended")

	wantHas(t, run(d, "move_file", `{"from":"fresh.txt","to":"moved/f2.txt"}`), "moved")
	wantHas(t, run(d, "move_file", `{"from":"moved/f2.txt","to":"a.txt"}`), "already exists")
	wantHas(t, run(d, "move_file", `{"from":"nope","to":"z"}`), "error:")
	wantHas(t, run(d, "move_file", `{"from":"a.txt","to":"../out.txt"}`), "outside")

	wantHas(t, run(d, "delete_file", `{"path":"moved/f2.txt"}`), "deleted")
	wantHas(t, run(d, "delete_file", `{"path":"sub"}`), "error:") // non-empty dir
	wantHas(t, run(d, "delete_file", `{"path":"sub/deep"}`), "deleted")
	wantHas(t, run(d, "delete_file", `{"path":"nope"}`), "error:")
	wantHas(t, run(d, "delete_file", `{"path":"."}`), "root")
	if _, err := os.Stat(d); err != nil {
		t.Fatal("project root was deleted!")
	}
}

func TestRunCommand(t *testing.T) {
	d := newProj(t)
	wantHas(t, run(d, "run_command", `{"command":"echo xin-chao"}`), "xin-chao")
	// runs inside the project dir
	cmd := "pwd"
	if runtime.GOOS == "windows" {
		cmd = "cd"
	}
	out := run(d, "run_command", `{"command":"`+cmd+`"}`)
	real, _ := filepath.EvalSymlinks(d)
	if !strings.Contains(out, filepath.Base(real)) {
		t.Errorf("not in project dir: %q", out)
	}
	// quotes survive
	wantHas(t, run(d, "run_command", `{"command":"echo \"a b\""}`), "a b")
	// non-zero exit is reported
	ex := "exit 3"
	wantHas(t, run(d, "run_command", `{"command":"`+ex+`"}`), "exit status 3")
	// stderr is captured
	wantHas(t, run(d, "run_command", `{"command":"echo oops 1>&2"}`), "oops")
	wantHas(t, run(d, "run_command", `{"command":""}`), "")
	if runtime.GOOS == "windows" {
		return
	}
	// timeout kills the command (and returns on time)
	st := time.Now()
	out = run(d, "run_command", `{"command":"echo start; sleep 30","timeout_sec":1}`)
	if time.Since(st) > 5*time.Second {
		t.Fatalf("timeout not enforced: %v", time.Since(st))
	}
	wantHas(t, out, "hết giờ sau 1s")
	wantHas(t, out, "start") // partial output kept
	// background child must not block us
	st = time.Now()
	out = run(d, "run_command", `{"command":"sleep 20 &"}`)
	if time.Since(st) > 5*time.Second {
		t.Fatalf("background child blocked: %v", time.Since(st))
	}
	wantHas(t, out, "tiến trình nền")
	// cancel (Stop button) returns quickly
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(500 * time.Millisecond); cancel() }()
	st = time.Now()
	out = runTool(ctx, d, "run_command", `{"command":"sleep 30"}`)
	if time.Since(st) > 4*time.Second {
		t.Fatalf("cancel not honoured: %v", time.Since(st))
	}
	wantHas(t, out, "đã dừng")
}

func TestEffectiveTimeout(t *testing.T) {
	defaultCmdTimeout = 60
	for in, want := range map[int]int{0: 60, -5: 60, 5: 5, 120: 120, 99999: maxCmdTimeout} {
		if got := effectiveTimeout(in); got != want {
			t.Errorf("effectiveTimeout(%d)=%d want %d", in, got, want)
		}
	}
	if commandTimeoutOf(`{"command":"x","timeout_sec":7}`) != 7 || commandTimeoutOf(`{"command":"x"}`) != 60 {
		t.Error("commandTimeoutOf wrong")
	}
}
