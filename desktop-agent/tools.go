package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const (
	maxReadBytes   = 200 * 1024
	maxOutputBytes = 20 * 1024
)

// toolDefs are sent to the LLM (OpenAI function-calling format).
var toolDefs = []map[string]any{
	fn("list_dir", "List files and folders in a directory of the project (relative path, '.' for root).",
		props{"path": str("relative directory path")}, "path"),
	fn("read_file", "Read a text file from the project.",
		props{"path": str("relative file path")}, "path"),
	fn("write_file", "Create or overwrite a file in the project with the given content.",
		props{"path": str("relative file path"), "content": str("full file content")}, "path", "content"),
	fn("edit_file", "Replace exactly one occurrence of old_string with new_string in a file. old_string must be unique.",
		props{"path": str("relative file path"), "old_string": str("text to find (must occur exactly once)"), "new_string": str("replacement text")},
		"path", "old_string", "new_string"),
	fn("tree", "Show a directory tree (max depth 3).",
		props{"path": str("relative directory, default '.'")}),
	fn("append_file", "Append text to the end of a file (creates it if missing).",
		props{"path": str("relative file path"), "content": str("text to append")}, "path", "content"),
	fn("delete_file", "Delete a file or an empty directory.",
		props{"path": str("relative path")}, "path"),
	fn("move_file", "Rename or move a file/directory inside the project.",
		props{"from": str("relative source path"), "to": str("relative destination path")}, "from", "to"),
	fn("search_files", "Search project files for a regex pattern. Returns file:line: text.",
		props{"pattern": str("regular expression"), "path": str("relative directory, default '.'")}, "pattern"),
	fn("run_command", "Run a shell command in the project directory (cmd on Windows, sh elsewhere). Returns combined output.",
		props{"command": str("command line"), "timeout_sec": map[string]any{"type": "integer", "description": "default 60, max 300"}}, "command"),
}

type props map[string]any

func str(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }

func fn(name, desc string, p props, required ...string) map[string]any {
	return map[string]any{"type": "function", "function": map[string]any{
		"name": name, "description": desc,
		"parameters": map[string]any{"type": "object", "properties": p, "required": required},
	}}
}

// needsApproval: tools that change state.
func needsApproval(name string) bool {
	switch name {
	case "write_file", "edit_file", "run_command", "append_file", "delete_file", "move_file":
		return true
	}
	return false
}

// resolve maps a user/LLM supplied path to an absolute path guaranteed to stay inside root.
func resolve(root, p string) (string, error) {
	if root == "" {
		return "", fmt.Errorf("no project selected")
	}
	if p == "" {
		p = "."
	}
	abs := p
	if !filepath.IsAbs(p) {
		abs = filepath.Join(root, p)
	}
	abs = filepath.Clean(abs)
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", err
	}
	// resolve symlinks of the deepest existing ancestor
	probe, tail := abs, ""
	for {
		if _, err := os.Lstat(probe); err == nil {
			break
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			break
		}
		tail = filepath.Join(filepath.Base(probe), tail)
		probe = parent
	}
	realProbe, err := filepath.EvalSymlinks(probe)
	if err != nil {
		return "", err
	}
	real := filepath.Join(realProbe, tail)
	rel, err := filepath.Rel(realRoot, real)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q is outside the project", p)
	}
	return real, nil
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + fmt.Sprintf("\n...[truncated %d bytes]", len(s)-n)
}

// runTool executes one tool call and returns text for the LLM.
func runTool(ctx context.Context, root, name, rawArgs string) string {
	var a struct {
		Path       string `json:"path"`
		Content    string `json:"content"`
		OldString  string `json:"old_string"`
		NewString  string `json:"new_string"`
		Pattern    string `json:"pattern"`
		Command    string `json:"command"`
		TimeoutSec int    `json:"timeout_sec"`
		From       string `json:"from"`
		To         string `json:"to"`
	}
	if err := json.Unmarshal([]byte(rawArgs), &a); err != nil {
		return "error: invalid arguments: " + err.Error()
	}
	switch name {
	case "list_dir":
		p, err := resolve(root, a.Path)
		if err != nil {
			return "error: " + err.Error()
		}
		ents, err := os.ReadDir(p)
		if err != nil {
			return "error: " + err.Error()
		}
		var out []string
		for _, e := range ents {
			if e.Name() == ".git" {
				continue
			}
			n := e.Name()
			if e.IsDir() {
				n += "/"
			}
			out = append(out, n)
			if len(out) >= 500 {
				out = append(out, "...[more entries omitted]")
				break
			}
		}
		if len(out) == 0 {
			return "(empty directory)"
		}
		return strings.Join(out, "\n")
	case "read_file":
		p, err := resolve(root, a.Path)
		if err != nil {
			return "error: " + err.Error()
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return "error: " + err.Error()
		}
		if bytes.IndexByte(b[:min(len(b), 8000)], 0) >= 0 {
			return "error: binary file"
		}
		return clip(string(b), maxReadBytes)
	case "write_file":
		p, err := resolve(root, a.Path)
		if err != nil {
			return "error: " + err.Error()
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return "error: " + err.Error()
		}
		if err := os.WriteFile(p, []byte(a.Content), 0o644); err != nil {
			return "error: " + err.Error()
		}
		return fmt.Sprintf("wrote %d bytes to %s", len(a.Content), a.Path)
	case "edit_file":
		p, err := resolve(root, a.Path)
		if err != nil {
			return "error: " + err.Error()
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return "error: " + err.Error()
		}
		if a.OldString == "" {
			return "error: old_string is empty"
		}
		n := strings.Count(string(b), a.OldString)
		if n != 1 {
			return fmt.Sprintf("error: old_string found %d times, must be exactly 1", n)
		}
		out := strings.Replace(string(b), a.OldString, a.NewString, 1)
		if err := os.WriteFile(p, []byte(out), 0o644); err != nil {
			return "error: " + err.Error()
		}
		return "edited " + a.Path
	case "tree":
		base, err := resolve(root, a.Path)
		if err != nil {
			return "error: " + err.Error()
		}
		var lines []string
		var walk func(dir, prefix string, depth int)
		walk = func(dir, prefix string, depth int) {
			ents, _ := os.ReadDir(dir)
			for i, e := range ents {
				if e.Name() == ".git" || e.Name() == "node_modules" || len(lines) >= 300 {
					continue
				}
				branch, next := "├── ", "│   "
				if i == len(ents)-1 {
					branch, next = "└── ", "    "
				}
				n := e.Name()
				if e.IsDir() {
					n += "/"
				}
				lines = append(lines, prefix+branch+n)
				if e.IsDir() && depth < 3 {
					walk(filepath.Join(dir, e.Name()), prefix+next, depth+1)
				}
			}
		}
		walk(base, "", 1)
		if len(lines) == 0 {
			return "(empty)"
		}
		return strings.Join(lines, "\n")
	case "append_file":
		p, err := resolve(root, a.Path)
		if err != nil {
			return "error: " + err.Error()
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return "error: " + err.Error()
		}
		f, err := os.OpenFile(p, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return "error: " + err.Error()
		}
		defer f.Close()
		if _, err := f.WriteString(a.Content + "\n"); err != nil {
			return "error: " + err.Error()
		}
		return "appended to " + a.Path
	case "delete_file":
		p, err := resolve(root, a.Path)
		if err != nil {
			return "error: " + err.Error()
		}
		if r, _ := filepath.EvalSymlinks(root); p == r {
			return "error: cannot delete the project root"
		}
		if err := os.Remove(p); err != nil {
			return "error: " + err.Error()
		}
		return "deleted " + a.Path
	case "move_file":
		src, err := resolve(root, a.From)
		if err != nil {
			return "error: " + err.Error()
		}
		dst, err := resolve(root, a.To)
		if err != nil {
			return "error: " + err.Error()
		}
		if _, err := os.Stat(dst); err == nil {
			return "error: destination already exists"
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return "error: " + err.Error()
		}
		if err := os.Rename(src, dst); err != nil {
			return "error: " + err.Error()
		}
		return fmt.Sprintf("moved %s -> %s", a.From, a.To)
	case "search_files":
		re, err := regexp.Compile(a.Pattern)
		if err != nil {
			return "error: bad pattern: " + err.Error()
		}
		base, err := resolve(root, a.Path)
		if err != nil {
			return "error: " + err.Error()
		}
		var hits []string
		deadline := time.Now().Add(15 * time.Second)
		cut := false
		_ = filepath.WalkDir(base, func(path string, d fs.DirEntry, err error) error {
			if ctx.Err() != nil {
				return filepath.SkipAll
			}
			if time.Now().After(deadline) {
				cut = true
				return filepath.SkipAll
			}
			if err != nil || len(hits) >= 200 {
				return nil
			}
			if d.IsDir() {
				if d.Name() == ".git" || d.Name() == "node_modules" {
					return filepath.SkipDir
				}
				return nil
			}
			b, err := os.ReadFile(path)
			if err != nil || len(b) > maxReadBytes || bytes.IndexByte(b[:min(len(b), 8000)], 0) >= 0 {
				return nil
			}
			rel, _ := filepath.Rel(root, path)
			for i, line := range strings.Split(string(b), "\n") {
				if re.MatchString(line) {
					hits = append(hits, fmt.Sprintf("%s:%d: %s", rel, i+1, strings.TrimSpace(clip(line, 200))))
					if len(hits) >= 200 {
						break
					}
				}
			}
			return nil
		})
		sort.Strings(hits)
		if len(hits) == 0 && !cut {
			return "no matches"
		}
		if cut {
			hits = append(hits, "[dừng sau 15 giây, kết quả chưa đầy đủ – hãy tìm trong thư mục nhỏ hơn]")
		}
		return strings.Join(hits, "\n")
	case "run_command":
		if root == "" {
			return "error: no project selected"
		}
		t := a.TimeoutSec
		if t <= 0 {
			t = 60
		}
		if t > 300 {
			t = 300
		}
		cctx, cancel := context.WithTimeout(ctx, time.Duration(t)*time.Second)
		defer cancel()
		c := newShellCmd(cctx, a.Command)
		c.Dir = root
		// If the command leaves a background/GUI process holding our output pipe
		// (notepad, `start x`, ping -t ...), don't wait for it forever.
		c.WaitDelay = 2 * time.Second
		var buf bytes.Buffer
		c.Stdout, c.Stderr = &buf, &buf
		err := c.Run()
		out := clip(buf.String(), maxOutputBytes)
		switch {
		case cctx.Err() == context.DeadlineExceeded:
			return out + fmt.Sprintf("\n[hết giờ sau %ds, đã dừng lệnh]", t)
		case ctx.Err() != nil:
			return out + "\n[đã dừng theo yêu cầu]"
		case errors.Is(err, exec.ErrWaitDelay):
			return out + "\n[lệnh đã khởi chạy tiến trình nền, bot không chờ nó kết thúc]"
		case err != nil:
			return out + "\n[" + err.Error() + "]"
		}
		if out == "" {
			return "(không có output, exit 0)"
		}
		return out
	}
	return "error: unknown tool " + name
}
