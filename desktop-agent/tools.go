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
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// defaultCmdTimeout is set from config at startup.
var defaultCmdTimeout = 60

const maxCmdTimeout = 3600

const (
	maxReadBytes   = 200 * 1024
	maxOutputBytes = 20 * 1024
)

// toolDefs are sent to the LLM (OpenAI function-calling format).
var toolDefs = []map[string]any{
	fn("list_dir", "List files and folders in a directory of the project (relative path, '.' for root).",
		props{"path": str("relative directory path")}, "path"),
	fn("read_file", "Read a text file, or the text of a .pdf/.docx/.xlsx/.pptx, from the project. Long files are paged: pass offset from the hint to continue.",
		props{"path": str("relative file path"), "offset": map[string]any{"type": "integer", "description": "character offset to start from (default 0)"}}, "path"),
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
	fn("find_files", "Find files/folders by NAME (case-insensitive substring, or a glob like *.py). Returns relative paths.",
		props{"pattern": str("name substring or glob"), "path": str("relative directory, default '.'")}, "pattern"),
	fn("create_task", "Create a scheduled task that runs a shell command (or a script you provide) on a schedule, with alert rules. Schedule examples: 'every 30s', 'every 5m', 'every 1h', 'daily 08:30'. Alerts fire on non-zero exit, or when output contains alert_contains / lacks alert_not_contains.",
		props{"name": str("short unique task name"), "schedule": str("every 30s | every 5m | every 1h | daily 08:30"),
			"command": str("shell command to run (use this OR script)"), "script": str("source code to save and run instead of command"),
			"language":       str("python | powershell | bat | sh (with script)"),
			"alert_contains": str("alert if output contains this text"), "alert_not_contains": str("alert if output does NOT contain this text"),
			"timeout_sec": map[string]any{"type": "integer", "description": "kill the run after N seconds (default 60)"}}, "name", "schedule"),
	fn("list_tasks", "List scheduled tasks with status; pass id to see one task's recent runs.", props{"id": str("task id or name (optional)")}),
	fn("run_task", "Run a scheduled task right now and return its output.", props{"id": str("task id or name")}, "id"),
	fn("delete_task", "Delete a scheduled task.", props{"id": str("task id or name")}, "id"),
	fn("toggle_task", "Pause or resume a scheduled task.", props{"id": str("task id or name"), "enabled": map[string]any{"type": "boolean"}}, "id", "enabled"),
	fn("web_search", "Search the web (news, facts). Returns up to 8 results with title, URL and snippet. Then use fetch_url to read a result.",
		props{"query": str("search keywords; add the year or 'tin tức' for news")}, "query"),
	fn("fetch_url", "Download a web page and return its readable text (paged with offset).",
		props{"url": str("http(s) URL"), "offset": map[string]any{"type": "integer", "description": "character offset to continue reading"}}, "url"),
	fn("search_files", "Search INSIDE files for a regex (case-insensitive). Also searches text inside .pdf/.docx/.xlsx/.pptx. Returns file:line: text. To find a file by NAME use find_files.",
		props{"pattern": str("regular expression"), "path": str("relative directory, default '.'")}, "pattern"),
	fn("run_command", "Run a shell command in the project directory (cmd on Windows, sh elsewhere). Returns combined output.",
		props{"command": str("command line"), "timeout_sec": map[string]any{"type": "integer", "description": "seconds before the command is killed (default from config, 60; max 3600)"}}, "command"),
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
	case "write_file", "edit_file", "run_command", "append_file", "delete_file", "move_file", "create_task", "delete_task", "fetch_url":
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
	a, err0 := parseArgs(rawArgs)
	if err0 != nil {
		return "error: invalid arguments: " + err0.Error()
	}
	switch name {
	case "list_dir":
		p, err := resolve(root, a.str("path"))
		if err != nil {
			return "error: " + err.Error()
		}
		ents, err := os.ReadDir(p)
		if isNotExist(err) {
			return "error: " + notFoundMsg(ctx, root, a.str("path"))
		}
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
		p, err := resolve(root, a.str("path"))
		if err != nil {
			return "error: " + err.Error()
		}
		st, err := os.Stat(p)
		if isNotExist(err) {
			return "error: " + notFoundMsg(ctx, root, a.str("path"))
		}
		if err != nil {
			return "error: " + err.Error()
		}
		if st.IsDir() {
			return "error: đây là thư mục, hãy dùng list_dir"
		}
		var text string
		if isDocExt(p) {
			text, err = readDocument(p)
			if err != nil {
				return "error: " + err.Error()
			}
		} else {
			if st.Size() > 50<<20 {
				return "error: tệp quá lớn (>50MB)"
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return "error: " + err.Error()
			}
			if bytes.IndexByte(b[:min(len(b), 8000)], 0) >= 0 {
				return "error: tệp nhị phân (chỉ đọc được văn bản, .pdf, .docx, .xlsx, .pptx)"
			}
			text = string(b)
		}
		return window(text, a.num("offset"), a.str("path"))
	case "write_file":
		p, err := resolve(root, a.str("path"))
		if err != nil {
			return "error: " + err.Error()
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return "error: " + err.Error()
		}
		if err := os.WriteFile(p, []byte(a.str("content")), 0o644); err != nil {
			return "error: " + err.Error()
		}
		return fmt.Sprintf("wrote %d bytes to %s", len(a.str("content")), a.str("path"))
	case "edit_file":
		p, err := resolve(root, a.str("path"))
		if err != nil {
			return "error: " + err.Error()
		}
		b, err := os.ReadFile(p)
		if isNotExist(err) {
			return "error: " + notFoundMsg(ctx, root, a.str("path"))
		}
		if err != nil {
			return "error: " + err.Error()
		}
		if a.str("old_string") == "" {
			return "error: old_string is empty"
		}
		n := strings.Count(string(b), a.str("old_string"))
		if n != 1 {
			return fmt.Sprintf("error: old_string found %d times, must be exactly 1", n)
		}
		out := strings.Replace(string(b), a.str("old_string"), a.str("new_string"), 1)
		if err := os.WriteFile(p, []byte(out), 0o644); err != nil {
			return "error: " + err.Error()
		}
		return "edited " + a.str("path")
	case "tree":
		base, err := resolve(root, a.str("path"))
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
		p, err := resolve(root, a.str("path"))
		if err != nil {
			return "error: " + err.Error()
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return "error: " + err.Error()
		}
		f, err := os.OpenFile(p, os.O_RDWR|os.O_APPEND|os.O_CREATE, 0o644)
		if err != nil {
			return "error: " + err.Error()
		}
		defer f.Close()
		text := a.str("content") + "\n"
		// file không kết thúc bằng xuống dòng → xuống dòng trước, để không dính chữ cũ
		if st, err := f.Stat(); err == nil && st.Size() > 0 {
			last := make([]byte, 1)
			if _, err := f.ReadAt(last, st.Size()-1); err == nil && last[0] != '\n' {
				text = "\n" + text
			}
		}
		if _, err := f.WriteString(text); err != nil {
			return "error: " + err.Error()
		}
		return "appended to " + a.str("path")
	case "delete_file":
		p, err := resolve(root, a.str("path"))
		if err != nil {
			return "error: " + err.Error()
		}
		if r, _ := filepath.EvalSymlinks(root); p == r {
			return "error: cannot delete the project root"
		}
		if err := os.Remove(p); err != nil {
			if isNotExist(err) {
				return "error: " + notFoundMsg(ctx, root, a.str("path"))
			}
			return "error: " + err.Error()
		}
		return "deleted " + a.str("path")
	case "move_file":
		src, err := resolve(root, a.str("from"))
		if err != nil {
			return "error: " + err.Error()
		}
		dst, err := resolve(root, a.str("to"))
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
		return fmt.Sprintf("moved %s -> %s", a.str("from"), a.str("to"))
	case "find_files":
		pat := strings.ToLower(strings.TrimSpace(a.str("pattern")))
		if pat == "" {
			return "error: empty pattern"
		}
		base, err := resolve(root, a.str("path"))
		if err != nil {
			return "error: " + err.Error()
		}
		glob := strings.ContainsAny(pat, "*?[")
		deadline := time.Now().Add(15 * time.Second)
		var found []string
		cut := false
		_ = filepath.WalkDir(base, func(path string, d fs.DirEntry, err error) error {
			if ctx.Err() != nil {
				return filepath.SkipAll
			}
			if time.Now().After(deadline) {
				cut = true
				return filepath.SkipAll
			}
			if err != nil {
				return nil
			}
			if d.IsDir() && (d.Name() == ".git" || d.Name() == "node_modules") {
				return filepath.SkipDir
			}
			if path == base {
				return nil
			}
			name := strings.ToLower(d.Name())
			ok := false
			if glob {
				ok, _ = filepath.Match(pat, name)
			} else {
				ok = strings.Contains(name, pat)
			}
			if ok {
				rel, _ := filepath.Rel(root, path)
				rel = filepath.ToSlash(rel)
				if d.IsDir() {
					rel += "/"
				}
				found = append(found, rel)
				if len(found) >= 200 {
					return filepath.SkipAll
				}
			}
			return nil
		})
		sort.Strings(found)
		if cut {
			found = append(found, "[dừng sau 15 giây, kết quả chưa đầy đủ]")
		}
		if len(found) == 0 {
			if !glob { // nothing contains it: try typos / missing diacritics
				if m := suggestFiles(ctx, root, a.str("pattern"), 8); len(m) > 0 {
					var ps []string
					for _, x := range m {
						ps = append(ps, x.Path)
					}
					return "không có tên nào chứa đúng \"" + a.str("pattern") + "\". Tên gần giống:\n" + strings.Join(ps, "\n")
				}
			}
			return "no matches"
		}
		return strings.Join(found, "\n")
	case "search_files":
		pat := a.str("pattern")
		re, err := regexp.Compile("(?i)" + pat)
		if err != nil {
			return "error: bad pattern: " + err.Error()
		}
		base, err := resolve(root, a.str("path"))
		if err != nil {
			return "error: " + err.Error()
		}
		plain := regexp.QuoteMeta(pat) == pat
		fp := foldName(pat)
		var hits []string
		deadline := time.Now().Add(15 * time.Second)
		cut := false
		docsScanned := 0
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
			var b []byte
			if isDocExt(path) { // search inside PDF / Word / Excel / PowerPoint too
				if info, e := d.Info(); e != nil || info.Size() > 30<<20 || docsScanned >= 300 {
					return nil
				}
				docsScanned++
				txt, e := readDocument(path)
				if e != nil {
					return nil
				}
				b = []byte(txt)
			} else {
				var err error
				b, err = os.ReadFile(path)
				if err != nil || len(b) > maxReadBytes || bytes.IndexByte(b[:min(len(b), 8000)], 0) >= 0 {
					return nil
				}
			}
			rel, _ := filepath.Rel(root, path)
			for i, line := range strings.Split(string(b), "\n") {
				if re.MatchString(line) || (plain && fp != "" && strings.Contains(foldName(line), fp)) { // also matches without Vietnamese diacritics
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
		tsec := effectiveTimeout(a.num("timeout_sec"))
		r := runShell(ctx, root, a.str("command"), tsec)
		out := clip(r.Out, maxOutputBytes)
		switch {
		case r.TimedOut:
			return out + fmt.Sprintf("\n[hết giờ sau %ds, đã dừng lệnh]", tsec)
		case r.Canceled:
			return out + "\n[đã dừng theo yêu cầu]"
		case r.Detached:
			return out + "\n[lệnh đã khởi chạy tiến trình nền, bot không chờ nó kết thúc]"
		case r.Err != nil:
			return out + "\n[" + r.Err.Error() + "]"
		}
		if out == "" {
			return "(không có output, exit 0)"
		}
		return out
	case "web_search":
		return webSearch(ctx, a.str("query")+a.str("pattern"))
	case "fetch_url":
		return fetchURL(ctx, a.str("url"), a.num("offset"))
	case "create_task", "list_tasks", "run_task", "delete_task", "toggle_task":
		return taskTool(ctx, root, name, a)
	}
	return "error: unknown tool " + name
}

// effectiveTimeout returns the seconds a run_command may take.
func effectiveTimeout(req int) int {
	if req <= 0 {
		req = defaultCmdTimeout
	}
	if req > maxCmdTimeout {
		req = maxCmdTimeout
	}
	return req
}

// commandTimeoutOf extracts the effective timeout from raw tool args (for the UI).
func commandTimeoutOf(raw string) int {
	a, _ := parseArgs(raw)
	return effectiveTimeout(a.num("timeout_sec"))
}

// targs are tool arguments as sent by the model. Weak models send numbers as strings,
// use other key names, etc. – be forgiving.
type targs map[string]any

func parseArgs(raw string) (targs, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return targs{}, nil
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return nil, err
	}
	return targs(m), nil
}

var argAliases = map[string][]string{
	"url":                {"link", "uri", "href", "address"},
	"query":              {"q", "keyword", "keywords", "search", "text"},
	"path":               {"file_path", "filepath", "file", "filename", "dir", "directory", "folder"},
	"content":            {"contents", "text", "data", "body"},
	"old_string":         {"old", "find", "search", "old_text"},
	"new_string":         {"new", "replace", "replacement", "new_text"},
	"pattern":            {"query", "regex", "keyword", "name", "q"},
	"command":            {"cmd", "script", "shell", "commandline"},
	"timeout_sec":        {"timeout", "timeoutsec", "seconds"},
	"schedule":           {"cron", "interval", "when", "every"},
	"alert_contains":     {"contains", "alert_if_contains"},
	"alert_not_contains": {"not_contains", "alert_if_missing"},
	"language":           {"lang"},
	"from":               {"src", "source", "old_path"},
	"to":                 {"dst", "dest", "destination", "new_path"},
}

func (a targs) get(k string) (any, bool) {
	if v, ok := a[k]; ok && v != nil {
		return v, true
	}
	for _, al := range argAliases[k] {
		if v, ok := a[al]; ok && v != nil {
			return v, true
		}
	}
	return nil, false
}

func (a targs) str(k string) string {
	v, ok := a.get(k)
	if !ok {
		return ""
	}
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(x)
	}
	b, _ := json.Marshal(v)
	return string(b)
}

func (a targs) num(k string) int {
	v, ok := a.get(k)
	if !ok {
		return 0
	}
	switch x := v.(type) {
	case float64:
		return int(x)
	case string:
		n, _ := strconv.Atoi(strings.TrimSpace(strings.TrimSuffix(x, "s")))
		return n
	}
	return 0
}

func (a targs) boolean(k string, def bool) bool {
	v, ok := a.get(k)
	if !ok {
		return def
	}
	switch x := v.(type) {
	case bool:
		return x
	case string:
		switch strings.ToLower(strings.TrimSpace(x)) {
		case "true", "1", "yes", "có", "co":
			return true
		case "false", "0", "no", "không", "khong":
			return false
		}
	case float64:
		return x != 0
	}
	return def
}

// readChunkChars is how much of a file one read_file call returns (set from the context budget).
var readChunkChars = 6000

// window returns text[offset:offset+readChunkChars] (UTF-8 safe) with a hint for the next page.
func window(text string, offset int, path string) string {
	if offset < 0 {
		offset = 0
	}
	if offset >= len(text) {
		if offset == 0 {
			return "(tệp rỗng)"
		}
		return "(hết tệp)"
	}
	start := offset
	for start < len(text) && !utf8.RuneStart(text[start]) {
		start++
	}
	end := start + readChunkChars
	if end >= len(text) {
		return text[start:]
	}
	for end < len(text) && !utf8.RuneStart(text[end]) {
		end++
	}
	return fmt.Sprintf("%s\n[... còn %d ký tự nữa. Đọc tiếp: read_file path=%q offset=%d]", text[start:end], len(text)-end, path, end)
}

type shellResult struct {
	Out                          string
	Exit                         int
	TimedOut, Canceled, Detached bool
	Err                          error
}

// runShell runs one shell command line in dir with a hard timeout. It never blocks on
// leftover background processes and kills the whole process tree on timeout/cancel.
func runShell(ctx context.Context, dir, command string, timeoutSec int) shellResult {
	cctx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
	defer cancel()
	c := newShellCmd(cctx, command)
	c.Dir = dir
	c.WaitDelay = 2 * time.Second
	var buf bytes.Buffer
	c.Stdout, c.Stderr = &buf, &buf
	err := c.Run()
	r := shellResult{Out: buf.String()}
	switch {
	case cctx.Err() == context.DeadlineExceeded:
		r.TimedOut, r.Exit = true, -1
	case ctx.Err() != nil:
		r.Canceled, r.Exit = true, -1
	case errors.Is(err, exec.ErrWaitDelay):
		r.Detached = true
	case err != nil:
		r.Err = err
		r.Exit = -1
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			r.Exit = ee.ExitCode()
		}
	}
	return r
}
