package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fuzzyProj(t *testing.T) string {
	d := t.TempDir()
	for p, c := range map[string]string{
		"báo-cáo/Báo cáo doanh thu Q1.txt": "doanh thu quý 1 là 500 triệu\nNguyễn Văn Ánh phụ trách",
		"báo-cáo/Báo cáo chi phí Q1.txt":   "chi phí quý 1",
		"docs/abc_notes.txt":               "hello notes",
		"docs/readme.md":                   "# Readme",
		"src/main.go":                      "package main",
	} {
		must(t, os.MkdirAll(filepath.Dir(filepath.Join(d, p)), 0o755))
		mustWrite(t, filepath.Join(d, p), c)
	}
	copyFile(t, "testdata/hd.docx", filepath.Join(d, "docs", "hợp-đồng.docx"))
	copyFile(t, "testdata/dt.xlsx", filepath.Join(d, "docs", "doanh-thu.xlsx"))
	copyFile(t, "testdata/vi.pdf", filepath.Join(d, "docs", "bao-cao.pdf"))
	return d
}

func TestFoldName(t *testing.T) {
	for in, want := range map[string]string{
		"Báo cáo doanh thu Q1.txt": "bao cao doanh thu q1 txt", "Hợp-Đồng_Mới": "hop dong moi", "  ABC__notes ": "abc notes",
		"NGUYỄN VĂN ÁNH": "nguyen van anh", "": "",
	} {
		if got := foldName(in); got != want {
			t.Errorf("foldName(%q)=%q want %q", in, got, want)
		}
	}
}

func TestNameScore(t *testing.T) {
	type c struct {
		want, cand string
		min, max   float64
	}
	for _, x := range []c{
		{"abc_notes.txt", "abc_notes.txt", 1, 1},
		{"abd_notes.txt", "abc_notes.txt", 0.85, 0.99},               // one-letter typo
		{"bao cao doanh thu q1", "Báo cáo doanh thu Q1.txt", 0.8, 1}, // no diacritics
		{"doanh thu", "Báo cáo doanh thu Q1.txt", 0.8, 1},            // part of the name
		{"hop dong", "hợp-đồng.docx", 0.8, 1},
		{"readme", "README.md", 0.8, 1},
		{"main", "main.go", 0.8, 1},
		{"abc_notes.txt", "totally_different.csv", 0, 0.5},
		{"report", "báo-cáo", 0, 0.5},
	} {
		got := nameScore(x.want, x.cand)
		if got < x.min || got > x.max {
			t.Errorf("nameScore(%q,%q)=%.2f want in [%.2f,%.2f]", x.want, x.cand, got, x.min, x.max)
		}
	}
}

func TestSuggestAndNotFound(t *testing.T) {
	d := fuzzyProj(t)
	m := suggestFiles(t.Context(), d, "abd_notes.txt", 5)
	if len(m) == 0 || m[0].Path != "docs/abc_notes.txt" {
		t.Fatalf("typo: %+v", m)
	}
	m = suggestFiles(t.Context(), d, "bao cao doanh thu q1", 5)
	if len(m) == 0 || m[0].Path != "báo-cáo/Báo cáo doanh thu Q1.txt" {
		t.Fatalf("no diacritics: %+v", m)
	}
	if m = suggestFiles(t.Context(), d, "zzzzqqqq", 5); len(m) != 0 {
		t.Errorf("garbage must not match: %+v", m)
	}
	out := run(d, "read_file", `{"path":"abd_notes.txt"}`)
	wantHas(t, out, "không thấy")
	wantHas(t, out, "docs/abc_notes.txt")
	wantHas(t, out, "Hãy dùng đúng tên")
	wantHas(t, run(d, "read_file", `{"path":"zzzzqqqq"}`), "list_dir")
	wantHas(t, run(d, "list_dir", `{"path":"doc"}`), "docs/")
	wantHas(t, run(d, "edit_file", `{"path":"docs/abc_note.txt","old_string":"a","new_string":"b"}`), "docs/abc_notes.txt")
	wantHas(t, run(d, "delete_file", `{"path":"docs/abc_note.txt"}`), "docs/abc_notes.txt")
	// find_files: exact substring first, fuzzy only when nothing contains it
	wantHas(t, run(d, "find_files", `{"pattern":"doanh"}`), "Báo cáo doanh thu Q1.txt")
	out = run(d, "find_files", `{"pattern":"abd_notes"}`)
	wantHas(t, out, "gần giống")
	wantHas(t, out, "docs/abc_notes.txt")
	wantHas(t, run(d, "find_files", `{"pattern":"zzzzqqqq"}`), "no matches")
}

func TestSearchInsideDocuments(t *testing.T) {
	d := fuzzyProj(t)
	for _, c := range []struct{ pat, want string }{
		{"Nguyễn Văn Ánh", "hợp-đồng.docx"},    // inside .docx
		{"Nguyễn Văn Ánh", "bao-cao.pdf"},      // inside .pdf
		{"Nguyễn Văn Ánh", "doanh-thu.xlsx"},   // inside .xlsx
		{"nguyen van anh", "hợp-đồng.docx"},    // no diacritics still matches
		{"NGUYỄN", "Báo cáo doanh thu Q1.txt"}, // case-insensitive
		{"hello NOTES", "abc_notes.txt"},
		{"0901234567", "doanh-thu.xlsx"},
		{"1.250.000.000", "bao-cao.pdf"},
		{"tăng trưởng 12%", "bao-cao.pdf"},
	} {
		wantHas(t, run(d, "search_files", `{"pattern":`+q(c.pat)+`}`), c.want)
	}
	wantHas(t, run(d, "search_files", `{"pattern":"không-có-chữ-này-đâu"}`), "no matches")
	// a corrupt document must not break the whole search
	mustWrite(t, d+"/docs/hỏng.docx", "not a zip")
	wantHas(t, run(d, "search_files", `{"pattern":"hello"}`), "abc_notes.txt")
}

func TestOfflineFuzzyOpen(t *testing.T) {
	b := &bot{t: t}
	b = newBot(t)
	d := fuzzyProj(t)
	b.s.project, b.d = d, d
	// clear typo, single clear winner → opens it and says so
	b.expect("đọc abd_notes.txt", "gần giống nhất", "docs/abc_notes.txt", "hello notes")
	b.expect("đọc bao cao doanh thu q1", "Báo cáo doanh thu Q1.txt", "quý 1")
	b.expect("đọc hop dong", "hợp-đồng.docx", "Bên A")
	// ambiguous (two Q1 reports) → does NOT guess, lists candidates
	out := b.say("đọc bao cao q1")
	if strings.Contains(out, "hello") || !strings.Contains(out, "gần giống") {
		t.Errorf("ambiguous name must list candidates, not guess: %s", out)
	}
	wantHas(t, out, "doanh thu Q1.txt")
	wantHas(t, out, "chi phí Q1.txt")
	// nothing similar
	b.expect("đọc zzzzqqqq", "list_dir")
	// find by name tolerant
	b.expect("tìm tên abd_notes", "abc_notes.txt")
	b.expect("tìm tên hop dong", "hợp-đồng.docx")
	// content search inside documents
	b.expect("tìm nội dung nguyen van anh", "hợp-đồng.docx", "bao-cao.pdf", "doanh-thu.xlsx")
}
