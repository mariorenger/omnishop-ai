package main

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Fuzzy file-name matching so that "bao cao doanh thu q1", "abd_notes" or "Báo cáo doanh thu"
// still find "báo-cáo/Báo cáo doanh thu Q1.txt".

var viFold = strings.NewReplacer(
	"à", "a", "á", "a", "ạ", "a", "ả", "a", "ã", "a", "â", "a", "ầ", "a", "ấ", "a", "ậ", "a", "ẩ", "a", "ẫ", "a", "ă", "a", "ằ", "a", "ắ", "a", "ặ", "a", "ẳ", "a", "ẵ", "a",
	"è", "e", "é", "e", "ẹ", "e", "ẻ", "e", "ẽ", "e", "ê", "e", "ề", "e", "ế", "e", "ệ", "e", "ể", "e", "ễ", "e",
	"ì", "i", "í", "i", "ị", "i", "ỉ", "i", "ĩ", "i",
	"ò", "o", "ó", "o", "ọ", "o", "ỏ", "o", "õ", "o", "ô", "o", "ồ", "o", "ố", "o", "ộ", "o", "ổ", "o", "ỗ", "o", "ơ", "o", "ờ", "o", "ớ", "o", "ợ", "o", "ở", "o", "ỡ", "o",
	"ù", "u", "ú", "u", "ụ", "u", "ủ", "u", "ũ", "u", "ư", "u", "ừ", "u", "ứ", "u", "ự", "u", "ử", "u", "ữ", "u",
	"ỳ", "y", "ý", "y", "ỵ", "y", "ỷ", "y", "ỹ", "y", "đ", "d",
)

// foldName lowercases, strips Vietnamese diacritics and turns separators into single spaces.
func foldName(s string) string {
	s = viFold.Replace(strings.ToLower(s))
	var sb strings.Builder
	space := true
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r > 127 {
			sb.WriteRune(r)
			space = false
		} else if !space {
			sb.WriteByte(' ')
			space = true
		}
	}
	return strings.TrimSpace(sb.String())
}

func levenshtein(a, b []rune) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			c := 1
			if a[i-1] == b[j-1] {
				c = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+c)
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

// nameScore returns 0..1 how well candidate matches what the user typed.
func nameScore(want, cand string) float64 {
	w, c := foldName(want), foldName(cand)
	if w == "" || c == "" {
		return 0
	}
	if w == c {
		return 1
	}
	best := 0.0
	// containment, e.g. "doanh thu" inside "bao cao doanh thu q1"
	if strings.Contains(c, w) {
		best = 0.80 + 0.19*float64(len(w))/float64(len(c))
	} else if strings.Contains(w, c) && len(c) >= 3 {
		best = 0.70 + 0.19*float64(len(c))/float64(len(w))
	}
	// whole-string edit distance (typos)
	rw, rc := []rune(w), []rune(c)
	if d := levenshtein(rw, rc); d <= max(len(rw), len(rc)) {
		best = max(best, 1-float64(d)/float64(max(len(rw), len(rc))))
	}
	// token overlap (order independent)
	wt, ct := strings.Fields(w), strings.Fields(c)
	if len(wt) > 0 && len(ct) > 0 {
		hit := 0
		for _, x := range wt {
			for _, y := range ct {
				if x == y || (len(x) >= 4 && levenshtein([]rune(x), []rune(y)) <= 1) {
					hit++
					break
				}
			}
		}
		best = max(best, 0.5+0.45*float64(hit)/float64(len(wt))*boolf(hit >= min(2, len(wt))))
	}
	return best
}

func boolf(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

type fileMatch struct {
	Path  string
	Score float64
}

// suggestFiles walks the project (bounded in time and count) and returns the closest names.
func suggestFiles(ctx context.Context, root, want string, limit int) []fileMatch {
	want = strings.TrimSpace(want)
	if root == "" || foldName(filepath.Base(want)) == "" {
		return nil
	}
	base := filepath.Base(filepath.FromSlash(want))
	wantStem := strings.TrimSuffix(base, filepath.Ext(base))
	deadline := time.Now().Add(3 * time.Second)
	seen := 0
	var out []fileMatch
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || ctx.Err() != nil || time.Now().After(deadline) || seen > 50000 {
			if d != nil && d.IsDir() && err != nil {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() && (d.Name() == ".git" || d.Name() == "node_modules") {
			return filepath.SkipDir
		}
		if p == root {
			return nil
		}
		seen++
		stem := strings.TrimSuffix(d.Name(), filepath.Ext(d.Name()))
		sc := max(nameScore(base, d.Name()), nameScore(wantStem, stem))
		if sc >= 0.55 {
			rel, _ := filepath.Rel(root, p)
			rel = filepath.ToSlash(rel)
			if d.IsDir() {
				rel += "/"
			}
			out = append(out, fileMatch{rel, sc})
		}
		return nil
	})
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return len(out[i].Path) < len(out[j].Path)
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

// notFoundMsg builds the error text for a missing path, with "did you mean" suggestions.
func notFoundMsg(ctx context.Context, root, userPath string) string {
	msg := fmt.Sprintf("không thấy %q trong dự án.", userPath)
	if m := suggestFiles(ctx, root, userPath, 5); len(m) > 0 {
		var ps []string
		for _, x := range m {
			ps = append(ps, x.Path)
		}
		msg += " Tệp/thư mục gần giống: " + strings.Join(ps, " ; ") + ". Hãy dùng đúng tên ở trên."
	} else {
		msg += " Không có tên nào gần giống – hãy dùng list_dir, tree hoặc find_files để xem có những gì."
	}
	return msg
}

func isNotExist(err error) bool { return err != nil && os.IsNotExist(err) }
