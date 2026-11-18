package main

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/ledongthuc/pdf"
)

// Basic readers for PDF / Word / Excel / PowerPoint. Text only, best effort.
// (.doc / .xls / .ppt – the old binary formats – are not supported.)

const (
	maxDocChars   = 400 * 1024 // upper bound of extracted text kept in memory
	maxSheetRows  = 3000
	maxDocPages   = 200
	maxZipEntryMB = 64
)

func isDocExt(p string) bool {
	switch strings.ToLower(filepath.Ext(p)) {
	case ".pdf", ".docx", ".xlsx", ".pptx":
		return true
	}
	return false
}

// readDocument extracts the text of a supported document.
func readDocument(p string) (string, error) {
	var out string
	var err error
	switch strings.ToLower(filepath.Ext(p)) {
	case ".pdf":
		out, err = readPDF(p)
	case ".docx":
		out, err = readDOCX(p)
	case ".xlsx":
		out, err = readXLSX(p)
	case ".pptx":
		out, err = readPPTX(p)
	default:
		return "", fmt.Errorf("unsupported document type %q", filepath.Ext(p))
	}
	if err != nil {
		return "", err
	}
	out = strings.TrimSpace(out)
	if len(out) > maxDocChars {
		out = out[:maxDocChars] + "\n...[tài liệu quá dài, đã cắt]"
	}
	return out, nil
}

// ---------------------------------------------------------------- PDF

func readPDF(p string) (text string, err error) {
	defer func() { // the parser can panic on malformed files
		if r := recover(); r != nil {
			err = fmt.Errorf("không đọc được PDF (%v)", r)
		}
	}()
	f, r, err := pdf.Open(p)
	if err != nil {
		return "", fmt.Errorf("không đọc được PDF: %v", err)
	}
	defer f.Close()
	var sb strings.Builder
	n := r.NumPage()
	if n > maxDocPages {
		n = maxDocPages
	}
	empty := 0
	for i := 1; i <= n; i++ {
		pg := r.Page(i)
		if pg.V.IsNull() {
			continue
		}
		t, e := pg.GetPlainText(nil)
		if e != nil {
			continue
		}
		t = strings.TrimSpace(t)
		if t == "" {
			empty++
			continue
		}
		fmt.Fprintf(&sb, "--- Trang %d ---\n%s\n\n", i, t)
		if sb.Len() > maxDocChars {
			break
		}
	}
	if strings.TrimSpace(sb.String()) == "" {
		return "", fmt.Errorf("PDF này không có lớp chữ (có thể là bản scan/ảnh) – cần OCR, bot chưa hỗ trợ")
	}
	return sb.String(), nil
}

// ---------------------------------------------------------------- zip helpers

func openZip(p string) (*zip.ReadCloser, error) {
	z, err := zip.OpenReader(p)
	if err != nil {
		return nil, fmt.Errorf("không mở được tệp (có thể bị hỏng, đặt mật khẩu hoặc là định dạng cũ .doc/.xls/.ppt): %v", err)
	}
	return z, nil
}

func zipFile(z *zip.ReadCloser, name string) ([]byte, error) {
	for _, f := range z.File {
		if f.Name == name {
			if f.UncompressedSize64 > maxZipEntryMB<<20 {
				return nil, fmt.Errorf("%s quá lớn", name)
			}
			rc, err := f.Open()
			if err != nil {
				return nil, err
			}
			defer rc.Close()
			return io.ReadAll(io.LimitReader(rc, maxZipEntryMB<<20))
		}
	}
	return nil, os.ErrNotExist
}

func local(n xml.Name) string { return n.Local }

// ---------------------------------------------------------------- DOCX

func readDOCX(p string) (string, error) {
	z, err := openZip(p)
	if err != nil {
		return "", err
	}
	defer z.Close()
	b, err := zipFile(z, "word/document.xml")
	if err != nil {
		return "", fmt.Errorf("không phải tệp .docx hợp lệ")
	}
	d := xml.NewDecoder(bytes.NewReader(b))
	var sb strings.Builder
	inT := false
	inTable, firstCell := 0, true
	for {
		tok, err := d.Token()
		if err != nil {
			break
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch local(t.Name) {
			case "t":
				inT = true
			case "tab":
				sb.WriteByte('\t')
			case "br", "cr":
				sb.WriteByte('\n')
			case "tbl":
				inTable++
			case "tc":
				if !firstCell {
					sb.WriteString(" | ")
				}
				firstCell = false
			case "tr":
				firstCell = true
			}
		case xml.CharData:
			if inT {
				sb.Write(t)
			}
		case xml.EndElement:
			switch local(t.Name) {
			case "t":
				inT = false
			case "p":
				if inTable == 0 {
					sb.WriteString("\n")
				} else {
					sb.WriteString(" ")
				}
			case "tr":
				sb.WriteString("\n")
			case "tbl":
				inTable--
			}
		}
		if sb.Len() > maxDocChars {
			break
		}
	}
	return collapseBlank(sb.String()), nil
}

var reCellSep = regexp.MustCompile(`[ \t]*\|[ \t]*`)

func collapseBlank(s string) string {
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		if strings.Contains(l, "|") {
			lines[i] = strings.TrimSpace(reCellSep.ReplaceAllString(l, " | "))
		}
	}
	var out []string
	blank := 0
	for _, l := range lines {
		l = strings.TrimRight(l, " \t")
		if l == "" {
			blank++
			if blank > 1 {
				continue
			}
		} else {
			blank = 0
		}
		out = append(out, l)
	}
	return strings.Join(out, "\n")
}

// ---------------------------------------------------------------- PPTX

func readPPTX(p string) (string, error) {
	z, err := openZip(p)
	if err != nil {
		return "", err
	}
	defer z.Close()
	type slide struct {
		n    int
		name string
	}
	var slides []slide
	for _, f := range z.File {
		if strings.HasPrefix(f.Name, "ppt/slides/slide") && strings.HasSuffix(f.Name, ".xml") {
			num, _ := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(f.Name, "ppt/slides/slide"), ".xml"))
			slides = append(slides, slide{num, f.Name})
		}
	}
	if len(slides) == 0 {
		return "", fmt.Errorf("không phải tệp .pptx hợp lệ")
	}
	sort.Slice(slides, func(i, j int) bool { return slides[i].n < slides[j].n })
	var sb strings.Builder
	for _, s := range slides {
		b, err := zipFile(z, s.name)
		if err != nil {
			continue
		}
		fmt.Fprintf(&sb, "--- Slide %d ---\n", s.n)
		d := xml.NewDecoder(bytes.NewReader(b))
		inT := false
		for {
			tok, err := d.Token()
			if err != nil {
				break
			}
			switch t := tok.(type) {
			case xml.StartElement:
				if local(t.Name) == "t" {
					inT = true
				}
			case xml.CharData:
				if inT {
					sb.Write(t)
				}
			case xml.EndElement:
				switch local(t.Name) {
				case "t":
					inT = false
				case "p":
					sb.WriteByte('\n')
				}
			}
		}
		sb.WriteString("\n")
		if sb.Len() > maxDocChars {
			break
		}
	}
	return collapseBlank(sb.String()), nil
}

// ---------------------------------------------------------------- XLSX

func readXLSX(p string) (string, error) {
	z, err := openZip(p)
	if err != nil {
		return "", err
	}
	defer z.Close()
	wb, err := zipFile(z, "xl/workbook.xml")
	if err != nil {
		return "", fmt.Errorf("không phải tệp .xlsx hợp lệ")
	}
	shared := readSharedStrings(z)

	// sheet name -> relationship id, relationship id -> target
	type sheetRef struct{ name, rid string }
	var refs []sheetRef
	d := xml.NewDecoder(bytes.NewReader(wb))
	for {
		tok, err := d.Token()
		if err != nil {
			break
		}
		if se, ok := tok.(xml.StartElement); ok && local(se.Name) == "sheet" {
			var r sheetRef
			for _, a := range se.Attr {
				switch {
				case a.Name.Local == "name":
					r.name = a.Value
				case a.Name.Local == "id": // r:id
					r.rid = a.Value
				}
			}
			refs = append(refs, r)
		}
	}
	targets := map[string]string{}
	if rb, err := zipFile(z, "xl/_rels/workbook.xml.rels"); err == nil {
		rd := xml.NewDecoder(bytes.NewReader(rb))
		for {
			tok, err := rd.Token()
			if err != nil {
				break
			}
			if se, ok := tok.(xml.StartElement); ok && local(se.Name) == "Relationship" {
				var id, tg string
				for _, a := range se.Attr {
					switch a.Name.Local {
					case "Id":
						id = a.Value
					case "Target":
						tg = a.Value
					}
				}
				targets[id] = tg
			}
		}
	}
	var sb strings.Builder
	for i, r := range refs {
		tg := targets[r.rid]
		if tg == "" {
			tg = fmt.Sprintf("worksheets/sheet%d.xml", i+1)
		}
		full := path.Clean(path.Join("xl", tg))
		if strings.HasPrefix(tg, "/") {
			full = strings.TrimPrefix(tg, "/")
		}
		sb.WriteString(fmt.Sprintf("## Sheet: %s\n", r.name))
		b, err := zipFile(z, full)
		if err != nil {
			sb.WriteString("(không đọc được sheet)\n\n")
			continue
		}
		sb.WriteString(sheetToText(b, shared))
		sb.WriteString("\n")
		if sb.Len() > maxDocChars {
			break
		}
	}
	return sb.String(), nil
}

func readSharedStrings(z *zip.ReadCloser) []string {
	b, err := zipFile(z, "xl/sharedStrings.xml")
	if err != nil {
		return nil
	}
	var out []string
	d := xml.NewDecoder(bytes.NewReader(b))
	var cur strings.Builder
	inSI, inT := false, false
	for {
		tok, err := d.Token()
		if err != nil {
			break
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch local(t.Name) {
			case "si":
				inSI = true
				cur.Reset()
			case "t":
				inT = inSI
			}
		case xml.CharData:
			if inT {
				cur.Write(t)
			}
		case xml.EndElement:
			switch local(t.Name) {
			case "t":
				inT = false
			case "si":
				out = append(out, cur.String())
				inSI = false
			}
		}
	}
	return out
}

// colIndex converts "AB12" -> 27 (0-based column).
func colIndex(ref string) int {
	n := 0
	for _, c := range ref {
		if c >= 'A' && c <= 'Z' {
			n = n*26 + int(c-'A') + 1
		} else if c >= 'a' && c <= 'z' {
			n = n*26 + int(c-'a') + 1
		} else {
			break
		}
	}
	return n - 1
}

func sheetToText(b []byte, shared []string) string {
	d := xml.NewDecoder(bytes.NewReader(b))
	var sb strings.Builder
	var row []string
	rows := 0
	var cellType, cellRef string
	var val strings.Builder
	inV, inT, inCell, inF := false, false, false, false
	var formula strings.Builder
	flushRow := func() {
		// trim trailing empty cells
		for len(row) > 0 && row[len(row)-1] == "" {
			row = row[:len(row)-1]
		}
		if len(row) > 0 {
			sb.WriteString(strings.Join(row, " | "))
			sb.WriteByte('\n')
			rows++
		}
		row = row[:0]
	}
	for {
		tok, err := d.Token()
		if err != nil {
			break
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch local(t.Name) {
			case "c":
				inCell = true
				cellType, cellRef = "", ""
				val.Reset()
				formula.Reset()
				for _, a := range t.Attr {
					switch a.Name.Local {
					case "t":
						cellType = a.Value
					case "r":
						cellRef = a.Value
					}
				}
			case "v":
				inV = inCell
			case "t":
				inT = inCell
			case "f":
				inF = inCell
			}
		case xml.CharData:
			if inV || inT {
				val.Write(t)
			} else if inF {
				formula.Write(t)
			}
		case xml.EndElement:
			switch local(t.Name) {
			case "v":
				inV = false
			case "t":
				inT = false
			case "f":
				inF = false
			case "c":
				s := val.String()
				if s == "" && formula.Len() > 0 { // formula without a cached value: show the formula
					s = "=" + formula.String()
				}
				switch cellType {
				case "s":
					if i, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && i >= 0 && i < len(shared) {
						s = shared[i]
					}
				case "b":
					if s == "1" {
						s = "TRUE"
					} else {
						s = "FALSE"
					}
				}
				s = strings.ReplaceAll(strings.ReplaceAll(s, "\n", " "), "|", "/")
				ci := colIndex(cellRef)
				if ci < 0 || ci > 500 {
					ci = len(row)
				}
				for len(row) < ci {
					row = append(row, "")
				}
				row = append(row, s)
				inCell = false
			case "row":
				flushRow()
				if rows >= maxSheetRows {
					sb.WriteString(fmt.Sprintf("...[sheet còn nhiều dòng, chỉ lấy %d dòng đầu]\n", maxSheetRows))
					return sb.String()
				}
			}
		}
	}
	return sb.String()
}
