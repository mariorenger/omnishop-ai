package main

import (
	"strings"
	"testing"
)

func TestReadDOCX(t *testing.T) {
	s, err := readDocument("testdata/hd.docx")
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []string{"Hợp đồng dịch vụ", "Công ty ABC", "Nguyễn Văn Ánh", "Hạng mục | SL | Giá", "Máy chủ | 2 | 15.000.000", "30 ngày"} {
		wantHas(t, s, w)
	}
}

func TestReadXLSX(t *testing.T) {
	s, err := readDocument("testdata/dt.xlsx")
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []string{"## Sheet: Doanh thu", "Tháng | Doanh thu | Ghi chú", "T1 | 100.5 | tốt", "T2 | 200", "TRUE", "có \"dấu\" & <tag>", "## Sheet: Khách hàng", "Nguyễn Văn Ánh | 0901234567"} {
		wantHas(t, s, w)
	}
	// the formula cell E2 sits in column E (index 4): blank cells in between must be kept as separators
	wantHas(t, s, "T1 | 100.5 | tốt |  | =B2+B3")
}

func TestReadPDF(t *testing.T) {
	s, err := readDocument("testdata/vi.pdf")
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []string{"--- Trang 1 ---", "BÁO CÁO DOANH THU QUÝ 1", "1.250.000.000", "Nguyễn Văn Ánh", "--- Trang 2 ---", "tăng trưởng 12%"} {
		wantHas(t, s, w)
	}
	s, err = readDocument("testdata/en.pdf")
	if err != nil || !strings.Contains(s, "Invoice 42") {
		t.Fatalf("en.pdf: %v %q", err, s)
	}
	s, err = readDocument("testdata/hd_lo.pdf") // PDF produced by LibreOffice (closer to Word/Office PDFs)
	if err != nil {
		t.Fatal(err)
	}
	wantHas(t, s, "Hợp đồng")
	wantHas(t, s, "Công ty ABC")
	if _, err = readDocument("testdata/scan.pdf"); err == nil || !strings.Contains(err.Error(), "OCR") {
		t.Fatalf("scan pdf should explain OCR, got %v", err)
	}
}

func TestReadDocumentErrors(t *testing.T) {
	d := newProj(t)
	// a text file renamed to .docx / .xlsx / .pdf must give a clean error, never panic
	for _, n := range []string{"fake.docx", "fake.xlsx", "fake.pdf", "fake.pptx"} {
		mustWrite(t, d+"/"+n, "not a real document")
		out := run(d, "read_file", `{"path":"`+n+`"}`)
		if !strings.HasPrefix(out, "error:") {
			t.Errorf("%s: want error, got %q", n, out)
		}
	}
}

func TestReadFileViaTool(t *testing.T) {
	d := newProj(t)
	copyFile(t, "testdata/hd.docx", d+"/hd.docx")
	copyFile(t, "testdata/dt.xlsx", d+"/dt.xlsx")
	copyFile(t, "testdata/vi.pdf", d+"/vi.pdf")
	wantHas(t, run(d, "read_file", `{"path":"hd.docx"}`), "Máy chủ")
	wantHas(t, run(d, "read_file", `{"path":"dt.xlsx"}`), "Khách hàng")
	wantHas(t, run(d, "read_file", `{"path":"vi.pdf"}`), "DOANH THU")
}

func TestPaging(t *testing.T) {
	d := newProj(t)
	old := readChunkChars
	readChunkChars = 50
	defer func() { readChunkChars = old }()
	big := strings.Repeat("xin chào thế giới. ", 40) // multi-byte text
	mustWrite(t, d+"/big.txt", big)
	var got strings.Builder
	off := 0
	for i := 0; i < 40; i++ {
		out := run(d, "read_file", `{"path":"big.txt","offset":`+itoa(off)+`}`)
		i2 := strings.Index(out, "\n[... còn ")
		if i2 < 0 {
			got.WriteString(out)
			break
		}
		got.WriteString(out[:i2])
		var next int
		idx := strings.Index(out, "offset=")
		next = atoi(out[idx+7 : len(out)-1])
		if next <= off {
			t.Fatalf("offset does not advance: %d -> %d", off, next)
		}
		off = next
	}
	if got.String() != big {
		t.Fatalf("paged read lost/garbled text:\n got %d bytes want %d", got.Len(), len(big))
	}
	wantHas(t, run(d, "read_file", `{"path":"big.txt","offset":999999}`), "hết tệp")
}
