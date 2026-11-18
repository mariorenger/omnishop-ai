# Cách tạo một bot desktop kiểu này – chi tiết kỹ thuật

Tài liệu giải thích **toàn bộ cách làm**: kiến trúc, từng file code, cách thêm luật if/else, thêm công cụ, nối model/RAG, đóng thành **một file exe** và phát hành. Đọc xong bạn tự dựng lại từ đầu được.

---

## 1. Ý tưởng tổng thể

```
┌─────────────── desktop-agent.exe (1 file, ~7 MB) ───────────────┐
│                                                                 │
│  Trình duyệt ──HTTP/SSE──►  Web server (127.0.0.1:7878)         │
│  (UI nhúng sẵn          │      ├─ /            → file UI (go:embed)│
│   trong exe)            │      └─ /api/*       → handlers         │
│                         │                                       │
│                         ▼                                       │
│                    Vòng lặp agent                               │
│              ┌──────────┴───────────┐                           │
│        Chế độ OFFLINE           Chế độ LLM                      │
│     (if/else: offline.go)   (gọi API OpenAI-compatible)         │
│              └──────────┬───────────┘                           │
│                         ▼                                       │
│              Công cụ (tools.go): đọc/ghi/sửa file, tìm, chạy lệnh│
│                (sandbox: chỉ trong thư mục dự án)               │
└─────────────────────────────────────────────────────────────────┘
```

Vì sao thiết kế như vậy:

- **UI là trang web, server là chính exe.** Không cần framework desktop (Electron ~100 MB). Exe mở cổng localhost rồi gọi trình duyệt có sẵn của Windows.
- **Cả UI lẫn logic nằm trong một binary** nhờ `go:embed` → "một file exe" đúng nghĩa, không kèm DLL hay runtime.
- **Hai "bộ não" dùng chung một bộ công cụ**: if/else (không model) và LLM (có model). Chuyển qua lại chỉ bằng `config.json`.
- **Mọi thao tác nguy hiểm đi qua một cổng duyệt duy nhất** (`execTool`), nên thêm công cụ mới là tự động có hộp xin duyệt.

---

## 2. Công nghệ & yêu cầu

| Thứ | Chọn | Lý do |
|---|---|---|
| Ngôn ngữ | **Go ≥ 1.22** | Build ra binary tĩnh, build chéo sang Windows từ Linux/macOS chỉ bằng biến môi trường |
| Thư viện | **stdlib + 1 thư viện PDF** (`github.com/ledongthuc/pdf`, MIT, thuần Go) | Mọi thứ khác (HTTP, JSON, zip/XML cho docx/xlsx, lập lịch) dùng stdlib. Thư viện PDF được `go mod vendor` vào thư mục `vendor/` nên **build offline được** và không kéo mã lạ lúc build |
| UI | HTML + CSS + JS thuần, 1 file | Không cần npm/webpack; nhúng bằng `go:embed` |
| Realtime | **SSE** (Server-Sent Events qua `fetch` stream) | Đơn giản hơn WebSocket, đủ cho "bot đẩy sự kiện về trình duyệt" |

Cài Go: https://go.dev/dl/ (Windows: file `.msi`). Kiểm tra `go version`.

---

## 3. Cấu trúc thư mục

```
desktop-agent/
├── main.go           # khởi động: config, mở cổng, nhúng UI, khởi động bộ lập lịch, mở trình duyệt
├── config.go         # config.json + biến môi trường
├── server.go         # /api/*: chat (SSE), duyệt, dừng, tasks; bảo vệ token + Host
├── agent.go          # VÒNG LẶP AGENT cho LLM (giao thức khối ```tool, native, auto) + execTool/xin duyệt
├── context.go        # ngân sách token, cắt lịch sử cho LLM 16k (fitContext)
├── offline.go        # bộ não if/else (không LLM) – nơi thêm luật/lệnh
├── tools.go          # tất cả công cụ + sandbox + runShell + đọc trang (window)
├── docs.go           # đọc PDF / DOCX / XLSX / PPTX
├── tasks.go           # task định kỳ: lịch, chạy, báo động, lưu JSON, tool create_task…
├── web.go            # web_search (DuckDuckGo/Google/SearXNG) + fetch_url
├── proc_windows.go   # chạy cmd.exe đúng cách + giết cả cây tiến trình
├── proc_other.go     # bản Linux/macOS
├── web/index.html    # toàn bộ giao diện (nhúng vào exe)
├── vendor/           # thư viện PDF (đã vendor)
├── testdata/         # PDF/DOCX/XLSX mẫu cho test
├── *_test.go         # ~40 test (tools, offline, docs, tasks, agent với LLM giả, web với server giả)
├── build.sh · config.example.json
```

---

## 4. Từng phần code

### 4.1 `main.go` – khởi động và nhúng UI

```go
//go:embed web
var webFS embed.FS            // toàn bộ thư mục web/ bị "nuốt" vào exe

func main() {
    cfg := loadConfig()
    token := randomHex(16)     // mã bí mật mỗi lần chạy
    ln, err := net.Listen("tcp", "127.0.0.1:7878")   // CHỈ localhost
    if err != nil { ln, _ = net.Listen("tcp", "127.0.0.1:0") } // bận → cổng ngẫu nhiên

    sub, _ := fs.Sub(webFS, "web")
    mux := http.NewServeMux()
    mux.Handle("/", http.FileServerFS(sub))           // phục vụ UI
    s.routes(mux)                                      // /api/*
    go openBrowser("http://" + ln.Addr().String() + "/?t=" + token)
    http.Serve(ln, mux)
}
```

Điểm mấu chốt:
- `//go:embed web` (dòng comment đặc biệt) biến thư mục `web/` thành dữ liệu trong binary.
- `127.0.0.1` thay vì `0.0.0.0` → máy khác trong mạng **không** vào được.
- Token đi kèm URL; JS lưu vào `sessionStorage` rồi gửi header `X-Token` cho mọi lời gọi `/api`.
- Mở trình duyệt trên Windows: `rundll32 url.dll,FileProtocolHandler <url>`.
- Log ghi vào `%AppData%\desktop-agent\agent.log` (vì bản GUI không có console).

### 4.2 `server.go` – API và bảo mật

| Endpoint | Việc |
|---|---|
| `GET /api/state` | project hiện tại, chế độ (offline/LLM), tự duyệt |
| `POST /api/project` | chọn thư mục dự án, xóa lịch sử |
| `GET /api/browse` | liệt kê thư mục con (hộp chọn thư mục trong UI) + danh sách ổ đĩa Windows |
| `GET /api/tree`, `/api/file` | cây tệp và xem nội dung (qua sandbox) |
| `POST /api/chat` | **gửi tin nhắn; trả về luồng SSE** các sự kiện |
| `POST /api/approve` | trả lời thẻ xin duyệt |
| `POST /api/reset`, `/api/auto` | cuộc trò chuyện mới / bật tự duyệt |
| `POST /api/cancel` | **dừng ngay lượt đang chạy**: hủy context → giết lệnh đang chạy → nhả cờ `busy` |

Hàm `guard` bọc mọi `/api`:

```go
hostOK := r.Host == s.addr || r.Host == "localhost:"+port    // chống DNS-rebinding
tokOK  := subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Token")), []byte(s.token)) == 1
if !hostOK || !tokOK { http.Error(w, "forbidden", 403); return }
```

Vì sao cần: nếu không có, một trang web bất kỳ bạn mở trong trình duyệt có thể `fetch("http://127.0.0.1:7878/api/...")` để điều khiển bot của bạn.

**Luồng SSE** (`handleChat`): mỗi sự kiện là một dòng `data: {json}\n\n`, sau đó `Flush()`:

```go
emit := func(ev string, data any) {
    j, _ := json.Marshal(map[string]any{"event": ev, "data": data})
    fmt.Fprintf(w, "data: %s\n\n", j); flusher.Flush()
}
```

Các sự kiện: `assistant` (chữ bot nói), `tool_call`, `approval` (xin duyệt), `tool_result`, `error`, `done`.
Phía JS đọc bằng `fetch(...).body.getReader()` và tách theo `\n\n` (dùng `fetch` thay `EventSource` vì cần POST + header token).

### 4.3 `tools.go` – công cụ và sandbox

Mỗi công cụ = (1) khai báo mô tả cho LLM + (2) một `case` trong `runTool`.

**Sandbox đường dẫn – phần quan trọng nhất:**

```go
func resolve(root, p string) (string, error) {
    abs := filepath.Clean(filepath.Join(root, p))
    realRoot, _ := filepath.EvalSymlinks(root)
    // đi lên tìm thư mục cha tồn tại sâu nhất, giải symlink rồi ghép phần đuôi lại
    ...
    rel, _ := filepath.Rel(realRoot, real)
    if rel == ".." || strings.HasPrefix(rel, ".."+sep) { return "", errors.New("outside project") }
    return real, nil
}
```

Chặn được: `../..`, đường dẫn tuyệt đối ngoài dự án, symlink trỏ ra ngoài, và cả trường hợp file chưa tồn tại (khi `ghi`).

**Chạy lệnh (phần dễ sai nhất – đọc kỹ):**

Ba lỗi kinh điển khi gọi lệnh shell từ Go, đều đã được xử lý trong `proc_windows.go` / `proc_other.go` (hàm `newShellCmd`):

1. **Treo vì tiến trình con giữ ống output.** `cmd /C notepad`, `start x`, `ping -t`… sinh tiến trình con thừa hưởng ống stdout; `c.Run()` chờ ống đóng nên chờ mãi. Cách sửa: `c.WaitDelay = 2 * time.Second` (Go ≥ 1.20) – sau khi tiến trình chính xong, chờ tối đa 2 giây rồi tự đóng ống và trả về `exec.ErrWaitDelay`.
2. **Hủy/timeout chỉ giết `cmd.exe`, không giết con của nó.** Cách sửa: gán `c.Cancel` để giết cả **cây tiến trình** – Windows: `taskkill /T /F /PID <pid>`; Linux/macOS: chạy trong process group riêng (`Setpgid: true`) rồi `kill(-pgid, SIGKILL)`.
3. **Dấu nháy kép bị Go escape sai khi gọi `cmd /C`.** Go bọc tham số theo quy tắc `CommandLineToArgvW`, `cmd.exe` không hiểu. Cách sửa: tự đặt `SysProcAttr.CmdLine = `cmd.exe /S /C "chcp 65001>nul & <lệnh>"``; `/S` giữ nguyên dấu nháy bên trong, `chcp 65001` ép output UTF-8 để tiếng Việt không vỡ chữ.

Và để nút **Dừng** thật sự hiệu quả: `handleChat` lưu `context.CancelFunc` vào `Server.cancel`; `POST /api/cancel` gọi nó. Mọi nơi chờ (chạy lệnh, chờ duyệt, gọi LLM, quét tìm kiếm) đều nhận `ctx` nên thoát ngay, handler return → cờ `busy` được nhả. Ngoài ra `tìm` có giới hạn 15 giây để không quét cả ổ đĩa vô tận.

Code gốc:

```go
if runtime.GOOS == "windows" { c = exec.CommandContext(ctx, "cmd", "/C", command) }
else                         { c = exec.CommandContext(ctx, "sh", "-c", command) }
c.Dir = root                      // luôn chạy tại thư mục dự án
hideWindow(c)                     // Windows: CREATE_NO_WINDOW, không nháy cửa sổ đen
// timeout 60s (tối đa 300s), cắt output 20 KB
```

`proc_windows.go` dùng build tag `//go:build windows` để chỉ biên dịch trên Windows (cờ `0x08000000` = `CREATE_NO_WINDOW`).

**Công cụ tìm kiếm:** `find_files` so khớp theo *tên* (chuỗi con không phân biệt hoa/thường, hoặc glob `*.py` qua `filepath.Match`); `search_files` so khớp theo *nội dung* từng dòng bằng `regexp`. Lệnh `tìm x` trong `offline.go` gọi cả hai và ghép thành một câu trả lời hai phần.

**Giới hạn an toàn có sẵn:** đọc tối đa 200 KB/tệp, từ chối tệp nhị phân (phát hiện byte `\x00`), tìm kiếm (theo tên và theo nội dung) tối đa 200 kết quả, 15 giây, bỏ qua `.git` và `node_modules`, `sửa` chỉ khi chuỗi cũ xuất hiện **đúng 1 lần**.

### 4.4 `agent.go` – cổng duyệt dùng chung

```go
func (s *Server) execTool(ctx, root, tc, emit) string {
    emit("tool_call", ...)
    if needsApproval(name) && !autoApprove {
        if !s.askApproval(ctx, tc.ID, name, args, emit) { return "error: the user denied this action" }
    }
    result := runTool(ctx, root, name, args)
    emit("tool_result", ...)
    return result
}
```

`askApproval` phát sự kiện `approval`, rồi **chờ trên một channel** cho đến khi UI gọi `POST /api/approve` (hoặc hết 10 phút / khách đóng kết nối):

```go
ch := make(chan bool, 1); s.pending[id] = ch
emit("approval", ...)
select { case ok := <-ch: return ok; case <-ctx.Done(): return false; case <-time.After(10*time.Minute): return false }
```

Cả chế độ if/else và chế độ LLM đều gọi `execTool` → **một chỗ duy nhất** quyết định việc gì cần duyệt (`needsApproval`).

Chế độ LLM (`runAgent`) là vòng lặp tool-calling chuẩn:

```
lặp tối đa 25 lần:
    gửi [system + lịch sử + định nghĩa tools] đến POST {base_url}/chat/completions
    nếu model trả tool_calls  → chạy từng cái qua execTool → thêm kết quả vào lịch sử → lặp
    nếu chỉ trả chữ           → hiển thị và dừng
```

### 4.5 `offline.go` – bộ não if/else (phần bạn sẽ sửa nhiều nhất)

```go
func (s *Server) runOffline(ctx, msg string, emit emitFn) {
    fields := strings.Fields(strings.TrimLeft(msg, "/!"))
    verb := strings.ToLower(fields[0])                     // từ khóa đầu
    rest := strings.TrimSpace(msg[len(fields[0]):])        // phần còn lại

    switch {
    case verb == "ls" || verb == "dir" || verb == "list":  call("list_dir", map[string]any{"path": rest}, true)
    case verb == "đọc" || verb == "read" || verb == "cat": call("read_file", map[string]any{"path": rest}, true)
    case verb == "tạo" || verb == "write":
        path, content, ok := strings.Cut(rest, ":")        // "tạo a.txt : nội dung"
        ...
    default:
        emit("assistant", "Mình chưa hiểu ...")
    }
}
```

`call(tên_công_cụ, tham_số, hiện_kết_quả_trong_chat)`:
- `true`  → in kết quả vào khung code (dùng cho lệnh đọc).
- `false` → chỉ in "✅ …" (dùng cho lệnh ghi).

#### Thêm một lệnh if/else mới (ví dụ: `đếm <tệp>` đếm số dòng)

1. **Thêm công cụ** trong `tools.go` (nếu chưa có công cụ phù hợp):
   ```go
   case "count_lines":
       p, err := resolve(root, a.Path); if err != nil { return "error: " + err.Error() }
       b, err := os.ReadFile(p);        if err != nil { return "error: " + err.Error() }
       return fmt.Sprintf("%d dòng", strings.Count(string(b), "\n"))
   ```
2. **Khai báo cho LLM** (để chế độ có model cũng dùng được), thêm vào `toolDefs`:
   ```go
   fn("count_lines", "Count lines in a file.", props{"path": str("relative file path")}, "path"),
   ```
3. **Thêm luật** trong `offline.go`:
   ```go
   case verb == "đếm" || verb == "count":
       call("count_lines", map[string]any{"path": rest}, true)
   ```
4. (Tuỳ chọn) thêm dòng vào `offlineHelp` và mảng `CMDS` trong `web/index.html` để hiện trong menu `/`.
5. Nếu lệnh thay đổi dữ liệu: thêm tên công cụ vào `needsApproval` → tự có hộp xin duyệt.

#### Luật nâng cao hơn (regex, nhiều biến thể câu)

Thay `switch` bằng danh sách luật:

```go
type rule struct{ re *regexp.Regexp; do func(m []string) }
var rules = []rule{
    {regexp.MustCompile(`(?i)^(đọc|xem|mở)\s+(?:file\s+)?(.+)$`), func(m []string){ call("read_file", ..., true) }},
    {regexp.MustCompile(`(?i)^tạo\s+(\S+)\s*:\s*(.*)$`),           func(m []string){ ... }},
}
for _, r := range rules { if m := r.re.FindStringSubmatch(text); m != nil { r.do(m); return } }
```

Cần bot nhớ ngữ cảnh ("mở file đó")? Dùng `s.history`/thêm trường `lastFile string` vào `Server` và ghi lại sau mỗi lệnh đọc.

### 4.6 `web/index.html` – giao diện

Một file, không thư viện. Các khối chính:

- **Design tokens** (biến CSS `--bg`, `--fg`, …) và `html[data-theme=dark]` ghi đè → đổi sáng/tối chỉ bằng cách đổi thuộc tính `data-theme`. Lưu lựa chọn trong `localStorage` (bọc `try/catch`).
- **Bố cục:** thanh bên (thu gọn được) + cột chat rộng tối đa 768px giữa màn hình + ô nhập dạng viên thuốc cố định dưới đáy – giống ChatGPT/Gemini.
- **Markdown tối giản** (`md()`): tách theo ``` để dựng khung code có nút Sao chép; xử lý `code`, `**đậm**`, danh sách gạch đầu dòng. Mọi chữ đều được `esc()` (escape HTML) **trước** khi chèn `innerHTML` → không bị XSS từ nội dung tệp.
- **Nhận SSE:** `fetch('/api/chat', {method:'POST'})` → `response.body.getReader()` → ghép buffer → tách theo `\n\n` → `onEvent()`.
- **Menu `/`:** mảng `CMDS` (tên, mô tả, mẫu điền sẵn); lọc theo chữ đã gõ; ↑↓/Tab/Enter chọn.
- **Responsive:** dưới 860px thanh bên thành ngăn kéo; lưới gợi ý 4 cột → 2 cột.

Sửa giao diện: chỉnh trực tiếp `web/index.html`, `go build` lại là xong (có thể chạy `go run .` và F5 trình duyệt).

---

## 4b. Các phần mở rộng (agent thật, đọc tài liệu, task, web)

### 4b.1 Vòng lặp agent cho LLM nhỏ (`agent.go`, `context.go`)

**Vấn đề:** nhiều LLM nội bộ không hỗ trợ `tools`/function-calling, và chỉ có 16k ngữ cảnh. Cách xử lý:

1. **Giao thức công cụ bằng văn bản** (`tool_mode: "text"`, mặc định). System prompt liệt kê công cụ dạng `read_file(path, offset?)` và yêu cầu trả lời bằng khối ```` ```tool {"name":..,"args":{..}} ``` ````. `parseToolText` đọc rất rộng lượng: khối fenced (`tool`/`json`), `<tool_call>…</tool_call>`, JSON "trần" lẫn trong văn bản; khóa `args`/`arguments`/`parameters`/`input`; tham số có thể là chuỗi JSON hoặc đặt phẳng; bí danh tên (`bash`→`run_command`, `cat`→`read_file`…). JSON thường (không có `name` là công cụ có thật) **không bị coi là lệnh**.
2. **Mỗi lượt một công cụ**; kết quả trả lại dưới dạng tin nhắn `user` ("Kết quả của `x`: …") – nên lịch sử chỉ gồm `user`/`assistant`, không có tin nhắn `tool` mồ côi, rất dễ cắt.
3. **Tham số khoan dung** (`targs` trong `tools.go`): số dạng chuỗi (`"30"`), bí danh khóa (`cmd`→`command`, `file`→`path`, `q`→`query`…), boolean dạng chữ (`"false"`).
4. **Vòng lặp** tối đa 15 bước; chặn lặp lại cùng một lệnh; nhắc sửa khối JSON hỏng (tối đa 2 lần); tự chuyển `auto`→`text` nếu máy chủ từ chối `tools`.

```go
for step := 0; step < maxAgentSteps; step++ {
    msgs := fitContext(systemPrompt, history, budget)   // cắt cho vừa 16k
    m, err := callLLM(ctx, msgs, native)                // POST /chat/completions (max_tokens, temperature 0.2)
    call := m.ToolCalls ?? parseToolText(m.Content)     // native hoặc khối ```tool
    if no call { emit(answer); return }
    result := execTool(ctx, root, call, emit)           // xin duyệt → chạy THẬT → kết quả
    history += assistant(m) + result(clipped)
}
```

**Ngân sách token (`context.go`)**: `estTokens` ước lượng *bi quan* (ASCII ≈ 3 ký tự/token, ký tự có dấu ≈ 1.5 ký tự/token) – thà gửi ít còn hơn bị lỗi vượt context. `fitContext(system, history, budget)`:
- luôn giữ **system** và **lượt hỏi mới nhất**; nếu lượt mới quá lớn thì rút gọn dần kết quả công cụ (6000→2500→1000→400→150 ký tự), sau đó bỏ nhóm (assistant+kết quả) cũ nhất;
- lượt cũ hơn: chỉ giữ nếu còn chỗ, kết quả công cụ rút còn 500 ký tự;
- không bao giờ bắt đầu giữa một lượt; ở chế độ `native` không để lại tin nhắn `tool` mồ côi.
Ngoài ra: `read_file` trả theo trang (`readChunkChars` ≈ `context_tokens/5` ký tự) kèm gợi ý `offset=`; lịch sử lưu tối đa ~120 tin; lỗi 4xx có chữ "context/token/maximum" → thử lại với ngân sách ×0.6.

### 4b.2 Đọc tài liệu (`docs.go`)

DOCX/XLSX/PPTX **là file zip chứa XML** nên chỉ cần `archive/zip` + `encoding/xml` (đọc dạng luồng):
- **DOCX**: `word/document.xml` → gom các thẻ `<w:t>`; `</w:p>` xuống dòng; `<w:tc>` nối ` | `, `</w:tr>` xuống dòng.
- **XLSX**: `xl/workbook.xml` (tên sheet + id) → `xl/_rels/workbook.xml.rels` (id → tệp) → `xl/sharedStrings.xml` (chuỗi dùng chung) → từng `sheetN.xml`. Ô kiểu `t="s"` tra bảng chuỗi; vị trí cột lấy từ thuộc tính `r="C5"` (nên ô trống ở giữa vẫn giữ đúng cột); ô `<f>` không có giá trị tính sẵn → hiện `=công_thức`.
- **PPTX**: `ppt/slides/slideN.xml` → các `<a:t>`.
- **PDF**: giải mã luồng/font phức tạp nên dùng thư viện `ledongthuc/pdf` (`GetPlainText`), bọc `recover()` vì parser có thể panic với tệp hỏng; trang không có chữ → báo "bản scan, cần OCR".
- Giới hạn: 400 KB chữ, 200 trang, 3000 dòng/sheet, mỗi mục zip ≤ 64 MB (chống zip-bomb).
`read_file` tự nhận đuôi tệp và gọi `readDocument`, nên cả chế độ lệnh lẫn LLM đều đọc được.

### 4b.3 Task định kỳ (`tasks.go`)

```go
type Task struct { ID, Name, Schedule, Command, Dir string; TimeoutSec int
    AlertOnFail bool; AlertContains, AlertNotContains string
    Enabled bool; NextRun time.Time; History []Run /* ≤20 */ }
```
- `parseSchedule` (regex) hiểu "mỗi 5 phút", "every 30s", "hàng ngày 8h30"…; `schedule.next(after)` tính lần chạy kế (interval: +Every; daily: hôm nay hoặc ngày mai lúc HH:MM).
- `Scheduler.Start()` chạy một goroutine, **tick mỗi giây**: task tới hạn (`Enabled && !Running && NextRun<=now`) → đặt `Running`, tính `NextRun` ngay, rồi chạy trong goroutine riêng (không bao giờ chạy chồng lên chính nó).
- `execute` dùng lại `runShell` (timeout, giết cây tiến trình) → tạo `Run` → đánh giá điều kiện (mã thoát, `contains`, `not contains`, hết giờ) → ghi lịch sử → lưu `tasks.json` (ghi tệp tạm rồi `rename` để không hỏng nửa chừng).
- **Chống spam**: chỉ phát cảnh báo khi trạng thái *đổi* (ổn→lỗi, lỗi→ổn). Cảnh báo vào hàng đợi `alerts` (UI poll `/api/tasks` mỗi 5 giây, hiện toast + huy hiệu) và tùy chọn `POST` tới `alert_webhook`.
- Khi mở lại chương trình, task có `NextRun` quá khứ được **tính lại từ bây giờ** (không chạy bù hàng loạt).
- Task dạng *script*: `scriptCommand` lưu mã ra `%AppData%\desktop-agent\tasks\<id>.py|.ps1|.bat|.sh` rồi tạo lệnh chạy nó.
- Duyệt: `create_task`/`delete_task` nằm trong `needsApproval` → thẻ duyệt hiện toàn bộ lệnh; task chạy sau đó không cần duyệt lại.
- Giới hạn: 30 task, chu kỳ ≥ 10 giây.

### 4b.4 Web (`web.go`)

`web_search`: DuckDuckGo (`html.duckduckgo.com/html/?q=`, tách `result__a`/`result__snippet` bằng regex, gỡ lớp chuyển hướng `uddg=`), Google Programmable Search (JSON API), hoặc SearXNG (`format=json`). `fetch_url`: tải (giới hạn 2 MB, timeout 20 giây), bỏ `script/style/nav/footer`, đổi thẻ khối thành xuống dòng, bỏ thẻ còn lại, `html.UnescapeString`, đọc theo trang. Chỉ nhận `http(s)://`. **`fetch_url` cần duyệt** (chống rò dữ liệu qua URL / prompt-injection).

## 5. Build ra file exe

### 5.1 Build trên Windows

```powershell
cd desktop-agent
go build -trimpath -ldflags "-s -w -H=windowsgui" -o desktop-agent.exe .
```

### 5.2 Build chéo từ Linux/macOS (cách dùng ở đây)

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 \
  go build -trimpath -ldflags "-s -w -H=windowsgui" -o dist/desktop-agent.exe .
```

Hoặc chạy `./build.sh` (đã gồm bước copy `config.json`).

| Cờ | Tác dụng |
|---|---|
| `GOOS=windows GOARCH=amd64` | nhắm tới Windows 64-bit (`arm64` cho Windows ARM) |
| `CGO_ENABLED=0` | không phụ thuộc thư viện C → binary tĩnh thật sự |
| `-ldflags "-s -w"` | bỏ bảng ký hiệu/debug → nhỏ hơn ~30% |
| `-ldflags "-H=windowsgui"` | **bản GUI: không hiện cửa sổ console** khi bấm đúp. Bỏ cờ này khi debug để thấy log ra màn hình |
| `-trimpath` | bỏ đường dẫn máy build khỏi binary (build lặp lại được, không lộ tên thư mục) |

Kết quả: `desktop-agent.exe` khoảng 7 MB, chạy được trên Windows 10/11 mà không cần cài gì.

### 5.3 Đặt icon, tên, phiên bản cho exe (tuỳ chọn, nên làm)

Dùng `goversioninfo` (công cụ build, không vào code chạy):

```bash
go install github.com/josephspurrier/goversioninfo/cmd/goversioninfo@latest
```

Tạo `versioninfo.json`:

```json
{ "FixedFileInfo": { "FileVersion": {"Major":1,"Minor":0,"Patch":0,"Build":0},
                     "ProductVersion": {"Major":1,"Minor":0,"Patch":0,"Build":0} },
  "StringFileInfo": { "ProductName": "Trợ lý Desktop", "CompanyName": "Công ty bạn",
                      "FileDescription": "Trợ lý Desktop", "OriginalFilename": "desktop-agent.exe" },
  "IconPath": "app.ico" }
```

Thêm dòng `//go:generate goversioninfo -64` vào `main.go`, chạy `go generate` → sinh `resource_windows_amd64.syso` (Go tự link). Sau đó build như bình thường. Exe sẽ có icon riêng và thông tin phiên bản (giúp giảm bị Defender nghi ngờ).

### 5.4 Ký số (khuyến nghị khi đưa cho người khác dùng)

Exe không ký sẽ bị SmartScreen cảnh báo. Cách ký:

```powershell
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /f cert.pfx /p MATKHAU desktop-agent.exe
```

- **Trong công ty:** dùng chứng chỉ từ CA nội bộ, IT đẩy chứng chỉ gốc xuống máy bằng GPO/Intune → hết cảnh báo.
- **Phát hành ra ngoài:** mua chứng chỉ OV/EV code-signing hoặc dùng Azure Trusted Signing.
- Luôn **timestamp** (`/tr`) để chữ ký không hết hạn khi chứng chỉ hết hạn.

### 5.5 Đóng gói phát hành

Cách đơn giản nhất: nén zip gồm `desktop-agent.exe`, `config.json`, `HUONG_DAN.md`.
Cần trình cài đặt có Start Menu/gỡ cài đặt → dùng **Inno Setup**:

```iss
[Setup]
AppName=Tro Ly Desktop
AppVersion=1.0
DefaultDirName={autopf}\TroLy
OutputBaseFilename=TroLy-Setup
[Files]
Source: "desktop-agent.exe"; DestDir: "{app}"
Source: "config.json";       DestDir: "{app}"; Flags: onlyifdoesntexist
[Icons]
Name: "{autoprograms}\Tro Ly"; Filename: "{app}\desktop-agent.exe"
```

Muốn tự chạy khi đăng nhập Windows: thêm `Name: "{userstartup}\Tro Ly"; Filename: "{app}\desktop-agent.exe"` vào `[Icons]`.

### 5.6 Build tự động bằng GitHub Actions

```yaml
name: build
on: { push: { tags: ['v*'] } }
tasks:
  win:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: '1.24' }
      - run: cd desktop-agent && ./build.sh
      - uses: actions/upload-artifact@v4
        with: { name: desktop-agent, path: desktop-agent/dist/ }
```

---

## 6. Nâng cấp: nối model AI và RAG

### 6.1 Có model → chỉ cần sửa `config.json`

```json
{ "base_url": "http://llm.noibo:8000/v1", "model": "ten-model", "api_key": "" }
```

`base_url` không rỗng → `runAgent` thay `runOffline`. Cần máy chủ tương thích OpenAI (`/chat/completions`) và model hỗ trợ **tool calling** (Ollama, vLLM, LM Studio, llama.cpp server đều có).

### 6.2 Thêm RAG nội bộ như một công cụ

Trong `tools.go`:

```go
// toolDefs
fn("rag_search", "Search the company knowledge base.", props{"query": str("question")}, "query"),

// runTool
case "rag_search":
    body, _ := json.Marshal(map[string]any{"query": a.Query, "top_k": 5})
    req, _ := http.NewRequestWithContext(ctx, "POST", ragURL, bytes.NewReader(body))
    req.Header.Set("Authorization", "Bearer "+ragToken)
    resp, err := http.DefaultClient.Do(req)
    if err != nil { return "error: " + err.Error() }
    defer resp.Body.Close()
    b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
    return string(b)
```

(Thêm trường `Query string `json:"query"`` vào struct `a`, đọc `ragURL`, `ragToken` từ `Config`.)

Với **chế độ offline** có thể gắn luôn bằng luật: `case verb == "hỏi": call("rag_search", map[string]any{"query": rest}, true)` – bot if/else nhưng vẫn tra cứu được tri thức công ty, không cần LLM.

### 6.3 Hướng khác: dùng framework agent có sẵn

Nếu muốn nhiều tính năng (kỹ năng, bộ nhớ, MCP) thay vì tự viết: Hermes Agent (Python) đóng được bằng PyInstaller (~70 MB, đã thử trên Linux, phải build trên Windows để dùng cho Windows, cần `--collect-all openai` và `--copy-metadata`); OpenClaw (Node) có Web UI sẵn nhưng cần Node runtime. Đánh đổi: nặng hơn, bề mặt tấn công rộng hơn, cập nhật upstream thường xuyên. Với nhu cầu "chat + thao tác tệp + terminal" thì bản Go này gọn và dễ kiểm soát hơn.

---

## 7. Bảo mật – checklist trước khi phát hành

- [ ] Chỉ `127.0.0.1`; không bao giờ đổi thành `0.0.0.0` nếu chưa có đăng nhập.
- [ ] Giữ `guard` (token + kiểm tra Host) cho **mọi** `/api`.
- [ ] Mọi đường dẫn đi qua `resolve()`. Không bao giờ `os.ReadFile(userPath)` trực tiếp.
- [ ] Công cụ mới có thay đổi dữ liệu → thêm vào `needsApproval`.
- [ ] Không nhét API key vào mã nguồn/exe; để ở `config.json` hoặc cấp theo người dùng.
- [ ] Chạy bằng quyền người dùng thường, **không** chạy Administrator.
- [ ] Chế độ LLM: coi nội dung tệp là **không tin cậy** (có thể chứa "prompt injection"); giữ duyệt thủ công với `chạy`/`xóa`.
- [ ] Ký số exe và công bố mã băm SHA-256 (`certutil -hashfile desktop-agent.exe SHA256`).

---

## 8. Kiểm thử

Dự án có sẵn bộ test tự động – **chạy trước mỗi lần build**:

```bash
go test ./...        # ~10 giây
go test -race ./...  # kiểm tra tranh chấp dữ liệu (nên chạy trước khi phát hành)
```

- `agent_test.go`: **LLM giả** (httptest) chạy đủ 3 chế độ – text, native, auto; kiểm tra lệnh chạy thật và kết quả quay lại LLM, 8 biến thể cú pháp khối tool, sửa khối hỏng, chặn lặp, giới hạn bước, từ chối duyệt, lỗi 500, **hội thoại 25 lượt không vượt ngân sách 16k**, kết quả khổng lồ bị cắt, **thử lại khi server báo tràn context**.
- `docs_test.go`: PDF tiếng Việt, PDF do LibreOffice xuất, PDF scan, DOCX có bảng, XLSX nhiều sheet/ô trống/công thức/ký tự đặc biệt, tệp giả đuôi `.docx`, phân trang UTF-8.
- `tasks_test.go`: phân tích lịch (hợp lệ/không hợp lệ), tính lần chạy kế, báo động + hồi phục + không spam, các luật `contains`, timeout, lưu/nạp JSON, không chạy bù, tick, lệnh `task …`.
- `web_test.go`: DuckDuckGo/Google/SearXNG bằng server giả, bóc chữ HTML, URL sai, duyệt.
- `tools_test.go`: sandbox (thoát bằng `../`, đường dẫn tuyệt đối, symlink), đọc/ghi/sửa/thêm/đổi tên/xóa, chạy lệnh (dấu nháy, mã thoát, stderr, **timeout**, **tiến trình nền**, **hủy giữa chừng**).
- `offline_test.go`: hơn 100 câu lệnh người dùng hay gõ (hoa/thường, tiếng Việt/Anh, tên tệp có dấu và khoảng trắng, nội dung chứa dấu `:`, thiếu tham số, từ chối duyệt).

Khi thêm lệnh/luật mới, thêm một dòng `b.expect("câu gõ", "chữ mong đợi")` vào `TestOfflineCommands`. Hai lỗi thật đã được test này bắt được: từ đệm "thư mục" bị cắt nhầm khỏi đường dẫn khi `tạo`, và `thêm` dính chữ khi tệp không kết thúc bằng xuống dòng.

Ghi chú: các test chạy lệnh shell chỉ chạy đủ trên Linux/macOS; trên Windows chúng bỏ qua phần `sleep`. Luôn thử bản exe trên Windows thật trước khi phát hành.

Chạy thử thủ công không cần Windows (trên Linux/macOS):

```bash
cd desktop-agent
go vet ./...
go run . --project /duong/dan/thu/muc   # mở trình duyệt tự động
```

Kiểm thử qua API (bỏ qua UI) – lấy token từ dòng `Open: http://…/?t=TOKEN` in ra:

```bash
curl -N -H "X-Token: $T" -X POST -d '{"message":"ls"}' http://127.0.0.1:7878/api/chat
# data: {"event":"tool_call",...}  data: {"event":"tool_result",...}  data: {"event":"assistant",...}
```

Các điểm cần thử khi sửa code: lệnh đọc ngoài dự án (`đọc ../../x` phải bị chặn); `sửa` với chuỗi xuất hiện 0 và 2 lần; từ chối thẻ duyệt không được làm thay đổi tệp; gọi `/api/state` thiếu token phải trả 403.

---

## 9. Lỗi thường gặp khi tự dựng

| Lỗi | Nguyên nhân / cách sửa |
|---|---|
| `pattern web: cannot embed directory web: contains no embeddable files` | Thư mục `web/` rỗng – thêm `index.html` trước khi build |
| `go:embed` không thấy tệp bắt đầu bằng `_` hoặc `.` | Mặc định bị bỏ qua; đổi tên hoặc dùng `//go:embed all:web` |
| Exe bấm lên không có gì | Build với `-H=windowsgui` thì không có console; xem `agent.log` hoặc build bản không có cờ này để thấy lỗi |
| Trình duyệt không tự mở | `rundll32` bị chặn bởi chính sách; mở thủ công URL trong log |
| UI cũ dù đã sửa | Trình duyệt cache; `Ctrl+F5`. Nhớ build lại vì UI nằm *trong* exe |
| Defender đánh dấu virus | Do exe chưa ký + có hành vi "chạy lệnh"; ký số, thêm icon/version info, nhờ IT whitelist theo SHA-256 |
| Streaming bị "đứng" sau proxy công ty | Proxy gom đệm; tắt buffering ở proxy hoặc chạy trực tiếp localhost |


---

## 10. Có nên dùng framework agent có sẵn?

Phần "agent" ở đây (vòng lặp + công cụ + cắt ngữ cảnh) chỉ khoảng 300 dòng Go, nên tôi **tự viết** để: ra được **một exe nhỏ (~8 MB)**, kiểm soát chặt quyền hạn/duyệt, chạy với LLM nhỏ không có function-calling, và hiểu từng dòng. Nếu bạn vẫn muốn dùng framework, đây là các lựa chọn phổ biến (**tôi chưa thử đóng gói các framework này thành exe trong dự án này** – nhận định dưới đây dựa trên cách chúng được thiết kế, hãy tự kiểm chứng):

| Framework | Ngôn ngữ | Hợp với | Lưu ý khi muốn ra `.exe` |
|---|---|---|---|
| **smolagents** (Hugging Face) | Python | Agent nhỏ gọn, mô hình viết *mã Python* để làm việc; dễ đọc | Đóng bằng PyInstaller ~50–100 MB; để model chạy mã tùy ý thì **phải sandbox** |
| **PydanticAI** | Python | Công cụ có kiểu dữ liệu chặt, đầu ra có cấu trúc | Như trên |
| **LangGraph / LangChain** | Python | Luồng nhiều bước, đồ thị trạng thái, lưu/khôi phục tiến trình | Phụ thuộc nhiều, exe nặng; hay đổi API |
| **OpenAI Agents SDK** | Python | Agent + công cụ + handoff | Thiết kế cho API OpenAI; dùng LLM nội bộ cần endpoint tương thích |
| **AutoGen/AG2, CrewAI** | Python | Nhiều agent phối hợp | Thường thừa với nhu cầu "một trợ lý làm việc với tệp" |
| **langchaingo, Eino** | Go | Nếu muốn ở lại Go và có sẵn nhiều tích hợp | Thêm phụ thuộc; vẫn ra được một binary |
| **Vercel AI SDK, Mastra** | TypeScript | Nếu đã có hệ sinh thái Node/React | Cần Node runtime hoặc `bun build --compile` |

**Gợi ý chọn:**
- Cần *một exe gọn, kiểm soát quyền, LLM nhỏ 16k* → giữ bản Go này.
- Muốn **thử ý tưởng nhanh** và không quan tâm kích thước exe → viết prototype bằng **smolagents hoặc PydanticAI** (Python), sau này mới quyết định có chuyển sang Go.
- Cần **quy trình nhiều bước, nhiều agent, lưu trạng thái** → LangGraph, nhưng cân nhắc nó trên máy Windows nhân viên (nặng, nhiều phụ thuộc).
- Dù chọn gì, giữ nguyên các nguyên tắc ở mục 7: *sandbox đường dẫn, duyệt trước khi ghi/chạy, giới hạn thời gian, coi nội dung đọc từ web/tệp là không tin cậy*. Framework không tự làm các việc này cho bạn.

**Việc "đặt lịch kiểm tra công việc"** nên tách khỏi framework agent: dùng bộ lập lịch đơn giản (như `tasks.go`) chạy *lệnh/script cố định*, chỉ gọi LLM khi cần diễn giải kết quả. Như vậy task vẫn chạy ổn khi LLM tắt hoặc chậm.
