# Nhật ký phát triển & ghi chú bàn giao

Tài liệu này ghi lại **quy trình đã code ra dự án**, từng chặng đã làm gì và vì sao, các lỗi thật đã gặp, **cách xuất ra mọi định dạng** (exe, zip, tar.gz, dịch vụ, tài liệu), và những việc nên làm tiếp. Mục đích: người (hoặc AI) tiếp quản chỉ cần đọc file này + `git log` + chạy `go test ./...` là làm tiếp được.

> Phạm vi: thư mục `desktop-agent/` của repo. Nhánh làm việc: `claude/practical-gates-ndvrc8`. Ngôn ngữ giao tiếp/tài liệu: tiếng Việt.

---

## 0. Tóm tắt một trang

**Sản phẩm:** một chương trình chạy trên máy người dùng (Windows `.exe`, Linux), **một file duy nhất**, mở giao diện chat trong trình duyệt tại `http://127.0.0.1:7878`. Làm việc với một hoặc nhiều **thư mục dự án**, mỗi dự án có nhiều **phiên chat**:
- đọc / ghi / sửa / xóa / đổi tên tệp, tìm theo tên (chịu gõ sai, không dấu) và theo nội dung (kể cả trong PDF/Word/Excel/PowerPoint), chạy lệnh terminal thật;
- hai "bộ não": **bot if/else** (không cần model) hoặc **trợ lý LLM** (vòng lặp agent gọi công cụ, chạy được với LLM không có function-calling và ngữ cảnh 16k);
- **task định kỳ** (monitor, lịch hàng ngày) có báo động; tìm web + đọc trang; xem PDF/ảnh gốc ngay trong giao diện;
- mọi thao tác ghi/chạy đều qua **thẻ xin duyệt**; mọi đường dẫn bị khóa trong thư mục dự án.

**Quy mô (lúc bàn giao):** ~9019 dòng (Go gồm cả test + `web/index.html`), **77 test**, 1 thư viện ngoài (`ledongthuc/pdf`, đã vendor), exe ≈ 8 MB.

**Đọc theo thứ tự này khi tiếp quản:** mục 1 (bản đồ mã) → mục 5 (quyết định thiết kế) → mục 4 (lỗi đã gặp) → mục 9 (kiểm thử) → mục 11 (việc nên làm).

---

## 1. Bản đồ mã nguồn

| File | Trách nhiệm | Test tương ứng |
|---|---|---|
| `main.go` | Khởi động: đọc config, tạo `App`, nghe `127.0.0.1`, nhúng UI (`go:embed web`), mở trình duyệt, cờ `--project` `--no-browser` `--selftest`. Định nghĩa `Server` = *bộ chạy của MỘT phiên chat* | — |
| `config.go` | `config.json` cạnh exe + biến môi trường (`LLM_BASE_URL`…) | — |
| `instance.go` | **Một bản duy nhất** (`instance.json`), tự tắt khi không còn tab, `/api/quit` | `instance_test.go`, `workspace_test.go` |
| `assets/` | Logo con vịt (`duck.svg`), `app.ico`, `make_syso.py` (nhúng icon vào exe) | kiểm bằng `pefile` |
| `server.go` | `App` (HTTP): xác thực token + Host, workspace/phiên/project, `/api/chat` (SSE), duyệt, dừng, xem tệp (`/api/file`, `/api/raw`, `/api/open`), tasks | `workspace_test.go` |
| `workspace.go` | `Workspace`→`Project`→`Session`; lưu `workspace.json`; ghi lại log để vẽ lại hội thoại; chuyển lịch sử LLM ⇄ đĩa | `workspace_test.go` |
| `agent.go` | **Vòng lặp agent** cho LLM: giao thức khối ```` ```tool ````, native/auto, parse khoan dung, chặn lặp, `execTool` + xin duyệt | `agent_test.go`, `agentic_test.go` |
| `context.go` | Ngân sách token, `fitContext` (cắt lịch sử cho LLM 16k), `compactHistory` | `agent_test.go` |
| `offline.go` | **Bot if/else**: phân tích câu lệnh tiếng Việt/Anh → gọi công cụ | `offline_test.go` |
| `tools.go` | Mọi công cụ + **sandbox đường dẫn** + `runShell` + đọc theo trang (`window`) + tham số khoan dung (`targs`) | `tools_test.go` |
| `fuzzy.go` | So khớp tên tệp gần đúng (bỏ dấu, Levenshtein), gợi ý "có phải ý bạn là…" | `fuzzy_test.go` |
| `docs.go` | Đọc PDF / DOCX / XLSX / PPTX | `docs_test.go` |
| `tasks.go` | Task định kỳ: lịch, thực thi, báo động, lưu `tasks.json`, tool `create_task`… | `tasks_test.go` |
| `web.go` | `web_search` (DuckDuckGo/Google/SearXNG) + `fetch_url` | `web_test.go` |
| `selftest.go` | `--selftest`: 3 mục tiêu nhiều bước chạy với LLM thật, in ĐẠT/KHÔNG ĐẠT | `agentic_test.go` |
| `proc_windows.go` / `proc_other.go` | Chạy shell đúng cách theo hệ điều hành, giết cả cây tiến trình | `tools_test.go` |
| `web/index.html` | Toàn bộ giao diện (HTML+CSS+JS thuần, 1 file) | kiểm bằng Playwright (mục 9.4) |
| `build.sh` | Build ra cả 3 gói phát hành | — |
| `desktop-agent.service` | Mẫu dịch vụ systemd | — |
| `docs/export-docs.sh` | Xuất `.md` → HTML/DOCX/PDF | — |
| `testdata/make_fixtures.py` | Tạo lại PDF/DOCX/XLSX mẫu cho test | — |
| `vendor/` | Thư viện PDF (đã vendor → build offline) | — |

**Mô hình đối tượng:** `App` giữ `Workspace` và một map `sessionID → *Server`. Mỗi `Server` có thư mục dự án, lịch sử LLM, cờ `busy`, `cancel`, các thẻ duyệt đang chờ. Nhờ vậy vòng lặp agent / bot offline / công cụ **không cần biết** có nhiều dự án hay nhiều phiên.

---

## 2. Quy trình đã dùng để code (lặp lại được)

Mỗi tính năng đi qua **cùng một vòng**; không bước nào bị bỏ:

1. **Làm rõ yêu cầu → chọn phương án rẻ nhất chạy được.** (vd: tự viết agent Go thay vì bọc Hermes/OpenClaw – xem 3.1).
2. **Viết lõi trước, giao diện sau.** Công cụ (`tools.go`) → bộ não (`offline.go`/`agent.go`) → API (`server.go`) → UI.
3. **`go vet ./...` + `GOOS=windows go vet ./...`** sau mỗi lần sửa (bắt lỗi biên dịch riêng Windows mà không cần máy Windows).
4. **Viết test NGAY CÙNG LÚC với tính năng**, ưu tiên test các ca xấu (tên sai, tệp hỏng, hết giờ, bấm Dừng, từ chối duyệt).
5. **Chạy binary thật** (`go build -o /tmp/da .`), gọi API bằng `curl` để xem từng sự kiện SSE.
6. **Kiểm tra giao diện bằng Playwright (Chromium headless)**: điều khiển như người dùng, in kết quả từng bước, chụp ảnh và *nhìn ảnh*. (Lưu ý: Chromium headless **không có bộ xem PDF** – mục 9.4.)
7. **`go test -race ./...`** trước khi đóng gói.
8. **`./build.sh`** → mở gói vừa tạo, giải nén vào thư mục sạch, chạy thử (`env -i` để chắc chắn không phụ thuộc môi trường).
9. **Cập nhật tài liệu** (`HUONG_DAN.md` cho người dùng, `HUONG_DAN_TAO_BOT.md` cho kỹ thuật, file này cho bàn giao).
10. **Commit nhỏ, có mô tả; push** lên nhánh làm việc.

**Quy tắc rút ra (đã tốn công mới hiểu):**
- *Test mà vượt qua ngay lần đầu thì nghi ngờ test.* Nhiều lần kiểm tra thêm mới phát hiện test quá lỏng (xem lỗi PDF ở 4).
- *Mô phỏng người dùng, không chỉ gọi hàm:* lỗi treo lệnh, lỗi "bận", lỗi JSON `null` đều chỉ lộ khi chạy end-to-end.
- *Tránh test phụ thuộc thời gian/goroutine mà không chờ đúng điều kiện* (đã bị treo 10 phút vì goroutine "bấm Từ chối" chạy trước khi thẻ duyệt xuất hiện → dùng `denyAll()` thăm dò lặp).
- *Không tải/chạy file nhị phân lạ để thử:* môi trường chặn việc tải Ollama; thay bằng `--selftest` để người dùng tự kiểm tra với LLM thật.

---

## 3. Dòng thời gian các chặng

### 3.1 Chọn hướng (không có code)
Hỏi: đóng một agent thành `.exe`. So sánh: Python+PyInstaller (50–100 MB, antivirus báo nhầm), Go (một file tĩnh vài MB), Electron (~100 MB), bọc Hermes Agent (Python 3.11) hoặc OpenClaw (Node, có web UI sẵn).
- Đã thử đóng **Hermes Agent 0.19.0** bằng PyInstaller trên Linux: được (70 MB), có dashboard web, nhưng phải thêm `--collect-all openai --copy-metadata …`; tài liệu chính thức nói Windows native còn "experimental"; PyInstaller không build chéo.
- **Quyết định:** tự viết bằng Go (build chéo từ Linux ra Windows, một file, kiểm soát quyền). Bảng so sánh các framework khác nằm ở `HUONG_DAN_TAO_BOT.md` mục 10.

### 3.2 Lõi + UI đầu tiên
Server HTTP nội bộ + UI nhúng bằng `go:embed`; 6 công cụ (list/read/write/edit/search/run); thẻ xin duyệt; sandbox đường dẫn; token + kiểm tra Host.

### 3.3 Chế độ offline (bot if/else)
Người dùng chưa có model → thêm `offline.go`: phân tích lệnh (`ls`, `đọc`, `tìm`, `tạo`, `sửa`, `chạy`…). Mặc định khi `base_url` trống. Cùng bộ công cụ & cùng cổng duyệt với chế độ LLM.

### 3.4 Giao diện kiểu ChatGPT/Gemini
Màn hình chào, thẻ gợi ý, ô nhập dạng viên thuốc, menu `/`, khung code có nút sao chép, chế độ sáng/tối, responsive.

### 3.5 Lỗi treo & "bot đang bận"
Xem mục 4 (lỗi #1–#3). Sinh ra: `WaitDelay`, giết cây tiến trình, `/api/cancel`, nhãn trạng thái thẻ lệnh, giới hạn thời gian hiển thị `đang chạy… 12s / tối đa 60s`, `-t N`.

### 3.6 Tìm kiếm
`search_files` (nội dung) + `find_files` (tên). `tìm x` gọi cả hai.

### 3.7 Agent LLM thật
Giao thức khối ```` ```tool ```` (chạy với mọi LLM), parse khoan dung, tham số khoan dung, cắt ngữ cảnh 16k, thử lại khi tràn context, chặn lặp, giới hạn 15 bước.

### 3.8 Đọc tài liệu, task định kỳ, web
`docs.go`, `tasks.go`, `web.go`. Đổi tên "job" → "task" (giữ `job` làm gõ tắt, nạp được `jobs.json` cũ).

### 3.9 Chịu lỗi tên tệp
`fuzzy.go`: bỏ dấu + Levenshtein; `đọc` tự mở khi chỉ có một ứng viên rõ ràng, còn nhập nhằng thì liệt kê, không đoán. `search_files` tìm trong tài liệu.

### 3.10 Linux
Build tĩnh `CGO_ENABLED=0`; gói `.tar.gz`; `desktop-agent.service`; `access_token` cố định cho chế độ dịch vụ.

### 3.12 Streaming câu trả lời
`callLLM` đọc SSE, `streamFilter` ẩn khối công cụ thô, sự kiện `delta` + `assistant` cuối thay bong bóng, Dừng giữ phần đã viết. Chi tiết: `HUONG_DAN_TAO_BOT.md` mục 4b.6.

### 3.13 Chạy ngầm, bấm lần hai, thoát
Người dùng hỏi "đóng trình duyệt có tắt không, bấm lần hai có ra cổng khác không". Kiểm tra lại thì **có lỗi thật**: cổng bận thì tự chọn cổng khác và chạy **bản thứ hai** (chung `workspace.json`, **task chạy hai lần**). Sửa: một bản duy nhất, tự tắt khi hết tab, nút Thoát, dòng báo "đang chạy nền". Chi tiết: mục 4b.7.

### 3.14 Logo con vịt
Theo yêu cầu: vịt cao su vàng trên nền nước. Dùng cho logo, avatar bot, favicon và icon exe (mục 6.5).

### 3.11 Nhiều project, nhiều phiên chat (chặng cuối)
Tách `App` (HTTP + workspace) khỏi `Server` (một phiên). Lưu `workspace.json` (project, phiên, log hiển thị, lịch sử LLM). Giao diện 3 cột: **tệp (trái) · chat (giữa) · dự án & phiên (phải)**. Viewer: PDF/ảnh xem **bản gốc** (blob + `<object>`), Word/Excel xem chữ trích xuất + nút "Mở bằng ứng dụng", tệp văn bản xem **toàn bộ** (không bị cắt trang như LLM).

---

## 4. Các lỗi thật đã gặp (đọc để khỏi lặp lại)

| # | Triệu chứng | Nguyên nhân gốc | Cách sửa | Test chặn lại |
|---|---|---|---|---|
| 1 | Lệnh `chạy notepad`, `ping -t`, `start x` **treo "đang chạy" mãi** | Tiến trình con thừa hưởng ống stdout; `cmd.Run()` chờ ống đóng | `cmd.WaitDelay = 2s`; trả `exec.ErrWaitDelay` thành "đã khởi chạy tiến trình nền" | `TestRunCommand` (sleep 20 &) |
| 2 | Bấm **Dừng** rồi chat tiếp báo *"agent is busy"* | Dừng chỉ ngắt kết nối trình duyệt, lệnh vẫn chạy → cờ `busy` còn | Hủy `context` + `cmd.Cancel` giết **cả cây tiến trình** (`taskkill /T /F` / `kill(-pgid)`); `/api/cancel` | `TestRunCommand` (cancel), `TestCancelIsPerSession` |
| 3 | Lệnh có dấu nháy kép lỗi, tiếng Việt vỡ chữ trên Windows | Go đóng gói tham số theo `CommandLineToArgvW`, `cmd.exe` không hiểu; bảng mã OEM | `SysProcAttr.CmdLine = cmd.exe /S /C "chcp 65001>nul & …"` | (chỉ kiểm được trên Windows thật) |
| 4 | `tạo thư mục mới/a.txt : x` tạo sai chỗ | Bước "bỏ từ đệm" cắt chữ *"thư mục"* ở **mọi** lệnh | Chỉ bỏ từ đệm ở lệnh có đối số đầu là đường dẫn (`ls`, `xem`, `xóa`…) | `TestOfflineCommands` |
| 5 | `thêm` vào tệp không kết thúc bằng xuống dòng → chữ dính nhau | `append` không kiểm tra byte cuối | Đọc byte cuối, thêm `\n` nếu thiếu | `TestOfflineCommands` |
| 6 | `đọc abd_notes.txt` báo `stat …: no such file`; `tìm tên abd` không ra | Không có so khớp gần đúng | `fuzzy.go` + lỗi kèm "tệp gần giống" | `fuzzy_test.go` |
| 7 | `tìm nội dung Ánh` không thấy dù có trong .docx/.pdf | `search_files` bỏ qua tệp nhị phân | Gọi `readDocument` cho đuôi tài liệu; khớp không dấu | `TestSearchInsideDocuments` |
| 8 | **PDF tiếng Việt từ LibreOffice vỡ chữ** (`Hạng mục`→`Hng mcạụ`, `Nguyê�n`) | `GetPlainText/GetTextByRow` của thư viện xếp sai ký tự dấu; ToUnicode ghép dấu bị giải mã thành U+FFFD | Tự ghép từ ký tự thô (`Content().Text`) theo Y rồi X (`pageText2`); phần ToUnicode còn sai → giữ `�` và **thêm cảnh báo** | `TestReadPDF` (còn giới hạn, xem 10) |
| 9 | Hộp chọn thư mục báo lỗi JS khi thư mục không có thư mục con | Go trả `null` cho slice rỗng | `[]entry{}` không nil + JS phòng thủ `|| []` | `TestEmptyListsAreArraysNotNull` |
| 10 | API mới: `project_id` bị bỏ qua, mọi chat trả 400 | Go khớp tên JSON không bỏ dấu `_` (`ProjectID` ≠ `project_id`) | Khai báo `json:"project_id"` cho mọi struct yêu cầu | `TestProjectsAndSessions` |
| 11 | Test treo 10 phút | Goroutine giả lập "bấm Từ chối" chạy xong trước khi thẻ duyệt xuất hiện | `denyAll()` thăm dò lặp đến khi dừng | `agent_test.go`, `offline_test.go` |
| 12 | Viewer tệp dài hiện cả chuỗi `[... còn N ký tự … offset=…]` | Viewer dùng `read_file` (đã phân trang cho LLM) | `/api/file` trả **toàn bộ** (≤2 MB) | `TestFileViewerEndpoints` |
| 13 | Khung PDF trắng | Chromium headless không có bộ xem PDF | `<object>` + nội dung dự phòng (mở tab mới / mở bằng ứng dụng / xem chữ) | kiểm bằng Playwright |
| 15 | Bấm exe lần hai → **chạy bản thứ hai** ở cổng khác; task định kỳ chạy hai lần, hai bản ghi đè `workspace.json` | `net.Listen` thất bại thì tự chọn cổng ngẫu nhiên, không kiểm tra bản đang chạy | `instance.json` + thăm dò token; chờ 3 giây trước khi chọn cổng khác; bản thứ hai chỉ mở tab rồi thoát | `TestSingleInstanceDiscovery`, `TestSecondLaunchOpensProject…`; kiểm tay: 3 bản khởi động cùng lúc chỉ còn 1 |
| 16 | Bấm Dừng giữa lúc stream: giữ chữ đã viết nhưng **mất dòng "(đã dừng)"** | Giao diện ngắt kết nối ngay nên không nhận được sự kiện cuối | Giao diện chỉ gọi `/api/cancel` và để máy chủ tự đóng luồng (ngắt cưỡng chế sau 3 giây) | `TestAgentStreamStopKeepsPartialText` + kiểm bằng Playwright (58 ms) |
| 17 | Lần chạy `go run` công cụ bên thứ ba để nhúng icon bị hệ thống chặn | Tải và chạy mã lạ | Tự viết `assets/make_syso.py` (không tải gì) và kiểm bằng `pefile` | — |
| 14 | PDF thật có ký tự thừa cuối dòng (`…đồngă`) | `GetPlainText` | Thay bằng `pageText2` | `TestReadPDF` (so khớp dòng chính xác) |

---

## 5. Quyết định thiết kế quan trọng (và lý do)

1. **Go, một binary tĩnh, UI nhúng.** Build chéo Windows/Linux chỉ bằng biến môi trường; không cần runtime; ~8 MB.
2. **Giao diện = trang web trên `127.0.0.1`**, không Electron. Nhẹ; trình duyệt có sẵn. Đánh đổi: chưa có cửa sổ riêng/tray (backlog).
3. **Chỉ nghe `127.0.0.1` + token ngẫu nhiên + kiểm tra Host** cho mọi `/api`. Chặn trang web khác và DNS-rebinding. Chế độ dịch vụ có `access_token` cố định.
4. **Một cổng duyệt duy nhất** (`execTool` + `needsApproval`). Thêm công cụ ghi/chạy mới = thêm vào `needsApproval`, tự có thẻ duyệt. `fetch_url` cũng cần duyệt (chống rò dữ liệu qua URL).
5. **Sandbox đường dẫn** (`resolve`): chặn `../`, đường dẫn tuyệt đối, symlink thoát ra, cả khi tệp chưa tồn tại.
6. **Giao thức công cụ bằng văn bản** làm mặc định vì không biết LLM của người dùng có function-calling không; `native`/`auto` là tùy chọn.
7. **Ngân sách token bi quan** (ước lượng thừa): thà gửi ít còn hơn lỗi tràn context; có thử lại khi vẫn tràn.
8. **Parse khoan dung ở mọi nơi LLM chạm vào** (khối tool, khóa JSON, kiểu số, bí danh tên công cụ), nhưng **nghiêm ngặt ở nơi quyết định hành động** (chỉ chạy tên công cụ có thật; JSON thường không bị coi là lệnh).
9. **Không đoán khi nhập nhằng** (tên tệp na ná → liệt kê, không tự mở).
10. **Task chỉ báo khi trạng thái đổi** (ổn→lỗi, lỗi→ổn) để khỏi spam; không chạy bù khi mở lại.
11b. **Một bản duy nhất** + **tự tắt khi hết tab** (trừ khi có việc/task bật): chương trình không âm thầm nằm lại, nhưng không phá việc chạy nền mà người dùng chủ động bật.
11. **Phiên chat = runner riêng** (`Server`): chạy song song nhiều phiên, `busy`/`cancel` theo phiên, thẻ duyệt tra theo ID toàn cục.
12. **Lưu log hiển thị riêng với lịch sử LLM.** Log (để vẽ lại UI) có thể cắt/rút gọn; lịch sử LLM giữ `Meta` ("result") qua `pMsg` để khôi phục đúng lượt.
13. **PDF/ảnh phục vụ qua `blob:` URL lấy bằng `fetch` + header token** (không đưa token lên URL); chỉ cho phép danh sách đuôi an toàn (không SVG vì có thể chứa script); `nosniff`.
14. **`/api/open` (mở bằng ứng dụng mặc định) chỉ cho tài liệu/ảnh**, không bao giờ `.exe/.bat/.ps1`.

---

## 6. Cách xuất ra các định dạng

> Mọi lệnh chạy từ thư mục `desktop-agent/`. Cần Go ≥ 1.24 (thư viện PDF đã vendor → **không cần mạng** để build). Python 3 để `build.sh` nén zip.

### 6.1 Cả ba gói phát hành (khuyên dùng)
```bash
./build.sh            # Windows + Linux amd64 + Linux arm64
./build.sh windows    # chỉ Windows       ./build.sh linux   # chỉ Linux
```
Sinh ra (ở thư mục gốc `desktop-agent/`, đồng thời trong `dist/`):

| Gói | Nội dung |
|---|---|
| `desktop-agent-win.zip` | `desktop-agent.exe`, `config.json`, 3 file hướng dẫn |
| `desktop-agent-linux-amd64.tar.gz` | thư mục `desktop-agent/` gồm binary, `config.json`, 3 file hướng dẫn, `desktop-agent.service` |
| `desktop-agent-linux-arm64.tar.gz` | như trên cho ARM64 |

`dist/` và các gói đã nằm trong `.gitignore` (là sản phẩm build, không commit).

### 6.2 Windows `.exe` thủ công
```bash
# từ Linux/macOS (build chéo)
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -H=windowsgui" -o desktop-agent.exe .
# trên Windows (PowerShell)
$env:CGO_ENABLED="0"; go build -trimpath -ldflags "-s -w -H=windowsgui" -o desktop-agent.exe .
```

| Cờ | Ý nghĩa |
|---|---|
| `CGO_ENABLED=0` | không phụ thuộc thư viện C → file tĩnh |
| `-ldflags "-s -w"` | bỏ ký hiệu/debug → nhỏ hơn ~30% |
| `-H=windowsgui` | **không hiện cửa sổ console** khi bấm đúp. Bỏ cờ này khi debug để thấy log. (Bản GUI không có stdout → `--selftest` ghi thêm `selftest-report.txt`) |
| `-trimpath` | bỏ đường dẫn máy build khỏi binary |
| `GOARCH=arm64` | Windows ARM |

### 6.3 Linux
```bash
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o desktop-agent .    # arm64: GOARCH=arm64
ldd desktop-agent            # → "not a dynamic executable" (đúng ý)
```
Dịch vụ nền: xem `desktop-agent.service` (đầu file có hướng dẫn cài `systemctl --user enable --now …`, `loginctl enable-linger`). Truy cập từ xa qua `ssh -L 7878:127.0.0.1:7878 user@server`; đặt `access_token` cố định trong `config.json`.

### 6.4 macOS (chưa kiểm thử)
`GOOS=darwin GOARCH=arm64 go build …` (hoặc `amd64`) build được vì code có nhánh `open`; **chưa chạy thử trên máy Mac** nên không phát hành. Nếu cần: thêm vào `build.sh`, ký bằng `codesign` + notarize.

### 6.5 Icon cho `.exe` (ĐÃ LÀM) và phiên bản (chưa)

Icon con vịt đã được nhúng: `resource_windows_amd64.syso` nằm trong thư mục gói nên `go build` cho Windows tự liên kết (xem `HUONG_DAN_TAO_BOT.md` mục 5.2c). Tạo lại ảnh khi đổi logo:
```bash
# 1) vẽ PNG trong suốt các cỡ 16,24,32,48,64,128,256 từ assets/duck.svg (đã dùng Playwright: đặt SVG vào trang, screenshot với omitBackground)
# 2) gói ICO bằng Pillow:
python3 -c "from PIL import Image; ims=[Image.open(f'duck-{n}.png').convert('RGBA') for n in (16,24,32,48,64,128,256)]; ims[-1].save('assets/app.ico', format='ICO', sizes=[(n,n) for n in (16,24,32,48,64,128,256)], append_images=ims[:-1])"
# 3) nhúng:
python3 assets/make_syso.py
# 4) kiểm: pip install pefile; mở exe, duyệt DIRECTORY_ENTRY_RESOURCE, so các RT_ICON với app.ico
```
Phần **thông tin phiên bản/công ty** (ProductName, FileVersion) vẫn dùng `goversioninfo` như bên dưới (chưa làm).

#### (Chưa làm) Tên, phiên bản cho `.exe`
```bash
go install github.com/josephspurrier/goversioninfo/cmd/goversioninfo@latest
```
Tạo `versioninfo.json` (`ProductName`, `CompanyName`, `FileVersion`, `IconPath: "app.ico"`), thêm `//go:generate goversioninfo -64` vào `main.go`, chạy `go generate` → sinh `resource_windows_amd64.syso` (Go tự link). Có icon + thông tin phiên bản giúp giảm bị Defender nghi ngờ. *(Chưa làm trong dự án — backlog.)*

### 6.6 Ký số
```powershell
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /f cert.pfx /p MATKHAU desktop-agent.exe
signtool verify /pa /v desktop-agent.exe
```
Trong công ty: chứng chỉ từ CA nội bộ + đẩy chứng chỉ gốc bằng GPO/Intune. Phát hành ngoài: OV/EV code-signing hoặc Azure Trusted Signing. Luôn **timestamp** (`/tr`).

### 6.7 Mã băm để người dùng kiểm tra
```bash
sha256sum desktop-agent-win.zip desktop-agent-linux-*.tar.gz > SHA256SUMS.txt          # Linux/macOS
certutil -hashfile desktop-agent.exe SHA256                                              # Windows
```

### 6.8 Trình cài đặt Windows (Inno Setup) — chưa thử
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
Name: "{userstartup}\Tro Ly";  Filename: "{app}\desktop-agent.exe"; Parameters: "--no-browser"   ; tùy chọn: chạy khi đăng nhập
```
MSI (WiX) cũng được nếu IT yêu cầu triển khai bằng SCCM/Intune.

### 6.9 CI/CD với GitHub Actions — chưa chạy thử
```yaml
name: release
on: { push: { tags: ['v*'] } }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: '1.24' }
      - run: cd desktop-agent && go vet ./... && go test -race ./...
      - run: cd desktop-agent && ./build.sh && sha256sum *.zip *.tar.gz > SHA256SUMS.txt
      - uses: softprops/action-gh-release@v2
        with:
          files: |
            desktop-agent/desktop-agent-win.zip
            desktop-agent/desktop-agent-linux-*.tar.gz
            desktop-agent/SHA256SUMS.txt
```
(Các test chạy lệnh shell `sleep` chỉ chạy đầy đủ trên Linux; ký số Windows cần thêm bước `signtool` trên runner `windows-latest`.)

### 6.10 Xuất tài liệu sang HTML / Word / PDF
```bash
pip install markdown                      # (hoặc cài pandoc: script tự dùng nếu có)
docs/export-docs.sh dist/docs             # ra HUONG_DAN*.html/.docx/.pdf + NHAT_KY_PHAT_TRIEN.*
```
Cách làm: Markdown → HTML (thư viện `markdown`, hoặc `pandoc`) → LibreOffice headless `--convert-to docx|pdf`. Đã chạy thử trong môi trường dev: ra đủ 3 định dạng.

### 6.11 Tạo lại dữ liệu mẫu cho test
```bash
pip install reportlab python-docx openpyxl      # cần LibreOffice + font DejaVu cho hd_lo.pdf
cd testdata && python3 make_fixtures.py
```

### 6.12 Kích thước & mẹo
Không dùng `upx` (nén exe) — antivirus rất hay báo nhầm. Kích thước hiện tại ≈ 8 MB (Windows), ≈ 7.7 MB (Linux amd64).

---

## 7. Cách thêm tính năng (checklist)

**Thêm một công cụ cho agent**
1. `tools.go`: thêm mô tả vào `toolDefs` (`fn("tên", "mô tả", props{…}, "tham_số_bắt_buộc")`) và một `case` trong `runTool`. Đọc tham số bằng `a.str("…")`, `a.num("…")`, `a.boolean("…", mặc_định)` (tự chịu bí danh + kiểu sai).
2. Nếu **ghi/xóa/chạy/gửi dữ liệu ra ngoài** → thêm tên vào `needsApproval`.
3. Mọi đường dẫn đi qua `resolve(root, p)`; thiếu tệp → trả `notFoundMsg(ctx, root, p)` để có gợi ý.
4. Kết quả dài → dùng `window()` (phân trang) hoặc `clip()`.
5. (Tùy chọn) luật if/else trong `offline.go` + dòng trong `offlineHelp` + mảng `CMDS` trong `web/index.html` (menu `/`).
6. Test trong `tools_test.go` (+ `offline_test.go`); thêm 1 dòng `b.expect("câu gõ", "chữ mong đợi")`.

**Thêm một lệnh if/else:** `offline.go` → thêm `case verb == "…"` trong `runOffline`; dùng `call(tên, tham_số, hiện_output)`.

**Thêm một API + UI:** `server.go` (`api("/api/…", handler)` — mọi `/api` tự có token/Host guard) → struct yêu cầu **bắt buộc có `json:"snake_case"`** → `web/index.html` dùng `api()`/`post()`. Mảng rỗng trả `[]` không `nil`.

**Thêm kiểu tài liệu đọc được:** `docs.go` → thêm đuôi vào `isDocExt` + hàm `readXXX`; test với tệp mẫu trong `testdata/`.

**Thêm ngôn ngữ cho task dạng script:** `scriptCommand` trong `tasks.go`.

**Thêm nguồn tìm web:** `web.go` → hàm `searchXXX` + nhánh trong `webSearch` + khóa config trong `config.go`/`main.go`.

**Đổi giao thức LLM:** `agent.go` (`systemPrompt`, `parseToolText`, `runAgent`). Chạy `TestAgentToolSyntaxVariants` + `--selftest`.

---

## 8. Cấu hình, dữ liệu, API

### 8.1 `config.json` (cạnh exe; biến môi trường ghi đè một số khóa)
| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `base_url`, `api_key`, `model` | rỗng | LLM kiểu OpenAI. **Rỗng = bot if/else** |
| `context_tokens` / `max_output_tokens` | 16000 / 1500 | ngân sách ngữ cảnh |
| `tool_mode` | `text` | `text` \| `native` \| `auto` |
| `port` | 7878 | bận → cổng ngẫu nhiên |
| `auto_approve` | false | bỏ qua thẻ duyệt |
| `command_timeout_sec` | 60 (tối đa 3600) | giới hạn lệnh `chạy` |
| `search_provider`, `google_api_key`, `google_cx`, `searxng_url` | duckduckgo | tìm web |
| `alert_webhook` | rỗng | POST `{"text","content"}` khi task báo động |
| `access_token` | rỗng | token cố định (≥16 ký tự) cho dịch vụ |
| `stream` | true | câu trả lời LLM hiện dần |
| `auto_exit_seconds` | 180 | tự tắt sau ngần này giây không còn tab (0 = không); không áp dụng khi đang có việc/task bật hoặc `--no-browser` |

Biến môi trường: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `AGENT_PORT`.

### 8.2 Dữ liệu người dùng
`%AppData%\desktop-agent\` (Windows) hoặc `~/.config/desktop-agent/` (Linux): `workspace.json` (project + phiên + log + lịch sử LLM), `instance.json` (địa chỉ + token của bản đang chạy, 0600), `tasks.json`, `agent.log`, `jobs/` (mã script của task). Ghi bằng tệp tạm + `rename` (không hỏng nửa chừng). Tệp hỏng → nạp rỗng, không sập.

`workspace.json` (rút gọn):
```json
{"projects":[{"id":"a1b2c3d4","path":"D:\\duan","name":"duan",
  "sessions":[{"id":"…","title":"…","created":"…","updated":"…",
    "log":[{"t":"user","text":"…"},{"t":"tool","id":"…","name":"run_command","args":"…","result":"…","done":true},{"t":"assistant","text":"…"}],
    "history":[{"role":"user","content":"…","meta":"result"}]}]}],
 "active_project":"…","active_session":"…"}
```
Giới hạn: log ≤ 300 mục, lịch sử LLM lưu ≤ 60 tin (mỗi tin ≤ 4000 ký tự), ≤ 60 phiên/project.

### 8.3 API (mọi đường dẫn cần header `X-Token`, và `Host` đúng)
| Phương thức · đường dẫn | Việc |
|---|---|
| `GET /api/state` | chế độ (offline/LLM), model, tự duyệt, `auto_exit_seconds` |
| `POST /api/quit` | tắt hẳn chương trình (hủy lệnh đang chạy, lưu dữ liệu, xóa `instance.json`) |
| `GET /api/workspace` · `POST /api/workspace/active` | danh sách project/phiên · đặt phiên đang mở |
| `POST /api/project/open` · `/remove` | mở (dùng lại nếu trùng đường dẫn) · bỏ project |
| `GET /api/session?project_id&id` | log để vẽ lại + thẻ duyệt đang chờ + `busy` |
| `POST /api/session/new` · `/delete` · `/rename` | phiên mới (dùng lại phiên trống) · xóa · đổi tên |
| `POST /api/chat` | gửi tin; trả **SSE** (`delta` = từng đoạn chữ đang viết, `assistant` = câu cuối sạch, `tool_call`, `tool_result`, `approval`, `error`, `done`); 409 nếu phiên đang bận |
| `POST /api/approve` · `/api/cancel` · `/api/auto` | duyệt · dừng một phiên · bật tự duyệt |
| `GET /api/browse` | chọn thư mục (chỉ thư mục con; có danh sách ổ đĩa trên Windows) |
| `GET /api/tree` | cây tệp (qua sandbox) |
| `GET /api/file` | **toàn bộ** chữ của tệp (≤2 MB; tài liệu thì chữ trích xuất) |
| `GET /api/raw` | tệp gốc, chỉ PDF/ảnh (hỗ trợ Range) |
| `POST /api/open` | mở bằng ứng dụng mặc định (chỉ tài liệu/ảnh) |
| `GET /api/tasks` · `POST /api/task` | danh sách task + cảnh báo · chạy/tạm dừng/xóa |

---

## 9. Kiểm thử

### 9.1 Chạy
```bash
go vet ./... && GOOS=windows go vet ./...      # biên dịch cả hai hệ điều hành
go test -race -timeout 200s ./...              # ~45 giây, 77 test
```

### 9.2 Có gì
`tools_test` (sandbox, công cụ, chạy lệnh, hết giờ, tiến trình nền, hủy) · `offline_test` (hơn 100 câu lệnh) · `fuzzy_test` · `docs_test` · `tasks_test` · `web_test` (server giả) · `agent_test` (LLM giả 3 chế độ, 8 kiểu cú pháp, tràn context 16k, thử lại) · `agentic_test` (LLM giả **phản ứng theo kết quả thật**: viết code lỗi → chạy → thấy lỗi → sửa → chạy lại) · `workspace_test` (HTTP end-to-end: guard, nhiều project/phiên cô lập, chạy song song, hủy theo phiên, duyệt qua HTTP, lưu/khôi phục sau khởi động lại, viewer, `/api/raw`, `/api/open`, `[]` không `null`).

### 9.3 Mẫu viết test cho agent
`newFakeLLM(t, func(n int, req map[string]any) (status int, body any))` dựng server `/chat/completions` giả; `s.drive("yêu cầu")` chạy vòng lặp và trả các sự kiện; `solver(buggyFirst)` là mô hình giả phản ứng theo kết quả công cụ. Với HTTP: `newTestApp(t, "")` + `a.chat(pid, sid, "…", onEvent)` đọc SSE và cho phép "bấm duyệt" trong callback.

### 9.4 Kiểm tra giao diện (Playwright, Chromium headless)
Mẫu đã dùng: chạy binary thật với `XDG_CONFIG_HOME` trỏ thư mục tạm (workspace sạch), `node script.js` dùng `require('playwright')`; thao tác `fill/click/waitForSelector`, in kết quả, `screenshot` rồi **mở ảnh xem**. Điểm cần nhớ:
- Chromium headless **không hiển thị PDF** (khung trắng) → chỉ kiểm được `<object>` có `data=blob:` và `type=application/pdf`, blob đúng MIME/kích thước. **Chưa xem được PDF render thật trong Chrome/Edge.**
- Test song song nhiều phiên: dùng `chạy sleep N` (bật tự duyệt) để có thao tác dài.

### 9.5 `--selftest` với LLM thật
```bash
desktop-agent --selftest        # ghi cả selftest-report.txt
```
Ba mục tiêu: mở tệp theo tên gần đúng và trả lời; tìm TODO + ghi báo cáo; viết `tong.py` rồi chạy và báo kết quả. In từng bước + ĐẠT/KHÔNG ĐẠT; thoát với mã ≠ 0 nếu có mục trượt. **Đây là cách duy nhất để biết LLM thật của bạn điều khiển được vòng lặp.**

---

## 10. Những gì CHƯA được kiểm chứng (nói thẳng)

| Hạng mục | Tình trạng |
|---|---|
| Chạy `.exe` trên **Windows thật** (chọn ổ đĩa, `cmd`, `taskkill`, `chcp 65001`, dấu nháy kép, Defender/SmartScreen) | Chưa. Chỉ build chéo + `go vet` cho Windows. Toàn bộ test chạy trên Linux |
| **LLM thật** của bạn | Chưa. Chỉ có LLM giả; dùng `--selftest` |
| **Tìm web thật** (DuckDuckGo/Google) | Chưa (sandbox không ra internet). Parser DuckDuckGo kiểm với mẫu HTML tự viết; Google CSE dùng API chính thức |
| **PDF tiếng Việt từ Word** | Chưa. PDF từ ReportLab: chính xác; từ LibreOffice (Carlito/Caladea): đúng thứ tự, **một số dấu thanh ghép (`ễ`, `ề`, `ả`) vẫn sai** (bot cảnh báo) |
| Xem PDF **render thật** trong Chrome/Edge | Chưa (headless không có). `<object>` + blob đúng, có phương án dự phòng |
| Linux **systemd thật**, Linux arm64 | Chưa. amd64 đã chạy thử trong môi trường sạch |
| macOS | Chưa |
| GitHub Actions workflow, Inno Setup, goversioninfo, signtool | Có hướng dẫn, **chưa chạy** |
| Task định kỳ chạy **nhiều ngày** liên tục, đổi giờ hệ thống | Chưa; có test tick/lịch bằng đồng hồ giả |

---

## 11. Hạn chế đã biết & việc nên làm tiếp (theo ưu tiên)

1. **Chạy thử trên Windows thật** và sửa lỗi phát sinh (mục 10) — việc đầu tiên.
2. **Chạy `--selftest` với LLM thật**; chỉnh `systemPrompt`/`parseToolText` theo cách model đó viết khối tool.
3. **PDF tiếng Việt chính xác:** tự viết bộ phân tích CMap/ToUnicode (xử lý chuỗi nhiều code point), hoặc dùng MuPDF/pdfium (cần cgo/DLL), hoặc OCR cho bản scan.
4. **Ký số + icon/version** (6.5, 6.6) và **trình cài đặt** (6.8).
5. **Cửa sổ riêng thay vì trình duyệt** (WebView2 trên Windows) + **icon khay hệ thống**; thông báo **toast Windows** cho task.
6. **Xác thực với dịch vụ công ty:** OIDC Authorization Code + PKCE (loopback redirect) để lấy token gọi LLM/API; lưu bằng Windows Credential Manager/DPAPI, tự refresh (đã bàn, chưa làm).
7. **Streaming câu trả lời** (hiện trả cả khối) và **render markdown đầy đủ** (bảng, tiêu đề, liên kết).
8. **Task:** chạy bù khi mở lại (tùy chọn), lịch theo thứ trong tuần/cron, chạy như dịch vụ Windows để không cần mở giao diện.
9. **OCR** (Tesseract) cho PDF scan; đọc `.doc/.xls/.ppt` cũ; **ghi** được `.docx/.xlsx`.
10. **Tự cập nhật** (kiểm tra phiên bản + hash trước khi thay file).
11. Tìm nội dung dùng **chỉ mục** (hiện quét tuần tự, giới hạn 15 giây/300 tài liệu) cho dự án rất lớn.
12. Xóa tệp hiện **không vào Thùng rác** (xóa là mất) — cân nhắc.

---

## 12. Checklist bảo mật trước khi phát hành

- [ ] Chỉ `127.0.0.1`; không đổi thành `0.0.0.0` khi chưa có đăng nhập thật.
- [ ] Giữ `guard` (token + Host) cho **mọi** `/api`.
- [ ] Mọi đường dẫn đi qua `resolve()`; không bao giờ `os.ReadFile(đường_dẫn_người_dùng)` trực tiếp.
- [ ] Công cụ mới có hậu quả → `needsApproval`.
- [ ] Không nhúng khóa/secret vào exe; `access_token` và `api_key` coi như mật khẩu.
- [ ] Chạy bằng quyền người dùng thường, không Administrator.
- [ ] Nội dung đọc từ web/tệp là **không tin cậy** (prompt injection): giữ duyệt thủ công cho `chạy`, `xóa`, `fetch_url`.
- [ ] `/api/raw`: chỉ PDF/ảnh, không SVG; `/api/open`: chỉ tài liệu/ảnh.
- [ ] Ký số exe, công bố SHA-256.

---

## 13. Phụ lục

### Lệnh hay dùng khi phát triển
```bash
go run . --project ~/thu-muc --no-browser           # chạy nhanh, in dòng "Open: http://127.0.0.1:7878/?t=…"
go build -o /tmp/da . && XDG_CONFIG_HOME=/tmp/cfg AGENT_PORT=7899 /tmp/da --no-browser --project /tmp/p   # dữ liệu tách biệt
curl -N -H "X-Token: $T" -X POST -d '{"project_id":"…","session_id":"…","message":"ls"}' http://127.0.0.1:7899/api/chat
go test -run TestAgent -v ./...                     # chạy một nhóm test
```

### Gợi ý để tiếp tục làm việc với một AI trợ lý
Đưa cho AI: (1) file này, (2) `git log --oneline` của `desktop-agent/`, (3) kết quả `go test ./...`. Yêu cầu nó **đọc test trước khi sửa code**, giữ nguyên quy trình ở mục 2 (vet → test → chạy binary thật → chụp UI → đóng gói), và **nói rõ mục nào chưa kiểm chứng** thay vì tuyên bố thành công.
