# Trợ lý Desktop – Hướng dẫn sử dụng

Bot chạy **một file exe duy nhất**, không cần cài đặt. Có hai chế độ:

| Chế độ | Khi nào | Cách dùng |
|---|---|---|
| **Bot if/else (offline)** – mặc định | `base_url` trong `config.json` để trống | Gõ lệnh ngắn: `ls`, `đọc`, `tìm`, `chạy`, `task`… (mục 3) |
| **Trợ lý LLM** | Điền `base_url` + `model` của LLM nội bộ | Nói chuyện bình thường, trợ lý tự gọi công cụ thật: đọc/sửa tệp, **chạy lệnh/viết code rồi chạy**, đọc PDF/Word/Excel, tìm web, **tạo task định kỳ** (mục 8–11) |

Cả hai chế độ dùng chung bộ công cụ, cùng thẻ xin duyệt và cùng bảng Tasks.

---

## 1. Chạy lần đầu (1 phút)

1. Giải nén file zip ra một thư mục bất kỳ (ví dụ `D:\TroLy\`).
2. Bấm đúp **`desktop-agent.exe`**.
   - Không có cửa sổ đen nào hiện ra. Trình duyệt mặc định sẽ tự mở trang `http://127.0.0.1:7878`.
3. Nếu Windows hiện cảnh báo xanh **"Windows protected your PC"** (do exe chưa ký số):
   bấm **More info** → **Run anyway**.
4. Bấm **"Chọn thư mục dự án để bắt đầu"** → duyệt tới thư mục muốn làm việc → **Chọn thư mục này**.
5. Gõ lệnh vào ô chat, ví dụ `ls`.

> Muốn tắt bot: mở Task Manager → kết thúc tiến trình `desktop-agent.exe`.
> Đóng tab trình duyệt **không** tắt bot; mở lại exe để lấy lại trang chat.

---

## 2. Giao diện

| Khu vực | Chức năng |
|---|---|
| **Thanh bên trái** | *Cuộc trò chuyện mới*, thư mục **Dự án** đang chọn (bấm để đổi), cây **Tệp** (bấm tệp để xem nội dung), công tắc **Tự duyệt thao tác**, nút **sáng/tối** |
| **Màn hình chào** | 4 thẻ gợi ý – bấm để điền sẵn lệnh |
| **Khung chat** | Câu bạn gõ ở bên phải, câu trả lời của bot bên trái. Kết quả dạng văn bản nằm trong khung code có nút **Sao chép** |
| **Thẻ công cụ** (dòng nhỏ có biểu tượng cờ-lê) | Cho biết bot vừa chạy thao tác gì; bấm để xem chi tiết |
| **Thẻ xin duyệt** (nền vàng) | Hiện trước mọi thao tác làm thay đổi dữ liệu |
| **Ô nhập** | `Enter` gửi · `Shift+Enter` xuống dòng · gõ `/` để mở danh sách lệnh · nút vuông = dừng |

Mẹo: gõ `/` rồi dùng phím ↑ ↓ và `Tab`/`Enter` để chọn lệnh, bot sẽ điền sẵn mẫu cú pháp.

---

## 3. Danh sách lệnh

Lệnh gõ **tiếng Việt hoặc tiếng Anh** đều được. Không phân biệt hoa/thường ở từ khóa đầu.
Đường dẫn luôn **tương đối với thư mục dự án** (ví dụ `src/main.py`).

### Lệnh chỉ đọc (chạy ngay, không hỏi)

| Gõ | Tác dụng | Ví dụ |
|---|---|---|
| `help` | Xem danh sách lệnh | `help` |
| `project` | Cho biết thư mục dự án hiện tại | `project` |
| `ls [thư mục]` (hoặc `dir`, `liệt kê`) | Liệt kê tệp, thư mục | `ls`, `ls src` |
| `tree [thư mục]` | Cây thư mục sâu tối đa 3 cấp | `tree` |
| `đọc <tệp>` (hoặc `read`, `cat`, `xem`, `mở`) | Xem nội dung tệp văn bản | `đọc README.md` |
| `tìm <từ khóa>` (hoặc `find`, `grep`) | Tìm **cả tên tệp lẫn nội dung** trong toàn bộ dự án, **kể cả chữ bên trong PDF/Word/Excel/PowerPoint**. Kết quả chia 2 phần: *Tên tệp khớp* và *Nội dung khớp* (`tệp:dòng: nội dung`) | `tìm hoadon`, `tìm TODO` |
| `tìm tên <mẫu>` (hoặc `tìm file`) | **Chỉ theo tên** tệp/thư mục. Không phân biệt hoa/thường, **không cần gõ dấu**, dùng được `*` `?`. Nếu không tên nào chứa đúng chữ bạn gõ, bot liệt kê **tên gần giống** (gõ sai 1–2 chữ cái cũng ra) | `tìm tên báo cáo`, `tìm tên bao cao`, `tìm tên *.pdf` |
| `tìm nội dung <regex>` (hoặc `tìm chữ`) | **Chỉ theo nội dung** bên trong tệp (văn bản + PDF/Word/Excel/PowerPoint). Không phân biệt hoa/thường và **khớp cả khi bạn gõ không dấu** (`nguyen van anh` tìm ra `Nguyễn Văn Ánh`) | `tìm nội dung password`, `tìm nội dung TODO\|FIXME` |

### Lệnh thay đổi dữ liệu (**luôn hỏi bạn duyệt trước**)

| Gõ | Tác dụng | Ví dụ |
|---|---|---|
| `tạo <tệp> : <nội dung>` (`write`) | Tạo mới hoặc **ghi đè** tệp; tự tạo thư mục cha | `tạo ghi_chu.txt : mua sữa` |
| `thêm <tệp> : <dòng>` (`append`) | Thêm 1 dòng vào cuối tệp | `thêm ghi_chu.txt : mua trứng` |
| `sửa <tệp> : <cũ> => <mới>` (`edit`) | Thay chữ. Chữ cũ phải **xuất hiện đúng 1 lần** | `sửa a.txt : world => agent` |
| `đổi tên <cũ> => <mới>` (`mv`, `rename`) | Đổi tên hoặc di chuyển trong dự án | `đổi tên a.txt => b/a.txt` |
| `xóa <tệp>` (`rm`, `delete`) | Xóa tệp hoặc **thư mục rỗng** | `xóa b.txt` |
| `chạy <lệnh>` (`run`, `cmd`) | Chạy lệnh terminal (Windows: `cmd`) tại thư mục dự án. **Tự dừng sau 60 giây** | `chạy dir`, `chạy git status` |
| `chạy -t <giây> <lệnh>` | Như trên nhưng đặt giới hạn thời gian riêng (tối đa 3600 giây) | `chạy -t 300 npm install` |

Ghi chú về `tìm`: tối đa 200 kết quả mỗi phần và dừng sau 15 giây nếu thư mục quá lớn (sẽ có dòng báo "kết quả chưa đầy đủ" – hãy tìm trong thư mục con: `tìm` không có tham số đường dẫn, nên chọn lại dự án là thư mục nhỏ hơn).

Cách nhớ cú pháp: **dấu `:`** ngăn *tên tệp* với *nội dung*, **dấu `=>`** ngăn *cũ* với *mới*.

### Giới hạn thời gian của lệnh `chạy`

- Mặc định **60 giây**. Trên thẻ công cụ bạn thấy đồng hồ đếm, ví dụ **`đang chạy… 12s / tối đa 60s`**.
- Hết giờ bot tự dừng lệnh (giết cả tiến trình con), giữ lại phần output đã có, và báo ⏱️ kèm gợi ý. Nhãn thẻ chuyển thành **hết giờ**.
- Cần lâu hơn một lần: `chạy -t 300 <lệnh>`. Muốn đổi mặc định: sửa `command_timeout_sec` trong `config.json`.
- Muốn dừng sớm: bấm nút **Dừng** ở ô nhập.
- Lệnh mở ứng dụng (`chạy notepad`) không bị chờ: bot trả về sau khoảng 2 giây và để ứng dụng chạy tiếp.
- Trình soạn thảo/lệnh tương tác (ví dụ `python` không tham số, `vim`) không dùng được vì bot không có bàn phím cho chúng; hãy chạy lệnh một dòng như `python script.py`.

### Duyệt thao tác

Với lệnh thay đổi dữ liệu, bot hiện thẻ vàng mô tả việc sắp làm:

- **Cho phép** – làm một lần.
- **Từ chối** – bỏ qua, không làm gì.
- **Luôn cho phép** – bật *Tự duyệt thao tác* cho đến khi tắt công tắc ở thanh bên hoặc khởi động lại.

---

### Đọc tài liệu và web trong chế độ lệnh

| Gõ | Tác dụng |
|---|---|
| `đọc bao_cao.pdf` / `.docx` / `.xlsx` / `.pptx` | Đọc chữ trong PDF, Word, Excel, PowerPoint (mục 9) |
| `đọc <tệp> ` rồi làm theo gợi ý `offset=` | Đọc tiếp phần sau của tệp dài |
| `đọc abd_notes` (gõ sai/không dấu/thiếu thư mục) | Nếu chỉ có **một** tệp rõ ràng gần giống nhất, bot mở luôn và nói *"mở tệp gần giống nhất: …"*. Nếu có **nhiều** tệp na ná, bot **không đoán** mà liệt kê để bạn chọn |
| `google <từ khóa>`, `tin tức <chủ đề>` | Tìm trên web (mục 11) |
| `đọc https://…` | Đọc nội dung một trang web (cần duyệt) |
| `task`, `task thêm …` | Quản lý tác vụ định kỳ (mục 10) |

## 4. An toàn dữ liệu

- Bot **chỉ thao tác trong thư mục dự án** bạn chọn. Đường dẫn trỏ ra ngoài (`../..`, ổ đĩa khác, symlink thoát ra) bị chặn.
- Chỉ lắng nghe ở `127.0.0.1` (máy bạn). Người khác trong mạng LAN **không** truy cập được.
- Mỗi lần chạy có một mã bí mật ngẫu nhiên; trang web khác trên trình duyệt không gọi trộm được bot.
- Lệnh `chạy …` thực thi với **quyền của tài khoản Windows bạn đang dùng** – hãy đọc kỹ trước khi bấm *Cho phép*, và đừng bật *Tự duyệt* khi chưa chắc.
- `xóa` không đưa vào Thùng rác – xóa là mất.

---

## 5. Cấu hình (`config.json`)

Đặt cạnh `desktop-agent.exe`. Mặc định:

```json
{
  "base_url": "",
  "api_key": "",
  "model": "",
  "port": 7878,
  "auto_approve": false,
  "command_timeout_sec": 60
}
```

| Khóa | Ý nghĩa |
|---|---|
| `base_url` | **Để trống = bot if/else offline.** Điền URL API kiểu OpenAI (ví dụ `http://llm.noibo:8000/v1`) để chuyển sang chế độ có model AI |
| `model` | Tên model (phải hỗ trợ *tool calling*) |
| `api_key` | Khóa API nếu máy chủ yêu cầu |
| `port` | Cổng mở trang chat (mặc định 7878; bận thì tự chọn cổng khác) |
| `auto_approve` | `true` = không hỏi duyệt (chỉ nên dùng khi bạn tin tưởng hoàn toàn) |
| `command_timeout_sec` | Số giây tối đa cho mỗi lệnh `chạy` (mặc định 60, tối đa 3600) |
| `context_tokens` | Cửa sổ ngữ cảnh của LLM (mặc định **16000**). Bot tự cắt đầu vào cho vừa |
| `max_output_tokens` | Phần dành cho câu trả lời (mặc định 1500) |
| `tool_mode` | `text` (mặc định, chạy với **mọi** LLM kể cả loại không hỗ trợ function-calling) · `native` · `auto` |
| `search_provider`, `google_api_key`, `google_cx`, `searxng_url` | Tìm web (mục 11) |
| `alert_webhook` | URL nhận cảnh báo của task (mục 10) |
| `access_token` | (tùy chọn) mã truy cập cố định ≥16 ký tự, để mở lại cùng một đường link khi chạy như dịch vụ (mục 12). Để trống = mã ngẫu nhiên mỗi lần chạy |

Có thể ghi đè bằng biến môi trường: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `AGENT_PORT`.
Tham số dòng lệnh: `desktop-agent.exe --project D:\duan --no-browser`.

Khi có model, mọi thứ giữ nguyên – chỉ khác là bạn nói chuyện bình thường ("đọc giúp mình file README rồi tóm tắt"), model tự chọn công cụ.

---

## 6. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| Bấm exe, không thấy gì | Đợi 2–3 giây. Không mở? Vào `http://127.0.0.1:7878` bằng tay. Nếu bị chặn, xem log |
| Trình duyệt báo "Không thể truy cập" | Bot không chạy hoặc cổng đã đổi. Mở Task Manager đóng `desktop-agent.exe` rồi chạy lại |
| Cảnh báo SmartScreen / Defender | Exe chưa ký số. *More info → Run anyway*. Nếu Defender xóa file: thêm thư mục vào *Exclusions* (hoặc nhờ IT) |
| "Chưa hiểu …" | Gõ `help` hoặc `/` xem cú pháp đúng |
| `tìm` không thấy tệp nhị phân/ảnh/PDF | Tìm theo **tên** thì thấy; tìm theo **nội dung** chỉ áp dụng cho tệp văn bản (≤ 200 KB, bỏ qua `.git`, `node_modules`) |
| `sửa` báo "found 0 times" | Chữ cũ không có trong tệp (kiểm tra hoa/thường, khoảng trắng) |
| `sửa` báo "found 2 times" | Chữ cũ xuất hiện nhiều lần; gõ dài hơn để duy nhất |
| `đọc` báo "binary file" | Tệp nhị phân (ảnh, exe…) – bot chỉ đọc văn bản |
| `google`/`tin tức` báo "máy này có ra được internet không?" | Máy không ra được mạng ngoài hoặc cần proxy. Thử `search_provider: searxng` với máy chủ nội bộ, hoặc nhờ IT mở đường ra |
| LLM trả lời nhưng không gọi công cụ | Xem log; kiểm tra `tool_mode` là `text`; model quá nhỏ có thể không tuân theo định dạng khối ```tool — thử model lớn hơn |
| Lệnh cứ ghi "đang chạy… Ns" mãi | Bấm nút **Dừng** (hình vuông ở ô nhập): bot dừng ngay và giết cả tiến trình con. Lệnh tự dừng sau 60 giây nếu bạn không bấm. Lệnh mở ứng dụng (`chạy notepad`) trả về sau ~2 giây và để ứng dụng chạy tiếp |
| "agent is busy" | Bot đang xử lý lệnh trước; bấm **Dừng** rồi gửi lại. Nếu vẫn báo bận: đóng tab, mở lại trang chat |
| Thẻ công cụ ghi **chờ bạn duyệt** | Bot đang đợi bạn bấm *Cho phép / Từ chối* ở thẻ vàng bên dưới |
| Lệnh `chạy` có dấu nháy kép, ví dụ `chạy dir "C:\Program Files"` | Dùng được; bot truyền nguyên văn cho `cmd.exe`. Output tiếng Việt hiển thị đúng (bot tự đặt bảng mã UTF-8) |

**Log** nằm ở: `%AppData%\desktop-agent\agent.log` (gõ đường dẫn này vào thanh địa chỉ File Explorer).

---

## 7. Gỡ bỏ

Xóa thư mục chứa exe và thư mục `%AppData%\desktop-agent\`. Bot không ghi gì vào Registry, không cài dịch vụ.


---

## 8. Chế độ trợ lý LLM (có model)

Điền `config.json`:

```json
{ "base_url": "http://llm.noibo:8000/v1", "model": "ten-model", "api_key": "",
  "context_tokens": 16000, "max_output_tokens": 1500, "tool_mode": "text" }
```

`base_url` là địa chỉ API kiểu OpenAI (có `/chat/completions`). Khởi động lại exe, góc trên giao diện sẽ hiện tên model.

**Trợ lý làm việc như agent thật:** nhận yêu cầu → quyết định công cụ → bot **chạy thật** → gửi kết quả lại → trợ lý chạy tiếp hoặc trả lời. Ví dụ bạn nói *"viết script python đếm số dòng các file .txt rồi chạy thử"*, trợ lý sẽ `write_file` ra `dem.py`, `run_command python dem.py`, đọc kết quả rồi báo lại. Mọi thao tác ghi/chạy vẫn hiện thẻ xin duyệt.

**Không cần LLM hỗ trợ "tool calling".** Ở `tool_mode: "text"` (mặc định), bot dạy LLM trả lời bằng một khối:

````
```tool
{"name":"run_command","args":{"command":"dir"}}
```
````

và tự đọc khối đó (chấp nhận cả vài biến thể JSON sai chuẩn mà model nhỏ hay viết). Nếu LLM của bạn hỗ trợ function-calling chuẩn OpenAI, có thể đặt `native`, hoặc `auto` (thử native, lỗi thì tự chuyển sang `text`).

### Khi bạn nhớ không chắc tên tệp
Trợ lý được hướng dẫn: *không chắc tên → dùng `find_files`/`list_dir`/`tree` để xem trước*; nếu `read_file` báo không thấy thì kết quả đã kèm **"tệp gần giống"** để nó thử lại đúng tên. Tìm chữ trong nhiều tệp thì dùng `search_files` (có cả PDF/Word/Excel). Với model nhỏ, nếu nó vẫn đoán bừa tên tệp, hãy nói rõ hơn: *"xem danh sách tệp trước rồi mới đọc"*.

### Giới hạn 16k token – bot xử lý thế nào
- Mỗi lần gọi, bot **cắt đầu vào cho vừa ngân sách** = `context_tokens` − `max_output_tokens` − phần dự phòng. Lịch sử cũ bị bỏ dần (giữ câu hỏi mới nhất), kết quả công cụ cũ bị rút gọn.
- Tệp dài được **đọc theo trang** (~3000 ký tự/lần); LLM tự gọi tiếp bằng `offset`.
- Kết quả công cụ gửi cho LLM bị giới hạn kích thước (giao diện vẫn thấy tối đa 8000 ký tự).
- Nếu máy chủ LLM vẫn báo "vượt context", bot tự **giảm 40% rồi gửi lại**.
- Nghĩa là: LLM sẽ **quên các đoạn hội thoại rất cũ** và không nhìn được cả tệp lớn một lúc. Muốn nhắc lại thứ cần nhớ, hãy nhắc trong câu hỏi mới.

### Cơ chế an toàn của vòng lặp
Tối đa 15 bước/lượt; lặp lại đúng một lệnh 3 lần thì dừng; khối `tool` sai JSON được nhắc sửa 2 lần; nút **Dừng** hủy ngay.

---

## 9. Đọc PDF / Word / Excel / PowerPoint

Dùng được qua `đọc <tệp>` hoặc nhờ trợ lý ("tóm tắt hợp đồng.docx").

| Định dạng | Kết quả |
|---|---|
| `.pdf` | Chữ theo từng trang (`--- Trang N ---`), hỗ trợ tiếng Việt |
| `.docx` | Văn bản, bảng (các ô cách nhau ` | `) |
| `.xlsx` | Từng sheet (`## Sheet: tên`), mỗi dòng là một hàng, ô cách nhau ` | `; tối đa 3000 dòng/sheet; ô công thức chưa có giá trị tính sẵn hiện dạng `=B2+B3` |
| `.pptx` | Chữ của từng slide |

**Về độ chính xác của PDF tiếng Việt (quan trọng):** PDF không lưu "văn bản" mà lưu từng ký tự đặt ở vị trí, nên bot phải ghép lại. Đã thử: PDF tạo từ phần mềm lập trình (font Unicode đầy đủ) → **chính xác**; PDF xuất từ LibreOffice với font Carlito/Caladea → đúng thứ tự dòng, bảng và hầu hết chữ, nhưng **một số dấu thanh ghép (ễ, ề, ả…) bị hỏng** (hiện thành `�` hoặc thiếu chữ) và bot tự thêm dòng *"[cảnh báo: …]"*. PDF xuất từ Word chưa được thử. Vì vậy: số liệu, tên riêng, nội dung quan trọng → **hãy kiểm tra lại với bản gốc**, và ưu tiên đưa bản `.docx`/`.xlsx` nếu có.

**Không đọc được:** PDF dạng ảnh/scan (cần OCR – bot sẽ báo rõ), tệp đặt mật khẩu, định dạng cũ `.doc` `.xls` `.ppt` (hãy lưu lại thành `.docx` `.xlsx` `.pptx`), hình ảnh và biểu đồ trong tài liệu. Ngày trong Excel hiện dạng số serial. Chỉ **đọc**, chưa ghi vào các định dạng này; muốn xuất dữ liệu hãy tạo `.csv` bằng `tạo`.

---

## 10. Task định kỳ (monitor, lịch hàng ngày)

Task = *một lệnh (hoặc đoạn code)* + *lịch* + *điều kiện báo động*. Task được lưu trong `%AppData%\desktop-agent\tasks.json` nên **còn nguyên sau khi mở lại**. Task **chỉ chạy khi chương trình đang mở** (bỏ lỡ lúc tắt thì không chạy bù).

### Tạo bằng lệnh
_Ghi chú: gõ tắt `job …` vẫn dùng được như `task …`; dữ liệu lưu bởi bản cũ (`jobs.json`) tự được nạp._

```
task thêm <tên> | <lịch> | <lệnh> [báo nếu có <chữ>] [báo nếu thiếu <chữ>] [timeout <giây>]
```

| Ví dụ | Ý nghĩa |
|---|---|
| `task thêm mạng | mỗi 1 phút | ping -n 1 google.com báo nếu thiếu Reply` | Mỗi phút ping, báo nếu output không có "Reply" |
| `task thêm log lỗi | mỗi 5 phút | findstr /C:"ERROR" app.log báo nếu có ERROR` | Báo khi log xuất hiện ERROR |
| `task thêm ổ đĩa | hàng ngày 08:30 | wmic logicaldisk get freespace,caption` | Mỗi sáng 8:30 chạy lệnh |
| `task thêm backup | hàng ngày 23:00 | python backup.py timeout 600` | Chạy script hằng đêm, tối đa 10 phút |

**Lịch hiểu được:** `mỗi 30s`, `mỗi 90 giây`, `mỗi 5 phút`, `mỗi giờ`, `mỗi 2 giờ`, `mỗi ngày`, `hàng ngày 08:30`, `hàng ngày 8h30`, `daily 08:30`, `every 5m`. Chu kỳ tối thiểu **10 giây**; tối đa **30 task**.

**Điều kiện báo động** (mặc định luôn báo khi lệnh **thoát với mã lỗi ≠ 0** hoặc **hết giờ**): thêm `báo nếu có <chữ>` hoặc `báo nếu thiếu <chữ>` (không phân biệt hoa thường).

### Tạo bằng trợ lý (khi có LLM)
Nói: *"tạo task hàng ngày 8h sáng chạy script python lấy tỷ giá và lưu ra file"*. Trợ lý sẽ viết code, lưu vào `%AppData%\desktop-agent\tasks\`, và tạo task (có thẻ duyệt hiện **toàn bộ lệnh** để bạn xem trước). Hỗ trợ code `python`, `powershell`, `bat`, `sh` (cần máy đã cài Python/PowerShell tương ứng).

### Quản lý
| Gõ | Tác dụng |
|---|---|
| `task` | Danh sách task và trạng thái |
| `task chạy <id/tên>` | Chạy ngay một lần |
| `task log <id>` | Xem 5 lần chạy gần nhất kèm output |
| `task dừng <id>` / `task bật <id>` | Tạm dừng / chạy lại |
| `task xóa <id>` | Xóa (có thẻ duyệt) |

Hoặc bấm nút **Tasks định kỳ** ở thanh bên: bảng hiển thị chấm trạng thái (xanh = ổn, đỏ = đang báo động, xám = tạm dừng), lần chạy kế tiếp, lịch sử và các nút *Chạy ngay / Tạm dừng / Lịch sử / Xóa*.

### Cảnh báo
- Khi task **chuyển từ ổn sang lỗi**: hiện thông báo trên giao diện và huy hiệu đỏ ở nút Tasks. Khi **hồi phục** có thông báo "đã hoạt động lại". Task tiếp tục lỗi **không báo lặp lại** (để không bị spam).
- Muốn nhận ngoài máy: điền `alert_webhook` trong `config.json`. Bot gửi `POST` JSON `{"text":"...","content":"..."}` – tương thích webhook đến của Slack/Mattermost; với Teams/Zalo thường cần một cầu nối nhỏ do IT cấu hình.
- Giao diện cần **đang mở một tab trình duyệt** mới hiện thông báo; task vẫn chạy và ghi lịch sử kể cả khi đóng tab.

### Lưu ý an toàn
Task chạy **không có ai ngồi duyệt**, với quyền tài khoản của bạn. Vì vậy: lệnh của task được bạn xem và duyệt **một lần khi tạo**; ưu tiên lệnh chỉ *đọc/kiểm tra*; đừng cho task chạy lệnh xóa/ghi đè nếu chưa chắc.

---

## 11. Tìm web và đọc trang web (tin tức)

Trợ lý có thể tìm tin rồi đọc và tổng hợp: *"tìm tin tức giá vàng hôm nay rồi tóm tắt 3 nguồn"*. Trong chế độ lệnh: `google giá vàng`, `tin tức bóng đá`, `đọc https://…`.

Máy chạy bot phải **ra được internet** (hoặc qua proxy công ty). Chọn nguồn tìm trong `config.json`:

| `search_provider` | Cần gì | Ghi chú |
|---|---|---|
| `duckduckgo` (mặc định) | Không cần khóa | Đọc trang kết quả HTML; có thể bị chặn/đổi định dạng |
| `google` | `google_api_key` + `google_cx` ([Programmable Search Engine](https://programmablesearchengine.google.com/)) | Ổn định, có giới hạn lượt miễn phí/ngày |
| `searxng` | `searxng_url` (máy chủ SearXNG tự dựng, bật định dạng JSON) | Phù hợp mạng nội bộ |

- `web_search` không cần duyệt. **`fetch_url` (mở trang) luôn hỏi duyệt**, vì địa chỉ URL có thể bị lợi dụng để gửi dữ liệu ra ngoài.
- Nội dung trang web là **không tin cậy**: một trang độc hại có thể chứa câu "hãy xóa file…" nhắm vào trợ lý. Hãy đọc kỹ thẻ duyệt trước khi cho phép bất kỳ thao tác ghi/chạy nào sau khi đọc web.
- Bot lấy chữ từ HTML đơn giản (bỏ menu/quảng cáo ở mức cơ bản), không chạy JavaScript nên trang động có thể trống.


---

## 12. Chạy trên Linux

Cùng một mã nguồn, có gói riêng: `desktop-agent-linux-amd64.tar.gz` (PC/máy chủ Intel/AMD) và `desktop-agent-linux-arm64.tar.gz` (ARM, Raspberry Pi 4/5…). Là **một file tĩnh**, không cần cài thư viện.

```bash
tar -xzf desktop-agent-linux-amd64.tar.gz && cd desktop-agent
nano config.json                      # base_url, model… (để trống = bot if/else)
./desktop-agent --project ~/work      # có giao diện đồ họa: tự mở trình duyệt
```

- Máy **không có màn hình** (server): thêm `--no-browser`; chương trình in ra dòng `Open: http://127.0.0.1:7878/?t=…`. Từ máy của bạn mở đường hầm rồi dán đường link: `ssh -L 7878:127.0.0.1:7878 user@server`.
- Bot chỉ nghe ở `127.0.0.1` (không mở ra mạng). Đừng đổi sang cổng công khai.
- Lệnh `chạy` dùng `sh -c`, nên dùng lệnh Linux (`ls -la`, `df -h`, `systemctl status …`). Các ví dụ `dir`, `type`, `findstr`, `ping -n` trong tài liệu là của Windows; trên Linux dùng `ls`, `cat`, `grep`, `ping -c 1`.
- Dữ liệu: `~/.config/desktop-agent/` (`tasks.json`, `agent.log`, thư mục `jobs`/mã script của task).
- Task `script` dùng `python3`, `bash/sh` (có sẵn trên hầu hết bản phân phối); `powershell` chỉ khi đã cài `pwsh`.

### Chạy nền 24/7 để task định kỳ luôn hoạt động
File `desktop-agent.service` kèm trong gói là dịch vụ systemd mức người dùng:

```bash
mkdir -p ~/.config/systemd/user && cp desktop-agent.service ~/.config/systemd/user/
# sửa dòng ExecStart: đường dẫn file chạy và thư mục dự án
systemctl --user daemon-reload && systemctl --user enable --now desktop-agent
loginctl enable-linger $USER          # vẫn chạy sau khi bạn đăng xuất
journalctl --user -u desktop-agent -f # xem log
```

Chạy dịch vụ thì mã truy cập ngẫu nhiên mỗi lần khởi động sẽ gây khó chịu; hãy đặt `"access_token": "một-chuỗi-dài-ngẫu-nhiên-từ-16-ký-tự-trở-lên"` trong `config.json`, rồi vào `http://127.0.0.1:7878/?t=<access_token>` qua đường hầm SSH. Coi mã này như mật khẩu: ai vào được máy chủ *và* biết mã thì điều khiển được bot.

Cũng có thể dùng **systemd cho chính task** thay vì bộ lập lịch của bot; bộ lập lịch của bot tiện khi bạn muốn quản lý và xem lịch sử ngay trong giao diện chat.

macOS: Go build được (`GOOS=darwin`) nhưng gói này chưa kiểm thử trên macOS nên không phát hành.
