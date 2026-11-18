# Trợ lý Desktop – Hướng dẫn sử dụng

Bot chạy **một file exe duy nhất**, không cần cài đặt, không cần model AI.
Bạn gõ lệnh, bot xử lý bằng luật if/else và thao tác trực tiếp lên thư mục dự án của bạn.

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
| `tìm <regex>` (hoặc `find`, `grep`) | Tìm trong toàn bộ dự án, trả về `tệp:dòng: nội dung` | `tìm TODO` |

### Lệnh thay đổi dữ liệu (**luôn hỏi bạn duyệt trước**)

| Gõ | Tác dụng | Ví dụ |
|---|---|---|
| `tạo <tệp> : <nội dung>` (`write`) | Tạo mới hoặc **ghi đè** tệp; tự tạo thư mục cha | `tạo ghi_chu.txt : mua sữa` |
| `thêm <tệp> : <dòng>` (`append`) | Thêm 1 dòng vào cuối tệp | `thêm ghi_chu.txt : mua trứng` |
| `sửa <tệp> : <cũ> => <mới>` (`edit`) | Thay chữ. Chữ cũ phải **xuất hiện đúng 1 lần** | `sửa a.txt : world => agent` |
| `đổi tên <cũ> => <mới>` (`mv`, `rename`) | Đổi tên hoặc di chuyển trong dự án | `đổi tên a.txt => b/a.txt` |
| `xóa <tệp>` (`rm`, `delete`) | Xóa tệp hoặc **thư mục rỗng** | `xóa b.txt` |
| `chạy <lệnh>` (`run`, `cmd`) | Chạy lệnh terminal (Windows: `cmd`) tại thư mục dự án, tối đa 60 giây | `chạy dir`, `chạy git status` |

Cách nhớ cú pháp: **dấu `:`** ngăn *tên tệp* với *nội dung*, **dấu `=>`** ngăn *cũ* với *mới*.

### Duyệt thao tác

Với lệnh thay đổi dữ liệu, bot hiện thẻ vàng mô tả việc sắp làm:

- **Cho phép** – làm một lần.
- **Từ chối** – bỏ qua, không làm gì.
- **Luôn cho phép** – bật *Tự duyệt thao tác* cho đến khi tắt công tắc ở thanh bên hoặc khởi động lại.

---

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
  "auto_approve": false
}
```

| Khóa | Ý nghĩa |
|---|---|
| `base_url` | **Để trống = bot if/else offline.** Điền URL API kiểu OpenAI (ví dụ `http://llm.noibo:8000/v1`) để chuyển sang chế độ có model AI |
| `model` | Tên model (phải hỗ trợ *tool calling*) |
| `api_key` | Khóa API nếu máy chủ yêu cầu |
| `port` | Cổng mở trang chat (mặc định 7878; bận thì tự chọn cổng khác) |
| `auto_approve` | `true` = không hỏi duyệt (chỉ nên dùng khi bạn tin tưởng hoàn toàn) |

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
| `sửa` báo "found 0 times" | Chữ cũ không có trong tệp (kiểm tra hoa/thường, khoảng trắng) |
| `sửa` báo "found 2 times" | Chữ cũ xuất hiện nhiều lần; gõ dài hơn để duy nhất |
| `đọc` báo "binary file" | Tệp nhị phân (ảnh, exe…) – bot chỉ đọc văn bản |
| "agent is busy" | Bot đang xử lý lệnh trước; bấm nút Dừng hoặc đợi |
| Chữ tiếng Việt lỗi khi `chạy dir` | Console Windows dùng bảng mã cũ; thử `chạy chcp 65001 >nul && dir` |

**Log** nằm ở: `%AppData%\desktop-agent\agent.log` (gõ đường dẫn này vào thanh địa chỉ File Explorer).

---

## 7. Gỡ bỏ

Xóa thư mục chứa exe và thư mục `%AppData%\desktop-agent\`. Bot không ghi gì vào Registry, không cài dịch vụ.
