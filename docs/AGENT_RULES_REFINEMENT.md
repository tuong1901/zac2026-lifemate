# Cải tiến agent theo tài liệu BTC ngày 06/11/2026

Mốc code trước đợt này: `a1dc83c`. Điểm BTC người dùng đã cung cấp vẫn là
**34,73/100**. Mục tiêu 95 chưa có kết quả thực nghiệm xác nhận.

Nguồn yêu cầu là phần Description, Data, Evaluation và Rules do người dùng gửi
trong phiên này, gồm `Pasted text.txt`. Website BTC chưa truy cập được từ môi
trường; không coi việc nhận tài liệu là đã kiểm chứng trực tiếp website.

## Điều quyết định điểm

Với mẫu có gold calls, sai state hash hoặc không có call khớp thì điểm bằng 0.
Nếu vượt điều kiện này, điểm là F1 với `P=M/(M+U)` và `R=M/|G|`. Mỗi call chỉ
khớp tối đa một call gold; tên tool và các tham số không fuzzy phải giống nhau;
fuzzy token-F1 phải **lớn hơn** 0,6 sau lowercase và Unicode NFC. Call lỗi schema
vẫn tăng U. Mẫu gold rỗng chỉ đạt điểm nếu submission cũng rỗng.

Do đó, tối ưu có hai việc đồng thời: tránh ghi sai đối tượng/trạng thái và tránh
thiếu/thừa lời gọi. Kết quả unit test không thay thế state hash hoặc điểm BTC.

## Các lỗi đã sửa

| Cơ chế trước đây | Thay đổi và lý do |
| --- | --- |
| `maps.save_place` luôn chọn địa điểm đầu tiên | Giữ ID đã quan sát, hoặc chọn theo tên duy nhất. Không thay ID lạ bằng một kết quả đơn lẻ không liên quan. Nhiều ứng viên chưa phân biệt được thì phản hồi cho planner. |
| Transfer chọn người từ lượt search thứ hai; số điện thoại bị đổi theo danh bạ | Ghép kết quả với đúng tool nguồn, giữ người nhận/điện thoại đã chọn; bỏ chọn theo vị trí search. |
| “phần tôi” biến transfer thành QR cho merchant/số tiền cố định | Bỏ đổi công cụ và số tiền cố định; transfer và QR giữ thông tin của yêu cầu hiện tại. |
| ID người nhận chưa thấy khiến agent tự tìm tên Lan | Phản hồi yêu cầu tìm đúng người trong câu/lịch sử hoặc hỏi phần thiếu; không tự tạo tên mặc định. |
| Chỉ cần search ra nhiều người là bắt buộc hỏi lại | Không ghi đè một lựa chọn đã có ID quan sát hợp lệ; trường hợp chưa rõ vẫn cần planner phân biệt/hỏi. |
| Bài mẫu quyết định số trường, số lần gọi, OOS và thời điểm kết thúc | Mẫu chỉ tham khảo; schema, yêu cầu và kết quả thực tế quyết định. Đếm từng lần xuất hiện thay vì đánh dấu xong toàn bộ theo tên tool. |
| Model trả rỗng nhưng executor lấy lệnh trực tiếp từ prompt mẫu | Bỏ đường thực thi fallback này để ID/giá trị của ví dụ không trở thành hành động. |
| BM25 tự bỏ hard filter khi không tìm thấy bài phù hợp | Giữ bộ lọc, xét cả danh sách tool cụ thể và target của `sys.ask_missing_info`; làm mới cache phụ khi xây lại index. |
| Guardrail đọc `env.tools`; wrapper chuyển tiếp mọi thuộc tính env | Bind schema từ catalog riêng cho mỗi mẫu. Raw ToolEnv chỉ được gọi qua `validate`, `call`, `get_tool_calls`, `get_state_hash`. |
| Sanitizer xóa nội dung object mở, sửa cả args hợp lệ, ép int thành float | Validate bản gốc trước; chỉ sửa khi cần. Giữ object mở và schema tổ hợp cho validator quyết định, không đoán nhánh. |
| Cache số dư còn hiệu lực sau thanh toán qua app khác | Invalidate ngân hàng/ví khi write chỉ định `source=bank/banking/ewallet`, áp dụng cho cache live, lịch sử và bước hiện tại. |
| Phân loại read/write chỉ dựa danh sách tên cố định | Execution lấy `kind` từ catalog, chỉ dùng danh sách cũ khi catalog thiếu kind. |
| Một mệnh đề hủy/ngoài phạm vi loại cả yêu cầu nhiều phần | Chuyển yêu cầu có dấu hiệu nhiều mệnh đề cho planner; không ép kết thúc sau khi hỏi thiếu thông tin của một phần. |
| HF có thể tải model khi inference | Chỉ tải file local, không chạy remote model code; kiểm tra LLM `<5B`, context `<=16384`, seed Python/NumPy/Torch. |

Agent tiếp tục dùng đúng một ToolEnv do runner truyền vào. Wrapper không tạo,
clone hoặc reset env. Submission lấy toàn bộ log và hash do env trả về; các call
thực sự chạy, kể cả thất bại, không bị lọc khỏi submission.

## Kiểm chứng và giới hạn

Chạy `python -m unittest discover -s tests -v`: **65 kiểm thử** gồm 40 kiểm thử
trước đó và 25 hồi quy mới. Các ca mới dùng yêu cầu/quan sát tổng hợp, không dùng
nhãn public/private. Kiểm thử Transformers dùng Qwen rất nhỏ với trọng số ngẫu
nhiên được tạo local để kiểm chứng đường chạy, không đo năng lực hiểu yêu cầu.

Hai fixture cũ được sửa thành `additionalProperties: false` khi kỳ vọng loại
trường ngoài schema; object không có ràng buộc đó phải giữ dữ liệu theo JSON Schema.

Những điều chưa được chứng minh:

- Chưa có A/B trên train holdout với LLM đã huấn luyện và zaic_tools đúng split.
  Workspace có transcript MultiASR, nhưng thiếu dataset gốc/context/catalog,
  zaic_tools và weights local để chạy benchmark chính thức.
- Chưa đo latency trên A30, tổng số tham số ASR + LLM + thành phần khác `<6B`,
  giấy phép/ngày phát hành model trước 26/10/2026, hoặc Docker hoàn toàn offline.
  Các kiểm tra của HF client chỉ bao phủ LLM mà client trực tiếp tải.
- Ngân sách 28 giây là thời gian trong `ReActAgent.run`, không bao gồm ASR trước
  đó. Không ngắt cứng native kernel/tool đang chạy. Cần đo cả pipeline dưới
  giới hạn 30 giây. Client server phải dùng model local với context tối đa 16K;
  không sử dụng endpoint bên ngoài khi inference thi đấu.
- Code còn heuristic theo miền và các ví dụ tổng hợp cũ. Prompt mới bớt ép theo
  mẫu có thể đổi hành vi LLM; chưa thể lượng hóa lợi/hại nếu thiếu đánh giá thật.
- Invalidation chéo mới xử lý nguồn thanh toán tường minh; chưa phải mô hình đầy
  đủ mọi tác động chéo giữa ứng dụng.
- Script kiểm tra compliance hiện có chỉ quét pattern, không chứng nhận toàn bộ
  quy định. Các script benchmark/train cũ có truy cập riêng như `env.log` và tạo
  env đối chiếu gold; không dùng chúng làm đường inference cuối. Scorer nội bộ
  cũng cần đối chiếu thuật toán matching với BTC trước khi dùng làm thước đo.

Không thay ASR, không gán nhãn public/private, không tạo submission hoặc sửa
artifact submission cũ trong đợt chỉ cải tiến agent này.
