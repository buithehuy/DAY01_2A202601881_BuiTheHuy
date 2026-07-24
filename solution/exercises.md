# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0 nội dung chắc chắn, từ ngữ rất ổn định. Khi tăng lên 0.5 và 1.0, phản hồi bắt đầu đa dạng hơn về cách diễn đạt nhưng vẫn chính xác về nội dung. Ở temperature 1.5, câu trả lời trở nên sáng tạo hơn, đôi khi dùng từ bất ngờ hoặc thêm chi tiết ngoài lề, nhưng cũng có thể kém nhất quán hoặc ít chính xác hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Em sẽ đặt temperature = 0.2. Chatbot hỗ trợ khách hàng cần ưu tiên độ chính xác và nhất quán và ổn định.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Vì openAI api key bị hết nên em dùng Gemini, với bảng giá Gemini trong bài, gemini-2.5-pro tốn $0.010/1K token output, còn gemini-2.5-flash tốn $0.0025/1K token output — tức Pro đắt gấp **4 lần** Flash. Với workload 10.000 users × 3 lần/ngày × 350 token = 10,5 triệu token output/ngày, chi phí Pro ≈ $105/ngày so với Flash ≈ $26/ngày. **Nên dùng Pro** khi cần phân tích hợp đồng pháp lý, viết báo cáo nghiên cứu hoặc lập luận đa bước phức tạp — nơi chất lượng ảnh hưởng trực tiếp đến kết quả kinh doanh. **Nên dùng Flash** cho chatbot FAQ, phân loại email, tóm tắt ngắn — những tác vụ lặp lặp, khối lượng lớn mà độ chính xác tuyệt đối không phải yếu tố sống còn.



---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với system prompt "giáo viên tiểu học", phản hồi ngắn hơn (~100–150 từ), dùng ví dụ gần gũi như "blockchain giống cuốn sổ ghi nợ mà cả lớp cùng giữ, không ai có thể sửa lén", hoàn toàn không có thuật ngữ kỹ thuật. Với system prompt "chuyên gia tài chính", phản hồi dài hơn (~200–300 từ), dày đặc các thuật ngữ. System prompt đóng vai trò như một nhân cách, nó định hướng toàn bộ giọng văn, lựa chọn từ vựng, độ phức tạp và loại ví dụ mà model sử dụng, dù nội dung câu hỏi hoàn toàn giống nhau. Điều này chứng minh system prompt là công cụ kiểm soát hành vi model mạnh mẽ hơn nhiều so với việc chỉnh sửa user prompt.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Thử nghiệm với đoạn văn tiếng Việt 100 từ: "Việt Nam là một quốc gia nằm ở Đông Nam Á, có diện tích khoảng 331.000 km² và dân số hơn 97 triệu người. Đất nước có hình chữ S, trải dài từ Bắc xuống Nam với chiều dài bờ biển hơn 3.200 km. Thủ đô Hà Nội nằm ở miền Bắc, còn Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất ở miền Nam. Văn hóa Việt Nam chịu ảnh hưởng sâu sắc từ Trung Quốc trong hàng nghìn năm lịch sử, đồng thời mang đậm bản sắc riêng qua các lễ hội truyền thống, ẩm thực phong phú và nghệ thuật dân gian đa dạng." — `count_tokens` đếm được khoảng **170 token** (do tiktoken fallback dùng len//4 ≈ 175), còn ước lượng `số từ / 0.75 ≈ 133 token`, chênh nhau khoảng **~28%**. Nguyên nhân: tiếng Việt có dấu thanh điệu và ký tự Unicode đặc thù (ă, â, ơ, ư, đ...), mỗi âm tiết thường bị tokenizer GPT-based tách thành nhiều byte-pair tokens thay vì một token duy nhất như với tiếng Anh, khiến tỉ lệ token/từ của tiếng Việt cao hơn hẳn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming đặc biệt quan trọng khi phản hồi dài và người dùng cần đọc ngay — như chatbot trợ lý, viết văn, giải thích code — vì nó giảm **perceived latency**: người dùng thấy chữ xuất hiện ngay lập tức thay vì chờ 10–30 giây mới thấy kết quả. Ngược lại, non-streaming phù hợp hơn khi cần xử lý toàn bộ phản hồi trước khi làm gì đó tiếp theo — ví dụ: phân tích JSON từ model, chạy batch processing tự động, gọi API trong pipeline backend không có UI, hoặc khi cần đếm token / tính chi phí chính xác trên toàn bộ output trước khi lưu vào database.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff (delay = `base_delay * 2^attempt`) tăng dần thời gian chờ qua mỗi lần thất bại, cho phép server có đủ thời gian phục hồi trước khi nhận thêm request. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (ví dụ đều chờ 1 giây rồi retry đồng loạt), chúng sẽ tạo ra hiện tượng **"thundering herd"** — server vừa hồi phục lại lập tức bị dội bom bởi hàng nghìn request cùng lúc, khiến nó quá tải lại ngay và không bao giờ thoát ra được vòng lặp lỗi. Thêm jitter ngẫu nhiên vào exponential backoff (như `delay + random(0, delay)`) giúp rải đều các request theo thời gian, tránh hiệu ứng đồng bộ hóa nguy hiểm này.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona **trợ giảng AI thân thiện** cho khóa học lập trình. System prompt: *"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."* — Đây chính xác là prompt được dùng trong `run_assistant()` của solution. **"Trả lời ngắn gọn"**: giúp tiết kiệm token và giữ cho hội thoại không bị ngập trong văn bản dài — context window chỉ giữ 3 lượt (6 message), nên mỗi lượt càng gọn thì lịch sử càng có ích lâu hơn. **"Bằng tiếng Việt"**: học viên là người Việt, dùng tiếng mẹ đẻ giúp họ hiểu nhanh hơn và không mất thời gian dịch thuật; nếu không chỉ định, model có thể trả lời tiếng Anh khi nhận được câu hỏi tiếng Anh, gây không nhất quán trong trải nghiệm.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là **history chỉ giữ 3 lượt gần nhất (6 message)** — nếu cuộc trò chuyện kéo dài, model sẽ "quên" toàn bộ bối cảnh trước đó, dẫn đến câu trả lời lạc đề hoặc mâu thuẫn. **Cải thiện đề xuất: tóm tắt tự động (conversation summarization).** Thay vì cắt cứng `history[-6:]`, sau mỗi 3 lượt ta gọi thêm một lời gọi API nhỏ để tóm tắt lịch sử cũ thành 2–3 câu, rồi chèn bản tóm tắt đó vào đầu history dưới dạng system message. Triển khai: thêm biến `summary: str = ""`, sau mỗi `n` lượt gọi `call_gemini(f"Tóm tắt cuộc trò chuyện sau trong 3 câu: {old_history}")` rồi đặt `history = [{"role": "user", "parts": [{"text": f"[Bối cảnh trước: {summary}]"}]}, ...]` trước khi gắn lượt mới — chi phí tăng nhẹ nhưng model giữ được ngữ cảnh dài hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
