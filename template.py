"""
K3 — Ngày 1: Khám Phá LLM API (9h00–13h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import Google Gen AI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from google import genai

from dotenv import load_dotenv

# Nạp cấu hình Gemini từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.010},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
}

GEMINI_MODEL = os.getenv("LAB_MODEL", "gemini-2.5-pro")
GEMINI_FLASH_MODEL = os.getenv("LAB_MINI_MODEL", "gemini-2.5-flash")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 10h00–10h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi Gemini
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi Gemini API bằng Google Gen AI SDK, trả về nội dung phản hồi + độ trễ.

    Args:
        prompt:      Tin nhắn của người dùng.
        model:       Model Gemini sử dụng.
        temperature: Độ ngẫu nhiên khi lấy mẫu (0.0 – 2.0).
        top_p:       Ngưỡng nucleus sampling.
        max_tokens:  Số token tối đa được sinh ra.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        from google import genai             # import BÊN TRONG hàm
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # đo thời gian bằng time.time() trước và sau lời gọi API
    """

    client = genai.Client(api_key=GEMINI_API_KEY)
    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        },
    )
    latency = max(time.time() - start, 1e-9)
    return response.text or "", latency


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi Gemini Flash
# ---------------------------------------------------------------------------
def call_gemini_flash(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với Gemini Flash — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        Tái sử dụng call_gemini() với model=GEMINI_FLASH_MODEL — 1 dòng code.
    """
    return call_gemini(
        prompt,
        model=GEMINI_FLASH_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh Gemini Pro vs Gemini Flash
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.

    Returns:
        Dict với các key:
            - "gemini_response":      str
            - "flash_response":       str
            - "gemini_latency":       float
            - "flash_latency":        float
            - "gemini_cost_estimate": float  (USD ước tính cho phản hồi)

    Gợi ý:
        cost = (len(response.split()) / 0.75) / 1000 \\
               * PRICING_PER_1K_TOKENS[GEMINI_MODEL]["output"]
        (0.75 từ ≈ 1 token — ước lượng thô; Part 2 sẽ tính chính xác hơn)
    """
    gemini_text, gemini_latency = call_gemini(prompt)
    flash_text, flash_latency = call_gemini_flash(prompt)
    output_price = PRICING_PER_1K_TOKENS[GEMINI_MODEL]["output"]
    cost = (len(gemini_text.split()) / 0.75) / 1000 * output_price

    return {
        "gemini_response": gemini_text,
        "flash_response": flash_text,
        "gemini_latency": gemini_latency,
        "flash_latency": flash_latency,
        "gemini_cost_estimate": cost,
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 10h40–11h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi Gemini với system_instruction (định hình vai trò/persona)
    và nội dung câu hỏi của người dùng.

    Args:
        system_prompt: Chỉ dẫn vai trò, ví dụ "Bạn là giáo viên tiểu học,
                       giải thích mọi thứ thật đơn giản."
        user_prompt:   Tin nhắn của người dùng.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        config = {"system_instruction": system_prompt, ...}
        contents = user_prompt
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )
    latency = max(time.time() - start, 1e-9)
    return response.text or "", latency


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = GEMINI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Args:
        text:  Đoạn text cần đếm.
        model: Model dùng để chọn bộ mã hóa (encoding).

    Returns:
        Số token (int).

    Gợi ý:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

        tiktoken cần tải bộ mã hóa từ mạng ở lần chạy đầu. Hãy bọc trong
        try/except — nếu lỗi (offline, model lạ), dùng ước lượng dự phòng:
        max(1, len(text) // 4)   (trung bình 1 token ≈ 4 ký tự)
    """
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return max(1, len(encoding.encode(text)))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = GEMINI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT (đếm bằng
    count_tokens) và bảng giá PRICING_PER_1K_TOKENS — tách riêng chi phí
    input (prompt) và output (response).

    Returns:
        Dict với các key:
            - "input_tokens":  int
            - "output_tokens": int
            - "input_cost":    float  (USD)
            - "output_cost":   float  (USD)
            - "total_cost":    float  (USD)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS[GEMINI_MODEL])
        input_cost = input_tokens / 1000 * pricing["input"]
        (.get với fallback về giá model Gemini chính)
    """
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    pricing = PRICING_PER_1K_TOKENS.get(
        model,
        PRICING_PER_1K_TOKENS[GEMINI_MODEL],
    )
    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 11h30–12h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming.

    Hành vi:
        - Stream token từ Gemini ngay khi chúng được sinh ra (in từng chunk).
        - Duy trì 3 lượt hội thoại gần nhất trong history.
        - Gõ 'quit' hoặc 'exit' để thoát.

    Gợi ý:
        - Giữ list `history` theo định dạng Gemini gồm role và parts.
        - Dùng client.models.generate_content_stream() và lặp:
            for chunk in stream:
                delta = chunk.text or ""
                print(delta, end="", flush=True)
        - Sau mỗi lượt, thêm phản hồi assistant vào history.
        - Cắt history còn 3 lượt cuối (6 message): history = history[-6:]
    """
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    history: list[dict] = []

    while True:
        user_msg = input("Bạn: ")
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        contents = history + [
            {"role": "user", "parts": [{"text": user_msg}]}
        ]
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
        )

        reply = ""
        print("Gemini: ", end="", flush=True)
        for chunk in stream:
            delta = chunk.text or ""
            print(delta, end="", flush=True)
            reply += delta
        print()

        history.extend(
            [
                {"role": "user", "parts": [{"text": user_msg}]},
                {"role": "model", "parts": [{"text": reply}]},
            ]
        )
        history = history[-6:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu ném exception, thử lại tối đa max_retries lần với
    exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Callable không tham số.
        max_retries: Số lần thử lại tối đa.
        base_delay:  Delay ban đầu (giây) trước lần thử lại đầu tiên.

    Returns:
        Giá trị trả về của fn() khi thành công.

    Raises:
        Exception cuối cùng của fn() sau khi hết số lần thử.
    """
    if max_retries < 0:
        raise ValueError("max_retries phải lớn hơn hoặc bằng 0")
    if base_delay < 0:
        raise ValueError("base_delay phải lớn hơn hoặc bằng 0")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))

    raise RuntimeError("Không thể thực thi retry_with_backoff")


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 12h10–12h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ bạn đã xây trong Part 1–3.

    Hành vi:
        1. Dùng `persona` làm system prompt cho TOÀN BỘ phiên chat.
        2. Mỗi lượt: đọc tin nhắn qua get_input(); nếu là 'quit'/'exit'
           (không phân biệt hoa thường) → kết thúc phiên.
        3. Gọi generate_content_stream với system_instruction, history và tin nhắn mới.
           Bọc lời gọi API trong retry_with_backoff để chịu lỗi tạm thời.
        4. In từng chunk khi stream về, ghép lại thành reply hoàn chỉnh.
        5. Cập nhật history (user + assistant), giữ tối đa 3 lượt cuối
           (6 message): history = history[-6:]
        6. Cộng dồn thống kê bằng count_tokens và estimate_cost.
        7. Dừng khi đạt max_turns (nếu được đặt).

    Args:
        persona:   Mô tả vai trò, dùng làm system prompt.
        get_input: Hàm đọc input (mặc định: input). Tham số này giúp
                   test tự động không cần bàn phím thật.
        max_turns: Số lượt tối đa (None = không giới hạn).

    Returns:
        Dict thống kê phiên chat:
            - "num_turns":    int   (số lượt hỏi–đáp đã thực hiện)
            - "total_tokens": int   (tổng token user + assistant)
            - "total_cost":   float (tổng USD ước tính)
            - "history":      list  (history còn lại sau khi cắt, ≤ 6 message)

    Gợi ý khung sườn:
        if get_input is None:
            get_input = input
        history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0
        while True:
            if max_turns is not None and num_turns >= max_turns:
                break
            user_msg = get_input()
            if user_msg.strip().lower() in ("quit", "exit"):
                break
            contents = history + [
                {"role": "user", "parts": [{"text": user_msg}]}
            ]
            # stream = retry_with_backoff(lambda:
            #     client.models.generate_content_stream(
            #         model=GEMINI_MODEL, contents=contents,
            #         config={"system_instruction": persona}))
            # reply = ghép các chunk...
            ...
        return {"num_turns": num_turns, "total_tokens": total_tokens,
                "total_cost": total_cost, "history": history}
    """
    if get_input is None:
        get_input = input

    client = genai.Client(api_key=GEMINI_API_KEY)
    history: list[dict] = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break

        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        contents = history + [
            {"role": "user", "parts": [{"text": user_msg}]}
        ]
        stream = retry_with_backoff(
            lambda: client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config={"system_instruction": persona},
            )
        )

        reply = ""
        print("Gemini: ", end="", flush=True)
        for chunk in stream:
            delta = chunk.text or ""
            print(delta, end="", flush=True)
            reply += delta
        print()

        history.extend(
            [
                {"role": "user", "parts": [{"text": user_msg}]},
                {"role": "model", "parts": [{"text": reply}]},
            ]
        )
        history = history[-6:]

        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.

    Returns:
        List các dict — mỗi dict là kết quả compare_models kèm thêm
        key "prompt" chứa prompt gốc.
    """
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({"prompt": prompt, **comparison})
    return results


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.

    Cột: Prompt | Gemini Response | Flash Response | Gemini Latency | Flash Latency
    Gợi ý: cắt text dài còn 40 ký tự cho dễ nhìn.
    """
    headers = [
        "Prompt",
        "Gemini Response",
        "Flash Response",
        "Gemini Latency",
        "Flash Latency",
    ]

    def shorten(value: Any, limit: int = 40) -> str:
        text = str(value).replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 3] + "..."

    rows = []
    for result in results:
        rows.append(
            [
                shorten(result.get("prompt", "")),
                shorten(result.get("gemini_response", "")),
                shorten(result.get("flash_response", "")),
                f"{float(result.get('gemini_latency', 0.0)):.3f}s",
                f"{float(result.get('flash_latency', 0.0)):.3f}s",
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        [format_row(headers), separator, *(format_row(row) for row in rows)]
    )


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần GEMINI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
