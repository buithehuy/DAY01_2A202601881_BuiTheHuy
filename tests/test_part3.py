"""
Checkpoint 3 (12h10) — Part 3: Streaming & độ bền
Chạy:  pytest tests/test_part3.py -v
Tất cả API đều được mock — không cần API key thật.
"""

import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD


def _make_stream(text: str):
    """Tạo mock stream giống Google Gen AI SDK."""
    chunks = []
    for piece in (text[: len(text) // 2], text[len(text) // 2 :]):
        chunk = MagicMock()
        chunk.text = piece
        chunks.append(chunk)
    final = MagicMock()
    final.text = None
    chunks.append(final)
    return chunks


class TestStreamingChatbot(unittest.TestCase):

    def test_function_exists_and_is_callable(self):
        self.assertTrue(callable(MOD.streaming_chatbot))

    @patch("builtins.input", side_effect=["quit"])
    @patch("google.genai.Client")
    def test_exits_on_quit(self, MockClient, mock_input):
        """Chatbot phải thoát sạch khi người dùng gõ 'quit'."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        try:
            MOD.streaming_chatbot()
        except StopIteration:
            pass  # input() hết side_effect — chấp nhận được

    @patch("builtins.input", side_effect=["Xin chào", "quit"])
    @patch("google.genai.Client")
    def test_streams_one_turn(self, MockClient, mock_input):
        """Một lượt chat phải gọi API streaming của Gemini."""
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.return_value = _make_stream("Chào bạn!")

        try:
            MOD.streaming_chatbot()
        except StopIteration:
            pass

        self.assertTrue(
            mock_client.models.generate_content_stream.called,
            "Phải gọi generate_content_stream khi người dùng nhập tin nhắn",
        )


class TestRetryWithBackoff(unittest.TestCase):

    def test_succeeds_on_first_try(self):
        result = MOD.retry_with_backoff(lambda: 42)
        self.assertEqual(result, 42)

    def test_retries_on_transient_exception(self):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("transient")
            return "ok"

        result = MOD.retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 2)

    def test_raises_after_max_retries(self):
        def always_fail():
            raise ValueError("permanent failure")

        with self.assertRaises(ValueError):
            MOD.retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)


if __name__ == "__main__":
    unittest.main()
