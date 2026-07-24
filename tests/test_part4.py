"""
Checkpoint 4 (12h50) — Part 4: Mini-project run_assistant
Chạy:  pytest tests/test_part4.py -v
    Nhóm cơ bản:   pytest tests/test_part4.py -k Basic -v
    Nhóm kịch bản: pytest tests/test_part4.py -k Scenario -v
Tất cả API đều được mock — không cần API key thật.
"""

import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD

REQUIRED_KEYS = {"num_turns", "total_tokens", "total_cost", "history"}


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


class TestRunAssistantBasic(unittest.TestCase):

    def test_function_exists_and_is_callable(self):
        self.assertTrue(callable(MOD.run_assistant))

    @patch("google.genai.Client")
    def test_quit_immediately_returns_stats_dict(self, MockClient):
        MockClient.return_value = MagicMock()
        get_input = MagicMock(side_effect=["quit"])

        result = MOD.run_assistant("Bạn là trợ lý.", get_input=get_input)

        self.assertIsInstance(result, dict)
        for key in REQUIRED_KEYS:
            self.assertIn(key, result, f"Thiếu key: {key}")
        self.assertEqual(result["num_turns"], 0)

    @patch("google.genai.Client")
    def test_exit_is_case_insensitive(self, MockClient):
        MockClient.return_value = MagicMock()
        get_input = MagicMock(side_effect=["EXIT"])

        result = MOD.run_assistant("Bạn là trợ lý.", get_input=get_input)

        self.assertEqual(result["num_turns"], 0)

    @patch("google.genai.Client")
    def test_max_turns_zero_returns_without_reading_input(self, MockClient):
        MockClient.return_value = MagicMock()
        get_input = MagicMock(side_effect=[])  # nếu bị gọi sẽ raise StopIteration

        result = MOD.run_assistant("Bạn là trợ lý.", get_input=get_input, max_turns=0)

        self.assertEqual(result["num_turns"], 0)


class TestRunAssistantScenario(unittest.TestCase):
    """Demo tự động: kịch bản hội thoại nhiều lượt có script."""

    PERSONA = "Bạn là trợ giảng thân thiện của khóa AI."

    def _run_conversation(self, MockClient, user_messages, replies):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.side_effect = [
            _make_stream(reply) for reply in replies
        ]
        get_input = MagicMock(side_effect=list(user_messages) + ["quit"])
        result = MOD.run_assistant(self.PERSONA, get_input=get_input)
        return result, mock_client

    @patch("google.genai.Client")
    def test_two_turns_counted_and_stats_positive(self, MockClient):
        result, _ = self._run_conversation(
            MockClient,
            ["Xin chào", "Kể một sự thật thú vị"],
            ["Chào bạn, mình giúp gì được?", "Việt Nam có hơn 3000 km bờ biển."],
        )
        self.assertEqual(result["num_turns"], 2)
        self.assertGreater(result["total_tokens"], 0)
        self.assertGreater(result["total_cost"], 0.0)

    @patch("google.genai.Client")
    def test_api_called_with_stream_and_persona(self, MockClient):
        _, mock_client = self._run_conversation(
            MockClient, ["Xin chào"], ["Chào bạn!"]
        )
        self.assertTrue(mock_client.models.generate_content_stream.called)
        _, kwargs = mock_client.models.generate_content_stream.call_args
        self.assertEqual(kwargs.get("config", {}).get("system_instruction"), self.PERSONA)

    @patch("google.genai.Client")
    def test_history_contains_last_turn(self, MockClient):
        result, _ = self._run_conversation(
            MockClient,
            ["Câu hỏi thứ nhất", "Câu hỏi thứ hai"],
            ["Trả lời thứ nhất.", "Trả lời thứ hai."],
        )
        history_text = " ".join(
            part["text"] for message in result["history"] for part in message["parts"]
        )
        self.assertIn("Câu hỏi thứ hai", history_text)
        self.assertIn("Trả lời thứ hai", history_text)

    @patch("google.genai.Client")
    def test_history_trimmed_to_three_turns(self, MockClient):
        user_messages = [f"Câu hỏi số {i}" for i in range(1, 6)]  # 5 lượt
        replies = [f"Trả lời số {i}." for i in range(1, 6)]
        result, _ = self._run_conversation(MockClient, user_messages, replies)

        self.assertEqual(result["num_turns"], 5)
        self.assertLessEqual(
            len(result["history"]),
            6,
            "History phải được cắt còn tối đa 3 lượt (6 message)",
        )

    @patch("google.genai.Client")
    def test_max_turns_limits_conversation(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.side_effect = [
            _make_stream(f"Trả lời {i}") for i in range(10)
        ]
        # Không có 'quit' — phiên phải tự dừng nhờ max_turns
        get_input = MagicMock(side_effect=[f"Câu {i}" for i in range(10)])

        result = MOD.run_assistant(self.PERSONA, get_input=get_input, max_turns=2)

        self.assertEqual(result["num_turns"], 2)


if __name__ == "__main__":
    unittest.main()
