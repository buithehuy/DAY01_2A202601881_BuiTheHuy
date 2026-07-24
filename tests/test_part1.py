"""Checkpoint 1 — Gemini API cơ bản (mọi API đều được mock)."""

import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD


def _make_gemini_response(text: str = "Hello from Gemini"):
    response = MagicMock()
    response.text = text
    return response


class TestCallGemini(unittest.TestCase):

    @patch("google.genai.Client")
    def test_returns_non_empty_string(self, MockClient):
        client = MockClient.return_value
        client.models.generate_content.return_value = _make_gemini_response("Test response")
        result, _ = MOD.call_gemini("Hello")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("google.genai.Client")
    def test_latency_is_positive_float(self, MockClient):
        client = MockClient.return_value
        client.models.generate_content.return_value = _make_gemini_response()
        _, latency = MOD.call_gemini("Hello")
        self.assertIsInstance(latency, float)
        self.assertGreater(latency, 0.0)

    @patch("google.genai.Client")
    def test_returns_tuple_of_two(self, MockClient):
        client = MockClient.return_value
        client.models.generate_content.return_value = _make_gemini_response()
        self.assertEqual(len(MOD.call_gemini("Hello")), 2)


class TestCallGeminiFlash(unittest.TestCase):

    @patch("google.genai.Client")
    def test_returns_non_empty_string(self, MockClient):
        client = MockClient.return_value
        client.models.generate_content.return_value = _make_gemini_response("Test response")
        result, _ = MOD.call_gemini_flash("Hello")
        self.assertGreater(len(result), 0)

    @patch("google.genai.Client")
    def test_uses_flash_model(self, MockClient):
        client = MockClient.return_value
        client.models.generate_content.return_value = _make_gemini_response()
        MOD.call_gemini_flash("Hello")
        _, kwargs = client.models.generate_content.call_args
        self.assertEqual(kwargs.get("model"), MOD.GEMINI_FLASH_MODEL)

    @patch("google.genai.Client")
    def test_returns_tuple_of_two(self, MockClient):
        client = MockClient.return_value
        client.models.generate_content.return_value = _make_gemini_response()
        self.assertEqual(len(MOD.call_gemini_flash("Hello")), 2)


class TestCompareModels(unittest.TestCase):

    def _compare(self, main="Gemini answer", flash="Flash answer"):
        with patch.object(MOD, "call_gemini", return_value=(main, 0.5)), \
             patch.object(MOD, "call_gemini_flash", return_value=(flash, 0.3)):
            return MOD.compare_models("Test prompt")

    def test_returns_dict_with_required_keys(self):
        result = self._compare()
        for key in {
            "gemini_response", "flash_response", "gemini_latency",
            "flash_latency", "gemini_cost_estimate",
        }:
            self.assertIn(key, result)

    def test_latency_values_are_positive(self):
        result = self._compare()
        self.assertGreater(result["gemini_latency"], 0)
        self.assertGreater(result["flash_latency"], 0)

    def test_responses_are_non_empty_strings(self):
        result = self._compare()
        self.assertGreater(len(result["gemini_response"]), 0)
        self.assertGreater(len(result["flash_response"]), 0)

    def test_cost_estimate_is_non_negative(self):
        self.assertGreaterEqual(self._compare("word " * 100)["gemini_cost_estimate"], 0)


if __name__ == "__main__":
    unittest.main()
