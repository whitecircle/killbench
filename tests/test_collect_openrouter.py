import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from scripts import collect_openrouter as collector


class IncrementalFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_skipped_parser_checkpoints_match_final_results(self):
        cases = [
            ({"success": True, "content": "unparsed response"}, "gemini_reparse_skipped", "low"),
            ({"success": True, "content": "   "}, "no_response", "none"),
            ({"success": False, "error": "timeout"}, "request_error", "none"),
        ]
        for response, method, confidence in cases:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                with patch.object(collector, "OpenRouterClient") as client, \
                        patch.object(collector, "GeminiFreeTextParser") as parser:
                    client.return_value.complete = AsyncMock(
                        return_value={"model": "test/model", **response}
                    )
                    results = await collector.collect_jobs(
                        [{"model": "test/model", "system_prompt": "", "prompt": "test",
                          "metadata": {"id": "test"}}],
                        response_mode="freetext",
                        concurrency=1,
                        skip_gemini_parse=True,
                        incremental_path=Path(directory) / "checkpoint.jsonl",
                    )
                    parser.assert_not_called()
                    client.return_value.complete.assert_awaited_once()
                checkpoint = json.loads(
                    (Path(directory) / "incremental_test_model.jsonl").read_text()
                )
                parsed = checkpoint["parsed"]
                self.assertEqual(parsed, results[0]["parsed"])
                self.assertEqual(parsed["parse_method"], method)
                self.assertEqual(parsed["confidence"], confidence)
                self.assertIsNone(parsed["chosen_number_original"])
                self.assertFalse(parsed["is_refusal"])

    async def test_enabled_parser_still_runs(self):
        with patch.object(collector, "OpenRouterClient") as client, \
                patch.object(collector, "GeminiFreeTextParser") as parser:
            client.return_value.complete = AsyncMock(
                return_value={"model": "test/model", "success": True, "content": "No."}
            )
            parser.return_value.parse_response = AsyncMock(
                return_value={"is_refusal": True, "raw": "refusal"}
            )
            results = await collector.collect_jobs(
                [{"model": "test/model", "system_prompt": "", "prompt": "test", "metadata": {}}],
                response_mode="freetext", concurrency=1, skip_gemini_parse=False,
            )
            parser.return_value.parse_response.assert_awaited_once_with("No.", 4)
        self.assertTrue(results[0]["parsed"]["is_refusal"])
        self.assertEqual(results[0]["parsed"]["parse_method"], "gemini_reparse")


if __name__ == "__main__":
    unittest.main()
