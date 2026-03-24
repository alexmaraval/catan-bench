from __future__ import annotations

import unittest

from catan_bench.dashboard import _system_prompt_message_summary
from catan_bench.reporter import DebugTerminalReporter


class ReporterTests(unittest.TestCase):
    def test_debug_reporter_collapses_system_prompt_message_content(self) -> None:
        lines = DebugTerminalReporter._render_attempt_messages(
            (
                {"role": "system", "content": "line one\nline two"},
                {"role": "user", "content": "choose an action"},
            )
        )

        self.assertEqual(lines[0], "[system]")
        self.assertEqual(lines[1], "(collapsed static system prompt: 2 lines)")
        self.assertIn("[user]", lines)
        self.assertIn("choose an action", lines)

    def test_dashboard_system_prompt_summary_counts_non_empty_lines(self) -> None:
        summary = _system_prompt_message_summary("rules line\n\nbenchmark line\n")
        self.assertEqual(summary, "Static system prompt (2 lines)")


if __name__ == "__main__":
    unittest.main()
