from __future__ import annotations

import unittest
from pathlib import Path


class UiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parents[1]
        cls.styles = (root / "ui" / "src" / "styles.css").read_text(encoding="utf-8")
        cls.application = (root / "ui" / "src" / "App.tsx").read_text(
            encoding="utf-8"
        )

    def test_comparison_results_have_one_bounded_scroll_owner(self):
        self.assertIn(
            ".comparison-dialog-results {\n"
            "  display: grid;\n"
            "  grid-template-rows: auto minmax(0, 1fr);",
            self.styles,
        )
        self.assertIn(
            ".comparison-dialog-results > .comparison-results {\n"
            "  display: grid;\n"
            "  align-content: start;\n"
            "  min-height: 0;",
            self.styles,
        )
        self.assertIn("overflow-y: auto;", self.styles)
        self.assertNotIn(
            ".comparison-dialog-results > .comparison-results {\n  height: 0;",
            self.styles,
        )
        dialog_source = self.application.split("function ComparisonDialog", 1)[1]
        self.assertIn('className="modal-backdrop comparison-backdrop"', dialog_source)


if __name__ == "__main__":
    unittest.main()
