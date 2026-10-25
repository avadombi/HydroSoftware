"""Release-level checks for public-facing source and version metadata."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest

from hydrosoftware import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "ui" / "src",
    PROJECT_ROOT / "examples",
)
PUBLIC_TEXT_FILES = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "ui" / "package.json",
    PROJECT_ROOT / "ui" / "index.html",
)


class ReleaseQualityTests(unittest.TestCase):
    def test_public_version_metadata_is_consistent(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
            python_version = tomllib.load(stream)["project"]["version"]
        with (PROJECT_ROOT / "ui" / "package.json").open(encoding="utf-8") as stream:
            interface_version = json.load(stream)["version"]
        config_source = (PROJECT_ROOT / "ui" / "src" / "config.ts").read_text(
            encoding="utf-8"
        )
        application_source = (PROJECT_ROOT / "ui" / "src" / "App.tsx").read_text(
            encoding="utf-8"
        )

        self.assertEqual(__version__, python_version)
        self.assertEqual(__version__, interface_version)
        self.assertIn(f'interfaceVersion = "{__version__}"', config_source)
        self.assertNotIn("Preview 0.", application_source)


if __name__ == "__main__":
    unittest.main()
