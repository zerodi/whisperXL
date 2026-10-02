"""Check installed distribution identity and both CLI namespaces."""

import importlib.metadata
import subprocess
import sys

import pytest
import whisperxl
import whisperx


def test_public_api_uses_existing_implementation():
    for name in whisperxl.__all__:
        assert getattr(whisperxl, name) is getattr(whisperx, name)


@pytest.mark.parametrize("module", ["whisperxl", "whisperx"])
@pytest.mark.parametrize("option", ["--version", "--help"])
def test_cli_works_with_renamed_distribution(module, option):
    result = subprocess.run([sys.executable, "-m", module, option],
                            capture_output=True, text=True, check=True)
    assert "whisperxl" in result.stdout
    if option == "--version":
        assert result.stdout.strip() == f"whisperxl {importlib.metadata.version('whisperxl')}"
