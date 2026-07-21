import os
import subprocess
import sys


def test_cli_can_write_unicode_status_to_a_cp1252_windows_stream():
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "cp1252"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from armcc.cli import _configure_console_encoding; _configure_console_encoding(); print('✓')",
        ],
        capture_output=True,
        check=True,
        env=environment,
    )

    assert result.stdout.decode("utf-8").strip() == "✓"
