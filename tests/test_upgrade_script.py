"""Validate the upgrade script and its generated Python code."""

from __future__ import annotations

import py_compile
import re
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "upgrade.sh"


def test_script_exists_and_executable() -> None:
    assert SCRIPT_PATH.exists(), f"upgrade.sh not found at {SCRIPT_PATH}"
    assert SCRIPT_PATH.stat().st_mode & 0o111, "upgrade.sh is not executable"


def test_bash_syntax_valid() -> None:
    """bash -n checks syntax without executing."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Bash syntax error: {result.stderr}"


def test_script_is_idempotent_guarded() -> None:
    """Every file creation should be guarded by an existence check."""
    content = SCRIPT_PATH.read_text()
    # Count heredoc file creations (cat > "$..._FILE") and skip guards (if [ -f)
    file_creates = len(re.findall(r'cat > "\$\w+_FILE"', content))
    file_creates += len(re.findall(r'cat > "\$\w+_TMPL"', content))
    file_creates += len(re.findall(r'cat > "\$\w+_SERVICE"', content))
    # Count skip calls (function name + space + quoted arg, excluding the definition)
    skip_guards = len(re.findall(r'^\s+skip "', content, re.MULTILINE))
    # Every file creation path should have a corresponding skip guard
    assert (
        skip_guards >= file_creates
    ), f"Found {file_creates} file creations but only {skip_guards} skip guards"


def _extract_heredoc(content: str, marker: str) -> list[str]:
    """Extract all heredoc blocks with a given end marker from shell script."""
    blocks = []
    pattern = rf"cat > .+? << '?{marker}'?\n(.*?)\n{marker}"
    for match in re.finditer(pattern, content, re.DOTALL):
        blocks.append(match.group(1))
    return blocks


def test_generated_python_syntax_valid(tmp_path: Path) -> None:
    """All Python heredocs in the script should be syntactically valid."""
    content = SCRIPT_PATH.read_text()
    python_blocks = _extract_heredoc(content, "PYEOF")
    assert len(python_blocks) >= 3, (
        f"Expected at least 3 Python heredocs (fastapi, pdf, chat_history), "
        f"found {len(python_blocks)}"
    )

    for i, block in enumerate(python_blocks):
        py_file = tmp_path / f"block_{i}.py"
        py_file.write_text(block)
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            raise AssertionError(
                f"Python heredoc block {i} has syntax error: {e}"
            ) from e


def test_generated_python_has_type_hints() -> None:
    """Python code in heredocs should have type hints on function signatures."""
    content = SCRIPT_PATH.read_text()
    python_blocks = _extract_heredoc(content, "PYEOF")

    for i, block in enumerate(python_blocks):
        # Find function definitions
        func_defs = re.findall(r"((?:async )?def \w+\(.*?\).*?:)", block, re.DOTALL)
        for func_def in func_defs:
            # Skip private/dunder methods that are simple
            func_name = re.search(r"def (\w+)", func_def).group(1)
            if func_name.startswith("_") and func_name != "__init__":
                continue
            # Check for return type hint (-> ...) or __init__ (no return needed)
            if func_name != "__init__" and func_name != "event_stream":
                assert (
                    "->" in func_def
                ), f"Block {i}: function '{func_name}' missing return type hint"


def test_generated_sql_uses_parameterized_queries() -> None:
    """SQL in generated code must use ? placeholders, never f-strings."""
    content = SCRIPT_PATH.read_text()
    python_blocks = _extract_heredoc(content, "PYEOF")

    for i, block in enumerate(python_blocks):
        if "sqlite3" not in block:
            continue
        # Check that execute calls use ? params
        execute_calls = re.findall(r"\.execute\((.*?)\)", block, re.DOTALL)
        for call in execute_calls:
            if "?" in call or "CREATE" in call or "executescript" in call.upper():
                continue  # DDL or parameterized — OK
            # Check it's not an f-string
            assert (
                'f"' not in call and "f'" not in call
            ), f"Block {i}: SQL uses f-string instead of parameterized query: {call}"


def test_script_has_set_euo_pipefail() -> None:
    """Script should fail fast on errors."""
    first_lines = SCRIPT_PATH.read_text()[:200]
    assert "set -euo pipefail" in first_lines


def test_script_creates_git_commit() -> None:
    """Script should commit generated files."""
    content = SCRIPT_PATH.read_text()
    assert 'git commit -m "feat: upgrade to production architecture"' in content
