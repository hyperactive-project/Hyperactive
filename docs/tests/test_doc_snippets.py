"""Test all documentation code snippets.

This module discovers and tests all Python snippet files in the documentation.
Only snippets in the 'getting_started/' directory are tested for execution,
as they contain complete, runnable examples.

User guide snippets may contain illustrative code with placeholders and are
not required to be executable - they serve documentation purposes.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the snippets directory
SNIPPETS_DIR = Path(__file__).parent.parent / "source" / "_snippets"
# Path to the docs source directory
SOURCE_DIR = Path(__file__).parent.parent / "source"
# Path to the repository root
REPO_ROOT = Path(__file__).parent.parent.parent


def get_testable_snippet_files():
    """Collect Python files in testable directories.

    Only includes files from directories that contain complete, runnable examples.
    Currently: getting_started/, examples/, installation/

    Returns
    -------
    list[Path]
        List of paths to testable Python snippet files.
    """
    testable_dirs = ["getting_started", "examples", "installation"]
    snippet_files = []

    for dir_name in testable_dirs:
        dir_path = SNIPPETS_DIR / dir_name
        if dir_path.exists():
            for path in dir_path.rglob("*.py"):
                if path.name not in ("__init__.py", "conftest.py"):
                    snippet_files.append(path)

    return sorted(snippet_files)


def get_all_snippet_files():
    """Collect all Python files in the snippets directory.

    Returns
    -------
    list[Path]
        List of paths to all Python snippet files.
    """
    snippet_files = []
    for path in SNIPPETS_DIR.rglob("*.py"):
        if path.name not in ("__init__.py", "conftest.py"):
            snippet_files.append(path)
    return sorted(snippet_files)


def _snippet_id(path):
    """Generate a readable test ID for a snippet file."""
    return str(path.relative_to(SNIPPETS_DIR))


@pytest.mark.parametrize("snippet_file", get_testable_snippet_files(), ids=_snippet_id)
def test_snippet_executes(snippet_file):
    """Test that each runnable snippet file executes without errors.

    This runs each snippet as a subprocess to ensure isolation between tests
    and to catch any import-time errors.

    Parameters
    ----------
    snippet_file : Path
        Path to the snippet file to test.
    """
    result = subprocess.run(
        [sys.executable, str(snippet_file)],
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout for optimization examples
        cwd=str(SNIPPETS_DIR),
    )

    # Provide helpful error message on failure
    if result.returncode != 0:
        error_msg = f"Snippet {snippet_file.name} failed to execute.\n"
        error_msg += f"stdout:\n{result.stdout}\n"
        error_msg += f"stderr:\n{result.stderr}"
        pytest.fail(error_msg)


@pytest.mark.parametrize("snippet_file", get_testable_snippet_files(), ids=_snippet_id)
def test_snippet_imports(snippet_file):
    """Test that each runnable snippet file can be imported as a module.

    This catches syntax errors and import-time errors in a more controlled way.

    Parameters
    ----------
    snippet_file : Path
        Path to the snippet file to test.
    """
    spec = importlib.util.spec_from_file_location(
        f"snippet_{snippet_file.stem}", snippet_file
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load spec for {snippet_file}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Snippet {snippet_file.name} failed to import: {e}")


def test_all_snippets_have_markers():
    """Test that all snippet files contain proper start/end markers.

    This ensures that literalinclude directives can extract code properly.
    """
    for snippet_file in get_all_snippet_files():
        content = snippet_file.read_text()

        # Check for at least one start/end pair
        has_start = "# [start:" in content
        has_end = "# [end:" in content

        if not (has_start and has_end):
            pytest.fail(
                f"Snippet {snippet_file.name} missing start/end markers. "
                f"has_start={has_start}, has_end={has_end}"
            )


def test_snippet_markers_are_balanced():
    """Test that start/end markers are properly paired in each snippet file."""
    import re

    for snippet_file in get_all_snippet_files():
        content = snippet_file.read_text()

        starts = re.findall(r"# \[start:(\w+)\]", content)
        ends = re.findall(r"# \[end:(\w+)\]", content)

        # Check that every start has a matching end
        for marker in starts:
            if marker not in ends:
                pytest.fail(
                    f"Snippet {snippet_file.name} has unmatched start marker: {marker}"
                )

        for marker in ends:
            if marker not in starts:
                pytest.fail(
                    f"Snippet {snippet_file.name} has unmatched end marker: {marker}"
                )


def get_rst_files():
    """Collect all RST files in the source directory.

    Returns
    -------
    list[Path]
        List of paths to all RST files.
    """
    return sorted(SOURCE_DIR.rglob("*.rst"))


def extract_github_file_links(content: str) -> list[tuple[str, str]]:
    """Extract GitHub file links from RST content.

    Finds links of the form:
    https://github.com/SimonBlanke/Hyperactive/blob/master/path/to/file.py

    Parameters
    ----------
    content : str
        RST file content.

    Returns
    -------
    list[tuple[str, str]]
        List of (full_url, relative_path) tuples.
    """
    # Pattern matches GitHub blob URLs to this repo
    pattern = r"https://github\.com/SimonBlanke/Hyperactive/blob/master/([^\s>`\"\']+)"
    matches = re.findall(pattern, content)
    return [
        (f"https://github.com/SimonBlanke/Hyperactive/blob/master/{path}", path)
        for path in matches
    ]


def test_github_example_links_exist():
    """Test that all GitHub example links in RST files point to existing files.

    This verifies that documentation links to example files are not broken.
    Only checks links to files within this repository.
    """
    broken_links = []

    for rst_file in get_rst_files():
        content = rst_file.read_text()
        links = extract_github_file_links(content)

        for full_url, rel_path in links:
            local_path = REPO_ROOT / rel_path
            if not local_path.exists():
                broken_links.append(f"{rst_file.name}: {rel_path} (file not found)")

    if broken_links:
        msg = f"Found {len(broken_links)} broken GitHub file link(s):\n"
        msg += "\n".join(f"  - {link}" for link in broken_links)
        pytest.fail(msg)


def extract_include_paths(content: str) -> list[str]:
    """Extract include and literalinclude paths from RST content.

    Parameters
    ----------
    content : str
        RST file content.

    Returns
    -------
    list[str]
        List of relative paths referenced by include directives.
    """
    # Match both include and literalinclude directives
    # Format: .. include:: path or .. literalinclude:: path
    pattern = r"\.\.\s+(?:include|literalinclude)::\s+([^\s\n]+)"
    return re.findall(pattern, content)


def test_rst_includes_exist():
    """Test that all include/literalinclude paths in RST files exist.

    This catches broken include directives that reference non-existent files.
    """
    broken_includes = []

    for rst_file in get_rst_files():
        # Skip auto-generated files and templates (they use Jinja placeholders)
        if "auto_generated" in str(rst_file) or "_templates" in str(rst_file):
            continue

        content = rst_file.read_text()
        includes = extract_include_paths(content)

        for include_path in includes:
            # Resolve path relative to the RST file's directory
            full_path = rst_file.parent / include_path
            if not full_path.exists():
                broken_includes.append(
                    f"{rst_file.relative_to(SOURCE_DIR)}: {include_path}"
                )

    if broken_includes:
        msg = f"Found {len(broken_includes)} broken include path(s):\n"
        msg += "\n".join(f"  - {inc}" for inc in broken_includes)
        pytest.fail(msg)
