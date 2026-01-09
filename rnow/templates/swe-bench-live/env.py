"""
SWE-bench Live Tools

Tools for the agent to interact with the repository inside the Docker container.
All tools use sandbox=True to execute in the per-instance Modal sandbox.
The repository is mounted at /testbed inside the container.
"""

import subprocess

from rnow.core.tool import tool


@tool(sandbox=True)
def bash(command: str) -> str:
    """Execute a bash command in the repository environment at /testbed.

    Use this for: running tests, git operations, searching files (grep/find),
    checking file structure, installing dependencies, etc.
    """
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/testbed",
        timeout=120,
    )
    output = f"Exit code: {result.returncode}\n"
    if result.stdout:
        output += f"Stdout:\n{result.stdout[-8000:]}\n"
    if result.stderr:
        output += f"Stderr:\n{result.stderr[-2000:]}"
    return output


@tool(sandbox=True)
def read_file(path: str, start_line: int = 1, num_lines: int = 200) -> str:
    """Read lines from a file in the repository.

    Args:
        path: Relative path from /testbed (e.g., "src/module.py")
        start_line: Line number to start reading from (1-indexed)
        num_lines: Number of lines to read (default 200)
    """
    import os

    full_path = os.path.join("/testbed", path)
    if not os.path.exists(full_path):
        return f"Error: File not found: {path}"

    try:
        with open(full_path) as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(start_idx + num_lines, total_lines)

        selected = lines[start_idx:end_idx]
        numbered = [f"{i + start_idx + 1:4d} | {line.rstrip()}" for i, line in enumerate(selected)]

        header = f"File: {path} (lines {start_idx + 1}-{end_idx} of {total_lines})\n"
        return header + "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


@tool(sandbox=True)
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories if needed.

    Args:
        path: Relative path from /testbed (e.g., "src/module.py")
        content: Full content to write to the file
    """
    import os

    full_path = os.path.join("/testbed", path)
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool(sandbox=True)
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Replace a string in a file. The old_str must match exactly (including whitespace).

    Args:
        path: Relative path from /testbed
        old_str: Exact string to find and replace (must be unique in the file)
        new_str: String to replace it with
    """
    import os

    full_path = os.path.join("/testbed", path)
    if not os.path.exists(full_path):
        return f"Error: File not found: {path}"

    try:
        with open(full_path) as f:
            content = f.read()

        count = content.count(old_str)
        if count == 0:
            return f"Error: String not found in {path}. Make sure whitespace matches exactly."
        if count > 1:
            return f"Error: String found {count} times in {path}. Provide more context to make it unique."

        new_content = content.replace(old_str, new_str, 1)
        with open(full_path, "w") as f:
            f.write(new_content)

        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {e}"


@tool(sandbox=True)
def search_files(pattern: str, path: str = ".", file_pattern: str = "*.py") -> str:
    """Search for a regex pattern in files using grep.

    Args:
        pattern: Regex pattern to search for
        path: Directory to search in (relative to /testbed)
        file_pattern: Glob pattern for files to search (default: *.py)
    """
    import subprocess

    cmd = f'grep -rn --include="{file_pattern}" "{pattern}" "{path}" | head -50'
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/testbed",
        timeout=30,
    )

    if result.returncode == 1 and not result.stdout:
        return "No matches found."

    output = result.stdout[:8000] if result.stdout else ""
    if result.stderr:
        output += f"\nErrors: {result.stderr[:500]}"
    return output or "No matches found."


@tool(sandbox=True)
def list_files(path: str = ".", max_depth: int = 2) -> str:
    """List files and directories in a tree structure.

    Args:
        path: Directory to list (relative to /testbed)
        max_depth: Maximum depth to traverse (default: 2)
    """
    import subprocess

    cmd = f'find "{path}" -maxdepth {max_depth} -type f -o -type d | head -100 | sort'
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/testbed",
        timeout=30,
    )

    return result.stdout[:5000] if result.stdout else "Directory is empty or not found."
