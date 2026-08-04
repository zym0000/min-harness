"""
Coding Tools MCP Server
提供: read_file, write_file, apply_patch, list_dir, grep, search_code, shell_exec
"""
import os
import re
import subprocess
import difflib
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("coding-tools-server")

# 安全沙箱根目录（限制文件操作范围）
WORKSPACE_ROOT = os.environ.get("AGENT_WORKSPACE", os.getcwd())


def _resolve_path(file_path: str) -> Path:
    """解析路径，确保在 workspace 内"""
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(WORKSPACE_ROOT) / p
    resolved = p.resolve()
    workspace = Path(WORKSPACE_ROOT).resolve()
    if not str(resolved).startswith(str(workspace)):
        raise PermissionError(f"Path '{file_path}' is outside workspace '{WORKSPACE_ROOT}'")
    return resolved

# Tool 1: read_file
@mcp.tool()
def read_file(file_path: str, start_line: int = 1, end_line: int = -1) -> str:
    """
    Read content of a file. Supports line range selection.
    
    Args:
        file_path: Path to the file (relative to workspace or absolute)
        start_line: Starting line number (1-based, default: 1)
        end_line: Ending line number (-1 means read to end)
    
    Returns:
        File content with line numbers prefixed.
    """
    resolved = _resolve_path(file_path)
    if not resolved.exists():
        return f"[ERROR] File not found: {file_path}"
    if not resolved.is_file():
        return f"[ERROR] Not a file: {file_path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR] Failed to read file: {e}"

    lines = content.splitlines(keepends=True)
    total = len(lines)

    if start_line > total:
        return f"[ERROR] start_line ({start_line}) exceeds total lines ({total}) in {file_path}"

    if end_line == -1:
        end_line = total
    start_line = max(1, start_line)
    end_line = min(total, end_line)

    if start_line > end_line:
        return f"[ERROR] start_line ({start_line}) > end_line ({end_line})"

    selected = lines[start_line - 1: end_line]
    numbered = [f"{i + start_line:>6} | {line}" for i, line in enumerate(selected)]

    header = f"[File: {file_path}] [Lines {start_line}-{end_line} of {total}]\n"
    return header + "".join(numbered)

# Tool 2: write_file
@mcp.tool()
def write_file(file_path: str, content: str, create_dirs: bool = True) -> str:
    """
    Write content to a file. Creates the file if it doesn't exist.
    Overwrites existing content entirely.
    
    Args:
        file_path: Path to the file
        content: The full content to write
        create_dirs: Whether to create parent directories if they don't exist
    
    Returns:
        Confirmation message with bytes written.
    """
    resolved = _resolve_path(file_path)

    if create_dirs:
        resolved.parent.mkdir(parents=True, exist_ok=True)

    try:
        resolved.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"[ERROR] Failed to write file: {e}"

    size = resolved.stat().st_size
    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"[OK] Written {size} bytes ({line_count} lines) to {file_path}"

# Tool 3: apply_patch (Unified Diff)
@mcp.tool()
def apply_patch(file_path: str, patch: str) -> str:
    """
    Apply a unified diff patch to a file.
    The patch should be in standard unified diff format (with --- and +++ headers).
    Supports creating new files (--- /dev/null) and deleting files (+++ /dev/null).
    
    Args:
        file_path: Path to the target file
        patch: Unified diff content (the patch text)
    
    Returns:
        Result of patch application.
    """
    resolved = _resolve_path(file_path)

    # Handle new file creation
    if re.search(r'^--- (?:a/)?/dev/null\s*$', patch, re.MULTILINE):
        # 使用统一的 hunk 解析，支持多 hunk
        hunks = _parse_unified_diff(patch)
        new_content_lines = []
        for h in hunks:
            new_content_lines.extend(h["new_lines"])
        new_content = "".join(new_content_lines)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(new_content, encoding="utf-8")
        return f"[OK] Created new file: {file_path} ({len(new_content)} bytes)"

    # Handle file deletion
    if re.search(r'^\+\+\+ (?:b/)?/dev/null\s*$', patch, re.MULTILINE):
        if resolved.exists():
            resolved.unlink()
            return f"[OK] Deleted file: {file_path}"
        return f"[WARN] File not found for deletion: {file_path}"

    # Standard patch application
    if not resolved.exists():
        return f"[ERROR] File not found: {file_path}"

    original_content = resolved.read_text(encoding="utf-8")
    original_lines = original_content.splitlines(keepends=True)

    # Parse hunks
    hunks = _parse_unified_diff(patch)
    if not hunks:
        return "[ERROR] No valid hunks found in patch"

    # Apply hunks in reverse order to preserve line numbers
    result_lines = list(original_lines)
    for hunk in reversed(hunks):
        start = hunk["orig_start"] - 1  # Convert to 0-based
        # Remove original lines
        del result_lines[start: start + hunk["orig_count"]]
        # Insert new lines
        for i, line in enumerate(hunk["new_lines"]):
            result_lines.insert(start + i, line)

    new_content = "".join(result_lines)
    resolved.write_text(new_content, encoding="utf-8")

    return f"[OK] Patch applied to {file_path} ({len(hunks)} hunk(s))"


def _parse_unified_diff(patch_text: str) -> list:
    """Parse unified diff into hunks"""
    hunks = []
    lines = patch_text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                orig_start = int(match.group(1))
                orig_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1

                hunk_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("---") and not lines[i].startswith("+++"):
                    hunk_lines.append(lines[i])
                    i += 1

                new_lines = []
                for hl in hunk_lines:
                    if hl.startswith("+"):
                        new_lines.append(hl[1:])
                    elif hl.startswith(" "):
                        new_lines.append(hl[1:])
                    # '-' lines and special lines like '\ No newline...' are omitted

                hunks.append({
                    "orig_start": orig_start,
                    "orig_count": orig_count,
                    "new_start": new_start,
                    "new_count": new_count,
                    "new_lines": new_lines,
                })
                continue
        i += 1
    return hunks

# Tool 4: list_dir
@mcp.tool()
def list_dir(dir_path: str = ".", recursive: bool = False, pattern: str = "*") -> str:
    """
    List directory contents.
    
    Args:
        dir_path: Directory path (relative to workspace or absolute)
        recursive: Whether to list recursively
        pattern: Glob pattern to filter entries (e.g., "*.py")
    
    Returns:
        Formatted directory listing with file sizes and types.
    """
    resolved = _resolve_path(dir_path)
    if not resolved.exists():
        return f"[ERROR] Directory not found: {dir_path}"
    if not resolved.is_dir():
        return f"[ERROR] Not a directory: {dir_path}"

    entries = []
    if recursive:
        for item in sorted(resolved.rglob(pattern)):
            rel = item.relative_to(resolved)
            if item.is_dir():
                entries.append(f"  [DIR]  {rel}/")
            else:
                size = item.stat().st_size
                entries.append(f"  [FILE] {rel} ({_format_size(size)})")
    else:
        for item in sorted(resolved.glob(pattern)):
            if item.is_dir():
                entries.append(f"  [DIR]  {item.name}/")
            else:
                size = item.stat().st_size
                entries.append(f"  [FILE] {item.name} ({_format_size(size)})")

    if not entries:
        return f"[INFO] No entries matching '{pattern}' in {dir_path}"

    header = f"[Directory: {dir_path}] ({len(entries)} entries)\n"
    return header + "\n".join(entries)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"

# Tool 5: grep
@mcp.tool()
def grep(pattern: str, path: str = ".", include: str = "*", case_sensitive: bool = True, max_results: int = 50) -> str:
    """
    Search for a regex pattern in files (like grep -rn).
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        include: File glob pattern to include (e.g., "*.py")
        case_sensitive: Whether search is case-sensitive
        max_results: Maximum number of results to return
    
    Returns:
        Matching lines with file paths and line numbers.
    """
    resolved = _resolve_path(path)
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"[ERROR] Invalid regex pattern: {e}"

    results = []
    files_to_search = []

    if resolved.is_file():
        files_to_search = [resolved]
    elif resolved.is_dir():
        files_to_search = sorted(resolved.rglob(include))
    else:
        return f"[ERROR] Path not found: {path}"

    for fpath in files_to_search:
        if not fpath.is_file():
            continue
        # Skip binary files
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for line_no, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                rel_path = fpath.relative_to(Path(WORKSPACE_ROOT).resolve()) if str(fpath).startswith(WORKSPACE_ROOT) else fpath
                results.append(f"  {rel_path}:{line_no}: {line.rstrip()}")
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:   # 外层提前退出
            break

    if not results:
        return f"[INFO] No matches found for pattern '{pattern}' in {path}"

    header = f"[Grep: '{pattern}' in {path}] ({len(results)} matches)\n"
    return header + "\n".join(results)

# Tool 6: search_code (semantic-ish search)
@mcp.tool()
def search_code(query: str, path: str = ".", file_pattern: str = "*.py", max_results: int = 20) -> str:
    """
    Search code by keywords/symbols. Looks for function/class/variable definitions
    and usages. More structured than grep for code exploration.
    
    Args:
        query: Search query (function name, class name, variable, or keyword)
        path: Directory to search in
        file_pattern: File glob pattern
        max_results: Maximum results
    
    Returns:
        Code search results with context.
    """
    resolved = _resolve_path(path)
    results = []

    # Build multiple search patterns
    patterns = [
        re.compile(rf"(def\s+{re.escape(query)}\s*\()", re.IGNORECASE),
        re.compile(rf"(class\s+{re.escape(query)}\s*[\(:])", re.IGNORECASE),
        re.compile(rf"(\b{re.escape(query)}\b\s*=)", re.IGNORECASE),
        re.compile(rf"(import\s+.*\b{re.escape(query)}\b)", re.IGNORECASE),
        re.compile(rf"(\b{re.escape(query)}\b\s*\()", re.IGNORECASE),  # function call
    ]

    files = sorted(resolved.rglob(file_pattern)) if resolved.is_dir() else [resolved]

    for fpath in files:
        if not fpath.is_file():
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = content.splitlines()
        for line_no, line in enumerate(lines, 1):
            for pat in patterns:
                if pat.search(line):
                    rel = fpath.relative_to(Path(WORKSPACE_ROOT).resolve()) if str(fpath).startswith(WORKSPACE_ROOT) else fpath
                    # Get context (1 line before, 1 after)
                    context_before = lines[line_no - 2].rstrip() if line_no > 1 else ""
                    context_after = lines[line_no].rstrip() if line_no < len(lines) else ""
                    results.append(
                        f"  {rel}:{line_no}:\n"
                        f"    {context_before}\n"
                        f"  > {line.rstrip()}\n"
                        f"    {context_after}"
                    )
                    break
            if len(results) >= max_results:
                break
        if len(results) >= max_results:   # 外层提前退出
            break

    if not results:
        return f"[INFO] No code matches for '{query}' in {path}"

    header = f"[Code Search: '{query}'] ({len(results)} results)\n"
    return header + "\n\n".join(results)

# Tool 7: shell_exec
@mcp.tool()
def shell_exec(command: str, timeout: int = 30, cwd: str = ".") -> str:
    """
    Execute a shell command and return stdout/stderr.
    Use for running tests, installing packages, git operations, etc.
    
    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        cwd: Working directory for the command
    
    Returns:
        Command output (stdout + stderr) and exit code.
    """
    work_dir = str(_resolve_path(cwd))

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
            env={**os.environ, "AGENT_WORKSPACE": WORKSPACE_ROOT},
        )

        output_parts = []
        if result.stdout:
            output_parts.append(f"[STDOUT]\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"[STDERR]\n{result.stderr}")
        output_parts.append(f"[EXIT CODE: {result.returncode}]")

        output = "\n".join(output_parts)

        # Truncate if too long
        if len(output) > 10000:
            output = output[:5000] + "\n... [TRUNCATED] ...\n" + output[-3000:]

        return output

    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout}s: {command}"
    except Exception as e:
        return f"[ERROR] Command execution failed: {e}"

# Tool 8: create_patch (generate diff)
@mcp.tool()
def create_patch(file_path: str, new_content: str) -> str:
    """
    Generate a unified diff patch between current file content and new content.
    Useful for reviewing changes before applying them.
    
    Args:
        file_path: Path to the existing file
        new_content: The proposed new content
    
    Returns:
        Unified diff patch string.
    """
    resolved = _resolve_path(file_path)

    if resolved.exists():
        original = resolved.read_text(encoding="utf-8")
    else:
        original = ""

    orig_lines = original.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        orig_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )

    patch_text = "".join(diff)
    if not patch_text:
        return "[INFO] No differences found."

    return f"[PATCH for {file_path}]\n{patch_text}"


if __name__ == "__main__":
    mcp.run()