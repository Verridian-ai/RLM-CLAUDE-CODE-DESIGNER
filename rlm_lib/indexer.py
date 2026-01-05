"""
RLM-C Indexer Module

Provides text search and indexing capabilities using ripgrep.
Enables fast, targeted retrieval from large datasets.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass

from .config import RLMConfig, load_config


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchMatch:
    """A single search match from ripgrep."""
    file_path: Path
    line_number: int
    line_content: str
    match_start: int
    match_end: int
    context_before: List[str]
    context_after: List[str]

    @property
    def highlighted(self) -> str:
        """Return line with match highlighted using markers."""
        before = self.line_content[:self.match_start]
        match = self.line_content[self.match_start:self.match_end]
        after = self.line_content[self.match_end:]
        return f"{before}>>>{match}<<<{after}"


@dataclass
class SearchResult:
    """Results from a search query."""
    query: str
    total_matches: int
    files_matched: int
    matches: List[SearchMatch]
    truncated: bool = False  # True if results were limited
    search_path: Optional[str] = None


@dataclass
class IndexEntry:
    """An entry in a simple file-based index."""
    file_path: str
    line_count: int
    size_bytes: int
    content_type: str
    keywords: List[str]


# =============================================================================
# Ripgrep Integration
# =============================================================================

def _check_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is available."""
    try:
        result = subprocess.run(
            ["rg", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _parse_ripgrep_json(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single line of ripgrep JSON output."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def search_with_ripgrep(
    query: str,
    search_path: str | Path = ".",
    max_results: int = 100,
    context_lines: int = 2,
    case_sensitive: bool = False,
    file_pattern: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> SearchResult:
    """
    Search for text using ripgrep.

    Args:
        query: Search query (regex supported)
        search_path: Path to search in
        max_results: Maximum number of matches to return
        context_lines: Number of context lines before/after match
        case_sensitive: Whether search is case-sensitive
        file_pattern: Glob pattern to filter files (e.g., "*.py")
        exclude_patterns: Patterns to exclude (e.g., ["node_modules", ".git"])

    Returns:
        SearchResult with matches
    """
    search_path = Path(search_path)

    if not _check_ripgrep_available():
        # Fallback to Python-based search
        return _python_search(
            query, search_path, max_results, context_lines, case_sensitive
        )

    # Build ripgrep command
    cmd = [
        "rg",
        "--json",  # JSON output for parsing
        f"--max-count={max_results}",
        f"--context={context_lines}",
    ]

    if not case_sensitive:
        cmd.append("--ignore-case")

    if file_pattern:
        cmd.extend(["--glob", file_pattern])

    if exclude_patterns:
        for pattern in exclude_patterns:
            cmd.extend(["--glob", f"!{pattern}"])

    cmd.extend([query, str(search_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return SearchResult(
            query=query,
            total_matches=0,
            files_matched=0,
            matches=[],
            truncated=True,
            search_path=str(search_path),
        )
    except FileNotFoundError:
        return _python_search(
            query, search_path, max_results, context_lines, case_sensitive
        )

    # Parse JSON output
    matches: List[SearchMatch] = []
    files_seen: set = set()
    context_buffer: Dict[str, List[str]] = {"before": [], "after": []}
    current_match: Optional[Dict] = None

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        data = _parse_ripgrep_json(line)
        if not data:
            continue

        msg_type = data.get("type")

        if msg_type == "match":
            match_data = data.get("data", {})
            file_path = Path(match_data.get("path", {}).get("text", ""))
            line_num = match_data.get("line_number", 0)
            line_text = match_data.get("lines", {}).get("text", "").rstrip()

            # Get match position from submatches
            submatches = match_data.get("submatches", [])
            if submatches:
                start = submatches[0].get("start", 0)
                end = submatches[0].get("end", len(line_text))
            else:
                start = 0
                end = len(line_text)

            files_seen.add(str(file_path))

            matches.append(SearchMatch(
                file_path=file_path,
                line_number=line_num,
                line_content=line_text,
                match_start=start,
                match_end=end,
                context_before=context_buffer["before"].copy(),
                context_after=[],  # Will be filled by subsequent context lines
            ))
            context_buffer["before"] = []

        elif msg_type == "context":
            ctx_data = data.get("data", {})
            line_text = ctx_data.get("lines", {}).get("text", "").rstrip()

            if matches and not matches[-1].context_after:
                # This is "after" context for the previous match
                if len(matches[-1].context_after) < context_lines:
                    matches[-1].context_after.append(line_text)
            else:
                # This is "before" context for a future match
                context_buffer["before"].append(line_text)
                if len(context_buffer["before"]) > context_lines:
                    context_buffer["before"].pop(0)

    return SearchResult(
        query=query,
        total_matches=len(matches),
        files_matched=len(files_seen),
        matches=matches[:max_results],
        truncated=len(matches) > max_results,
        search_path=str(search_path),
    )


def _python_search(
    query: str,
    search_path: Path,
    max_results: int,
    context_lines: int,
    case_sensitive: bool,
) -> SearchResult:
    """Fallback Python-based search when ripgrep is not available."""
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        pattern = re.compile(query, flags)
    except re.error:
        # If query is not valid regex, escape it
        pattern = re.compile(re.escape(query), flags)

    matches: List[SearchMatch] = []
    files_seen: set = set()

    # Walk through files
    for file_path in search_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip binary files and hidden directories
        if any(part.startswith(".") for part in file_path.parts):
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except (IOError, OSError):
            continue

        for i, line in enumerate(lines):
            match = pattern.search(line)
            if match:
                files_seen.add(str(file_path))

                # Get context
                start_ctx = max(0, i - context_lines)
                end_ctx = min(len(lines), i + context_lines + 1)

                matches.append(SearchMatch(
                    file_path=file_path,
                    line_number=i + 1,
                    line_content=line.rstrip(),
                    match_start=match.start(),
                    match_end=match.end(),
                    context_before=[l.rstrip() for l in lines[start_ctx:i]],
                    context_after=[l.rstrip() for l in lines[i+1:end_ctx]],
                ))

                if len(matches) >= max_results:
                    return SearchResult(
                        query=query,
                        total_matches=len(matches),
                        files_matched=len(files_seen),
                        matches=matches,
                        truncated=True,
                        search_path=str(search_path),
                    )

    return SearchResult(
        query=query,
        total_matches=len(matches),
        files_matched=len(files_seen),
        matches=matches,
        truncated=False,
        search_path=str(search_path),
    )


# =============================================================================
# Index Building
# =============================================================================

def build_index(
    source_path: str | Path,
    index_file: Optional[str | Path] = None,
    config: Optional[RLMConfig] = None,
) -> Path:
    """
    Build a simple index of files in a directory.

    Args:
        source_path: Path to index
        index_file: Optional output path for index
        config: Optional RLM configuration

    Returns:
        Path to the created index file
    """
    config = config or load_config()
    source_path = Path(source_path)

    if index_file:
        index_path = Path(index_file)
    else:
        index_path = config.get_cache_path() / "file_index.json"

    entries: List[Dict[str, Any]] = []

    for file_path in source_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip hidden files and common ignore patterns
        if any(part.startswith(".") for part in file_path.parts):
            continue
        if any(p in str(file_path) for p in ["node_modules", "__pycache__", ".git"]):
            continue

        try:
            stat = file_path.stat()
            size = stat.st_size

            # Detect content type
            from .chunker import detect_content_type
            content_type = detect_content_type(file_path)

            # Extract keywords from filename
            keywords = re.findall(r"\w+", file_path.stem.lower())

            # Count lines for text files
            line_count = 0
            if content_type != "binary" and size < 10_000_000:  # < 10MB
                try:
                    with open(file_path, "rb") as f:
                        line_count = sum(1 for _ in f)
                except (IOError, OSError):
                    pass

            entries.append({
                "file_path": str(file_path.relative_to(source_path)),
                "line_count": line_count,
                "size_bytes": size,
                "content_type": content_type,
                "keywords": keywords,
            })

        except (IOError, OSError):
            continue

    # Write index
    index_data = {
        "source_path": str(source_path),
        "total_files": len(entries),
        "indexed_at": __import__("datetime").datetime.now().isoformat(),
        "entries": entries,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    return index_path


def search_index(
    query: str,
    index_file: Optional[str | Path] = None,
    top_k: int = 10,
    config: Optional[RLMConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Search the file index for matching files.

    Args:
        query: Search query (matches against filename and keywords)
        index_file: Path to index file
        top_k: Maximum results to return
        config: Optional RLM configuration

    Returns:
        List of matching index entries
    """
    config = config or load_config()

    if index_file:
        index_path = Path(index_file)
    else:
        index_path = config.get_cache_path() / "file_index.json"

    if not index_path.exists():
        return []

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    entries = index_data.get("entries", [])
    query_terms = set(query.lower().split())

    # Score each entry
    scored = []
    for entry in entries:
        keywords = set(entry.get("keywords", []))
        file_path = entry.get("file_path", "").lower()

        # Simple scoring: count matching terms
        score = 0
        for term in query_terms:
            if term in keywords:
                score += 2
            if term in file_path:
                score += 1

        if score > 0:
            scored.append((score, entry))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [entry for _, entry in scored[:top_k]]


# =============================================================================
# Convenience Functions
# =============================================================================

def find_files(
    pattern: str,
    search_path: str | Path = ".",
    max_results: int = 100,
) -> List[Path]:
    """
    Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "*.py", "**/*.js")
        search_path: Base path to search from
        max_results: Maximum files to return

    Returns:
        List of matching file paths
    """
    search_path = Path(search_path)
    matches = []

    for match in search_path.glob(pattern):
        if match.is_file():
            matches.append(match)
            if len(matches) >= max_results:
                break

    return matches


def count_matches(
    query: str,
    search_path: str | Path = ".",
    case_sensitive: bool = False,
) -> int:
    """
    Count total matches for a query without returning details.

    Args:
        query: Search query
        search_path: Path to search in
        case_sensitive: Whether search is case-sensitive

    Returns:
        Total number of matches
    """
    if _check_ripgrep_available():
        cmd = ["rg", "--count-matches", query, str(search_path)]
        if not case_sensitive:
            cmd.insert(1, "--ignore-case")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            total = 0
            for line in result.stdout.strip().split("\n"):
                if ":" in line:
                    count = line.split(":")[-1]
                    try:
                        total += int(count)
                    except ValueError:
                        pass
            return total
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Fallback: do full search and count
    result = search_with_ripgrep(query, search_path, max_results=10000, case_sensitive=case_sensitive)
    return result.total_matches
