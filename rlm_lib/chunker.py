"""
RLM-C Chunker Module

Handles intelligent splitting of large files into manageable chunks.
Supports both code and document content with appropriate boundary detection.
"""

import re
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime

from .config import (
    RLMConfig,
    load_config,
    MAX_CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_LINES,
    CODE_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    LOG_EXTENSIONS,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FileInfo:
    """Information about a file without reading its full content."""
    path: Path
    size_bytes: int
    size_human: str
    line_count: Optional[int]
    content_type: str
    extension: str
    modified_time: datetime
    is_binary: bool
    encoding: Optional[str]


@dataclass
class ChunkInfo:
    """Information about a single chunk."""
    chunk_id: str
    chunk_path: Path
    source_path: Path
    start_line: int
    end_line: int
    size_bytes: int
    content_type: str


ContentType = Literal["code", "document", "log", "binary", "unknown"]


# =============================================================================
# Content Type Detection
# =============================================================================

def detect_content_type(path: Path) -> ContentType:
    """
    Auto-detect content type based on file extension and content sampling.

    Args:
        path: Path to the file

    Returns:
        Content type: 'code', 'document', 'log', 'binary', or 'unknown'
    """
    path = Path(path)
    ext = path.suffix.lower()

    # Check by extension first
    if ext in CODE_EXTENSIONS:
        return "code"
    elif ext in DOCUMENT_EXTENSIONS:
        return "document"
    elif ext in LOG_EXTENSIONS:
        return "log"

    # Sample content for binary detection
    try:
        with open(path, "rb") as f:
            sample = f.read(8192)

        # Check for null bytes (binary indicator)
        if b"\x00" in sample:
            return "binary"

        # Try to decode as text
        try:
            text = sample.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = sample.decode("latin-1")
            except UnicodeDecodeError:
                return "binary"

        # Heuristic: check for code patterns
        code_patterns = [
            r"^\s*(def |class |function |import |from |const |let |var |public |private )",
            r"[{}\[\]();]",
            r"^\s*#include",
            r"^\s*package\s+",
        ]
        for pattern in code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return "code"

        # Default to document for text files
        return "document"

    except (IOError, OSError):
        return "unknown"


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _count_lines_fast(path: Path, max_lines: int = 1_000_000) -> Optional[int]:
    """Quickly count lines in a file without reading entire content."""
    try:
        count = 0
        with open(path, "rb") as f:
            for _ in f:
                count += 1
                if count > max_lines:
                    return None  # Too many lines to count
        return count
    except (IOError, OSError):
        return None


def _detect_encoding(path: Path) -> Optional[str]:
    """Detect file encoding by sampling content."""
    try:
        with open(path, "rb") as f:
            sample = f.read(4096)

        # Check BOM markers
        if sample.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        elif sample.startswith(b"\xff\xfe"):
            return "utf-16-le"
        elif sample.startswith(b"\xfe\xff"):
            return "utf-16-be"

        # Try common encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                sample.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue

        return None
    except (IOError, OSError):
        return None


# =============================================================================
# File Information
# =============================================================================

def get_file_info(path: str | Path) -> FileInfo:
    """
    Get detailed file information without reading full content.

    Args:
        path: Path to the file

    Returns:
        FileInfo dataclass with file metadata
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    stat = path.stat()
    content_type = detect_content_type(path)
    is_binary = content_type == "binary"

    return FileInfo(
        path=path.absolute(),
        size_bytes=stat.st_size,
        size_human=_format_size(stat.st_size),
        line_count=None if is_binary else _count_lines_fast(path),
        content_type=content_type,
        extension=path.suffix.lower(),
        modified_time=datetime.fromtimestamp(stat.st_mtime),
        is_binary=is_binary,
        encoding=None if is_binary else _detect_encoding(path),
    )


def preview_file(
    path: str | Path,
    lines: int = 50,
    from_end: bool = False
) -> str:
    """
    Safely preview a file's content without loading the entire file.

    Args:
        path: Path to the file
        lines: Number of lines to preview (default 50)
        from_end: If True, show last N lines instead of first N

    Returns:
        String containing the preview content
    """
    path = Path(path)
    info = get_file_info(path)

    if info.is_binary:
        return f"[Binary file: {info.size_human}]"

    try:
        encoding = info.encoding or "utf-8"

        if from_end:
            # Read last N lines
            with open(path, "r", encoding=encoding, errors="replace") as f:
                all_lines = f.readlines()
                preview_lines = all_lines[-lines:]
        else:
            # Read first N lines
            preview_lines = []
            with open(path, "r", encoding=encoding, errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    preview_lines.append(line)

        content = "".join(preview_lines)
        total_lines = info.line_count or "unknown"

        header = f"# Preview of {path.name} ({info.size_human}, {total_lines} lines)\n"
        header += f"# Showing {'last' if from_end else 'first'} {len(preview_lines)} lines\n"
        header += "# " + "=" * 60 + "\n\n"

        return header + content

    except (IOError, OSError) as e:
        return f"[Error reading file: {e}]"


# =============================================================================
# Chunking Logic
# =============================================================================

def _generate_chunk_id(source_path: Path, start_line: int, end_line: int) -> str:
    """Generate a unique chunk ID based on source and position."""
    content = f"{source_path.absolute()}:{start_line}-{end_line}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _find_code_boundary(lines: List[str], target_idx: int, direction: int = 1) -> int:
    """
    Find a good code boundary (function/class definition) near target index.

    Args:
        lines: List of lines
        target_idx: Target line index
        direction: 1 for forward, -1 for backward

    Returns:
        Adjusted line index at a code boundary
    """
    # Patterns that indicate good split points in code
    boundary_patterns = [
        r"^\s*def\s+\w+",          # Python function
        r"^\s*class\s+\w+",        # Python class
        r"^\s*function\s+\w+",     # JavaScript function
        r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",  # Arrow function
        r"^\s*(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\(",  # Java/C# method
        r"^\s*#\s*region",         # Region markers
        r"^---+$",                 # Section dividers
    ]

    combined_pattern = "|".join(boundary_patterns)
    search_range = 50  # Lines to search in each direction

    start_idx = max(0, target_idx)
    end_idx = min(len(lines), target_idx + search_range * direction)

    if direction > 0:
        indices = range(start_idx, end_idx)
    else:
        indices = range(min(start_idx, end_idx), max(start_idx, end_idx))
        indices = reversed(list(indices))

    for idx in indices:
        if idx < len(lines) and re.match(combined_pattern, lines[idx], re.IGNORECASE):
            return idx

    return target_idx


def _find_document_boundary(lines: List[str], target_idx: int, direction: int = 1) -> int:
    """
    Find a good document boundary (paragraph break, heading) near target index.

    Args:
        lines: List of lines
        target_idx: Target line index
        direction: 1 for forward, -1 for backward

    Returns:
        Adjusted line index at a document boundary
    """
    # Patterns for document boundaries
    boundary_patterns = [
        r"^#{1,6}\s+",            # Markdown headings
        r"^\s*$",                 # Empty line (paragraph break)
        r"^[-=]{3,}$",            # Horizontal rules
        r"^\d+\.\s+",             # Numbered list start
        r"^[-*+]\s+",             # Bullet list start
    ]

    combined_pattern = "|".join(boundary_patterns)
    search_range = 30

    start_idx = max(0, target_idx - search_range if direction < 0 else target_idx)
    end_idx = min(len(lines), target_idx + search_range if direction > 0 else target_idx)

    for idx in range(start_idx, end_idx)[::direction]:
        if idx < len(lines) and re.match(combined_pattern, lines[idx]):
            return idx

    return target_idx


def chunk_data(
    source_path: str | Path,
    chunk_size: int = MAX_CHUNK_SIZE_CHARS,
    overlap_lines: int = CHUNK_OVERLAP_LINES,
    config: Optional[RLMConfig] = None,
) -> List[ChunkInfo]:
    """
    Split a large file into manageable chunks with intelligent boundary detection.

    Args:
        source_path: Path to the source file
        chunk_size: Maximum size of each chunk in characters
        overlap_lines: Number of overlapping lines between chunks
        config: Optional RLM configuration

    Returns:
        List of ChunkInfo objects describing each chunk
    """
    source_path = Path(source_path)
    config = config or load_config()
    config.ensure_directories()

    info = get_file_info(source_path)

    if info.is_binary:
        raise ValueError(f"Cannot chunk binary file: {source_path}")

    # Read all lines
    encoding = info.encoding or "utf-8"
    with open(source_path, "r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return []

    # Choose boundary finder based on content type
    if info.content_type == "code":
        find_boundary = _find_code_boundary
    else:
        find_boundary = _find_document_boundary

    # Calculate approximate lines per chunk
    avg_line_len = sum(len(line) for line in lines) / len(lines)
    lines_per_chunk = max(10, int(chunk_size / avg_line_len))

    chunks: List[ChunkInfo] = []
    current_start = 0

    while current_start < len(lines):
        # Calculate target end position
        target_end = min(current_start + lines_per_chunk, len(lines))

        # Find a good boundary for the end
        if target_end < len(lines):
            actual_end = find_boundary(lines, target_end, direction=1)
            # Don't let boundary adjustment make chunk too large
            if actual_end - current_start > lines_per_chunk * 1.5:
                actual_end = target_end
        else:
            actual_end = len(lines)

        # Extract chunk content
        chunk_lines = lines[current_start:actual_end]
        chunk_content = "".join(chunk_lines)

        # Generate chunk ID and path
        chunk_id = _generate_chunk_id(source_path, current_start, actual_end)
        chunk_filename = f"chunk_{chunk_id}_{source_path.stem}.txt"
        chunk_path = config.get_cache_path() / chunk_filename

        # Write chunk to cache
        with open(chunk_path, "w", encoding="utf-8") as f:
            # Add header with context
            f.write(f"# Chunk from: {source_path.name}\n")
            f.write(f"# Lines: {current_start + 1}-{actual_end} of {len(lines)}\n")
            f.write(f"# Content type: {info.content_type}\n")
            f.write("# " + "=" * 60 + "\n\n")
            f.write(chunk_content)

        chunks.append(ChunkInfo(
            chunk_id=chunk_id,
            chunk_path=chunk_path,
            source_path=source_path.absolute(),
            start_line=current_start + 1,  # 1-indexed for display
            end_line=actual_end,
            size_bytes=len(chunk_content.encode("utf-8")),
            content_type=info.content_type,
        ))

        # Move to next chunk with overlap
        current_start = max(current_start + 1, actual_end - overlap_lines)

    return chunks


def get_chunk_count_estimate(
    source_path: str | Path,
    chunk_size: int = MAX_CHUNK_SIZE_CHARS,
) -> int:
    """
    Estimate how many chunks a file will produce without actually chunking.

    Args:
        source_path: Path to the source file
        chunk_size: Maximum size of each chunk in characters

    Returns:
        Estimated number of chunks
    """
    info = get_file_info(source_path)
    if info.is_binary:
        return 0

    # Rough estimate based on file size
    return max(1, info.size_bytes // chunk_size + 1)


def cleanup_chunks(chunks: List[ChunkInfo]) -> int:
    """
    Remove chunk files from the cache.

    Args:
        chunks: List of ChunkInfo objects to clean up

    Returns:
        Number of chunks cleaned up
    """
    count = 0
    for chunk in chunks:
        try:
            if chunk.chunk_path.exists():
                chunk.chunk_path.unlink()
                count += 1
        except (IOError, OSError):
            pass
    return count
