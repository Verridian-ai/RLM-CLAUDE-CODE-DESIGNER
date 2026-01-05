"""Tests for RLM-C Chunker Module"""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm_lib.chunker import (
    detect_content_type,
    get_file_info,
    preview_file,
    chunk_data,
    get_chunk_count_estimate,
    cleanup_chunks,
)
from rlm_lib.config import RLMConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file."""
    file_path = temp_dir / "sample.py"
    content = '''"""Sample Python module."""

def hello_world():
    """Print hello world."""
    print("Hello, World!")

class Greeter:
    """A simple greeter class."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        """Greet the person."""
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    hello_world()
'''
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file."""
    file_path = temp_dir / "sample.txt"
    content = """# Introduction

This is a sample document for testing the RLM-C chunker.

## Section 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

## Section 2

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.

---

## Conclusion

That's all folks!
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def large_file(temp_dir):
    """Create a large file for chunking tests."""
    file_path = temp_dir / "large.txt"
    # Create ~100KB file
    lines = [f"Line {i}: This is some sample content for testing.\n" for i in range(2000)]
    file_path.write_text("".join(lines))
    return file_path


class TestContentTypeDetection:
    """Test content type detection."""

    def test_detect_python_file(self, sample_python_file):
        """Python files should be detected as code."""
        assert detect_content_type(sample_python_file) == "code"

    def test_detect_text_file(self, sample_text_file):
        """Text files should be detected as document."""
        assert detect_content_type(sample_text_file) == "document"

    def test_detect_by_extension(self, temp_dir):
        """Files should be detected by extension."""
        js_file = temp_dir / "app.js"
        js_file.write_text("const x = 1;")
        assert detect_content_type(js_file) == "code"

        md_file = temp_dir / "readme.md"
        md_file.write_text("# Hello")
        assert detect_content_type(md_file) == "document"


class TestFileInfo:
    """Test file info retrieval."""

    def test_get_file_info_exists(self, sample_python_file):
        """File info should be retrieved for existing files."""
        info = get_file_info(sample_python_file)
        assert info.path == sample_python_file.absolute()
        assert info.size_bytes > 0
        assert info.content_type == "code"
        assert info.extension == ".py"
        assert not info.is_binary

    def test_get_file_info_line_count(self, sample_python_file):
        """File info should include line count."""
        info = get_file_info(sample_python_file)
        assert info.line_count is not None
        assert info.line_count > 0

    def test_get_file_info_not_found(self, temp_dir):
        """Non-existent files should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_file_info(temp_dir / "nonexistent.txt")

    def test_size_human_readable(self, large_file):
        """Size should be formatted in human-readable form."""
        info = get_file_info(large_file)
        assert "KB" in info.size_human or "B" in info.size_human


class TestPreview:
    """Test file preview functionality."""

    def test_preview_first_lines(self, sample_text_file):
        """Preview should show first N lines."""
        preview = preview_file(sample_text_file, lines=5)
        assert "Introduction" in preview
        assert "Preview of" in preview

    def test_preview_last_lines(self, sample_text_file):
        """Preview with from_end should show last N lines."""
        preview = preview_file(sample_text_file, lines=5, from_end=True)
        assert "last" in preview.lower()

    def test_preview_respects_line_limit(self, large_file):
        """Preview should respect line limit."""
        preview = preview_file(large_file, lines=10)
        # Should have header + limited content
        lines = preview.split("\n")
        assert len(lines) < 20  # Header + 10 content lines


class TestChunking:
    """Test file chunking functionality."""

    def test_chunk_large_file(self, large_file, temp_dir):
        """Large files should be split into chunks."""
        config = RLMConfig(
            workspace_root=temp_dir,
            max_chunk_size_tokens=500,  # Small chunks for testing
        )
        config.ensure_directories()

        chunks = chunk_data(large_file, chunk_size=2000, config=config)

        assert len(chunks) > 1  # Should have multiple chunks
        assert all(c.chunk_path.exists() for c in chunks)
        assert all(c.source_path == large_file.absolute() for c in chunks)

    def test_chunk_small_file(self, sample_text_file, temp_dir):
        """Small files might produce single chunk."""
        config = RLMConfig(workspace_root=temp_dir)
        config.ensure_directories()

        chunks = chunk_data(sample_text_file, chunk_size=100000, config=config)

        assert len(chunks) >= 1

    def test_chunk_has_metadata(self, large_file, temp_dir):
        """Chunks should include metadata headers."""
        config = RLMConfig(workspace_root=temp_dir)
        config.ensure_directories()

        chunks = chunk_data(large_file, chunk_size=5000, config=config)

        for chunk in chunks:
            content = chunk.chunk_path.read_text()
            assert "Chunk from:" in content
            assert "Lines:" in content

    def test_cleanup_chunks(self, large_file, temp_dir):
        """Cleanup should remove chunk files."""
        config = RLMConfig(workspace_root=temp_dir)
        config.ensure_directories()

        chunks = chunk_data(large_file, chunk_size=5000, config=config)
        assert all(c.chunk_path.exists() for c in chunks)

        count = cleanup_chunks(chunks)

        assert count == len(chunks)
        assert not any(c.chunk_path.exists() for c in chunks)


class TestChunkEstimate:
    """Test chunk count estimation."""

    def test_estimate_large_file(self, large_file):
        """Estimate should give reasonable chunk count."""
        estimate = get_chunk_count_estimate(large_file, chunk_size=10000)
        assert estimate > 0

    def test_estimate_small_file(self, sample_text_file):
        """Small files should estimate 1 chunk."""
        estimate = get_chunk_count_estimate(sample_text_file, chunk_size=100000)
        assert estimate == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
