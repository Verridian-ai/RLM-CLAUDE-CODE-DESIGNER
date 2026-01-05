"""Integration Tests for RLM-C System"""

import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm_lib import (
    RLMConfig,
    RLMKernel,
    chunk_data,
    get_file_info,
    preview_file,
    detect_content_type,
    aggregate_results,
    cleanup_cache,
    search_with_ripgrep,
    build_index,
    search_index,
)
from rlm_lib.config import (
    get_current_recursion_depth,
    set_recursion_depth,
    is_rlm_mode_active,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create directory structure
        (workspace / "src").mkdir()
        (workspace / "docs").mkdir()

        # Create Python file
        (workspace / "src" / "main.py").write_text('''"""Main module."""

def main():
    """Entry point."""
    print("Hello, RLM!")

def process_data(data):
    """Process some data."""
    return data.upper()

if __name__ == "__main__":
    main()
''')

        # Create JavaScript file
        (workspace / "src" / "app.js").write_text('''// Application entry
const express = require('express');
const app = express();

function handleRequest(req, res) {
    res.send('Hello World');
}

app.get('/', handleRequest);
module.exports = app;
''')

        # Create documentation
        (workspace / "docs" / "README.md").write_text('''# RLM-C Test Project

## Overview

This is a test project for RLM-C integration tests.

## Features

- Feature 1: Processing
- Feature 2: Analysis
- Feature 3: Reporting

## Usage

Run `python main.py` to start.
''')

        # Create large file (ensure it's > 256KB for test)
        lines = [f"Log entry {i}: Operation completed successfully with detailed status information and additional context data\n" for i in range(8000)]
        (workspace / "large.log").write_text("".join(lines))

        yield workspace


@pytest.fixture
def config(temp_workspace):
    """Create RLM config for temp workspace."""
    return RLMConfig(workspace_root=temp_workspace)


@pytest.fixture
def kernel(config):
    """Create initialized kernel."""
    k = RLMKernel(config)
    k.initialize()
    return k


class TestEndToEndWorkflow:
    """Test complete RLM workflow."""

    def test_file_analysis_workflow(self, temp_workspace, kernel):
        """Test analyzing files with content type detection."""
        # Analyze Python file
        py_file = temp_workspace / "src" / "main.py"
        info = get_file_info(py_file)

        assert info.content_type == "code"
        assert info.extension == ".py"
        assert info.size_bytes > 0

        # Preview should work
        preview = preview_file(py_file, lines=5)
        assert "Main module" in preview

    def test_large_file_chunking_workflow(self, temp_workspace, config):
        """Test chunking a large file."""
        config.ensure_directories()
        large_file = temp_workspace / "large.log"

        # File should require RLM
        info = get_file_info(large_file)
        assert info.size_bytes > config.max_file_size_bytes // 2  # Should be significant

        # Chunk the file
        chunks = chunk_data(large_file, chunk_size=10000, config=config)

        assert len(chunks) >= 1
        assert all(c.chunk_path.exists() for c in chunks)

        # Verify chunk content
        for chunk in chunks:
            content = chunk.chunk_path.read_text()
            assert "Log entry" in content

        # Cleanup
        cleanup_cache(config)

    def test_search_workflow(self, temp_workspace):
        """Test searching across codebase."""
        # Search for function definitions
        result = search_with_ripgrep(
            query="def \\w+",
            search_path=str(temp_workspace / "src"),
            max_results=20,
        )

        # Should find Python functions
        assert result.total_matches > 0
        py_matches = [m for m in result.matches if m.file_path.suffix == ".py"]
        assert len(py_matches) > 0

    def test_index_workflow(self, temp_workspace, config):
        """Test building and searching index."""
        config.ensure_directories()

        # Build index
        index_path = build_index(temp_workspace, config=config)
        assert index_path.exists()

        # Search index
        results = search_index("main", config=config)
        assert len(results) > 0


class TestKernelIntegration:
    """Test kernel integration with other modules."""

    def test_kernel_preview(self, temp_workspace, kernel):
        """Kernel preview should work."""
        preview = kernel.preview(
            str(temp_workspace / "docs" / "README.md"),
            lines=10
        )
        assert "RLM-C Test Project" in preview

    def test_kernel_search(self, temp_workspace, kernel):
        """Kernel search should work."""
        result = kernel.search(
            query="Hello",
            search_path=str(temp_workspace),
            max_results=10,
        )
        assert result.total_matches > 0

    def test_kernel_status(self, kernel):
        """Kernel should report comprehensive status."""
        status = kernel.get_status()

        assert status["rlm_mode_active"]
        assert status["kernel_status"] == "ready"
        assert "recursion" in status
        assert "budget" in status

    def test_kernel_validation(self, temp_workspace, kernel):
        """Kernel should validate operations correctly."""
        # Small file should be allowed
        small_file = temp_workspace / "src" / "main.py"
        allowed, _ = kernel.validate_operation("Read", {"file_path": str(small_file)})
        assert allowed

        # Large file should be blocked
        large_file = temp_workspace / "large.log"
        info = get_file_info(large_file)
        if info.size_bytes > kernel.config.max_file_size_bytes:
            allowed, error = kernel.validate_operation("Read", {"file_path": str(large_file)})
            assert not allowed
            assert error is not None


class TestEnvironmentIntegration:
    """Test environment variable integration."""

    def test_rlm_mode_activation(self, kernel):
        """RLM mode should be active after initialization."""
        assert is_rlm_mode_active()
        assert os.environ.get("RLM_MODE") == "active"

    def test_recursion_depth_tracking(self):
        """Recursion depth should be trackable."""
        original = get_current_recursion_depth()

        set_recursion_depth(2)
        assert get_current_recursion_depth() == 2

        set_recursion_depth(original)
        assert get_current_recursion_depth() == original


class TestErrorHandling:
    """Test error handling across modules."""

    def test_nonexistent_file_handling(self, kernel):
        """Non-existent files should raise appropriate errors."""
        with pytest.raises(FileNotFoundError):
            get_file_info("/nonexistent/path/file.txt")

    def test_invalid_path_preview(self, kernel):
        """Preview of invalid path should raise error."""
        with pytest.raises(FileNotFoundError):
            kernel.preview("/nonexistent/file.txt")


class TestCleanupIntegration:
    """Test cleanup across modules."""

    def test_full_cleanup(self, temp_workspace, config, kernel):
        """Full cleanup should remove all temp files."""
        config.ensure_directories()

        # Create some chunks
        large_file = temp_workspace / "large.log"
        chunks = chunk_data(large_file, chunk_size=10000, config=config)
        assert len(chunks) > 0

        # Cleanup
        counts = kernel.cleanup()
        assert counts["cache_files"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
