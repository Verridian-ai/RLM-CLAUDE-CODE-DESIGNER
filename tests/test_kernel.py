"""Tests for RLM-C Kernel Module"""

import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm_lib.kernel import (
    RLMKernel,
    KernelStatus,
    KernelState,
    get_kernel,
    init_kernel,
)
from rlm_lib.config import RLMConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def kernel(temp_dir):
    """Create a kernel with temp workspace."""
    config = RLMConfig(workspace_root=temp_dir)
    k = RLMKernel(config)
    k.initialize()
    return k


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World!\nThis is a test file.\n")
    return file_path


@pytest.fixture
def large_file(temp_dir):
    """Create a large file for testing."""
    file_path = temp_dir / "large.txt"
    # Create ~1MB file (over default threshold)
    lines = [f"Line {i}: " + "x" * 100 + "\n" for i in range(10000)]
    file_path.write_text("".join(lines))
    return file_path


class TestKernelState:
    """Test KernelState dataclass."""

    def test_default_state(self):
        """Default state should be uninitialized."""
        state = KernelState()
        assert state.status == KernelStatus.UNINITIALIZED
        assert state.context_file is None
        assert state.chunks_created == 0

    def test_state_serialization(self, temp_dir):
        """State should serialize and deserialize correctly."""
        state = KernelState(
            status=KernelStatus.READY,
            context_file="test.txt",
            chunks_created=5,
            session_id="abc123",
        )

        state_file = temp_dir / "state.json"
        state.save(state_file)

        loaded = KernelState.load(state_file)
        assert loaded.status == KernelStatus.READY
        assert loaded.context_file == "test.txt"
        assert loaded.chunks_created == 5
        assert loaded.session_id == "abc123"


class TestKernelInitialization:
    """Test kernel initialization."""

    def test_kernel_creation(self, temp_dir):
        """Kernel should be creatable with config."""
        # Clean up any existing state file to ensure fresh start
        state_file = Path(".cache/rlm_state.json")
        if state_file.exists():
            state_file.unlink()

        config = RLMConfig(workspace_root=temp_dir)
        kernel = RLMKernel(config)
        assert kernel._state.status == KernelStatus.UNINITIALIZED

    def test_kernel_initialize(self, temp_dir):
        """Initialize should set up the kernel."""
        config = RLMConfig(workspace_root=temp_dir)
        kernel = RLMKernel(config)
        kernel.initialize()

        assert kernel._state.status == KernelStatus.READY
        assert kernel._state.session_id != ""
        assert os.environ.get("RLM_MODE") == "active"

    def test_initialize_with_context_file(self, temp_dir, sample_file):
        """Initialize should accept context file."""
        config = RLMConfig(workspace_root=temp_dir)
        kernel = RLMKernel(config)
        kernel.initialize(context_file=str(sample_file))

        assert kernel._state.context_file == str(sample_file)


class TestKernelStatus:
    """Test kernel status reporting."""

    def test_get_status(self, kernel):
        """Status should return comprehensive info."""
        status = kernel.get_status()

        assert "kernel_status" in status
        assert "session_id" in status
        assert "recursion" in status
        assert "budget" in status
        assert "statistics" in status

    def test_status_after_operations(self, kernel, sample_file):
        """Status should reflect operations."""
        status = kernel.get_status()
        assert status["statistics"]["chunks_created"] == 0


class TestValidation:
    """Test operation validation."""

    def test_validate_small_file_read(self, kernel, sample_file):
        """Small file reads should be allowed."""
        allowed, error = kernel.validate_operation(
            "Read",
            {"file_path": str(sample_file)}
        )
        assert allowed
        assert error is None

    def test_validate_large_file_read(self, kernel, large_file):
        """Large file reads should be blocked."""
        allowed, error = kernel.validate_operation(
            "Read",
            {"file_path": str(large_file)}
        )
        assert not allowed
        assert "too large" in error.lower() or "rlm" in error.lower()

    def test_validate_nonexistent_file(self, kernel, temp_dir):
        """Nonexistent files should be allowed (tool handles error)."""
        allowed, error = kernel.validate_operation(
            "Read",
            {"file_path": str(temp_dir / "nonexistent.txt")}
        )
        assert allowed


class TestRequiresRLM:
    """Test requires_rlm check."""

    def test_small_file_not_requires_rlm(self, kernel, sample_file):
        """Small files should not require RLM."""
        assert not kernel.requires_rlm(sample_file)

    def test_large_file_requires_rlm(self, kernel, large_file):
        """Large files should require RLM."""
        assert kernel.requires_rlm(large_file)


class TestPreview:
    """Test preview functionality."""

    def test_preview_file(self, kernel, sample_file):
        """Preview should return file content."""
        preview = kernel.preview(str(sample_file), lines=10)
        assert "Hello, World!" in preview
        assert "Preview of" in preview


class TestSearch:
    """Test search functionality."""

    def test_search_finds_matches(self, kernel, sample_file):
        """Search should find matching content."""
        result = kernel.search(
            query="Hello",
            search_path=str(sample_file.parent),
            max_results=10,
        )
        assert result.total_matches > 0

    def test_search_no_matches(self, kernel, sample_file):
        """Search should handle no matches."""
        result = kernel.search(
            query="NONEXISTENT_STRING_12345",
            search_path=str(sample_file.parent),
            max_results=10,
        )
        assert result.total_matches == 0


class TestCleanup:
    """Test cleanup functionality."""

    def test_cleanup_empty(self, kernel):
        """Cleanup should handle empty state."""
        counts = kernel.cleanup()
        assert "cache_files" in counts

    def test_reset(self, kernel):
        """Reset should clear all state."""
        kernel.reset()
        status = kernel.get_status()
        assert status["kernel_status"] == "uninitialized"


class TestGlobalKernel:
    """Test global kernel functions."""

    def test_get_kernel_singleton(self):
        """get_kernel should return singleton."""
        k1 = get_kernel()
        k2 = get_kernel()
        assert k1 is k2

    def test_init_kernel(self, temp_dir):
        """init_kernel should initialize global kernel."""
        # This may affect other tests, so just check it doesn't error
        kernel = init_kernel()
        assert kernel._state.status == KernelStatus.READY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
