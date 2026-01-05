"""Tests for RLM-C Configuration Module"""

import pytest
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm_lib.config import (
    MAX_FILE_SIZE_BYTES,
    MAX_CHUNK_SIZE_TOKENS,
    MAX_RECURSION_DEPTH,
    RLMConfig,
    load_config,
    get_current_recursion_depth,
    set_recursion_depth,
    is_rlm_mode_active,
)


class TestConstants:
    """Test configuration constants."""

    def test_max_file_size_is_reasonable(self):
        """Max file size should be between 100KB and 10MB."""
        assert 100_000 < MAX_FILE_SIZE_BYTES < 10_000_000

    def test_max_chunk_size_is_reasonable(self):
        """Max chunk size should be between 1k and 100k tokens."""
        assert 1_000 < MAX_CHUNK_SIZE_TOKENS < 100_000

    def test_max_recursion_depth_is_safe(self):
        """Max recursion should be between 1 and 10."""
        assert 1 <= MAX_RECURSION_DEPTH <= 10


class TestRLMConfig:
    """Test RLMConfig model."""

    def test_default_config_creation(self):
        """Default config should be valid."""
        config = RLMConfig()
        assert config.max_file_size_bytes == MAX_FILE_SIZE_BYTES
        assert config.max_chunk_size_tokens == MAX_CHUNK_SIZE_TOKENS
        assert config.max_recursion_depth == MAX_RECURSION_DEPTH

    def test_custom_config_creation(self):
        """Custom config values should be accepted."""
        config = RLMConfig(
            max_file_size_bytes=1_000_000,
            max_chunk_size_tokens=10_000,
            max_recursion_depth=5,
        )
        assert config.max_file_size_bytes == 1_000_000
        assert config.max_chunk_size_tokens == 10_000
        assert config.max_recursion_depth == 5

    def test_config_validation_min_file_size(self):
        """File size below minimum should fail validation."""
        with pytest.raises(Exception):
            RLMConfig(max_file_size_bytes=100)  # Below 1KB minimum

    def test_config_validation_max_recursion(self):
        """Recursion depth above maximum should fail validation."""
        with pytest.raises(Exception):
            RLMConfig(max_recursion_depth=100)  # Above 10 maximum

    def test_config_paths(self):
        """Config should generate correct paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLMConfig(workspace_root=Path(tmpdir))
            assert config.get_cache_path() == Path(tmpdir) / ".cache"
            assert config.get_results_path() == Path(tmpdir) / "results"

    def test_ensure_directories(self):
        """ensure_directories should create required dirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLMConfig(workspace_root=Path(tmpdir))
            config.ensure_directories()
            assert config.get_cache_path().exists()
            assert config.get_results_path().exists()


class TestEnvironmentFunctions:
    """Test environment-related functions."""

    def test_recursion_depth_default(self):
        """Default recursion depth should be 0."""
        os.environ.pop("RLM_RECURSION_DEPTH", None)
        assert get_current_recursion_depth() == 0

    def test_set_recursion_depth(self):
        """Setting recursion depth should update environment."""
        set_recursion_depth(2)
        assert get_current_recursion_depth() == 2
        set_recursion_depth(0)  # Reset

    def test_rlm_mode_inactive_by_default(self):
        """RLM mode should be inactive by default."""
        os.environ.pop("RLM_MODE", None)
        assert not is_rlm_mode_active()

    def test_rlm_mode_active_when_set(self):
        """RLM mode should be active when environment set."""
        os.environ["RLM_MODE"] = "active"
        assert is_rlm_mode_active()
        os.environ.pop("RLM_MODE", None)


class TestLoadConfig:
    """Test config loading function."""

    def test_load_config_creates_directories(self):
        """load_config should create required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(workspace_root=Path(tmpdir))
            assert config.get_cache_path().exists()
            assert config.get_results_path().exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
