"""Unit tests for the Context Cache module."""

import pytest
import tempfile
import time
from pathlib import Path

from rlm_lib.context_cache import (
    ContextCache,
    CacheEntry,
    CacheEntryType,
    CacheStats,
)
from rlm_lib.config import RLMConfig


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache(temp_cache_dir):
    """Create a context cache instance."""
    config = RLMConfig(cache_dir=str(temp_cache_dir))
    return ContextCache(config=config)


class TestContextCache:
    def test_set_and_get(self, cache):
        cache.set("test_key", "test_value", CacheEntryType.QUERY_RESULT)
        result = cache.get("test_key")
        assert result is not None
        assert result.content == "test_value"

    def test_get_nonexistent_key(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_delete(self, cache):
        cache.set("to_delete", "value", CacheEntryType.QUERY_RESULT)
        result = cache.get("to_delete")
        assert result is not None
        assert result.content == "value"
        cache.delete("to_delete")
        assert cache.get("to_delete") is None

    def test_invalidate_by_source(self, cache):
        cache.set("key1", "value1", CacheEntryType.FILE_ANALYSIS, source_path="file.py")
        cache.set("key2", "value2", CacheEntryType.FILE_ANALYSIS, source_path="file.py")
        cache.set("key3", "value3", CacheEntryType.FILE_ANALYSIS, source_path="other.py")

        count = cache.invalidate_by_source("file.py")
        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        result = cache.get("key3")
        assert result is not None
        assert result.content == "value3"

    def test_get_stats(self, cache):
        cache.set("key1", "value1", CacheEntryType.QUERY_RESULT)
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        assert isinstance(stats, CacheStats)
        assert stats.total_entries >= 1
        assert stats.hits >= 1
        assert stats.misses >= 1

    def test_ttl_expiration(self, cache):
        cache.set("short_lived", "value", CacheEntryType.QUERY_RESULT, ttl_seconds=1)
        result = cache.get("short_lived")
        assert result is not None
        assert result.content == "value"
        time.sleep(1.5)
        assert cache.get("short_lived") is None

    def test_set_with_metadata(self, cache):
        cache.set("with_meta", "content", CacheEntryType.QUERY_RESULT, metadata={"key": "val"})
        result = cache.get("with_meta")
        assert result is not None
        assert result.metadata == {"key": "val"}


class TestCacheEntry:
    def test_entry_creation(self):
        entry = CacheEntry(
            key="test",
            content="data",
            entry_type=CacheEntryType.QUERY_RESULT,
        )
        assert entry.key == "test"
        assert entry.content == "data"
        assert entry.entry_type == CacheEntryType.QUERY_RESULT

    def test_entry_is_expired(self):
        entry = CacheEntry(
            key="test",
            content="data",
            entry_type=CacheEntryType.QUERY_RESULT,
            ttl_seconds=0,  # Immediately expired
        )
        time.sleep(0.1)
        assert entry.is_expired() is True


class TestCacheStats:
    def test_hit_rate_calculation(self):
        stats = CacheStats(
            total_entries=10,
            total_size_bytes=1000,
            hits=80,
            misses=20,
            evictions=5,
            entries_by_type={"query_result": 10},
        )
        assert stats.hit_rate == 0.8

    def test_hit_rate_zero_requests(self):
        stats = CacheStats(
            total_entries=0,
            total_size_bytes=0,
            hits=0,
            misses=0,
            evictions=0,
            entries_by_type={},
        )
        assert stats.hit_rate == 0.0


class TestCacheEntryType:
    def test_entry_types_exist(self):
        assert CacheEntryType.QUERY_RESULT is not None
        assert CacheEntryType.FILE_ANALYSIS is not None
        assert CacheEntryType.CHUNK is not None
        assert CacheEntryType.SYMBOL_INFO is not None

