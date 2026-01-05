# Benchmark tests for RLM-CLAUDE Enterprise System
"""
Performance benchmarks to validate system performance targets.
Targets from IMPLEMENTATION_PLAN.md:
- Context build time: <30s for 10k files
- First-time correctness: >95%
"""

import pytest
import tempfile
import time
import os
from pathlib import Path

from rlm_lib import (
    EnterpriseKernel,
    SemanticChunker,
    ProjectKnowledgeGraph,
    ContextCache,
    create_enterprise_kernel,
)


class TestContextBuildPerformance:
    """Benchmark context building performance."""

    @pytest.fixture
    def large_project(self):
        """Create a simulated large project for benchmarking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 100 files to simulate a medium project
            # (Full 10k file test would be too slow for CI)
            for i in range(100):
                subdir = Path(tmpdir) / f"module_{i // 10}"
                subdir.mkdir(exist_ok=True)
                
                filepath = subdir / f"file_{i}.py"
                filepath.write_text(f'''
"""Module {i} for testing."""

class Class{i}:
    """A test class."""
    
    def __init__(self, value: int):
        self.value = value
    
    def method_{i}(self) -> int:
        """Process value."""
        return self.value * {i + 1}

def function_{i}(x: int, y: int) -> int:
    """Compute something."""
    return x + y + {i}

CONSTANT_{i} = {i * 100}
''')
            
            yield tmpdir

    def test_index_build_time(self, large_project):
        """Benchmark: Index building should complete in reasonable time."""
        kernel = create_enterprise_kernel(project_root=large_project)
        
        start_time = time.time()
        kernel.rebuild_index()
        elapsed = time.time() - start_time
        
        # 100 files should index in <5 seconds
        assert elapsed < 5.0, f"Index build took {elapsed:.2f}s, expected <5s"
        
        # Verify index was built
        status = kernel.get_status()
        assert status["initialized"] is True

    def test_context_build_time(self, large_project):
        """Benchmark: Context building should be fast."""
        kernel = create_enterprise_kernel(project_root=large_project)
        kernel.rebuild_index()
        
        start_time = time.time()
        context = kernel.build_context("Find all classes")
        elapsed = time.time() - start_time
        
        # Context build should complete in <2 seconds
        assert elapsed < 2.0, f"Context build took {elapsed:.2f}s, expected <2s"
        assert context is not None

    def test_symbol_search_time(self, large_project):
        """Benchmark: Symbol search should be fast."""
        kernel = create_enterprise_kernel(project_root=large_project)
        kernel.rebuild_index()
        
        start_time = time.time()
        results = kernel.find_symbol("Class50")
        elapsed = time.time() - start_time
        
        # Symbol search should complete in <0.5 seconds
        assert elapsed < 0.5, f"Symbol search took {elapsed:.2f}s, expected <0.5s"


class TestSemanticChunkerPerformance:
    """Benchmark semantic chunking performance."""

    def test_chunk_file_time(self, tmp_path):
        """Benchmark: Chunking a file should be fast."""
        # Create a large file
        large_file = tmp_path / "large_module.py"
        content = "# Large module\n\n"
        for i in range(100):
            content += f'''
class Class{i}:
    """Class {i}."""
    def method_{i}(self) -> int:
        return {i}

'''
        large_file.write_text(content)
        
        chunker = SemanticChunker()
        
        start_time = time.time()
        chunks = chunker.chunk_file(str(large_file))
        elapsed = time.time() - start_time
        
        # Chunking should complete in <1 second
        assert elapsed < 1.0, f"Chunking took {elapsed:.2f}s, expected <1s"
        assert len(chunks) > 0


class TestCachePerformance:
    """Benchmark cache performance."""

    def test_cache_hit_rate(self):
        """Benchmark: Cache should have high hit rate on repeated queries."""
        from rlm_lib.context_cache import CacheEntryType
        
        cache = ContextCache(max_size_mb=100)
        
        # Simulate repeated access pattern
        keys = [f"key_{i}" for i in range(50)]
        
        # Populate cache
        for key in keys:
            cache.set(key, f"content_{key}", CacheEntryType.FILE_CONTENT)
        
        # Access pattern: 80% hits on existing keys, 20% misses
        hits = 0
        misses = 0
        for i in range(100):
            if i % 5 == 0:
                # Miss
                result = cache.get(f"nonexistent_{i}")
                if result is None:
                    misses += 1
            else:
                # Hit
                key = keys[i % len(keys)]
                result = cache.get(key)
                if result is not None:
                    hits += 1
        
        # Hit rate should be >70%
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        assert hit_rate > 0.7, f"Hit rate {hit_rate:.2%} too low, expected >70%"

