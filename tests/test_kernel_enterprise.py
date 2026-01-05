"""Unit tests for the Enterprise Kernel module."""

import pytest
import tempfile
import shutil
from pathlib import Path

from rlm_lib.kernel_enterprise import (
    EnterpriseKernel,
    EnterpriseContext,
    ProcessingMode,
    ProcessingResult,
    create_enterprise_kernel,
)


@pytest.fixture
def temp_project():
    """Create a temporary project structure."""
    temp_dir = Path(tempfile.mkdtemp())
    
    (temp_dir / "main.py").write_text('''
from utils import helper

def main():
    return helper()
''')
    
    (temp_dir / "utils.py").write_text('''
def helper():
    return "Hello, World!"
''')
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def enterprise_kernel(temp_project):
    """Create an enterprise kernel for testing."""
    return EnterpriseKernel(
        project_root=temp_project,
        mode=ProcessingMode.FAST,
        auto_index=True,
    )


class TestEnterpriseKernel:
    def test_initialization(self, temp_project):
        kernel = EnterpriseKernel(
            project_root=temp_project,
            mode=ProcessingMode.BALANCED,
            auto_index=False,
        )
        assert kernel.project_root == temp_project
        assert kernel.mode == ProcessingMode.BALANCED

    def test_get_status(self, enterprise_kernel):
        status = enterprise_kernel.get_status()
        assert "initialized" in status
        assert "project_root" in status
        assert "mode" in status
        assert "graph" in status
        assert "cache" in status

    def test_rebuild_index(self, enterprise_kernel):
        stats = enterprise_kernel.rebuild_index()
        assert "files_indexed" in stats
        assert stats["files_indexed"] >= 2

    def test_build_context(self, enterprise_kernel):
        context = enterprise_kernel.build_context("What does helper do?")
        assert isinstance(context, EnterpriseContext)
        assert context.query == "What does helper do?"

    def test_find_symbol(self, enterprise_kernel):
        symbols = enterprise_kernel.find_symbol("helper")
        assert len(symbols) >= 1
        assert any(s["name"] == "helper" for s in symbols)

    def test_get_cache_stats(self, enterprise_kernel):
        stats = enterprise_kernel.get_cache_stats()
        assert "total_entries" in stats
        assert "hit_rate" in stats

    def test_cleanup(self, enterprise_kernel):
        result = enterprise_kernel.cleanup()
        assert "stale_entries_removed" in result


class TestEnterpriseContext:
    def test_context_creation(self):
        context = EnterpriseContext(query="Test query")
        assert context.query == "Test query"
        assert context.relevant_files == []
        assert context.relevant_symbols == []

    def test_to_prompt_context(self):
        context = EnterpriseContext(
            query="Test",
            relevant_files=["file1.py", "file2.py"],
            relevant_symbols=["func1", "func2"],
        )
        prompt = context.to_prompt_context(max_tokens=1000)
        assert isinstance(prompt, str)


class TestProcessingMode:
    def test_modes_exist(self):
        assert ProcessingMode.FAST is not None
        assert ProcessingMode.BALANCED is not None
        assert ProcessingMode.THOROUGH is not None
        assert ProcessingMode.EXHAUSTIVE is not None

    def test_mode_values(self):
        assert ProcessingMode.FAST.value == "fast"
        assert ProcessingMode.BALANCED.value == "balanced"
        assert ProcessingMode.THOROUGH.value == "thorough"
        assert ProcessingMode.EXHAUSTIVE.value == "exhaustive"


class TestProcessingResult:
    def test_result_creation(self):
        context = EnterpriseContext(query="Test")
        result = ProcessingResult(
            success=True,
            output="Result output",
            context_used=context,
            processing_time_ms=100.5,
            tokens_used=500,
            chunks_processed=3,
            cache_hits=1,
            errors=[],
        )
        assert result.success is True
        assert result.output == "Result output"
        assert result.processing_time_ms == 100.5


class TestConvenienceFunctions:
    def test_create_enterprise_kernel(self, temp_project):
        kernel = create_enterprise_kernel(
            project_root=temp_project,
            mode=ProcessingMode.FAST,
            auto_index=False,
        )
        assert isinstance(kernel, EnterpriseKernel)
        assert kernel.mode == ProcessingMode.FAST

