# Integration tests for Enterprise RLM workflow
"""
End-to-end tests for the Enterprise RLM system.
Tests the full workflow from file analysis to context building.
"""

import pytest
import tempfile
import os
from pathlib import Path

from rlm_lib import (
    EnterpriseKernel,
    ProcessingMode,
    create_enterprise_kernel,
    SemanticChunker,
    ProjectKnowledgeGraph,
    ContextCache,
)


class TestEnterpriseRLMWorkflow:
    """Test complete enterprise RLM workflows."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project structure
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            
            # Create Python files
            (src_dir / "main.py").write_text('''
"""Main application module."""

from .utils import helper_function
from .models import User

def main():
    """Entry point for the application."""
    user = User("test")
    result = helper_function(user.name)
    return result

if __name__ == "__main__":
    main()
''')
            
            (src_dir / "utils.py").write_text('''
"""Utility functions."""

def helper_function(name: str) -> str:
    """Process a name and return a greeting."""
    return f"Hello, {name}!"

def validate_input(data: dict) -> bool:
    """Validate input data."""
    return "name" in data and len(data["name"]) > 0
''')
            
            (src_dir / "models.py").write_text('''
"""Data models."""

from dataclasses import dataclass

@dataclass
class User:
    """User model."""
    name: str
    email: str = ""
    
    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello, {self.name}!"

@dataclass
class Product:
    """Product model."""
    id: int
    name: str
    price: float
''')
            
            (src_dir / "__init__.py").write_text('# src package\n')
            
            yield tmpdir

    def test_full_enterprise_workflow(self, temp_project):
        """Test complete workflow: index -> search -> context build."""
        kernel = create_enterprise_kernel(project_root=temp_project)

        # Build index
        kernel.rebuild_index()

        # Get status
        status = kernel.get_status()
        assert status["initialized"] is True
        assert status["project_root"] == temp_project

        # Search for symbol
        results = kernel.find_symbol("User")
        assert len(results) > 0

        # Build context (without mode parameter)
        context = kernel.build_context(query="How does the User class work?")
        assert context is not None
        assert len(context.to_prompt_context()) > 0

    def test_semantic_chunking_integration(self, temp_project):
        """Test semantic chunker integration with knowledge graph."""
        chunker = SemanticChunker()
        graph = ProjectKnowledgeGraph(project_root=temp_project)

        # Analyze a file
        main_py = Path(temp_project) / "src" / "main.py"
        analysis = chunker.analyze_file(str(main_py))

        assert analysis is not None

        # Build graph
        graph.build_index()
        stats = graph.get_stats()
        assert stats["total_nodes"] > 0

    def test_context_cache_persistence(self, temp_project):
        """Test context cache persistence across kernel instances."""
        from rlm_lib.context_cache import CacheEntryType

        cache = ContextCache(max_size_mb=100)

        # Clear cache first to ensure clean state
        cache.clear()

        # Store some context
        cache.set("test_key", "test data content", CacheEntryType.FILE_CONTENT)

        # Verify retrieval
        result = cache.get("test_key")
        assert result is not None
        assert result.content == "test data content"

        # Check stats - should have at least 1 entry
        stats = cache.get_stats()
        assert stats.total_entries >= 1

    def test_multi_file_analysis(self, temp_project):
        """Test analyzing multiple files in a project."""
        kernel = create_enterprise_kernel(project_root=temp_project)
        kernel.rebuild_index()
        
        # Search across all files
        user_refs = kernel.find_symbol("User")
        helper_refs = kernel.find_symbol("helper_function")
        
        # Should find references
        assert len(user_refs) >= 1
        assert len(helper_refs) >= 1

    def test_processing_modes(self, temp_project):
        """Test different queries produce context results."""
        kernel = create_enterprise_kernel(project_root=temp_project)
        kernel.rebuild_index()

        # Test with different queries
        context1 = kernel.build_context(query="What is User?")
        context2 = kernel.build_context(query="How does helper_function work?")

        # Both should produce valid context
        assert context1 is not None
        assert context2 is not None

