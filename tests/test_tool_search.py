"""
Unit tests for the Tool Search module.
"""

import pytest
from rlm_lib.tool_search import (
    ToolSearchIndex,
    ToolDefinition,
    ToolCategory,
    SearchResult,
    create_default_tool_index,
)


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""
    
    def test_tool_creation(self):
        """Test creating a tool definition."""
        tool = ToolDefinition(
            name="test-tool",
            description="A test tool",
            category=ToolCategory.GENERAL,
        )
        assert tool.name == "test-tool"
        assert tool.enabled
    
    def test_matches_query_name(self):
        """Test matching by name."""
        tool = ToolDefinition(
            name="file-reader",
            description="Reads files",
            category=ToolCategory.FILE_SYSTEM,
        )
        score = tool.matches_query("file-reader")
        assert score > 0.3
    
    def test_matches_query_keywords(self):
        """Test matching by keywords."""
        tool = ToolDefinition(
            name="editor",
            description="Edit files",
            category=ToolCategory.FILE_SYSTEM,
            keywords={"edit", "modify", "change"},
        )
        score = tool.matches_query("modify the file")
        assert score > 0.0


class TestToolSearchIndex:
    """Tests for ToolSearchIndex."""
    
    @pytest.fixture
    def index(self):
        return ToolSearchIndex()
    
    @pytest.fixture
    def populated_index(self):
        index = ToolSearchIndex()
        index.register_tool(ToolDefinition(
            name="view",
            description="View file contents",
            category=ToolCategory.FILE_SYSTEM,
            keywords={"read", "file", "cat"},
        ))
        index.register_tool(ToolDefinition(
            name="edit",
            description="Edit files",
            category=ToolCategory.FILE_SYSTEM,
            keywords={"modify", "change", "update"},
        ))
        index.register_tool(ToolDefinition(
            name="search",
            description="Search codebase",
            category=ToolCategory.CODE_ANALYSIS,
            keywords={"find", "code", "function"},
        ))
        return index
    
    def test_register_tool(self, index):
        """Test registering a tool."""
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            category=ToolCategory.GENERAL,
        )
        index.register_tool(tool)
        assert index.get_by_name("test") == tool
    
    def test_unregister_tool(self, populated_index):
        """Test unregistering a tool."""
        assert populated_index.unregister_tool("view")
        assert populated_index.get_by_name("view") is None
    
    def test_search_by_query(self, populated_index):
        """Test searching by query."""
        results = populated_index.search("read file contents")
        assert len(results) > 0
        assert results[0].tool.name == "view"
    
    def test_search_by_category(self, populated_index):
        """Test searching by category."""
        results = populated_index.search(
            "anything",
            category=ToolCategory.CODE_ANALYSIS,
        )
        # Should only return code analysis tools
        for result in results:
            assert result.tool.category == ToolCategory.CODE_ANALYSIS
    
    def test_search_with_limit(self, populated_index):
        """Test search with limit."""
        results = populated_index.search("file", limit=1)
        assert len(results) <= 1
    
    def test_get_tools_for_task(self, populated_index):
        """Test getting tools for a task."""
        tools = populated_index.get_tools_for_task("read the file contents")
        assert len(tools) > 0
        assert any(t.name == "view" for t in tools)
    
    def test_get_by_category(self, populated_index):
        """Test getting tools by category."""
        tools = populated_index.get_by_category(ToolCategory.FILE_SYSTEM)
        assert len(tools) == 2
    
    def test_list_all(self, populated_index):
        """Test listing all tools."""
        tools = populated_index.list_all()
        assert len(tools) == 3
    
    def test_get_stats(self, populated_index):
        """Test getting statistics."""
        stats = populated_index.get_stats()
        assert stats["total_tools"] == 3
        assert "categories" in stats


class TestCreateDefaultToolIndex:
    """Tests for the factory function."""
    
    def test_creates_populated_index(self):
        """Test creating default index."""
        index = create_default_tool_index()
        assert index.get_stats()["total_tools"] > 0
    
    def test_has_file_system_tools(self):
        """Test default index has file system tools."""
        index = create_default_tool_index()
        tools = index.get_by_category(ToolCategory.FILE_SYSTEM)
        assert len(tools) > 0
    
    def test_has_code_analysis_tools(self):
        """Test default index has code analysis tools."""
        index = create_default_tool_index()
        tools = index.get_by_category(ToolCategory.CODE_ANALYSIS)
        assert len(tools) > 0

