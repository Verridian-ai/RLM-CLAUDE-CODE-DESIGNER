"""
Tool Search Index for RLM-CLAUDE.

Implements Tool Search capability to avoid context pollution.
Tools are indexed and searched semantically, not all loaded into context.
This prevents plugin bloat and keeps context focused.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import re


class ToolCategory(str, Enum):
    """Categories of tools."""
    FILE_SYSTEM = "file_system"
    CODE_ANALYSIS = "code_analysis"
    TESTING = "testing"
    VERSION_CONTROL = "version_control"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    DATABASE = "database"
    API = "api"
    UI = "ui"
    GENERAL = "general"


@dataclass
class ToolDefinition:
    """Definition of a tool in the search index."""
    name: str
    description: str
    category: ToolCategory
    keywords: Set[str] = field(default_factory=set)
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    requires_plugins: List[str] = field(default_factory=list)
    enabled: bool = True

    def matches_query(self, query: str) -> float:
        """
        Calculate match score for a query.

        Args:
            query: Search query string.

        Returns:
            Match score between 0 and 1.
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        score = 0.0

        # Name match (highest weight)
        if self.name.lower() in query_lower:
            score += 0.4

        # Description match
        desc_words = set(re.findall(r'\w+', self.description.lower()))
        desc_overlap = len(query_words & desc_words) / max(len(query_words), 1)
        score += desc_overlap * 0.3

        # Keyword match
        keyword_overlap = len(query_words & {k.lower() for k in self.keywords})
        score += min(keyword_overlap * 0.1, 0.3)

        return min(score, 1.0)


@dataclass
class SearchResult:
    """Result from a tool search."""
    tool: ToolDefinition
    score: float
    matched_keywords: Set[str] = field(default_factory=set)


class ToolSearchIndex:
    """
    Implements Tool Search capability to avoid context pollution.
    Tools are indexed and searched, not all loaded into context.
    """

    def __init__(self):
        """Initialize the tool search index."""
        self._index: Dict[str, ToolDefinition] = {}
        self._category_index: Dict[ToolCategory, List[str]] = {}
        self._keyword_index: Dict[str, Set[str]] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a tool in the searchable index.

        Args:
            tool: Tool definition to register.
        """
        self._index[tool.name] = tool

        # Update category index
        if tool.category not in self._category_index:
            self._category_index[tool.category] = []
        self._category_index[tool.category].append(tool.name)

        # Update keyword index
        for keyword in tool.keywords:
            kw_lower = keyword.lower()
            if kw_lower not in self._keyword_index:
                self._keyword_index[kw_lower] = set()
            self._keyword_index[kw_lower].add(tool.name)

    def unregister_tool(self, name: str) -> bool:
        """
        Remove a tool from the index.

        Args:
            name: Name of tool to remove.

        Returns:
            True if tool was removed.
        """
        if name not in self._index:
            return False

        tool = self._index.pop(name)

        # Clean up category index
        if tool.category in self._category_index:
            self._category_index[tool.category] = [
                n for n in self._category_index[tool.category] if n != name
            ]

        # Clean up keyword index
        for keyword in tool.keywords:
            kw_lower = keyword.lower()
            if kw_lower in self._keyword_index:
                self._keyword_index[kw_lower].discard(name)

        return True

    def search(
        self,
        query: str,
        category: Optional[ToolCategory] = None,
        limit: int = 5,
        min_score: float = 0.1,
    ) -> List[SearchResult]:
        """
        Search for tools matching a query.

        Args:
            query: Search query string.
            category: Optional category filter.
            limit: Maximum results to return.
            min_score: Minimum match score.

        Returns:
            List of SearchResult sorted by score.
        """
        results: List[SearchResult] = []

        # Get candidate tools
        if category:
            candidates = self._category_index.get(category, [])
        else:
            candidates = list(self._index.keys())

        # Score each candidate
        for name in candidates:
            tool = self._index.get(name)
            if not tool or not tool.enabled:
                continue

            score = tool.matches_query(query)
            if score >= min_score:
                # Find matched keywords
                query_words = set(re.findall(r'\w+', query.lower()))
                matched = query_words & {k.lower() for k in tool.keywords}

                results.append(SearchResult(
                    tool=tool,
                    score=score,
                    matched_keywords=matched,
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    def get_tools_for_task(
        self,
        task_description: str,
        max_tools: int = 3,
    ) -> List[ToolDefinition]:
        """
        Get relevant tools for a task description.

        Args:
            task_description: Description of the task.
            max_tools: Maximum tools to return.

        Returns:
            List of relevant ToolDefinition objects.
        """
        results = self.search(task_description, limit=max_tools)
        return [r.tool for r in results]

    def get_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all tools in a category."""
        names = self._category_index.get(category, [])
        return [self._index[n] for n in names if n in self._index]

    def get_by_name(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._index.get(name)

    def list_all(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return list(self._index.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_tools": len(self._index),
            "categories": {
                cat.value: len(names)
                for cat, names in self._category_index.items()
            },
            "total_keywords": len(self._keyword_index),
        }


def create_default_tool_index() -> ToolSearchIndex:
    """
    Create a tool index with default RLM tools.

    Returns:
        Populated ToolSearchIndex.
    """
    index = ToolSearchIndex()

    # File system tools
    index.register_tool(ToolDefinition(
        name="view",
        description="View file contents or directory structure",
        category=ToolCategory.FILE_SYSTEM,
        keywords={"read", "file", "directory", "cat", "ls"},
    ))

    index.register_tool(ToolDefinition(
        name="str-replace-editor",
        description="Edit files by replacing text",
        category=ToolCategory.FILE_SYSTEM,
        keywords={"edit", "replace", "modify", "update"},
    ))

    index.register_tool(ToolDefinition(
        name="save-file",
        description="Create new files",
        category=ToolCategory.FILE_SYSTEM,
        keywords={"create", "new", "write", "save"},
    ))

    # Code analysis tools
    index.register_tool(ToolDefinition(
        name="codebase-retrieval",
        description="Search codebase for relevant code snippets",
        category=ToolCategory.CODE_ANALYSIS,
        keywords={"search", "find", "code", "function", "class"},
    ))

    index.register_tool(ToolDefinition(
        name="diagnostics",
        description="Get IDE diagnostics and errors",
        category=ToolCategory.CODE_ANALYSIS,
        keywords={"error", "warning", "lint", "diagnostic"},
    ))

    # Version control tools
    index.register_tool(ToolDefinition(
        name="git-commit-retrieval",
        description="Search git commit history",
        category=ToolCategory.VERSION_CONTROL,
        keywords={"git", "commit", "history", "change"},
    ))

    return index
