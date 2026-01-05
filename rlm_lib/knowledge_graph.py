"""
RLM-C Enterprise Knowledge Graph Module

Provides a project-wide knowledge graph for cross-file context awareness.
Tracks relationships between files, symbols, and dependencies to enable
comprehensive understanding of large codebases.

This module is part of the Enterprise RLM-CLAUDE system designed for
large-scale codebase management (10,000+ files, multi-million LOC).
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import threading

from .config import RLMConfig, load_config, CODE_EXTENSIONS
from .semantic_chunker import (
    SemanticChunker,
    SemanticBoundary,
    SemanticBoundaryType,
    PythonSemanticStrategy,
    TypeScriptSemanticStrategy,
)


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    PACKAGE = "package"


class EdgeType(Enum):
    """Types of edges (relationships) in the knowledge graph."""
    IMPORTS = "imports"
    EXPORTS = "exports"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    USES = "uses"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    DEFINED_IN = "defined_in"


@dataclass
class GraphNode:
    """A node in the knowledge graph representing a code entity."""
    node_id: str
    node_type: NodeType
    name: str
    qualified_name: str  # Full path including module/class
    file_path: Optional[Path] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_modified: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_path": str(self.file_path) if self.file_path else None,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "docstring": self.docstring,
            "metadata": self.metadata,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            qualified_name=data["qualified_name"],
            file_path=Path(data["file_path"]) if data.get("file_path") else None,
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            docstring=data.get("docstring"),
            metadata=data.get("metadata", {}),
            last_modified=datetime.fromisoformat(data["last_modified"]) if data.get("last_modified") else None,
        )


@dataclass
class GraphEdge:
    """An edge in the knowledge graph representing a relationship."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QueryResult:
    """Result of a knowledge graph query."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    query: str
    execution_time_ms: float



class ProjectKnowledgeGraph:
    """
    Enterprise-grade knowledge graph for project-wide context awareness.

    Indexes all code entities and their relationships to enable:
    - Cross-file dependency tracking
    - Symbol resolution and lookup
    - Impact analysis for changes
    - Semantic search across the codebase
    """

    def __init__(self, project_root: Path, config: Optional[RLMConfig] = None):
        """Initialize the knowledge graph for a project."""
        self.project_root = Path(project_root)
        self.config = config or load_config()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._adjacency: Dict[str, List[str]] = defaultdict(list)  # node_id -> connected node_ids
        self._reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        self._file_index: Dict[str, List[str]] = defaultdict(list)  # file_path -> node_ids
        self._name_index: Dict[str, List[str]] = defaultdict(list)  # name -> node_ids
        self._type_index: Dict[NodeType, List[str]] = defaultdict(list)
        self._chunker = SemanticChunker(config)
        self._lock = threading.RLock()
        self._indexed_files: Set[str] = set()
        self._last_build: Optional[datetime] = None

    def build_index(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build the knowledge graph by indexing all project files.

        Args:
            include_patterns: Glob patterns to include (default: code files)
            exclude_patterns: Glob patterns to exclude
            max_files: Maximum number of files to index

        Returns:
            Statistics about the indexing operation
        """
        start_time = datetime.now()
        stats = {"files_indexed": 0, "nodes_created": 0, "edges_created": 0, "errors": []}

        if include_patterns is None:
            include_patterns = [f"**/*{ext}" for ext in CODE_EXTENSIONS]

        if exclude_patterns is None:
            exclude_patterns = [
                "**/node_modules/**", "**/.git/**", "**/__pycache__/**",
                "**/venv/**", "**/.venv/**", "**/dist/**", "**/build/**",
            ]

        files_to_index: List[Path] = []
        for pattern in include_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    # Check exclusions
                    excluded = False
                    for excl in exclude_patterns:
                        if file_path.match(excl):
                            excluded = True
                            break
                    if not excluded:
                        files_to_index.append(file_path)

        if max_files:
            files_to_index = files_to_index[:max_files]

        for file_path in files_to_index:
            try:
                file_stats = self._index_file(file_path)
                stats["files_indexed"] += 1
                stats["nodes_created"] += file_stats.get("nodes", 0)
                stats["edges_created"] += file_stats.get("edges", 0)
            except Exception as e:
                stats["errors"].append({"file": str(file_path), "error": str(e)})

        self._last_build = datetime.now()
        stats["build_time_seconds"] = (self._last_build - start_time).total_seconds()
        return stats

    def _index_file(self, file_path: Path) -> Dict[str, int]:
        """Index a single file and add its entities to the graph."""
        with self._lock:
            rel_path = file_path.relative_to(self.project_root)
            file_key = str(rel_path)

            if file_key in self._indexed_files:
                self._remove_file_nodes(file_key)

            stats = {"nodes": 0, "edges": 0}

            # Create file node
            file_node = self._create_node(
                node_type=NodeType.FILE,
                name=file_path.name,
                qualified_name=file_key,
                file_path=file_path,
            )
            stats["nodes"] += 1

            # Analyze file semantics
            analysis = self._chunker.analyze_file(file_path)

            if analysis.get("semantic_support"):
                # Read and parse file
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                strategy = self._chunker._get_strategy(file_path)
                if strategy:
                    boundaries = strategy.extract_boundaries(content, file_path)
                    imports = strategy.extract_imports(content)

                    # Create nodes for each boundary
                    for boundary in boundaries:
                        entity_node = self._create_node(
                            node_type=self._boundary_to_node_type(boundary.boundary_type),
                            name=boundary.name,
                            qualified_name=f"{file_key}::{boundary.name}",
                            file_path=file_path,
                            line_start=boundary.start_line,
                            line_end=boundary.end_line,
                            docstring=boundary.docstring,
                        )
                        stats["nodes"] += 1

                        # Create CONTAINS edge from file to entity
                        self._add_edge(file_node.node_id, entity_node.node_id, EdgeType.CONTAINS)
                        stats["edges"] += 1

                    # Create IMPORTS edges
                    for imp in imports:
                        import_node = self._get_or_create_import_node(imp)
                        self._add_edge(file_node.node_id, import_node.node_id, EdgeType.IMPORTS)
                        stats["edges"] += 1

            self._indexed_files.add(file_key)
            return stats

    def _boundary_to_node_type(self, btype: SemanticBoundaryType) -> NodeType:
        """Convert semantic boundary type to node type."""
        mapping = {
            SemanticBoundaryType.CLASS: NodeType.CLASS,
            SemanticBoundaryType.FUNCTION: NodeType.FUNCTION,
            SemanticBoundaryType.METHOD: NodeType.METHOD,
            SemanticBoundaryType.MODULE: NodeType.MODULE,
        }
        return mapping.get(btype, NodeType.VARIABLE)

    def _create_node(self, node_type: NodeType, name: str, qualified_name: str, **kwargs) -> GraphNode:
        """Create and register a new node."""
        node_id = hashlib.md5(qualified_name.encode()).hexdigest()[:16]
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            qualified_name=qualified_name,
            last_modified=datetime.now(),
            **kwargs
        )
        self._nodes[node_id] = node
        self._name_index[name].append(node_id)
        self._type_index[node_type].append(node_id)
        if node.file_path:
            self._file_index[str(node.file_path)].append(node_id)
        return node

    def _get_or_create_import_node(self, import_name: str) -> GraphNode:
        """Get or create a node for an import."""
        qualified_name = f"import::{import_name}"
        node_id = hashlib.md5(qualified_name.encode()).hexdigest()[:16]
        if node_id in self._nodes:
            return self._nodes[node_id]
        return self._create_node(
            node_type=NodeType.IMPORT,
            name=import_name,
            qualified_name=qualified_name,
        )

    def _add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, **metadata) -> GraphEdge:
        """Add an edge between two nodes."""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            metadata=metadata,
        )
        self._edges.append(edge)
        self._adjacency[source_id].append(target_id)
        self._reverse_adjacency[target_id].append(source_id)
        return edge

    def _remove_file_nodes(self, file_key: str) -> None:
        """Remove all nodes associated with a file."""
        nodes_to_remove = [
            nid for nid, node in self._nodes.items()
            if node.file_path and str(node.file_path).endswith(file_key)
        ]
        for node_id in nodes_to_remove:
            node = self._nodes.pop(node_id, None)
            if node:
                self._name_index[node.name] = [
                    nid for nid in self._name_index[node.name] if nid != node_id
                ]
                self._type_index[node.node_type] = [
                    nid for nid in self._type_index[node.node_type] if nid != node_id
                ]
        self._edges = [e for e in self._edges if e.source_id not in nodes_to_remove and e.target_id not in nodes_to_remove]

    # =========================================================================
    # Query Methods
    # =========================================================================

    def find_by_name(self, name: str, node_type: Optional[NodeType] = None) -> List[GraphNode]:
        """Find nodes by name, optionally filtered by type."""
        node_ids = self._name_index.get(name, [])
        nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]
        if node_type:
            nodes = [n for n in nodes if n.node_type == node_type]
        return nodes

    def find_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Find all nodes of a specific type."""
        node_ids = self._type_index.get(node_type, [])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_file_entities(self, file_path: Path) -> List[GraphNode]:
        """Get all entities defined in a file."""
        node_ids = self._file_index.get(str(file_path), [])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_dependencies(self, node_id: str) -> List[GraphNode]:
        """Get all nodes that a given node depends on."""
        target_ids = self._adjacency.get(node_id, [])
        return [self._nodes[tid] for tid in target_ids if tid in self._nodes]

    def get_dependents(self, node_id: str) -> List[GraphNode]:
        """Get all nodes that depend on a given node."""
        source_ids = self._reverse_adjacency.get(node_id, [])
        return [self._nodes[sid] for sid in source_ids if sid in self._nodes]

    def search(self, query: str, max_results: int = 50) -> QueryResult:
        """Search the knowledge graph for matching entities."""
        import time
        start = time.time()
        query_lower = query.lower()
        matching_nodes: List[GraphNode] = []
        for node in self._nodes.values():
            if query_lower in node.name.lower() or query_lower in node.qualified_name.lower():
                matching_nodes.append(node)
                if len(matching_nodes) >= max_results:
                    break
        matching_edges = [
            e for e in self._edges
            if e.source_id in {n.node_id for n in matching_nodes}
            or e.target_id in {n.node_id for n in matching_nodes}
        ]
        return QueryResult(
            nodes=matching_nodes,
            edges=matching_edges[:100],
            query=query,
            execution_time_ms=(time.time() - start) * 1000,
        )

    def get_context_for_file(self, file_path: Path, depth: int = 2) -> str:
        """Get rich context for a file including its dependencies."""
        lines = [f"# Context for: {file_path.name}", ""]
        entities = self.get_file_entities(file_path)
        lines.append(f"## Entities ({len(entities)})")
        for entity in entities[:20]:
            lines.append(f"- {entity.node_type.value}: {entity.name}")
        file_node_ids = [e.node_id for e in entities]
        imports = []
        for nid in file_node_ids:
            for edge in self._edges:
                if edge.source_id == nid and edge.edge_type == EdgeType.IMPORTS:
                    target = self._nodes.get(edge.target_id)
                    if target:
                        imports.append(target.name)
        if imports:
            lines.append(f"\n## Imports ({len(imports)})")
            for imp in imports[:30]:
                lines.append(f"- {imp}")
        return "\n".join(lines)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the knowledge graph to disk."""
        if path is None:
            path = self.config.get_cache_path() / "knowledge_graph.json"
        data = {
            "project_root": str(self.project_root),
            "last_build": self._last_build.isoformat() if self._last_build else None,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "indexed_files": list(self._indexed_files),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def load(self, path: Optional[Path] = None) -> bool:
        """Load the knowledge graph from disk."""
        if path is None:
            path = self.config.get_cache_path() / "knowledge_graph.json"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._nodes = {n["node_id"]: GraphNode.from_dict(n) for n in data.get("nodes", [])}
        self._edges = [GraphEdge.from_dict(e) for e in data.get("edges", [])]
        self._indexed_files = set(data.get("indexed_files", []))
        self._last_build = datetime.fromisoformat(data["last_build"]) if data.get("last_build") else None
        self._rebuild_indices()
        return True

    def _rebuild_indices(self) -> None:
        """Rebuild internal indices from nodes and edges."""
        self._adjacency.clear()
        self._reverse_adjacency.clear()
        self._file_index.clear()
        self._name_index.clear()
        self._type_index.clear()
        for node in self._nodes.values():
            self._name_index[node.name].append(node.node_id)
            self._type_index[node.node_type].append(node.node_id)
            if node.file_path:
                self._file_index[str(node.file_path)].append(node.node_id)
        for edge in self._edges:
            self._adjacency[edge.source_id].append(edge.target_id)
            self._reverse_adjacency[edge.target_id].append(edge.source_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "indexed_files": len(self._indexed_files),
            "last_build": self._last_build.isoformat() if self._last_build else None,
            "nodes_by_type": {
                nt.value: len(ids) for nt, ids in self._type_index.items()
            },
        }
