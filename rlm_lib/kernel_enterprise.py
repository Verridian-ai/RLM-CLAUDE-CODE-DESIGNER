"""
RLM-C Enterprise Kernel Module

The enterprise-grade RLM kernel that integrates all Phase 1 components:
- Semantic Chunker for intelligent file splitting
- Knowledge Graph for cross-file context awareness
- Context Cache for persistent storage

This module is part of the Enterprise RLM-CLAUDE system designed for
large-scale codebase management (10,000+ files, multi-million LOC).
"""

import os
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading

from .config import RLMConfig, load_config, MAX_FILE_SIZE_BYTES, MAX_CHUNK_SIZE_CHARS
from .semantic_chunker import SemanticChunker, SemanticChunk, analyze_file_semantics
from .knowledge_graph import ProjectKnowledgeGraph, NodeType, QueryResult
from .context_cache import ContextCache, CacheEntryType
from .kernel import RLMKernel, KernelState
from .delegator import delegate_task, TaskType
from .aggregator import aggregate_results


class ProcessingMode(Enum):
    """Processing modes for the enterprise kernel."""
    FAST = "fast"           # Minimal context, quick responses
    BALANCED = "balanced"   # Moderate context, good quality
    THOROUGH = "thorough"   # Full context, highest quality
    EXHAUSTIVE = "exhaustive"  # Complete codebase awareness


@dataclass
class EnterpriseContext:
    """Rich context for enterprise-scale processing."""
    query: str
    relevant_files: List[Path] = field(default_factory=list)
    relevant_symbols: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    related_chunks: List[SemanticChunk] = field(default_factory=list)
    graph_context: Optional[str] = None
    cached_results: Optional[str] = None
    confidence_score: float = 0.0

    def to_prompt_context(self, max_tokens: int = 50000) -> str:
        """Convert to a context string for LLM prompts."""
        lines = ["# Enterprise Context", ""]
        if self.graph_context:
            lines.append("## Project Structure")
            lines.append(self.graph_context[:max_tokens // 4])
            lines.append("")
        if self.relevant_files:
            lines.append(f"## Relevant Files ({len(self.relevant_files)})")
            for f in self.relevant_files[:20]:
                lines.append(f"- {f}")
            lines.append("")
        if self.relevant_symbols:
            lines.append(f"## Relevant Symbols ({len(self.relevant_symbols)})")
            for s in self.relevant_symbols[:30]:
                lines.append(f"- {s}")
            lines.append("")
        if self.dependencies:
            lines.append(f"## Dependencies ({len(self.dependencies)})")
            for d in self.dependencies[:20]:
                lines.append(f"- {d}")
        return "\n".join(lines)


@dataclass
class ProcessingResult:
    """Result of enterprise kernel processing."""
    success: bool
    output: str
    context_used: EnterpriseContext
    processing_time_ms: float
    tokens_used: int = 0
    chunks_processed: int = 0
    cache_hits: int = 0
    errors: List[str] = field(default_factory=list)


class EnterpriseKernel:
    """
    Enterprise-grade RLM kernel for large codebase management.

    Integrates:
    - Semantic chunking for intelligent file splitting
    - Knowledge graph for cross-file context awareness
    - Context cache for persistent storage
    - Base RLM kernel for recursive processing
    """

    def __init__(
        self,
        project_root: Path,
        config: Optional[RLMConfig] = None,
        mode: ProcessingMode = ProcessingMode.BALANCED,
        auto_index: bool = True,
    ):
        """Initialize the enterprise kernel."""
        self.project_root = Path(project_root)
        self.config = config or load_config()
        self.mode = mode
        self._lock = threading.RLock()

        # Initialize components
        self.chunker = SemanticChunker(self.config)
        self.graph = ProjectKnowledgeGraph(self.project_root, self.config)
        self.cache = ContextCache(self.config)
        self.base_kernel = RLMKernel(self.config)

        # State
        self._initialized = False
        self._last_index_time: Optional[datetime] = None

        if auto_index:
            self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize or load the project index."""
        with self._lock:
            # Try to load existing graph
            if self.graph.load():
                self._initialized = True
                self._last_index_time = self.graph._last_build
            else:
                # Build new index
                self.rebuild_index()

    def rebuild_index(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Rebuild the project knowledge graph."""
        with self._lock:
            stats = self.graph.build_index(max_files=max_files)
            self.graph.save()
            self._initialized = True
            self._last_index_time = datetime.now()
            return stats

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the enterprise kernel."""
        graph_stats = self.graph.get_stats()
        cache_stats = self.cache.get_stats()
        return {
            "initialized": self._initialized,
            "project_root": str(self.project_root),
            "mode": self.mode.value,
            "last_index_time": self._last_index_time.isoformat() if self._last_index_time else None,
            "graph": graph_stats,
            "cache": {
                "total_entries": cache_stats.total_entries,
                "size_bytes": cache_stats.total_size_bytes,
                "hit_rate": cache_stats.hit_rate,
            },
        }

    # =========================================================================
    # Context Building
    # =========================================================================

    def build_context(self, query: str) -> EnterpriseContext:
        """Build rich context for a query using all available sources."""
        context = EnterpriseContext(query=query)

        # Check cache first
        cached = self.cache.get_query_result(query)
        if cached:
            context.cached_results = cached
            context.confidence_score = 0.8

        # Search knowledge graph
        graph_results = self.graph.search(query, max_results=50)
        if graph_results.nodes:
            context.relevant_symbols = [n.qualified_name for n in graph_results.nodes]
            context.relevant_files = list(set(
                n.file_path for n in graph_results.nodes if n.file_path
            ))

            # Get dependencies for found symbols
            for node in graph_results.nodes[:10]:
                deps = self.graph.get_dependencies(node.node_id)
                context.dependencies.extend([d.qualified_name for d in deps])

        # Build graph context string
        if context.relevant_files:
            file_contexts = []
            for file_path in context.relevant_files[:5]:
                file_ctx = self.graph.get_context_for_file(file_path)
                file_contexts.append(file_ctx)
            context.graph_context = "\n\n".join(file_contexts)

        # Calculate confidence based on available context
        context.confidence_score = self._calculate_confidence(context)
        return context

    def _calculate_confidence(self, context: EnterpriseContext) -> float:
        """Calculate confidence score for the context."""
        score = 0.0
        if context.cached_results:
            score += 0.3
        if context.relevant_files:
            score += min(0.3, len(context.relevant_files) * 0.03)
        if context.relevant_symbols:
            score += min(0.2, len(context.relevant_symbols) * 0.01)
        if context.graph_context:
            score += 0.2
        return min(1.0, score)

    # =========================================================================
    # Query Processing
    # =========================================================================

    def process_query(
        self,
        query: str,
        context_file: Optional[Path] = None,
        mode: Optional[ProcessingMode] = None,
    ) -> ProcessingResult:
        """Process a query with enterprise-grade context awareness."""
        start_time = time.time()
        mode = mode or self.mode
        errors: List[str] = []

        # Build context
        context = self.build_context(query)

        # If a specific file is provided, add its context
        if context_file:
            try:
                chunks = self.chunker.chunk_file(context_file)
                context.related_chunks = chunks
            except Exception as e:
                errors.append(f"Failed to chunk {context_file}: {e}")

        # Prepare prompt with context
        prompt_context = context.to_prompt_context(
            max_tokens=self._get_max_context_tokens(mode)
        )

        # Process with base kernel
        try:
            result = self.base_kernel.process_query(
                query=query,
                context_file=str(context_file) if context_file else None,
            )
            output = result.get("output", "")
            tokens_used = result.get("tokens_used", 0)
        except Exception as e:
            errors.append(f"Kernel processing failed: {e}")
            output = ""
            tokens_used = 0

        # Cache the result
        if output and not errors:
            self.cache.cache_query_result(query, output)

        processing_time = (time.time() - start_time) * 1000
        return ProcessingResult(
            success=len(errors) == 0,
            output=output,
            context_used=context,
            processing_time_ms=processing_time,
            tokens_used=tokens_used,
            chunks_processed=len(context.related_chunks),
            cache_hits=1 if context.cached_results else 0,
            errors=errors,
        )

    def _get_max_context_tokens(self, mode: ProcessingMode) -> int:
        """Get maximum context tokens based on processing mode."""
        limits = {
            ProcessingMode.FAST: 10000,
            ProcessingMode.BALANCED: 50000,
            ProcessingMode.THOROUGH: 100000,
            ProcessingMode.EXHAUSTIVE: 200000,
        }
        return limits.get(mode, 50000)

    # =========================================================================
    # File Operations
    # =========================================================================

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a file with caching."""
        file_path = Path(file_path)

        # Check cache
        cached = self.cache.get_file_analysis(file_path)
        if cached:
            return cached

        # Perform analysis
        analysis = analyze_file_semantics(file_path, self.config)

        # Add graph context
        entities = self.graph.get_file_entities(file_path)
        analysis["graph_entities"] = len(entities)
        analysis["entity_names"] = [e.name for e in entities[:20]]

        # Cache result
        self.cache.cache_file_analysis(file_path, analysis)
        return analysis

    def get_file_context(self, file_path: Path, depth: int = 2) -> str:
        """Get rich context for a file including dependencies."""
        return self.graph.get_context_for_file(file_path, depth)

    def find_symbol(self, name: str, symbol_type: Optional[NodeType] = None) -> List[Dict[str, Any]]:
        """Find a symbol across the codebase."""
        nodes = self.graph.find_by_name(name, symbol_type)
        return [
            {
                "name": n.name,
                "qualified_name": n.qualified_name,
                "type": n.node_type.value,
                "file": str(n.file_path) if n.file_path else None,
                "line": n.line_start,
            }
            for n in nodes
        ]

    # =========================================================================
    # Maintenance
    # =========================================================================

    def invalidate_file(self, file_path: Path) -> None:
        """Invalidate cache and re-index a file."""
        file_path = Path(file_path)
        self.cache.invalidate_by_source(str(file_path))
        # Re-index will happen on next access

    def cleanup(self) -> Dict[str, int]:
        """Clean up stale cache entries and temporary files."""
        stale_count = self.cache.invalidate_stale()
        return {"stale_entries_removed": stale_count}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        return {
            "total_entries": stats.total_entries,
            "size_bytes": stats.total_size_bytes,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "evictions": stats.evictions,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_enterprise_kernel(
    project_root: Union[str, Path],
    mode: ProcessingMode = ProcessingMode.BALANCED,
    auto_index: bool = True,
) -> EnterpriseKernel:
    """Create an enterprise kernel for a project."""
    return EnterpriseKernel(
        project_root=Path(project_root),
        mode=mode,
        auto_index=auto_index,
    )


def quick_query(
    project_root: Union[str, Path],
    query: str,
    mode: ProcessingMode = ProcessingMode.FAST,
) -> str:
    """Quick query against a project without full initialization."""
    kernel = create_enterprise_kernel(project_root, mode=mode, auto_index=False)
    result = kernel.process_query(query)
    return result.output
