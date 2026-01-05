"""
RLM-C Enterprise Semantic Chunker Module

Provides intelligent semantic-aware file chunking that respects language
boundaries (classes, functions, modules) rather than naive line/character splits.

This module is part of the Enterprise RLM-CLAUDE system designed for
large-scale codebase management (10,000+ files, multi-million LOC).
"""

import ast
import re
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .config import RLMConfig, load_config, MAX_CHUNK_SIZE_CHARS, CODE_EXTENSIONS
from .chunker import FileInfo, ChunkInfo, get_file_info


class SemanticBoundaryType(Enum):
    """Types of semantic boundaries in source code."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    BLOCK = "block"
    IMPORT = "import"
    UNKNOWN = "unknown"


@dataclass
class SemanticBoundary:
    """Represents a semantic boundary in source code."""
    boundary_type: SemanticBoundaryType
    name: str
    start_line: int
    end_line: int
    depth: int
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parent: Optional[str] = None


@dataclass
class SemanticChunk:
    """A chunk that respects semantic boundaries."""
    chunk_id: str
    chunk_path: Path
    source_path: Path
    start_line: int
    end_line: int
    size_bytes: int
    content_type: str
    boundaries: List[SemanticBoundary] = field(default_factory=list)
    primary_entity: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_chunk_info(self) -> ChunkInfo:
        """Convert to standard ChunkInfo for compatibility."""
        return ChunkInfo(
            chunk_id=self.chunk_id,
            chunk_path=self.chunk_path,
            source_path=self.source_path,
            start_line=self.start_line,
            end_line=self.end_line,
            size_bytes=self.size_bytes,
            content_type=self.content_type,
        )


class SemanticStrategy(Protocol):
    """Protocol for language-specific semantic parsing strategies."""
    def extract_boundaries(self, content: str, path: Path) -> List[SemanticBoundary]: ...
    def extract_imports(self, content: str) -> List[str]: ...
    def get_optimal_split_points(
        self, boundaries: List[SemanticBoundary], max_chunk_size: int, lines: List[str]
    ) -> List[int]: ...


class PythonSemanticStrategy:
    """Semantic parsing strategy for Python files using AST."""

    def __init__(self):
        self.language = "python"

    def extract_boundaries(self, content: str, path: Path) -> List[SemanticBoundary]:
        """Extract semantic boundaries from Python source code using AST."""
        boundaries: List[SemanticBoundary] = []
        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError:
            return self._extract_boundaries_regex(content)

        for node in ast.walk(tree):
            boundary = self._node_to_boundary(node)
            if boundary:
                boundaries.append(boundary)

        boundaries.sort(key=lambda b: b.start_line)
        return boundaries

    def _node_to_boundary(self, node: ast.AST) -> Optional[SemanticBoundary]:
        """Convert an AST node to a SemanticBoundary."""
        if isinstance(node, ast.ClassDef):
            return SemanticBoundary(
                boundary_type=SemanticBoundaryType.CLASS,
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                depth=0,
                docstring=ast.get_docstring(node),
                decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return SemanticBoundary(
                boundary_type=SemanticBoundaryType.FUNCTION,
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                depth=0,
                docstring=ast.get_docstring(node),
                decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            )
        return None

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        return "unknown"

    def _extract_boundaries_regex(self, content: str) -> List[SemanticBoundary]:
        """Fallback regex-based extraction for invalid Python syntax."""
        boundaries: List[SemanticBoundary] = []
        lines = content.split('\n')
        class_pattern = re.compile(r'^(\s*)class\s+(\w+)')
        func_pattern = re.compile(r'^(\s*)(async\s+)?def\s+(\w+)')

        for i, line in enumerate(lines, 1):
            class_match = class_pattern.match(line)
            if class_match:
                boundaries.append(SemanticBoundary(
                    boundary_type=SemanticBoundaryType.CLASS,
                    name=class_match.group(2),
                    start_line=i,
                    end_line=i,
                    depth=len(class_match.group(1)) // 4,
                ))
                continue
            func_match = func_pattern.match(line)
            if func_match:
                boundaries.append(SemanticBoundary(
                    boundary_type=SemanticBoundaryType.FUNCTION,
                    name=func_match.group(3),
                    start_line=i,
                    end_line=i,
                    depth=len(func_match.group(1)) // 4,
                ))
        return boundaries

    def extract_imports(self, content: str) -> List[str]:
        """Extract import statements from Python source code."""
        imports: List[str] = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
        except SyntaxError:
            import_pattern = re.compile(
                r'^(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)', re.MULTILINE
            )
            for match in import_pattern.finditer(content):
                from_module = match.group(1) or ""
                imported = match.group(2)
                for item in imported.split(','):
                    item = item.strip().split()[0]
                    if from_module:
                        imports.append(f"{from_module}.{item}")
                    else:
                        imports.append(item)
        return imports

    def get_optimal_split_points(
        self, boundaries: List[SemanticBoundary], max_chunk_size: int, lines: List[str]
    ) -> List[int]:
        """Determine optimal line numbers to split the file."""
        if not boundaries:
            return self._simple_split_points(lines, max_chunk_size)

        split_points: List[int] = []
        current_size = 0
        last_split = 0
        top_level = [b for b in boundaries if b.depth == 0]

        for boundary in top_level:
            chunk_lines = lines[last_split:boundary.end_line]
            chunk_size = sum(len(line) + 1 for line in chunk_lines)
            if current_size + chunk_size > max_chunk_size and current_size > 0:
                split_points.append(boundary.start_line - 1)
                last_split = boundary.start_line - 1
                current_size = chunk_size
            else:
                current_size += chunk_size
        return split_points

    def _simple_split_points(self, lines: List[str], max_chunk_size: int) -> List[int]:
        """Simple line-based split for files without clear boundaries."""
        split_points: List[int] = []
        current_size = 0
        for i, line in enumerate(lines):
            current_size += len(line) + 1
            if current_size >= max_chunk_size:
                split_points.append(i + 1)
                current_size = 0
        return split_points


class TypeScriptSemanticStrategy:
    """Semantic parsing strategy for TypeScript/JavaScript files."""

    def __init__(self):
        self.language = "typescript"

    def extract_boundaries(self, content: str, path: Path) -> List[SemanticBoundary]:
        """Extract semantic boundaries from TypeScript/JavaScript source code."""
        boundaries: List[SemanticBoundary] = []
        lines = content.split('\n')
        patterns = {
            'class': re.compile(r'^(\s*)(?:export\s+)?(?:abstract\s+)?class\s+(\w+)'),
            'function': re.compile(r'^(\s*)(?:export\s+)?(?:async\s+)?function\s+(\w+)'),
            'arrow': re.compile(r'^(\s*)(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\('),
            'interface': re.compile(r'^(\s*)(?:export\s+)?interface\s+(\w+)'),
        }

        for i, line in enumerate(lines, 1):
            for name, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    btype = SemanticBoundaryType.CLASS if name in ['class', 'interface'] \
                        else SemanticBoundaryType.FUNCTION
                    boundaries.append(SemanticBoundary(
                        boundary_type=btype,
                        name=match.group(2),
                        start_line=i,
                        end_line=i,
                        depth=len(match.group(1)) // 2,
                    ))
                    break
        return boundaries

    def extract_imports(self, content: str) -> List[str]:
        """Extract import statements from TypeScript/JavaScript source code."""
        imports: List[str] = []
        patterns = [
            re.compile(r"import\s+{[^}]+}\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"import\s+\w+\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"import\s+\*\s+as\s+\w+\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        ]
        for pattern in patterns:
            for match in pattern.finditer(content):
                imports.append(match.group(1))
        return list(set(imports))

    def get_optimal_split_points(
        self, boundaries: List[SemanticBoundary], max_chunk_size: int, lines: List[str]
    ) -> List[int]:
        """Determine optimal line numbers to split TypeScript/JavaScript files."""
        if not boundaries:
            return self._simple_split_points(lines, max_chunk_size)
        split_points: List[int] = []
        current_size = 0
        last_split = 0
        top_level = [b for b in boundaries if b.depth == 0]
        for boundary in top_level:
            chunk_lines = lines[last_split:boundary.end_line]
            chunk_size = sum(len(line) + 1 for line in chunk_lines)
            if current_size + chunk_size > max_chunk_size and current_size > 0:
                split_points.append(boundary.start_line - 1)
                last_split = boundary.start_line - 1
                current_size = chunk_size
            else:
                current_size += chunk_size
        return split_points

    def _simple_split_points(self, lines: List[str], max_chunk_size: int) -> List[int]:
        """Simple line-based split for files without clear boundaries."""
        split_points: List[int] = []
        current_size = 0
        for i, line in enumerate(lines):
            current_size += len(line) + 1
            if current_size >= max_chunk_size:
                split_points.append(i + 1)
                current_size = 0
        return split_points



# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGY_REGISTRY: Dict[str, type] = {
    ".py": PythonSemanticStrategy,
    ".pyw": PythonSemanticStrategy,
    ".ts": TypeScriptSemanticStrategy,
    ".tsx": TypeScriptSemanticStrategy,
    ".js": TypeScriptSemanticStrategy,
    ".jsx": TypeScriptSemanticStrategy,
    ".mjs": TypeScriptSemanticStrategy,
    ".cjs": TypeScriptSemanticStrategy,
}


def get_strategy_for_file(path: Path) -> Optional[SemanticStrategy]:
    """Get the appropriate semantic strategy for a file based on its extension."""
    ext = path.suffix.lower()
    strategy_class = STRATEGY_REGISTRY.get(ext)
    if strategy_class:
        return strategy_class()
    return None


# =============================================================================
# Main Semantic Chunker
# =============================================================================

class SemanticChunker:
    """
    Enterprise-grade semantic chunker for large codebase management.

    Splits files at semantic boundaries (classes, functions, modules) rather
    than naive line/character limits, preserving context and meaning.
    """

    def __init__(self, config: Optional[RLMConfig] = None):
        """Initialize the semantic chunker with optional configuration."""
        self.config = config or load_config()
        self._strategy_cache: Dict[str, SemanticStrategy] = {}

    def chunk_file(
        self,
        source_path: Path,
        max_chunk_size: int = MAX_CHUNK_SIZE_CHARS,
    ) -> List[SemanticChunk]:
        """
        Chunk a file using semantic-aware splitting.

        Args:
            source_path: Path to the source file
            max_chunk_size: Maximum size of each chunk in characters

        Returns:
            List of SemanticChunk objects
        """
        source_path = Path(source_path)
        self.config.ensure_directories()

        # Get file info
        info = get_file_info(source_path)
        if info.is_binary:
            raise ValueError(f"Cannot chunk binary file: {source_path}")

        # Read file content
        encoding = info.encoding or "utf-8"
        with open(source_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()
            lines = content.split('\n')

        if not lines:
            return []

        # Get appropriate strategy
        strategy = self._get_strategy(source_path)

        if strategy:
            return self._semantic_chunk(
                source_path, content, lines, info, strategy, max_chunk_size
            )
        else:
            # Fall back to line-based chunking for unsupported file types
            return self._fallback_chunk(source_path, lines, info, max_chunk_size)

    def _get_strategy(self, path: Path) -> Optional[SemanticStrategy]:
        """Get cached strategy for file type."""
        ext = path.suffix.lower()
        if ext not in self._strategy_cache:
            strategy = get_strategy_for_file(path)
            if strategy:
                self._strategy_cache[ext] = strategy
        return self._strategy_cache.get(ext)

    def _semantic_chunk(
        self,
        source_path: Path,
        content: str,
        lines: List[str],
        info: FileInfo,
        strategy: SemanticStrategy,
        max_chunk_size: int,
    ) -> List[SemanticChunk]:
        """Perform semantic-aware chunking."""
        # Extract boundaries
        boundaries = strategy.extract_boundaries(content, source_path)
        imports = strategy.extract_imports(content)

        # Get optimal split points
        split_points = strategy.get_optimal_split_points(boundaries, max_chunk_size, lines)

        # Create chunks
        chunks: List[SemanticChunk] = []
        chunk_starts = [0] + split_points
        chunk_ends = split_points + [len(lines)]

        for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
            chunk_content = '\n'.join(lines[start:end])
            chunk_id = self._generate_chunk_id(source_path, start, end)
            chunk_path = self.config.get_cache_path() / f"sem_chunk_{chunk_id}.txt"

            # Find boundaries within this chunk
            chunk_boundaries = [
                b for b in boundaries
                if start < b.start_line <= end
            ]

            # Write chunk to cache with metadata header
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(f"# Semantic Chunk from: {source_path.name}\n")
                f.write(f"# Lines: {start + 1}-{end} of {len(lines)}\n")
                f.write(f"# Content type: {info.content_type}\n")
                if chunk_boundaries:
                    entities = [b.name for b in chunk_boundaries]
                    f.write(f"# Entities: {', '.join(entities[:5])}\n")
                f.write("# " + "=" * 60 + "\n\n")
                f.write(chunk_content)

            primary = chunk_boundaries[0].name if chunk_boundaries else None

            chunks.append(SemanticChunk(
                chunk_id=chunk_id,
                chunk_path=chunk_path,
                source_path=source_path,
                start_line=start + 1,
                end_line=end,
                size_bytes=len(chunk_content.encode('utf-8')),
                content_type=info.content_type,
                boundaries=chunk_boundaries,
                primary_entity=primary,
                imports=imports if i == 0 else [],
            ))

        return chunks

    def _fallback_chunk(
        self,
        source_path: Path,
        lines: List[str],
        info: FileInfo,
        max_chunk_size: int,
    ) -> List[SemanticChunk]:
        """Fallback to simple line-based chunking."""
        chunks: List[SemanticChunk] = []
        current_start = 0
        current_size = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1
            if current_size + line_size > max_chunk_size and current_size > 0:
                # Create chunk
                chunk_content = '\n'.join(lines[current_start:i])
                chunk_id = self._generate_chunk_id(source_path, current_start, i)
                chunk_path = self.config.get_cache_path() / f"chunk_{chunk_id}.txt"

                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(chunk_content)

                chunks.append(SemanticChunk(
                    chunk_id=chunk_id,
                    chunk_path=chunk_path,
                    source_path=source_path,
                    start_line=current_start + 1,
                    end_line=i,
                    size_bytes=len(chunk_content.encode('utf-8')),
                    content_type=info.content_type,
                ))

                current_start = i
                current_size = 0

            current_size += line_size

        # Final chunk
        if current_start < len(lines):
            chunk_content = '\n'.join(lines[current_start:])
            chunk_id = self._generate_chunk_id(source_path, current_start, len(lines))
            chunk_path = self.config.get_cache_path() / f"chunk_{chunk_id}.txt"

            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk_content)

            chunks.append(SemanticChunk(
                chunk_id=chunk_id,
                chunk_path=chunk_path,
                source_path=source_path,
                start_line=current_start + 1,
                end_line=len(lines),
                size_bytes=len(chunk_content.encode('utf-8')),
                content_type=info.content_type,
            ))

        return chunks

    def _generate_chunk_id(self, source_path: Path, start: int, end: int) -> str:
        """Generate a unique chunk ID."""
        unique_str = f"{source_path}:{start}:{end}:{datetime.now().isoformat()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]

    def analyze_file(self, source_path: Path) -> Dict[str, Any]:
        """Analyze a file and return semantic information without chunking."""
        source_path = Path(source_path)
        info = get_file_info(source_path)

        if info.is_binary:
            return {"error": "Binary file", "path": str(source_path)}

        encoding = info.encoding or "utf-8"
        with open(source_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()

        strategy = self._get_strategy(source_path)
        if not strategy:
            return {
                "path": str(source_path),
                "content_type": info.content_type,
                "lines": info.line_count,
                "size_bytes": info.size_bytes,
                "semantic_support": False,
            }

        boundaries = strategy.extract_boundaries(content, source_path)
        imports = strategy.extract_imports(content)

        return {
            "path": str(source_path),
            "content_type": info.content_type,
            "lines": info.line_count,
            "size_bytes": info.size_bytes,
            "semantic_support": True,
            "language": getattr(strategy, 'language', 'unknown'),
            "boundaries": len(boundaries),
            "classes": len([b for b in boundaries if b.boundary_type == SemanticBoundaryType.CLASS]),
            "functions": len([b for b in boundaries if b.boundary_type == SemanticBoundaryType.FUNCTION]),
            "imports": len(imports),
            "top_level_entities": [b.name for b in boundaries if b.depth == 0][:10],
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def semantic_chunk_file(
    source_path: str | Path,
    max_chunk_size: int = MAX_CHUNK_SIZE_CHARS,
    config: Optional[RLMConfig] = None,
) -> List[SemanticChunk]:
    """Convenience function to chunk a file semantically."""
    chunker = SemanticChunker(config)
    return chunker.chunk_file(Path(source_path), max_chunk_size)


def analyze_file_semantics(
    source_path: str | Path,
    config: Optional[RLMConfig] = None,
) -> Dict[str, Any]:
    """Convenience function to analyze file semantics."""
    chunker = SemanticChunker(config)
    return chunker.analyze_file(Path(source_path))