# Core Modules

## 1. Semantic Chunker (`semantic_chunker.py`)

**Purpose**: Split code files into semantically meaningful chunks that preserve context.

**Key Classes**:

- `SemanticChunker`: Main chunker with language strategy selection
- `PythonSemanticStrategy`: AST-based Python chunking
- `TypeScriptSemanticStrategy`: TypeScript/JavaScript chunking

**Usage**:

```python
from rlm_lib import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk_file("path/to/file.py")
```

## 2. Knowledge Graph (`knowledge_graph.py`)

**Purpose**: Build a queryable graph of project entities and relationships.

**Features**:

- File discovery and indexing
- Symbol extraction (classes, functions, variables)
- Relationship analysis (imports, inheritance, calls)

**Usage**:

```python
from rlm_lib import ProjectKnowledgeGraph

graph = ProjectKnowledgeGraph(project_root=".")
results = graph.search("UserService")
```

## 3. Context Cache (`context_cache.py`)

**Purpose**: Persistent caching with LRU eviction to speed up context building.

**Features**:

- SQLite-based persistence
- Invalidates entries when files change
