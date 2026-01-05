# RLM-CLAUDE Enterprise System Wiki

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Modules](#core-modules)
4. [Quality Assurance](#quality-assurance)
5. [Plugin Architecture](#plugin-architecture)
6. [Design OS Integration](#design-os-integration)
7. [Mode Management](#mode-management)
8. [Usage Examples](#usage-examples)
9. [API Reference](#api-reference)

---

## Overview

### What is RLM-CLAUDE?

**RLM (Recursive Language Model)** is an enterprise-scale architecture designed to handle massive prompts (10M+ tokens) that exceed standard context window limits. It treats large data as an **external environment variable** that you query programmatically, rather than loading directly into the attention mechanism.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Semantic Chunking** | AST-aware code splitting preserving semantic boundaries |
| **Knowledge Graph** | Project-wide entity extraction and relationship mapping |
| **Context Caching** | Persistent cache with LRU eviction and invalidation |
| **Quality Gates** | Multi-validator system blocking low-confidence work |
| **Design OS** | Pixel-perfect UI validation against design tokens |
| **Mode Manager** | Enterprise workflow with review/commit gates |

### Performance Targets

- **Context build**: <30 seconds for 10,000 files
- **First-time correctness**: >95%
- **Cache hit rate**: >70%
- **Test coverage**: >95%

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RLM-CLAUDE Enterprise                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Semantic   │  │  Knowledge  │  │   Context   │         │
│  │   Chunker   │  │    Graph    │  │    Cache    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│              ┌─────────────────────┐                        │
│              │  Enterprise Kernel  │                        │
│              └─────────────────────┘                        │
│                          │                                  │
│         ┌────────────────┼────────────────┐                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Quality   │  │   Design    │  │    Mode     │         │
│  │    Gates    │  │     OS      │  │   Manager   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
rlm_lib/
├── kernel.py              # Base RLM kernel
├── kernel_enterprise.py   # Enterprise kernel (extends base)
├── semantic_chunker.py    # AST-aware chunking
├── knowledge_graph.py     # Entity/relationship graph
├── context_cache.py       # Persistent caching
├── quality_gate.py        # Quality validation
├── confidence_scorer.py   # Confidence calculation
├── effort_control.py      # Task effort classification
├── tool_search.py         # Tool discovery
├── design_os_adapter.py   # Design system integration
├── component_spec_parser.py  # Component specs
├── pixel_perfect_validator.py  # Style validation
├── review_orchestrator.py  # Multi-agent review
├── coherence_validator.py  # Architecture coherence
└── validators/            # Validation modules
    ├── base.py
    ├── requirements.py
    ├── architecture.py
    ├── design.py
    └── security.py
```

---

## Core Modules

### 1. Semantic Chunker (`semantic_chunker.py`)

**Purpose**: Split code files into semantically meaningful chunks that preserve context.

**Key Classes**:

- `SemanticChunker`: Main chunker with language strategy selection
- `PythonSemanticStrategy`: AST-based Python chunking
- `TypeScriptSemanticStrategy`: TypeScript/JavaScript chunking
- `SemanticChunk`: Dataclass representing a chunk

**Features**:

- AST parsing for accurate boundary detection
- Preserves function/class definitions intact
- Handles imports and module-level code
- Configurable chunk size limits

**Usage**:

```python
from rlm_lib import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk_file("path/to/file.py")
for chunk in chunks:
    print(f"Lines {chunk.start_line}-{chunk.end_line}: {chunk.chunk_type}")
```

### 2. Knowledge Graph (`knowledge_graph.py`)

**Purpose**: Build a queryable graph of project entities and relationships.

**Key Classes**:

- `ProjectKnowledgeGraph`: Main graph builder and query interface
- `GraphNode`: Entity representation (class, function, module)
- `GraphEdge`: Relationship between entities
- `NodeType` / `EdgeType`: Enums for entity/relationship types

**Features**:

- File discovery and indexing
- Symbol extraction (classes, functions, variables)
- Relationship analysis (imports, inheritance, calls)
- Semantic search across entities
- Persistence to disk

**Usage**:

```python
from rlm_lib import ProjectKnowledgeGraph

graph = ProjectKnowledgeGraph(project_root="/path/to/project")
graph.build_index()

# Search for entities
results = graph.search("UserService")
for node in results:
    print(f"{node.name} ({node.node_type})")
```

---

## Quality Assurance

### 4. Quality Gate (`quality_gate.py`)

**Purpose**: Block low-confidence work from proceeding through the pipeline.

**Key Classes**:

- `QualityGate`: Main gate with configurable thresholds
- `GateResult`: Pass/fail result with details
- `GateDecision`: Enum (PASS, FAIL, NEEDS_REVIEW)

**Features**:

- Configurable confidence thresholds
- Multiple validation dimensions
- Detailed failure reasons
- Integration with mode manager

**Usage**:

```python
from rlm_lib import QualityGate, create_quality_gate

gate = create_quality_gate(min_confidence=0.8)
result = gate.evaluate(confidence_score, validation_results)
if result.decision == GateDecision.PASS:
    proceed()
```

### 5. Confidence Scorer (`confidence_scorer.py`)

**Purpose**: Calculate confidence scores for generated outputs.

**Key Classes**:

- `ConfidenceScorer`: Multi-factor confidence calculation
- `ConfidenceScore`: Score with breakdown by factor
- `ConfidenceLevel`: Enum (HIGH, MEDIUM, LOW, VERY_LOW)

**Features**:

- Multiple scoring factors (syntax, semantics, coverage)
- Weighted aggregation
- Threshold-based classification
- Detailed factor breakdown

**Usage**:

```python
from rlm_lib import ConfidenceScorer, create_confidence_scorer

scorer = create_confidence_scorer()
score = scorer.calculate(code, context)
print(f"Confidence: {score.overall} ({score.level})")
```

### 6. Effort Control (`effort_control.py`)

**Purpose**: Classify tasks by effort level and select appropriate processing strategy.

**Key Classes**:

- `EffortController`: Task classification and strategy selection
- `EffortLevel`: Enum (TRIVIAL, LOW, MEDIUM, HIGH, CRITICAL)
- `Task`: Task representation with metadata
- `TaskType`: Enum for task categories

**Features**:

- Automatic effort estimation
- Strategy selection based on effort
- Resource allocation guidance
- Integration with mode manager

**Usage**:

```python
from rlm_lib import EffortController, get_effort_controller

controller = get_effort_controller()
effort = controller.classify_task(task_description)
strategy = controller.get_strategy(effort)
```

---

## Plugin Architecture

### 7. Validators (`validators/`)

**Purpose**: Modular validation system for different aspects of code quality.

**Available Validators**:

- `RequirementsValidator`: Validates against requirements
- `ArchitectureValidator`: Checks architectural patterns
- `DesignValidator`: Validates design compliance
- `SecurityValidator`: Security vulnerability detection
- `CodebaseValidator`: General codebase health

**Base Class**:

```python
from rlm_lib.validators import BaseValidator, ValidationResult

class CustomValidator(BaseValidator):
    def validate(self, content: str, context: dict) -> ValidationResult:
        # Custom validation logic
        return ValidationResult(passed=True, issues=[])
```

**Usage**:

```python
from rlm_lib.validators import (
    RequirementsValidator,
    SecurityValidator,
    ValidationSeverity,
)

req_validator = RequirementsValidator()
result = req_validator.validate(code, {"requirements": specs})

sec_validator = SecurityValidator()
result = sec_validator.validate(code, {})
for issue in result.issues:
    if issue.severity == ValidationSeverity.CRITICAL:
        print(f"CRITICAL: {issue.message}")
```

---

## Design OS Integration

### 8. Design OS Adapter (`design_os_adapter.py`)

**Purpose**: Integrate with design systems for consistent UI implementation.

**Key Classes**:

- `DesignOSAdapter`: Main adapter for design system integration
- `DesignTokens`: Design token storage and lookup
- `DesignTokenValidator`: Token usage validation
- `ProductVision`: Product vision document parser
- `ComponentSpec`: Component specification

**Features**:

- Design token management
- Token usage validation
- Product vision parsing
- Component spec loading

**Usage**:

```python
from rlm_lib import DesignOSAdapter, DesignTokens

tokens = DesignTokens(
    colors={"primary": "#007bff"},
    spacing={"md": "16px"},
    typography={"base": "16px"}
)
adapter = DesignOSAdapter(tokens)
```

### 9. Component Spec Parser (`component_spec_parser.py`)

**Purpose**: Parse component specifications for implementation guidance.

**Key Classes**:

- `ComponentSpecParser`: YAML/JSON spec parser
- `ComponentSpec`: Parsed specification
- `ImplementationGuide`: Generated implementation guide

**Usage**:

```python
from rlm_lib import ComponentSpecParser
from pathlib import Path

parser = ComponentSpecParser()
spec = parser.parse(Path("specs/Button.yaml"))
guide = parser.generate_implementation_guide(spec)
```

### 10. Pixel-Perfect Validator (`pixel_perfect_validator.py`)

**Purpose**: Validate UI implementations against design specifications.

**Key Classes**:

- `PixelPerfectValidator`: Main validator
- `PixelPerfectReport`: Validation report
- `PixelPerfectIssue`: Individual issue

**Features**:

- CSS validation
- JSX/TSX validation
- Vue component validation
- Design token compliance checking
- Tolerance configuration

**Usage**:

```python
from rlm_lib import create_pixel_perfect_validator

validator = create_pixel_perfect_validator(design_tokens)
report = validator.validate_css(css_code)
if not report.passed:
    for issue in report.issues:
        print(f"{issue.severity}: {issue.message}")
```

---

## Mode Management

### 11. Review Orchestrator (`review_orchestrator.py`)

**Purpose**: Coordinate multiple review agents for comprehensive code review.

**Key Classes**:

- `ReviewOrchestrator`: Main orchestrator
- `ReviewAgent`: Base class for review agents
- `CodeReviewAgent`: Code quality review
- `SecurityAuditAgent`: Security vulnerability detection
- `OrchestratedReviewResult`: Aggregated review result

**Features**:

- Parallel agent execution
- Fail-fast on critical issues
- Configurable agent selection
- Result aggregation

**Usage**:

```python
from rlm_lib import create_review_orchestrator

orchestrator = create_review_orchestrator()
result = orchestrator.review(code, context={"filename": "service.py"})
if result.passed:
    print("Review passed!")
else:
    for agent_name, agent_result in result.agent_results.items():
        for issue in agent_result.issues:
            print(f"[{agent_name}] {issue.message}")
```

### 12. Coherence Validator (`coherence_validator.py`)

**Purpose**: Validate architectural coherence across the codebase.

**Key Classes**:

- `ArchitecturalCoherenceValidator`: Main validator
- `PatternDetector`: Design pattern detection
- `LayerEnforcer`: Layer boundary enforcement
- `CoherenceReport`: Validation report

**Features**:

- Pattern detection (MVC, Repository, etc.)
- Layer boundary enforcement
- Dependency direction validation
- Circular dependency detection

**Usage**:

```python
from rlm_lib import create_coherence_validator

validator = create_coherence_validator()
report = validator.validate(code, context={"layer": "service"})
if not report.passed:
    for issue in report.issues:
        print(f"{issue.issue_type}: {issue.message}")
```

### 13. Mode Manager v2 (`.claude/mode_manager_v2.py`)

**Purpose**: Enterprise workflow management with mode transitions.

**Key Classes**:

- `EnterpriseModeManager`: Main mode manager
- `AgentMode`: Enum (EXPLORE, PLAN, CODE, REVIEW, COMMIT)
- `ModeConfig`: Mode configuration
- `ModeTransitionRules`: Transition validation

**Features**:

- Mode-based workflow
- Transition validation
- Quality gate integration
- Human approval gates

**Usage**:

```python
from .claude.mode_manager_v2 import EnterpriseModeManager, AgentMode

manager = EnterpriseModeManager()
success, message = manager.request_transition(AgentMode.CODE)
if success:
    print(f"Transitioned to CODE mode")
```

---

## API Reference

### Quick Import Reference

```python
# Core modules
from rlm_lib import (
    EnterpriseKernel,
    create_enterprise_kernel,
    SemanticChunker,
    ProjectKnowledgeGraph,
    ContextCache,
)

# Quality assurance
from rlm_lib import (
    QualityGate,
    create_quality_gate,
    ConfidenceScorer,
    create_confidence_scorer,
    EffortController,
    get_effort_controller,
)

# Design OS
from rlm_lib import (
    DesignOSAdapter,
    DesignTokens,
    ComponentSpecParser,
    create_pixel_perfect_validator,
)

# Review system
from rlm_lib import (
    ReviewOrchestrator,
    create_review_orchestrator,
    ArchitecturalCoherenceValidator,
    create_coherence_validator,
)

# Validators
from rlm_lib.validators import (
    RequirementsValidator,
    ArchitectureValidator,
    DesignValidator,
    SecurityValidator,
    CodebaseValidator,
    ValidationResult,
    ValidationSeverity,
)
```

### Factory Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `create_enterprise_kernel(project_root)` | `EnterpriseKernel` | Create configured kernel |
| `create_quality_gate(min_confidence)` | `QualityGate` | Create quality gate |
| `create_confidence_scorer()` | `ConfidenceScorer` | Create scorer |
| `get_effort_controller()` | `EffortController` | Get effort controller |
| `create_pixel_perfect_validator(tokens)` | `PixelPerfectValidator` | Create validator |
| `create_review_orchestrator()` | `ReviewOrchestrator` | Create orchestrator |
| `create_coherence_validator()` | `ArchitecturalCoherenceValidator` | Create validator |

---

## Testing

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/test_*.py -v

# Integration tests
python -m pytest tests/integration/ -v

# Benchmarks
python -m pytest tests/test_benchmarks.py -v

# With coverage
python -m pytest tests/ --cov=rlm_lib --cov-report=html
```

### Test Categories

| Category | Location | Count |
|----------|----------|-------|
| Unit Tests | `tests/test_*.py` | 221 |
| Integration Tests | `tests/integration/` | 30 |
| Benchmark Tests | `tests/test_benchmarks.py` | 5 |
| **Total** | | **256** |

---

## Troubleshooting

### Common Issues

**Import Error: Module not found**

```bash
# Ensure rlm_lib is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Cache corruption**

```python
cache = ContextCache()
cache.clear()  # Clear all entries
```

**Knowledge graph out of sync**

```python
graph = ProjectKnowledgeGraph(project_root=".")
graph.rebuild_index()  # Force rebuild
```

---

**Wiki Version**: 1.0
**Last Updated**: 2026-01-05
