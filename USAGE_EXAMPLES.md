# RLM-CLAUDE Usage Examples

Practical examples for common use cases with the RLM-CLAUDE Enterprise system.

---

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Context Building](#context-building)
3. [Code Review Pipeline](#code-review-pipeline)
4. [Design Validation](#design-validation)
5. [Quality Gates](#quality-gates)
6. [Large Codebase Processing](#large-codebase-processing)

---

## Basic Setup

### Initialize Enterprise Kernel

```python
from rlm_lib import create_enterprise_kernel

# Create kernel for your project
kernel = create_enterprise_kernel(project_root="/path/to/project")

# Build the index (first time or after major changes)
kernel.rebuild_index()

# Check status
status = kernel.get_status()
print(f"Initialized: {status['initialized']}")
print(f"Files indexed: {status.get('file_count', 0)}")
```

### Quick Context Query

```python
from rlm_lib import create_enterprise_kernel

kernel = create_enterprise_kernel(project_root=".")
kernel.rebuild_index()

# Build context for a query
context = kernel.build_context("How does user authentication work?")
print(context)
```

---

## Context Building

### Semantic Chunking

```python
from rlm_lib import SemanticChunker

chunker = SemanticChunker()

# Chunk a single file
chunks = chunker.chunk_file("src/services/user_service.py")

for chunk in chunks:
    print(f"Type: {chunk.chunk_type}")
    print(f"Lines: {chunk.start_line}-{chunk.end_line}")
    print(f"Content preview: {chunk.content[:100]}...")
    print("---")
```

### Knowledge Graph Queries

```python
from rlm_lib import ProjectKnowledgeGraph

graph = ProjectKnowledgeGraph(project_root=".")
graph.build_index()

# Find a specific symbol
results = graph.search("UserService")
for node in results:
    print(f"Found: {node.name} ({node.node_type}) in {node.file_path}")

# Get relationships
for node in results:
    edges = graph.get_edges(node.id)
    for edge in edges:
        print(f"  -> {edge.edge_type}: {edge.target_id}")
```

### Caching Results

```python
from rlm_lib import ContextCache
from rlm_lib.context_cache import CacheEntryType

cache = ContextCache(max_size_mb=100)

# Store processed context
cache.set(
    key="user_auth_context",
    content="Processed context about user authentication...",
    entry_type=CacheEntryType.QUERY_RESULT
)

# Retrieve later
result = cache.get("user_auth_context")
if result:
    print(f"Cache hit! Content: {result.content[:50]}...")

# Check cache stats
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
```

---

## Code Review Pipeline

### Basic Review

```python
from rlm_lib import create_review_orchestrator

orchestrator = create_review_orchestrator()

code = '''
def process_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
'''

result = orchestrator.review(code, context={"filename": "user_service.py"})

print(f"Review passed: {result.passed}")
print(f"Confidence: {result.overall_confidence:.2f}")

for agent_name, agent_result in result.agent_results.items():
    print(f"\n[{agent_name}]")
    for issue in agent_result.issues:
        print(f"  - {issue.severity}: {issue.message}")
```

### Coherence Validation

```python
from rlm_lib import create_coherence_validator

validator = create_coherence_validator()



---

## Quality Gates

### Confidence Scoring

```python
from rlm_lib import create_confidence_scorer

scorer = create_confidence_scorer()

# Score generated code
code = '''
def calculate_total(items):
    """Calculate total price of items."""
    return sum(item.price * item.quantity for item in items)
'''

score = scorer.calculate(code, context={"task": "implement total calculation"})

print(f"Overall confidence: {score.overall:.2f}")
print(f"Level: {score.level}")
print(f"Factors:")
for factor, value in score.factors.items():
    print(f"  - {factor}: {value:.2f}")
```

### Quality Gate Evaluation

```python
from rlm_lib import create_quality_gate, create_confidence_scorer
from rlm_lib.quality_gate import GateDecision

gate = create_quality_gate(min_confidence=0.8)
scorer = create_confidence_scorer()

code = "def example(): pass"
score = scorer.calculate(code, context={})

result = gate.evaluate(score)

if result.decision == GateDecision.PASS:
    print("✅ Quality gate passed - proceed with commit")
elif result.decision == GateDecision.NEEDS_REVIEW:
    print("⚠️ Needs human review before proceeding")
else:
    print(f"❌ Quality gate failed: {result.reason}")
```

### Effort Classification

```python
from rlm_lib import get_effort_controller
from rlm_lib.effort_control import EffortLevel

controller = get_effort_controller()

# Classify different tasks
tasks = [
    "Fix typo in README",
    "Add input validation to user form",
    "Implement OAuth2 authentication flow",
    "Refactor entire payment processing module",
]

for task in tasks:
    effort = controller.classify_task(task)
    strategy = controller.get_strategy(effort)
    print(f"Task: {task[:40]}...")
    print(f"  Effort: {effort}")
    print(f"  Strategy: {strategy}")
    print()
```

---

## Large Codebase Processing

### Processing 10k+ Files

```python
from rlm_lib import create_enterprise_kernel
from rlm_lib import chunk_data, delegate_task, aggregate_results

# For very large codebases, use the RLM pattern
kernel = create_enterprise_kernel(project_root="/large/codebase")

# Check if RLM processing is needed
status = kernel.get_status()
if status.get("file_count", 0) > 1000:
    print("Large codebase detected - using RLM processing")

    # Build index in background
    kernel.rebuild_index()

    # Use targeted queries instead of loading everything
    context = kernel.build_context("Find security vulnerabilities")

    # Process in chunks if needed
    # chunks = chunk_data(large_file, chunk_size=20000)
    # for chunk in chunks:
    #     result = delegate_task("Analyze for issues", chunk.chunk_path)
```

### Incremental Updates

```python
from rlm_lib import create_enterprise_kernel

kernel = create_enterprise_kernel(project_root=".")

# Initial full index
kernel.rebuild_index()

# After file changes, update incrementally
changed_files = ["src/services/user.py", "src/models/user.py"]
for file_path in changed_files:
    kernel.update_file(file_path)

# Invalidate related cache entries
from rlm_lib import ContextCache
cache = ContextCache()
for file_path in changed_files:
    cache.invalidate_by_source(file_path)
```

---

## Integration Patterns

### CI/CD Integration

```python
#!/usr/bin/env python
"""CI/CD quality check script."""

import sys
from rlm_lib import (
    create_review_orchestrator,
    create_quality_gate,
    create_confidence_scorer,
)
from rlm_lib.quality_gate import GateDecision

def check_code_quality(file_path: str) -> bool:
    """Run quality checks on a file."""
    with open(file_path) as f:
        code = f.read()

    # Run review
    orchestrator = create_review_orchestrator()
    review = orchestrator.review(code, context={"filename": file_path})

    if not review.passed:
        print(f"❌ Review failed for {file_path}")
        return False

    # Check quality gate
    scorer = create_confidence_scorer()
    score = scorer.calculate(code, context={})

    gate = create_quality_gate(min_confidence=0.8)
    result = gate.evaluate(score)

    if result.decision != GateDecision.PASS:
        print(f"❌ Quality gate failed: {result.reason}")
        return False

    print(f"✅ {file_path} passed all checks")
    return True

if __name__ == "__main__":
    files = sys.argv[1:]
    all_passed = all(check_code_quality(f) for f in files)
    sys.exit(0 if all_passed else 1)
```

---

## Best Practices

1. **Always rebuild index after major changes** - The knowledge graph needs to reflect current state
2. **Use caching for repeated queries** - Avoid redundant processing
3. **Set appropriate confidence thresholds** - Balance speed vs. quality
4. **Use semantic chunking for large files** - Preserve context boundaries
5. **Integrate quality gates in CI/CD** - Catch issues early

---

**Examples Version**: 1.0
**Last Updated**: 2026-01-05
