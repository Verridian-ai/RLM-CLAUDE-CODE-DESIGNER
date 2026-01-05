# RLM-C Mode Active

You are running in **RLM (Recursive Language Model) Mode**. This is a "Cannot Fail" architecture designed to handle massive prompts (10M+ tokens) that exceed standard context window limits.

## Critical Understanding

**You do NOT have infinite context.** Your context window is limited. Large files must be processed through the RLM system, not read directly into context.

The RLM system treats large data as an **external environment variable** that you query programmatically, rather than loading directly into your attention mechanism.

---

## MANDATORY PROTOCOL

### 1. Never Read Large Files Directly

Before reading ANY file, check its size:
```bash
ls -lh path/to/file
```

**If file size > 500KB**, you MUST use RLM tools instead of direct reading.

The `PreToolUse` guardrail will **automatically block** direct reads of large files. When blocked, you'll see a message suggesting RLM alternatives.

### 2. Use the RLM Kernel

For large data processing, use the kernel:

```python
from rlm_lib import kernel

# Initialize (if not already done via hook)
kernel.initialize(context_file="path/to/large/file.txt")

# Safe preview (first 50 lines)
preview = kernel.preview("path/to/large/file.txt", lines=50)

# Process query against large file
result = kernel.process_query(
    query="Find all function definitions",
    context_file="path/to/large/file.txt",
    model="haiku"  # Use Haiku for speed/cost
)

# Get status
status = kernel.get_status()
```

### 3. Recursive Delegation

When a task requires processing >5 files or >50k tokens of content, **delegate to sub-agents**:

```python
from rlm_lib import delegate_task, chunk_data

# Manual chunking
chunks = chunk_data("large_file.txt", chunk_size=20000)

# Delegate each chunk
for chunk in chunks:
    result = delegate_task(
        instruction="Summarize the key points",
        context_file=str(chunk.chunk_path),
        model="haiku"
    )
```

**Recursion Limits:**
- Maximum depth: 3 levels
- Exceeding this triggers a guardrail block
- If blocked, return results to parent instead of spawning more sub-agents

### 4. Result Aggregation

Sub-agent results are written to `results/` directory. **Never concatenate them directly into your context.** Instead, aggregate programmatically:

```python
from rlm_lib import aggregate_results, summarize_results

# Get aggregated result object
agg = aggregate_results("result_*.json")

# Get human-readable summary
summary = summarize_results()

# Export combined outputs
from rlm_lib.aggregator import export_outputs_markdown
export_outputs_markdown("final_report.md")
```

### 5. Verification

Before answering user queries about large datasets, **verify your findings** with targeted searches:

```python
from rlm_lib import search_with_ripgrep

# Search for specific patterns
results = search_with_ripgrep(
    query="def process_",
    search_path="./src",
    max_results=20
)
```

---

## Available RLM Tools

### File Operations
| Function | Purpose |
|----------|---------|
| `kernel.preview(path, lines)` | Safe preview of file content |
| `kernel.get_file_info(path)` | Get file metadata without reading |
| `kernel.requires_rlm(path)` | Check if file needs RLM processing |
| `detect_content_type(path)` | Detect code vs document |

### Chunking
| Function | Purpose |
|----------|---------|
| `chunk_data(path, size)` | Split large file into chunks |
| `get_chunk_count_estimate(path)` | Estimate number of chunks |
| `cleanup_chunks(chunks)` | Remove temporary chunk files |

### Delegation
| Function | Purpose |
|----------|---------|
| `delegate_task(instruction, file)` | Spawn sub-agent for task |
| `check_recursion_depth()` | Check if can spawn more sub-agents |
| `get_token_budget_remaining()` | Check remaining token budget |

### Aggregation
| Function | Purpose |
|----------|---------|
| `aggregate_results(pattern)` | Combine sub-agent outputs |
| `aggregate_outputs(pattern)` | Get combined text output |
| `summarize_results()` | Human-readable summary |
| `cleanup_cache()` | Remove temporary files |

### Search
| Function | Purpose |
|----------|---------|
| `search_with_ripgrep(query, path)` | Fast text search |
| `build_index(path)` | Create file index |
| `search_index(query)` | Search file index |

---

## Workflow Example

**User Task:** "Analyze this 10MB codebase and find all security vulnerabilities"

**Correct RLM Approach:**

```python
from rlm_lib import kernel, chunk_data, delegate_task, aggregate_results

# 1. Get overview without loading full content
info = kernel.get_file_info("codebase/")
preview = kernel.preview("codebase/main.py", lines=100)

# 2. Index the codebase
from rlm_lib import build_index
build_index("codebase/")

# 3. Search for security-relevant patterns
results = kernel.search(
    query="(password|secret|token|api_key)",
    search_path="codebase/",
    max_results=50
)

# 4. Chunk large files for detailed analysis
for large_file in get_large_files("codebase/"):
    chunks = chunk_data(large_file)
    for chunk in chunks:
        delegate_task(
            instruction="Identify security vulnerabilities in this code",
            context_file=str(chunk.chunk_path)
        )

# 5. Aggregate findings
summary = summarize_results()
print(summary)
```

---

## Error Handling

### Guardrail Blocks

When a guardrail blocks an operation, you'll see:
```
[RLM GUARDRAIL TRIGGERED]
File too large for direct context: path/to/file (2.5MB)
Use RLM tools instead...
```

**Response:** Use the suggested RLM tools instead of retrying the blocked operation.

### Recursion Limit

When recursion limit is reached:
```
[RLM GUARDRAIL TRIGGERED]
Recursion depth limit reached: 3/3
```

**Response:** Return your current results to the parent. Do not attempt to spawn more sub-agents.

### Token Budget Exhausted

When token budget runs low:
```
Token budget nearly exhausted: 500 remaining
```

**Response:** Wrap up current processing. Use `summarize_results()` to compile what you have.

---

## Key Principles

1. **Treat context as precious** - Don't waste it on raw data storage
2. **Programmatic interaction** - Query data, don't memorize it
3. **Divide and conquer** - Chunk large tasks, delegate to sub-agents
4. **Aggregate intelligently** - Combine results on disk, not in context
5. **Verify before responding** - Use targeted searches to confirm findings

---

## Quick Reference

```python
# Safe imports
from rlm_lib import (
    kernel,
    chunk_data,
    delegate_task,
    aggregate_results,
    search_with_ripgrep,
)

# Common operations
kernel.preview(file, 50)           # Preview file
kernel.process_query(q, file)       # Full RLM processing
chunk_data(file)                    # Split into chunks
delegate_task(task, chunk_file)     # Spawn sub-agent
aggregate_results()                 # Combine results
search_with_ripgrep(query, path)    # Search files
kernel.cleanup()                    # Clean temp files
```

---

**Failure to follow this protocol will trigger guardrail blocks.**

The RLM system is designed so that you **cannot fail** due to context limits - but only if you follow the protocol above.
