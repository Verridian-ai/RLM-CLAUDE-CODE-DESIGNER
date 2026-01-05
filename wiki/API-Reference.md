# API Reference

## Factory Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `create_enterprise_kernel(project_root)` | `EnterpriseKernel` | Create configured kernel |
| `create_quality_gate(min_confidence)` | `QualityGate` | Create quality gate |
| `create_confidence_scorer()` | `ConfidenceScorer` | Create scorer |
| `get_effort_controller()` | `EffortController` | Get effort controller |
| `create_pixel_perfect_validator(tokens)` | `PixelPerfectValidator` | Create validator |
| `create_review_orchestrator()` | `ReviewOrchestrator` | Create orchestrator |
| `create_coherence_validator()` | `ArchitecturalCoherenceValidator` | Create validator |

## Core Imports

```python
from rlm_lib import (
    EnterpriseKernel,
    SemanticChunker,
    ProjectKnowledgeGraph,
    ContextCache,
    QualityGate,
    DesignOSAdapter
)
```
