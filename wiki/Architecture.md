# System Architecture

## High-Level Diagram

```mermaid
graph TD
    subgraph Input
    SC[Semantic Chunker]
    KG[Knowledge Graph]
    CC[Context Cache]
    end

    subgraph Core
    EK[Enterprise Kernel]
    end

    subgraph Output
    QG[Quality Gates]
    DOS[Design OS]
    MM[Mode Manager]
    end

    SC --> EK
    KG --> EK
    CC --> EK
    
    EK --> QG
    EK --> DOS
    EK --> MM
```

## Module Dependencies

```text
rlm_lib/
├── kernel.py              # Base RLM kernel
├── kernel_enterprise.py   # Enterprise kernel (extends base)
├── semantic_chunker.py    # AST-aware chunking
├── knowledge_graph.py     # Entity/relationship graph
├── context_cache.py       # Persistent caching
├── quality_gate.py        # Quality validation
├── confidence_scorer.py   # Confidence calculation
├── design_os_adapter.py   # Design system integration
├── review_orchestrator.py  # Multi-agent review
└── validators/            # Validation modules
```

See [[Core Modules]] for detailed component descriptions.
