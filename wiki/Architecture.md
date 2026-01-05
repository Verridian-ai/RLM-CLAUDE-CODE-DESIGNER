# System Architecture

## High-Level Diagram

```mermaid
graph TD
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    classDef core fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#bf360c
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef cache fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    
    subgraph Data_Ingestion ["Data Ingestion Layer"]
        direction TB
        SC[Semantic Chunker]:::input
        KG[Knowledge Graph]:::input
    end

    subgraph Memory ["Memory & Context"]
        CC[Context Cache]:::cache
        
    end

    subgraph Engine ["Core Engine"]
        EK[Enterprise Kernel]:::core
    end

    subgraph Execution ["Execution & Output"]
        QG[Quality Gates]:::output
        DOS[Design OS]:::output
        MM[Mode Manager]:::output
    end

    %% Data Flow
    SC ==>|AST Chunks| EK
    KG ==>|Relationships| EK
    CC <==>|Cached Context| EK
    
    %% Control Flow
    EK -->|Eval 1| QG
    QG -->|Pass| DOS
    QG -.->|Fail| MM
    DOS -->|Validate UI| MM
    
    %% Feedback
    MM -.->|Retry| EK
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
