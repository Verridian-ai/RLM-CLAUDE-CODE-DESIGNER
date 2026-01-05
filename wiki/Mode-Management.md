# Mode Management

## 11. Review Orchestrator (`review_orchestrator.py`)

**Purpose**: Coordinate multiple review agents for comprehensive code review.

**Agents**:

- `CodeReviewAgent`: General code quality
- `SecurityAuditAgent`: Vulnerability detection

**Usage**:

```python
orchestrator = create_review_orchestrator()
result = orchestrator.review(code)
```

## 13. Mode Manager v2

**Purpose**: Enterprise workflow management with strict mode transitions.

```mermaid
stateDiagram-v2
    %% Styling
    classDef active fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    classDef gate fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#bf360c
    
    [*] --> EXPLORE:::active
    
    EXPLORE --> PLAN:::active: User Approval
    PLAN --> CODE:::active: Plan Approved
    
    CODE --> REVIEW:::gate: Implementation Complete
    
    state REVIEW {
        [*] --> AutomatedTests
        AutomatedTests --> CodeReviewAgent
        CodeReviewAgent --> SecurityAudit
        SecurityAudit --> [*]
    }
    
    REVIEW --> CODE: Fail
    REVIEW --> COMMIT:::active: Pass
    
    COMMIT --> [*]: Deploy
```

**Modes**:

- `EXPLORE`: Research and understanding
- `PLAN`: Solution design
- `CODE`: Implementation
- `REVIEW`: Verification and testing
- `COMMIT`: Finalization

Transitions between modes are gated by quality checks and human approval.
