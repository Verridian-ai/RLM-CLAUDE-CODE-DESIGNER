# Quality Assurance

## 4. Quality Gate (`quality_gate.py`)

**Purpose**: Block low-confidence work from proceeding through the pipeline.

**Features**:

- Configurable confidence thresholds
- Multiple validation dimensions
- detailed failure reasons

**Usage**:

```python
from rlm_lib import QualityGate, create_quality_gate

gate = create_quality_gate(min_confidence=0.8)
result = gate.evaluate(confidence_score, validation_results)
```

## 5. Confidence Scorer (`confidence_scorer.py`)

**Purpose**: Calculate confidence scores for generated outputs based on syntax, semantics, and test coverage.

## 6. Effort Control (`effort_control.py`)

**Purpose**: Classify tasks by effort level (TRIVIAL to CRITICAL) to select the appropriate processing strategy and resource allocation.
