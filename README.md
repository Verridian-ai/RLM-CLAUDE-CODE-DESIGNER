# 🧠 Cognition Engine

> **Enterprise-Scale Recursive Language Model Architecture for Claude**

A "Cannot Fail" architecture designed to handle massive prompts (10M+ tokens) and large codebases (10,000+ files, multi-million LOC) by treating data as external environment variables that you query programmatically.

[![Tests](https://img.shields.io/badge/tests-256%20passing-brightgreen)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## ✨ Features

### Core Capabilities

- **🔄 Semantic Chunking** - AST-aware code splitting preserving semantic boundaries
- **🕸️ Knowledge Graph** - Project-wide entity extraction and relationship mapping
- **💾 Context Caching** - Persistent SQLite cache with LRU eviction
- **🎯 Quality Gates** - Multi-validator system blocking low-confidence work

### Enterprise Features

- **🎨 Design OS Integration** - Pixel-perfect UI validation against design tokens
- **👥 Review Orchestrator** - Multi-agent code review with parallel execution
- **🏗️ Coherence Validator** - Architectural pattern enforcement
- **🔀 Mode Manager** - Enterprise workflow with review/commit gates

### Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Index Build (100 files) | <5s | ✅ <1s |
| Context Build | <2s | ✅ <0.5s |
| Symbol Search | <0.5s | ✅ <0.1s |
| Cache Hit Rate | >70% | ✅ >80% |
| Test Coverage | >95% | ✅ 100% |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Verridian-ai/RLM-CLAUDE-CODE-DESIGNER.git
cd RLM-CLAUDE-CODE-DESIGNER
pip install -e .
```

### Basic Usage

```python
from rlm_lib import create_enterprise_kernel

# Initialize for your project
kernel = create_enterprise_kernel(project_root=".")
kernel.rebuild_index()

# Build context for a query
context = kernel.build_context("How does user authentication work?")
print(context)
```

### Code Review

```python
from rlm_lib import create_review_orchestrator

orchestrator = create_review_orchestrator()
result = orchestrator.review(code, context={"filename": "service.py"})

if result.passed:
    print("✅ Review passed!")
else:
    for agent, result in result.agent_results.items():
        for issue in result.issues:
            print(f"[{agent}] {issue.message}")
```

### Design Validation

```python
from rlm_lib import DesignTokens, create_pixel_perfect_validator

tokens = DesignTokens(
    colors={"primary": "#007bff"},
    spacing={"md": "16px"},
    typography={"base": "16px"}
)

validator = create_pixel_perfect_validator(tokens)
report = validator.validate_css(css_code)
print(f"Score: {report.score:.2f}")
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Wiki](wiki/Home.md) | Comprehensive system documentation |
| [Usage Examples](USAGE_EXAMPLES.md) | Practical code examples |
| [Benchmarks](BENCHMARKS.md) | Performance validation results |
| [Implementation Plan](IMPLEMENTATION_PLAN.md) | Architecture and design decisions |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific categories
python -m pytest tests/integration/ -v  # Integration tests
python -m pytest tests/test_benchmarks.py -v  # Benchmarks

# With coverage
python -m pytest tests/ --cov=rlm_lib --cov-report=html
```

**Test Summary:**

- Unit Tests: 221
- Integration Tests: 30
- Benchmark Tests: 5
- **Total: 256 tests (100% passing)**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cognition Engine                          │
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
│  │   Quality   │  │   Design    │  │   Review    │         │
│  │    Gates    │  │     OS      │  │ Orchestrator│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Overview

| Module | Purpose |
|--------|---------|
| `kernel_enterprise.py` | Main enterprise kernel |
| `semantic_chunker.py` | AST-aware code chunking |
| `knowledge_graph.py` | Entity/relationship graph |
| `context_cache.py` | Persistent caching |
| `quality_gate.py` | Quality validation |
| `confidence_scorer.py` | Confidence calculation |
| `design_os_adapter.py` | Design system integration |
| `pixel_perfect_validator.py` | UI validation |
| `review_orchestrator.py` | Multi-agent review |
| `coherence_validator.py` | Architecture coherence |

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for enterprise-scale AI development**
