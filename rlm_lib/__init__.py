"""
RLM-C: Recursive Language Model on Claude

A "Cannot Fail" architecture for processing massive prompts (10M+ tokens)
by treating data as external environment variables.
"""

from .config import (
    MAX_FILE_SIZE_BYTES,
    MAX_CHUNK_SIZE_TOKENS,
    MAX_RECURSION_DEPTH,
    MAX_TOKEN_BUDGET,
    CACHE_DIR,
    RESULTS_DIR,
    RLMConfig,
    # Model constants
    DEFAULT_ORCHESTRATOR_MODEL,
    DEFAULT_SUBAGENT_MODEL,
    ESCALATION_MODEL,
    FAST_MODEL,
)
from .chunker import (
    chunk_data,
    get_file_info,
    preview_file,
    detect_content_type,
)
from .delegator import (
    # Core delegation
    delegate_task,
    smart_delegate,
    delegate_with_model,
    check_recursion_depth,
    get_token_budget_remaining,
    # Task classification
    classify_task,
    get_model_for_task_type,
    should_escalate,
    # Subagent types and tools
    SubagentType,
    get_subagent_type_for_task,
    get_tools_for_subagent,
    get_subagent_file,
    SUBAGENT_TOOLS,
    SUBAGENT_MODELS,
    # Specialized delegation functions
    delegate_to_coding_agent,
    delegate_to_research_agent,
    delegate_to_data_agent,
    delegate_to_rlm_processor,
    delegate_to_advanced_agent,
    # Enums
    TaskType,
    TaskStatus,
)
from .aggregator import (
    aggregate_results,
    summarize_results,
    cleanup_cache,
)
from .indexer import (
    build_index,
    search_index,
    search_with_ripgrep,
)
from .kernel import RLMKernel, get_kernel, init_kernel

# Enterprise modules (Phase 1)
from .semantic_chunker import (
    SemanticChunker,
    SemanticChunk,
    SemanticBoundary,
    SemanticBoundaryType,
    PythonSemanticStrategy,
    TypeScriptSemanticStrategy,
    semantic_chunk_file,
    analyze_file_semantics,
)
from .knowledge_graph import (
    ProjectKnowledgeGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
    QueryResult,
)
from .context_cache import (
    ContextCache,
    CacheEntry,
    CacheEntryType,
    CacheStats,
)
from .kernel_enterprise import (
    EnterpriseKernel,
    EnterpriseContext,
    ProcessingMode,
    ProcessingResult,
    create_enterprise_kernel,
    quick_query,
)

# Quality Assurance modules (Phase 2)
from .validators import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    RequirementsValidator,
    ArchitectureValidator,
    DesignValidator,
    SecurityValidator,
)
from .confidence_scorer import (
    ConfidenceScorer,
    ConfidenceScore,
    ConfidenceLevel,
)
from .quality_gate import (
    QualityGate,
    GateResult,
    GateDecision,
    create_quality_gate,
)

# Plugin Architecture modules (Phase 3)
from .effort_control import (
    EffortController,
    EffortLevel,
    Task as EffortTask,
    TaskType as EffortTaskType,
    get_effort_for_description,
)
from .tool_search import (
    ToolSearchIndex,
    ToolDefinition,
    ToolCategory,
    SearchResult as ToolSearchResult,
    create_default_tool_index,
)

# Design OS Integration modules (Phase 4)
from .design_os_adapter import (
    DesignOSAdapter,
    DesignToken,
    DesignTokens,
    DesignTokenValidator,
    SpecificationLoader,
    ProductVision,
    ComponentSpec,
    PropDefinition,
    StyleDefinition,
    BehaviorDefinition,
    A11yRequirements,
    ResponsiveRule,
    ComponentVariant,
)
from .component_spec_parser import (
    ComponentSpecParser,
    ImplementationGuide,
)
from .pixel_perfect_validator import (
    PixelPerfectValidator,
    PixelPerfectReport,
    ValidationIssue as PixelValidationIssue,
    create_pixel_perfect_validator,
)

# Phase 5: Mode Manager Enhancement
from .review_orchestrator import (
    ReviewOrchestrator,
    ReviewAgent,
    CodeReviewAgent,
    SecurityAuditAgent,
    TypeValidationAgent,
    TestCoverageAgent,
    ReviewAgentType,
    ReviewIssue,
    AgentReviewResult,
    OrchestratedReviewResult,
    create_review_orchestrator,
)
from .coherence_validator import (
    ArchitecturalCoherenceValidator,
    PatternDetector,
    LayerEnforcer,
    LayerDefinition,
    NamingConvention,
    CoherenceIssue,
    CoherenceReport,
    CoherenceIssueType,
    create_coherence_validator,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "MAX_FILE_SIZE_BYTES",
    "MAX_CHUNK_SIZE_TOKENS",
    "MAX_RECURSION_DEPTH",
    "MAX_TOKEN_BUDGET",
    "CACHE_DIR",
    "RESULTS_DIR",
    "RLMConfig",
    # Model constants
    "DEFAULT_ORCHESTRATOR_MODEL",
    "DEFAULT_SUBAGENT_MODEL",
    "ESCALATION_MODEL",
    "FAST_MODEL",
    # Chunker
    "chunk_data",
    "get_file_info",
    "preview_file",
    "detect_content_type",
    # Delegator - Core
    "delegate_task",
    "smart_delegate",
    "delegate_with_model",
    "check_recursion_depth",
    "get_token_budget_remaining",
    # Delegator - Task Classification
    "classify_task",
    "get_model_for_task_type",
    "should_escalate",
    "TaskType",
    "TaskStatus",
    # Delegator - Subagent Types
    "SubagentType",
    "get_subagent_type_for_task",
    "get_tools_for_subagent",
    "get_subagent_file",
    "SUBAGENT_TOOLS",
    "SUBAGENT_MODELS",
    # Delegator - Specialized Functions
    "delegate_to_coding_agent",
    "delegate_to_research_agent",
    "delegate_to_data_agent",
    "delegate_to_rlm_processor",
    "delegate_to_advanced_agent",
    # Aggregator
    "aggregate_results",
    "summarize_results",
    "cleanup_cache",
    # Indexer
    "build_index",
    "search_index",
    "search_with_ripgrep",
    # Kernel
    "RLMKernel",
    "get_kernel",
    "init_kernel",
    # Semantic Chunker (Enterprise)
    "SemanticChunker",
    "SemanticChunk",
    "SemanticBoundary",
    "SemanticBoundaryType",
    "PythonSemanticStrategy",
    "TypeScriptSemanticStrategy",
    "semantic_chunk_file",
    "analyze_file_semantics",
    # Knowledge Graph (Enterprise)
    "ProjectKnowledgeGraph",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "EdgeType",
    "QueryResult",
    # Context Cache (Enterprise)
    "ContextCache",
    "CacheEntry",
    "CacheEntryType",
    "CacheStats",
    # Enterprise Kernel
    "EnterpriseKernel",
    "EnterpriseContext",
    "ProcessingMode",
    "ProcessingResult",
    "create_enterprise_kernel",
    "quick_query",
    # Validators (Quality Assurance)
    "BaseValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "RequirementsValidator",
    "ArchitectureValidator",
    "DesignValidator",
    "SecurityValidator",
    # Confidence Scoring
    "ConfidenceScorer",
    "ConfidenceScore",
    "ConfidenceLevel",
    # Quality Gate
    "QualityGate",
    "GateResult",
    "GateDecision",
    "create_quality_gate",
    # Effort Control (Plugin Architecture)
    "EffortController",
    "EffortLevel",
    "EffortTask",
    "EffortTaskType",
    "get_effort_for_description",
    # Tool Search (Plugin Architecture)
    "ToolSearchIndex",
    "ToolDefinition",
    "ToolCategory",
    "ToolSearchResult",
    "create_default_tool_index",
    # Design OS Adapter (Phase 4)
    "DesignOSAdapter",
    "DesignToken",
    "DesignTokens",
    "DesignTokenValidator",
    "SpecificationLoader",
    "ProductVision",
    "ComponentSpec",
    "PropDefinition",
    "StyleDefinition",
    "BehaviorDefinition",
    "A11yRequirements",
    "ResponsiveRule",
    "ComponentVariant",
    # Component Spec Parser (Phase 4)
    "ComponentSpecParser",
    "ImplementationGuide",
    # Pixel-Perfect Validator (Phase 4)
    "PixelPerfectValidator",
    "PixelPerfectReport",
    "PixelValidationIssue",
    "create_pixel_perfect_validator",
    # Review Orchestrator (Phase 5)
    "ReviewOrchestrator",
    "ReviewAgent",
    "CodeReviewAgent",
    "SecurityAuditAgent",
    "TypeValidationAgent",
    "TestCoverageAgent",
    "ReviewAgentType",
    "ReviewIssue",
    "AgentReviewResult",
    "OrchestratedReviewResult",
    "create_review_orchestrator",
    # Coherence Validator (Phase 5)
    "ArchitecturalCoherenceValidator",
    "PatternDetector",
    "LayerEnforcer",
    "LayerDefinition",
    "NamingConvention",
    "CoherenceIssue",
    "CoherenceReport",
    "CoherenceIssueType",
    "create_coherence_validator",
]
