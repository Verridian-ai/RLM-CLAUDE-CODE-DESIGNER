"""
RLM-C Delegator Module

Handles sub-agent delegation for recursive processing.
Manages recursion depth tracking and token budget enforcement.
"""

import os
import json
import subprocess
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from .config import (
    RLMConfig,
    load_config,
    MAX_RECURSION_DEPTH,
    MAX_TOKEN_BUDGET,
    DEFAULT_SUBAGENT_MODEL,
    DEFAULT_ORCHESTRATOR_MODEL,
    ESCALATION_MODEL,
    FAST_MODEL,
    ENABLE_ULTRA_THINKING,
    MAX_RETRIES_BEFORE_ESCALATION,
    STRUGGLE_INDICATORS,
    ENV_RLM_RECURSION_DEPTH,
    ENV_RLM_MAX_DEPTH,
    get_current_recursion_depth,
    set_recursion_depth,
)


# =============================================================================
# Data Classes & Enums
# =============================================================================

class TaskStatus(str, Enum):
    """Status of a delegated task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Blocked by recursion limit


class TaskType(str, Enum):
    """
    Task type classification for intelligent model selection.

    Model assignments:
    - SIMPLE_RESEARCH → Haiku (fast, cost-effective)
    - COMPLEX_RESEARCH → Sonnet (synthesis, analysis)
    - CODING → Sonnet (implementation, debugging)
    - ADVANCED_REASONING → Opus (complex reasoning, escalation)
    """
    SIMPLE_RESEARCH = "simple_research"      # → Haiku
    COMPLEX_RESEARCH = "complex_research"    # → Sonnet
    CODING = "coding"                        # → Sonnet
    ADVANCED_REASONING = "advanced_reasoning"  # → Opus
    DATA_EXTRACTION = "data_extraction"      # → Haiku
    AGGREGATION = "aggregation"              # → Haiku
    RLM_PROCESSING = "rlm_processing"        # → Haiku (chunk processing)
    UNKNOWN = "unknown"                      # → Default (Sonnet)


class SubagentType(str, Enum):
    """
    Specialized sub-agent types with tool restrictions.

    Each type maps to a native Claude Code subagent configuration
    in .claude/agents/ directory.
    """
    CODING = "coding"           # Edit, View, Bash, Diagnostics, CodebaseRetrieval
    RESEARCH = "research"       # WebSearch, WebFetch, Read
    DATA_PROCESSING = "data"    # Bash, Read, Write
    RLM_PROCESSOR = "rlm"       # Read, Bash (for RLM chunk operations)
    ADVANCED = "advanced"       # All tools (Opus escalation)
    GENERAL = "general"         # All tools (default)


# =============================================================================
# Subagent Tool Restrictions
# =============================================================================

# Tool sets for each subagent type (matches .claude/agents/*.md configurations)
SUBAGENT_TOOLS: Dict[SubagentType, Optional[List[str]]] = {
    SubagentType.CODING: ["Edit", "View", "Bash", "Diagnostics", "CodebaseRetrieval"],
    SubagentType.RESEARCH: ["WebSearch", "WebFetch", "Read"],
    SubagentType.DATA_PROCESSING: ["Bash", "Read", "Write"],
    SubagentType.RLM_PROCESSOR: ["Read", "Bash"],
    SubagentType.ADVANCED: None,  # All tools
    SubagentType.GENERAL: None,   # All tools
}

# Model assignments for each subagent type
SUBAGENT_MODELS: Dict[SubagentType, str] = {
    SubagentType.CODING: DEFAULT_SUBAGENT_MODEL,      # Sonnet
    SubagentType.RESEARCH: FAST_MODEL,                 # Haiku
    SubagentType.DATA_PROCESSING: FAST_MODEL,          # Haiku
    SubagentType.RLM_PROCESSOR: FAST_MODEL,            # Haiku
    SubagentType.ADVANCED: ESCALATION_MODEL,           # Opus
    SubagentType.GENERAL: DEFAULT_SUBAGENT_MODEL,      # Sonnet
}

# Native subagent file names in .claude/agents/
SUBAGENT_FILES: Dict[SubagentType, str] = {
    SubagentType.CODING: "coding-agent.md",
    SubagentType.RESEARCH: "research-agent.md",
    SubagentType.DATA_PROCESSING: "data-agent.md",
    SubagentType.RLM_PROCESSOR: "rlm-processor.md",
    SubagentType.ADVANCED: "advanced-reasoning.md",
    SubagentType.GENERAL: None,  # Uses default agent
}


# =============================================================================
# Task Classification Keywords
# =============================================================================

# Keywords that indicate simple/fast tasks → Haiku
SIMPLE_TASK_KEYWORDS: set = {
    # Search and retrieval
    "search", "find", "locate", "lookup", "fetch", "retrieve", "get",
    # Simple extraction
    "extract", "list", "enumerate", "count", "scan", "check",
    # Quick operations
    "quick", "simple", "fast", "brief", "summarize", "overview",
    # Data gathering
    "gather", "collect", "compile", "aggregate",
    # Fact-checking
    "verify", "confirm", "validate", "exists", "contains",
}

# Keywords that indicate complex research → Sonnet
COMPLEX_RESEARCH_KEYWORDS: set = {
    # Analysis
    "analyze", "analyse", "investigate", "research", "study", "examine",
    # Synthesis
    "synthesize", "combine", "integrate", "correlate", "compare", "contrast",
    # Deep understanding
    "explain", "understand", "interpret", "evaluate", "assess",
    # Reports
    "report", "document", "comprehensive", "detailed", "thorough", "in-depth",
}

# Keywords that indicate coding tasks → Sonnet
CODING_TASK_KEYWORDS: set = {
    # Implementation
    "implement", "code", "write", "create", "build", "develop", "program",
    # Debugging
    "debug", "fix", "resolve", "troubleshoot", "patch", "repair",
    # Refactoring
    "refactor", "optimize", "improve", "enhance", "clean",
    # Testing
    "test", "unit test", "integration test", "coverage",
    # Code-specific
    "function", "class", "method", "api", "endpoint", "module",
    "variable", "parameter", "return", "import", "export",
}

# Keywords that indicate need for advanced reasoning → Opus
ADVANCED_REASONING_KEYWORDS: set = {
    # Complex reasoning
    "complex", "difficult", "challenging", "intricate", "nuanced",
    # Architecture
    "architect", "design", "structure", "plan", "strategy",
    # Decision making
    "decide", "choose", "recommend", "suggest", "advise", "best approach",
    # Problem solving
    "solve", "figure out", "determine", "deduce", "infer",
    # Edge cases
    "edge case", "corner case", "unusual", "exceptional", "rare",
}

# File extensions that indicate coding context
CODE_FILE_EXTENSIONS: set = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".vue", ".svelte", ".html", ".css", ".scss", ".sql", ".sh", ".bash",
}


# =============================================================================
# Task Type Classification
# =============================================================================

def classify_task(
    instruction: str,
    context_file: Optional[str] = None,
) -> TaskType:
    """
    Classify a task to determine the appropriate model.

    Args:
        instruction: The task instruction text
        context_file: Optional context file path

    Returns:
        TaskType indicating the classification
    """
    instruction_lower = instruction.lower()
    words = set(instruction_lower.split())

    # Check for coding indicators first (high priority)
    coding_matches = words & CODING_TASK_KEYWORDS
    if coding_matches:
        # Strong coding indicators
        if len(coding_matches) >= 2 or any(kw in instruction_lower for kw in ["implement", "debug", "refactor", "write code"]):
            return TaskType.CODING

    # Check if context file is code
    if context_file:
        ext = Path(context_file).suffix.lower()
        if ext in CODE_FILE_EXTENSIONS:
            # If working with code files, likely coding task
            if coding_matches:
                return TaskType.CODING

    # Check for advanced reasoning needs
    advanced_matches = words & ADVANCED_REASONING_KEYWORDS
    if advanced_matches:
        if len(advanced_matches) >= 2 or any(kw in instruction_lower for kw in ["architect", "complex", "best approach"]):
            return TaskType.ADVANCED_REASONING

    # Check for complex research
    complex_matches = words & COMPLEX_RESEARCH_KEYWORDS
    if complex_matches:
        if len(complex_matches) >= 2 or any(kw in instruction_lower for kw in ["comprehensive", "in-depth", "detailed analysis"]):
            return TaskType.COMPLEX_RESEARCH

    # Check for simple/fast tasks
    simple_matches = words & SIMPLE_TASK_KEYWORDS
    if simple_matches:
        if len(simple_matches) >= 1:
            # Additional check: if also has complex indicators, upgrade
            if complex_matches or advanced_matches:
                return TaskType.COMPLEX_RESEARCH
            return TaskType.SIMPLE_RESEARCH

    # Check for data extraction patterns
    if any(pattern in instruction_lower for pattern in ["extract all", "list all", "find all", "get all", "count"]):
        return TaskType.DATA_EXTRACTION

    # Check for aggregation patterns
    if any(pattern in instruction_lower for pattern in ["aggregate", "combine results", "merge", "consolidate"]):
        return TaskType.AGGREGATION

    # Default to UNKNOWN (will use default model)
    return TaskType.UNKNOWN


def get_model_for_task_type(task_type: TaskType) -> str:
    """
    Get the recommended model for a task type.

    Args:
        task_type: The classified task type

    Returns:
        Model name string
    """
    model_mapping = {
        TaskType.SIMPLE_RESEARCH: FAST_MODEL,       # Haiku
        TaskType.DATA_EXTRACTION: FAST_MODEL,       # Haiku
        TaskType.AGGREGATION: FAST_MODEL,           # Haiku
        TaskType.RLM_PROCESSING: FAST_MODEL,        # Haiku
        TaskType.COMPLEX_RESEARCH: DEFAULT_SUBAGENT_MODEL,  # Sonnet
        TaskType.CODING: DEFAULT_SUBAGENT_MODEL,    # Sonnet
        TaskType.ADVANCED_REASONING: ESCALATION_MODEL,  # Opus
        TaskType.UNKNOWN: DEFAULT_SUBAGENT_MODEL,   # Sonnet (safe default)
    }
    return model_mapping.get(task_type, DEFAULT_SUBAGENT_MODEL)


def get_subagent_type_for_task(task_type: TaskType) -> SubagentType:
    """
    Get the recommended subagent type for a task type.

    This determines which specialized subagent configuration to use,
    which in turn determines tool restrictions.

    Args:
        task_type: The classified task type

    Returns:
        SubagentType for the task
    """
    subagent_mapping = {
        TaskType.SIMPLE_RESEARCH: SubagentType.RESEARCH,
        TaskType.DATA_EXTRACTION: SubagentType.DATA_PROCESSING,
        TaskType.AGGREGATION: SubagentType.DATA_PROCESSING,
        TaskType.RLM_PROCESSING: SubagentType.RLM_PROCESSOR,
        TaskType.COMPLEX_RESEARCH: SubagentType.RESEARCH,
        TaskType.CODING: SubagentType.CODING,
        TaskType.ADVANCED_REASONING: SubagentType.ADVANCED,
        TaskType.UNKNOWN: SubagentType.GENERAL,
    }
    return subagent_mapping.get(task_type, SubagentType.GENERAL)


def get_tools_for_subagent(subagent_type: SubagentType) -> Optional[List[str]]:
    """
    Get the allowed tools for a subagent type.

    Args:
        subagent_type: The subagent type

    Returns:
        List of allowed tool names, or None for all tools
    """
    return SUBAGENT_TOOLS.get(subagent_type)


def get_subagent_file(subagent_type: SubagentType) -> Optional[str]:
    """
    Get the native subagent configuration file path.

    Args:
        subagent_type: The subagent type

    Returns:
        Path to the subagent markdown file, or None for default
    """
    filename = SUBAGENT_FILES.get(subagent_type)
    if filename:
        return f".claude/agents/{filename}"
    return None


def should_escalate(result: "DelegationResult") -> bool:
    """
    Determine if a task result indicates the sub-agent struggled
    and should be escalated to a more capable model.

    Args:
        result: The DelegationResult from a sub-agent

    Returns:
        True if escalation is recommended
    """
    # Already using the most capable model
    if result.model == ESCALATION_MODEL:
        return False

    # Check for failure status
    if result.status == TaskStatus.FAILED:
        return True

    # Check output for struggle indicators
    if result.output:
        output_lower = result.output.lower()
        for indicator in STRUGGLE_INDICATORS:
            if indicator.lower() in output_lower:
                return True

    # Check for empty or very short output (might indicate struggle)
    if result.status == TaskStatus.COMPLETED and result.output:
        if len(result.output.strip()) < 50:
            return True

    return False


def get_escalation_model(current_model: str) -> Optional[str]:
    """
    Get the next model tier for escalation.

    Args:
        current_model: Current model being used

    Returns:
        Next model tier, or None if already at highest
    """
    # Model escalation chain: Haiku → Sonnet → Opus
    escalation_chain = {
        FAST_MODEL: DEFAULT_SUBAGENT_MODEL,  # Haiku → Sonnet
        DEFAULT_SUBAGENT_MODEL: ESCALATION_MODEL,  # Sonnet → Opus
        ESCALATION_MODEL: None,  # Opus is highest, no escalation
    }
    return escalation_chain.get(current_model)


@dataclass
class DelegationResult:
    """Result from a delegated sub-agent task."""
    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
    duration_seconds: float = 0.0
    model: str = DEFAULT_SUBAGENT_MODEL
    context_file: Optional[str] = None
    recursion_depth: int = 0
    task_type: TaskType = TaskType.UNKNOWN
    subagent_type: SubagentType = SubagentType.GENERAL
    allowed_tools: Optional[List[str]] = None  # Tools this subagent can use
    escalated: bool = False  # True if this was escalated from a lower model
    original_model: Optional[str] = None  # Model before escalation
    retry_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums to strings for JSON
        data["status"] = self.status.value
        data["task_type"] = self.task_type.value
        data["subagent_type"] = self.subagent_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelegationResult":
        """Create from dictionary."""
        data["status"] = TaskStatus(data["status"])
        data["task_type"] = TaskType(data.get("task_type", "unknown"))
        data["subagent_type"] = SubagentType(data.get("subagent_type", "general"))
        return cls(**data)


@dataclass
class TokenBudget:
    """Track token usage for the session."""
    total_budget: int = MAX_TOKEN_BUDGET
    tokens_used: int = 0
    tasks_completed: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.tokens_used)

    @property
    def usage_percent(self) -> float:
        return (self.tokens_used / self.total_budget) * 100 if self.total_budget > 0 else 0

    def can_afford(self, estimated_tokens: int) -> bool:
        return self.remaining >= estimated_tokens

    def spend(self, tokens: int) -> None:
        self.tokens_used += tokens
        self.tasks_completed += 1


# =============================================================================
# Session State Management
# =============================================================================

_session_budget: Optional[TokenBudget] = None
_delegation_history: List[DelegationResult] = []


def _get_session_budget() -> TokenBudget:
    """Get or create session token budget."""
    global _session_budget
    if _session_budget is None:
        _session_budget = TokenBudget()
    return _session_budget


def _save_result(result: DelegationResult, config: RLMConfig) -> Path:
    """Save delegation result to results directory."""
    results_path = config.get_results_path()
    results_path.mkdir(parents=True, exist_ok=True)

    result_file = results_path / f"result_{result.task_id}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    return result_file


# =============================================================================
# Recursion Control
# =============================================================================

def check_recursion_depth() -> tuple[bool, int, int]:
    """
    Check if we can recurse further.

    Returns:
        Tuple of (can_recurse, current_depth, max_depth)
    """
    current = get_current_recursion_depth()
    max_depth = int(os.environ.get(ENV_RLM_MAX_DEPTH, str(MAX_RECURSION_DEPTH)))
    return current < max_depth, current, max_depth


def increment_recursion_depth() -> int:
    """Increment recursion depth and return new value."""
    current = get_current_recursion_depth()
    new_depth = current + 1
    set_recursion_depth(new_depth)
    return new_depth


def decrement_recursion_depth() -> int:
    """Decrement recursion depth and return new value."""
    current = get_current_recursion_depth()
    new_depth = max(0, current - 1)
    set_recursion_depth(new_depth)
    return new_depth


# =============================================================================
# Token Budget
# =============================================================================

def get_token_budget_remaining() -> int:
    """Get remaining token budget for the session."""
    return _get_session_budget().remaining


def get_token_budget_status() -> Dict[str, Any]:
    """Get detailed token budget status."""
    budget = _get_session_budget()
    return {
        "total_budget": budget.total_budget,
        "tokens_used": budget.tokens_used,
        "remaining": budget.remaining,
        "usage_percent": budget.usage_percent,
        "tasks_completed": budget.tasks_completed,
    }


def reset_token_budget(new_budget: Optional[int] = None) -> None:
    """Reset the session token budget."""
    global _session_budget
    _session_budget = TokenBudget(
        total_budget=new_budget or MAX_TOKEN_BUDGET
    )


# =============================================================================
# Sub-Agent Delegation
# =============================================================================

def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough approximation)."""
    # Rough estimate: ~4 characters per token
    return len(text) // 4


def _build_subagent_prompt(
    instruction: str,
    context_file: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> str:
    """Build the prompt for a sub-agent."""
    prompt_parts = [
        "You are an RLM Sub-agent processing a specific chunk of data.",
        "Focus ONLY on the task given. Do NOT attempt to access other files.",
        "Return a concise, synthesized answer - not a reproduction of the input.",
        "",
        "## Task:",
        instruction,
    ]

    if context_file:
        prompt_parts.extend([
            "",
            "## Context File:",
            f"Read and analyze: {context_file}",
        ])

    if additional_context:
        prompt_parts.extend([
            "",
            "## Additional Context:",
            additional_context,
        ])

    prompt_parts.extend([
        "",
        "## Output Requirements:",
        "- Be concise and focused",
        "- Extract only relevant information",
        "- Do NOT reproduce large chunks of input",
        "- If the answer is not in the context, say so clearly",
    ])

    return "\n".join(prompt_parts)


def delegate_task(
    instruction: str,
    context_file: Optional[str] = None,
    model: Optional[str] = None,
    subagent_type: Optional[SubagentType] = None,
    allowed_tools: Optional[List[str]] = None,
    additional_context: Optional[str] = None,
    timeout_seconds: int = 120,
    config: Optional[RLMConfig] = None,
    auto_select_model: bool = True,
    auto_escalate: bool = True,
) -> DelegationResult:
    """
    Delegate a task to a sub-agent for processing with intelligent model selection.

    Args:
        instruction: The task/question for the sub-agent
        context_file: Optional file path for the sub-agent to read
        model: Model to use (if None, auto-selects based on task type)
        subagent_type: Specific subagent type to use (determines tools)
        allowed_tools: Explicit list of allowed tools (overrides subagent_type)
        additional_context: Additional context string
        timeout_seconds: Timeout for the sub-agent call
        config: Optional RLM configuration
        auto_select_model: If True and model is None, automatically select model
        auto_escalate: If True, retry with more capable model on failure

    Returns:
        DelegationResult with the sub-agent's output
    """
    config = config or load_config()
    task_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()

    # Classify task and select model/subagent if not explicitly provided
    task_type = classify_task(instruction, context_file)

    # Determine subagent type
    if subagent_type is None:
        subagent_type = get_subagent_type_for_task(task_type)

    # Determine allowed tools
    if allowed_tools is None:
        allowed_tools = get_tools_for_subagent(subagent_type)

    # Determine model
    if model is None and auto_select_model:
        model = SUBAGENT_MODELS.get(subagent_type, get_model_for_task_type(task_type))
    elif model is None:
        model = DEFAULT_SUBAGENT_MODEL

    original_model = model

    # Check recursion depth
    can_recurse, current_depth, max_depth = check_recursion_depth()
    if not can_recurse:
        result = DelegationResult(
            task_id=task_id,
            status=TaskStatus.BLOCKED,
            error=f"Recursion limit reached: {current_depth}/{max_depth}",
            recursion_depth=current_depth,
            task_type=task_type,
            subagent_type=subagent_type,
            allowed_tools=allowed_tools,
        )
        _save_result(result, config)
        _delegation_history.append(result)
        return result

    # Estimate token cost
    prompt = _build_subagent_prompt(instruction, context_file, additional_context)
    estimated_tokens = _estimate_tokens(prompt) * 2  # Input + estimated output

    budget = _get_session_budget()
    if not budget.can_afford(estimated_tokens):
        result = DelegationResult(
            task_id=task_id,
            status=TaskStatus.BLOCKED,
            error=f"Token budget exceeded: {budget.remaining} remaining, ~{estimated_tokens} needed",
            recursion_depth=current_depth,
            task_type=task_type,
            subagent_type=subagent_type,
            allowed_tools=allowed_tools,
        )
        _save_result(result, config)
        _delegation_history.append(result)
        return result

    # Execute with potential escalation
    retry_count = 0
    escalated = False
    result = None

    while retry_count <= MAX_RETRIES_BEFORE_ESCALATION:
        # Increment recursion depth for sub-call
        new_depth = increment_recursion_depth()

        try:
            # Build the claude CLI command
            # Using claude -p (print mode) for headless operation
            cmd = [
                "claude",
                "-p",  # Print mode - non-interactive
                prompt,
            ]

            # Add model selection if not default
            if model != "sonnet":  # Claude Code default is sonnet
                cmd.extend(["--model", model])

            # Add allowed tools restriction if specified
            # Note: Claude Code uses --allowedTools flag for tool restrictions
            if allowed_tools:
                cmd.extend(["--allowedTools", ",".join(allowed_tools)])

            # Set environment for sub-process
            env = os.environ.copy()
            env[ENV_RLM_RECURSION_DEPTH] = str(new_depth)
            env[ENV_RLM_MAX_DEPTH] = str(max_depth)

            # Execute sub-agent
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=env,
                    cwd=str(Path.cwd()),
                )

                output = proc.stdout.strip()
                error = proc.stderr.strip() if proc.returncode != 0 else None
                status = TaskStatus.COMPLETED if proc.returncode == 0 else TaskStatus.FAILED

            except subprocess.TimeoutExpired:
                output = None
                error = f"Sub-agent timed out after {timeout_seconds}s"
                status = TaskStatus.FAILED

            except FileNotFoundError:
                # Claude CLI not found - provide helpful error
                output = None
                error = "Claude CLI not found. Ensure 'claude' command is available in PATH."
                status = TaskStatus.FAILED

            # Calculate duration and tokens
            duration = (datetime.now() - start_time).total_seconds()
            tokens_used = _estimate_tokens(prompt) + _estimate_tokens(output or "")
            budget.spend(tokens_used)

            result = DelegationResult(
                task_id=task_id,
                status=status,
                output=output,
                error=error,
                tokens_used=tokens_used,
                duration_seconds=duration,
                model=model,
                context_file=context_file,
                recursion_depth=new_depth,
                task_type=task_type,
                subagent_type=subagent_type,
                allowed_tools=allowed_tools,
                escalated=escalated,
                original_model=original_model if escalated else None,
                retry_count=retry_count,
            )

        except Exception as e:
            result = DelegationResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                recursion_depth=new_depth,
                task_type=task_type,
                subagent_type=subagent_type,
                allowed_tools=allowed_tools,
                escalated=escalated,
                original_model=original_model if escalated else None,
                retry_count=retry_count,
            )

        finally:
            # Always decrement recursion depth after sub-call completes
            decrement_recursion_depth()

        # Check if we should escalate
        if auto_escalate and should_escalate(result):
            next_model = get_escalation_model(model)
            if next_model:
                model = next_model
                escalated = True
                retry_count += 1
                continue  # Retry with more capable model

        # Success or no more escalation possible
        break

    # Save and track result
    _save_result(result, config)
    _delegation_history.append(result)

    return result


def smart_delegate(
    instruction: str,
    context_file: Optional[str] = None,
    additional_context: Optional[str] = None,
    timeout_seconds: int = 120,
    config: Optional[RLMConfig] = None,
) -> DelegationResult:
    """
    Smart delegation that automatically selects the best model for the task.

    This is a convenience function that wraps delegate_task with
    auto_select_model=True and auto_escalate=True.

    Model Selection:
    - Simple research/extraction → Haiku (fast, cheap)
    - Complex research/coding → Sonnet (balanced)
    - Advanced reasoning → Opus (most capable)

    Args:
        instruction: The task/question for the sub-agent
        context_file: Optional file path for the sub-agent to read
        additional_context: Additional context string
        timeout_seconds: Timeout for the sub-agent call
        config: Optional RLM configuration

    Returns:
        DelegationResult with the sub-agent's output
    """
    return delegate_task(
        instruction=instruction,
        context_file=context_file,
        model=None,  # Auto-select
        additional_context=additional_context,
        timeout_seconds=timeout_seconds,
        config=config,
        auto_select_model=True,
        auto_escalate=True,
    )


def delegate_with_model(
    instruction: str,
    model: str,
    context_file: Optional[str] = None,
    additional_context: Optional[str] = None,
    timeout_seconds: int = 120,
    config: Optional[RLMConfig] = None,
    auto_escalate: bool = False,
) -> DelegationResult:
    """
    Delegate a task with a specific model (no auto-selection).

    Args:
        instruction: The task/question for the sub-agent
        model: Model to use (haiku, sonnet, opus)
        context_file: Optional file path for the sub-agent to read
        additional_context: Additional context string
        timeout_seconds: Timeout for the sub-agent call
        config: Optional RLM configuration
        auto_escalate: If True, escalate on failure

    Returns:
        DelegationResult with the sub-agent's output
    """
    return delegate_task(
        instruction=instruction,
        context_file=context_file,
        model=model,
        additional_context=additional_context,
        timeout_seconds=timeout_seconds,
        config=config,
        auto_select_model=False,
        auto_escalate=auto_escalate,
    )


def delegate_task_batch(
    tasks: List[Dict[str, Any]],
    parallel: bool = False,
    config: Optional[RLMConfig] = None,
) -> List[DelegationResult]:
    """
    Delegate multiple tasks to sub-agents.

    Args:
        tasks: List of task dictionaries with keys: instruction, context_file, model
        parallel: If True, run tasks in parallel (uses more resources)
        config: Optional RLM configuration

    Returns:
        List of DelegationResults
    """
    config = config or load_config()
    results = []

    if parallel:
        # For parallel execution, we'd need to use concurrent.futures
        # But for safety in RLM, we run sequentially by default
        import warnings
        warnings.warn("Parallel delegation not yet implemented, running sequentially")

    for task in tasks:
        result = delegate_task(
            instruction=task.get("instruction", ""),
            context_file=task.get("context_file"),
            model=task.get("model", DEFAULT_SUBAGENT_MODEL),
            additional_context=task.get("additional_context"),
            config=config,
        )
        results.append(result)

        # Stop if we hit a blocking condition
        if result.status == TaskStatus.BLOCKED:
            break

    return results


def get_delegation_history() -> List[DelegationResult]:
    """Get the history of delegated tasks in this session."""
    return _delegation_history.copy()


def clear_delegation_history() -> int:
    """Clear delegation history and return count cleared."""
    global _delegation_history
    count = len(_delegation_history)
    _delegation_history = []
    return count


# =============================================================================
# Specialized Subagent Delegation Functions
# =============================================================================

def delegate_to_coding_agent(
    instruction: str,
    context_file: Optional[str] = None,
    additional_context: Optional[str] = None,
    timeout_seconds: int = 180,
    config: Optional[RLMConfig] = None,
) -> DelegationResult:
    """
    Delegate a coding task to a specialized coding sub-agent.

    Uses Sonnet model with tools: Edit, View, Bash, Diagnostics, CodebaseRetrieval

    Best for:
    - Implementation tasks
    - Debugging and bug fixes
    - Refactoring
    - Writing tests

    Args:
        instruction: The coding task
        context_file: Optional file to work with
        additional_context: Additional context
        timeout_seconds: Timeout (default 180s for complex coding)
        config: Optional RLM configuration

    Returns:
        DelegationResult
    """
    return delegate_task(
        instruction=instruction,
        context_file=context_file,
        subagent_type=SubagentType.CODING,
        additional_context=additional_context,
        timeout_seconds=timeout_seconds,
        config=config,
        auto_select_model=True,
        auto_escalate=True,
    )


def delegate_to_research_agent(
    instruction: str,
    context_file: Optional[str] = None,
    additional_context: Optional[str] = None,
    timeout_seconds: int = 60,
    config: Optional[RLMConfig] = None,
) -> DelegationResult:
    """
    Delegate a research task to a fast research sub-agent.

    Uses Haiku model with tools: WebSearch, WebFetch, Read

    Best for:
    - Web searches
    - Information gathering
    - Document scanning
    - Quick lookups

    Args:
        instruction: The research task
        context_file: Optional file to read
        additional_context: Additional context
        timeout_seconds: Timeout (default 60s for fast research)
        config: Optional RLM configuration

    Returns:
        DelegationResult
    """
    return delegate_task(
        instruction=instruction,
        context_file=context_file,
        subagent_type=SubagentType.RESEARCH,
        additional_context=additional_context,
        timeout_seconds=timeout_seconds,
        config=config,
        auto_select_model=True,
        auto_escalate=True,
    )


def delegate_to_data_agent(
    instruction: str,
    context_file: Optional[str] = None,
    additional_context: Optional[str] = None,
    timeout_seconds: int = 120,
    config: Optional[RLMConfig] = None,
) -> DelegationResult:
    """
    Delegate a data processing task to a data sub-agent.

    Uses Haiku model with tools: Bash, Read, Write

    Best for:
    - Data extraction
    - File transformations
    - Aggregation tasks
    - Batch processing

    Args:
        instruction: The data processing task
        context_file: Optional file to process
        additional_context: Additional context
        timeout_seconds: Timeout (default 120s)
        config: Optional RLM configuration

    Returns:
        DelegationResult
    """
    return delegate_task(
        instruction=instruction,
        context_file=context_file,
        subagent_type=SubagentType.DATA_PROCESSING,
        additional_context=additional_context,
        timeout_seconds=timeout_seconds,
        config=config,
        auto_select_model=True,
        auto_escalate=True,
    )


def delegate_to_rlm_processor(
    instruction: str,
    context_file: str,
    additional_context: Optional[str] = None,
    timeout_seconds: int = 90,
    config: Optional[RLMConfig] = None,
) -> DelegationResult:
    """
    Delegate an RLM chunk processing task to a specialized processor.

    Uses Haiku model with tools: Read, Bash

    Best for:
    - Processing individual chunks of large files
    - Extracting information from file segments
    - RLM recursive processing

    Args:
        instruction: The processing task
        context_file: The chunk file to process (required)
        additional_context: Additional context
        timeout_seconds: Timeout (default 90s)
        config: Optional RLM configuration

    Returns:
        DelegationResult
    """
    return delegate_task(
        instruction=instruction,
        context_file=context_file,
        subagent_type=SubagentType.RLM_PROCESSOR,
        additional_context=additional_context,
        timeout_seconds=timeout_seconds,
        config=config,
        auto_select_model=True,
        auto_escalate=True,
    )


def delegate_to_advanced_agent(
    instruction: str,
    context_file: Optional[str] = None,
    additional_context: Optional[str] = None,
    timeout_seconds: int = 300,
    config: Optional[RLMConfig] = None,
) -> DelegationResult:
    """
    Delegate a complex task to the advanced reasoning sub-agent.

    Uses Opus model with ALL tools available.

    Best for:
    - Complex architectural decisions
    - Difficult problem-solving
    - Tasks where simpler agents failed
    - Edge cases requiring deep analysis

    Args:
        instruction: The complex task
        context_file: Optional file to analyze
        additional_context: Additional context
        timeout_seconds: Timeout (default 300s for complex reasoning)
        config: Optional RLM configuration

    Returns:
        DelegationResult
    """
    return delegate_task(
        instruction=instruction,
        context_file=context_file,
        subagent_type=SubagentType.ADVANCED,
        additional_context=additional_context,
        timeout_seconds=timeout_seconds,
        config=config,
        auto_select_model=True,
        auto_escalate=False,  # Already at highest tier
    )
