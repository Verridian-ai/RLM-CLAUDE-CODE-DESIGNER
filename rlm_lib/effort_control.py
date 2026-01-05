"""
Effort Control for Opus 4.5 Extended Thinking.

Controls the "effort" parameter for Claude Opus 4.5 to manage
reasoning depth based on task complexity. Enterprise mode defaults
to HIGH effort for architectural and critical tasks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set


class EffortLevel(str, Enum):
    """
    Opus 4.5 effort levels for extended thinking.

    LOW: Fast, minimal reasoning (similar to Sonnet performance)
    MEDIUM: Balanced (76% fewer tokens than HIGH, matches Sonnet quality)
    HIGH: Maximum reasoning depth for complex architecture
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskType(str, Enum):
    """Types of tasks for effort classification."""
    ARCHITECTURE = "architecture"
    DEBUGGING = "debugging"
    SECURITY_REVIEW = "security_review"
    IMPLEMENTATION = "implementation"
    REFACTORING = "refactoring"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    SIMPLE_FIX = "simple_fix"
    FORMATTING = "formatting"
    TESTING = "testing"
    UNKNOWN = "unknown"


@dataclass
class Task:
    """Represents a task to be processed."""
    description: str
    task_type: Optional[TaskType] = None
    affects_critical_path: bool = False
    file_count: int = 0
    complexity_score: float = 0.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EffortController:
    """
    Controls Opus 4.5 effort based on task complexity.
    Enterprise mode: Default to HIGH for architectural tasks.
    Quality-first: When in doubt, use HIGH.
    """

    # Task type to effort mapping
    EFFORT_MAPPING: Dict[TaskType, EffortLevel] = {
        TaskType.ARCHITECTURE: EffortLevel.HIGH,
        TaskType.DEBUGGING: EffortLevel.HIGH,
        TaskType.SECURITY_REVIEW: EffortLevel.HIGH,
        TaskType.CODE_REVIEW: EffortLevel.HIGH,
        TaskType.IMPLEMENTATION: EffortLevel.MEDIUM,
        TaskType.REFACTORING: EffortLevel.MEDIUM,
        TaskType.TESTING: EffortLevel.MEDIUM,
        TaskType.DOCUMENTATION: EffortLevel.LOW,
        TaskType.SIMPLE_FIX: EffortLevel.LOW,
        TaskType.FORMATTING: EffortLevel.LOW,
        TaskType.UNKNOWN: EffortLevel.HIGH,  # Default to HIGH when uncertain
    }

    # Keywords that indicate high-effort tasks
    HIGH_EFFORT_KEYWORDS: Set[str] = {
        "architecture", "design", "security", "critical", "complex",
        "refactor", "migrate", "breaking", "api", "interface",
        "performance", "scalability", "concurrency", "threading",
        "database", "schema", "authentication", "authorization",
    }

    # Keywords that indicate low-effort tasks
    LOW_EFFORT_KEYWORDS: Set[str] = {
        "typo", "comment", "format", "lint", "style", "rename",
        "simple", "minor", "trivial", "quick", "fix",
    }

    def __init__(
        self,
        default_effort: EffortLevel = EffortLevel.HIGH,
        enterprise_mode: bool = True,
    ):
        """
        Initialize the effort controller.

        Args:
            default_effort: Default effort level when uncertain.
            enterprise_mode: If True, bias toward higher effort.
        """
        self.default_effort = default_effort
        self.enterprise_mode = enterprise_mode

    def get_effort_for_task(self, task: Task) -> EffortLevel:
        """
        Determine appropriate effort level for a task.
        Quality-first: When in doubt, use HIGH.

        Args:
            task: The task to evaluate.

        Returns:
            Appropriate EffortLevel.
        """
        # If task type is specified, use mapping
        if task.task_type:
            base_effort = self.EFFORT_MAPPING.get(task.task_type, self.default_effort)
        else:
            # Classify based on description
            base_effort = self._classify_from_description(task.description)

        # Upgrade effort if task affects critical path
        if task.affects_critical_path:
            return EffortLevel.HIGH

        # Upgrade effort if high complexity
        if task.complexity_score > 0.7:
            return EffortLevel.HIGH

        # Upgrade effort if many files affected
        if task.file_count > 10:
            return EffortLevel.HIGH

        # Enterprise mode biases toward higher effort
        if self.enterprise_mode and base_effort == EffortLevel.LOW:
            return EffortLevel.MEDIUM

        return base_effort

    def _classify_from_description(self, description: str) -> EffortLevel:
        """Classify effort level from task description."""
        desc_lower = description.lower()

        # Check for high-effort keywords
        high_count = sum(1 for kw in self.HIGH_EFFORT_KEYWORDS if kw in desc_lower)
        low_count = sum(1 for kw in self.LOW_EFFORT_KEYWORDS if kw in desc_lower)

        if high_count > low_count:
            return EffortLevel.HIGH
        elif low_count > high_count:
            return EffortLevel.LOW
        else:
            return self.default_effort

    def classify_task(self, description: str) -> TaskType:
        """
        Classify a task based on its description.

        Args:
            description: Task description text.

        Returns:
            Classified TaskType.
        """
        desc_lower = description.lower()

        # Architecture keywords
        if any(kw in desc_lower for kw in ["architecture", "design", "structure", "pattern"]):
            return TaskType.ARCHITECTURE

        # Security keywords
        if any(kw in desc_lower for kw in ["security", "auth", "permission", "vulnerability"]):
            return TaskType.SECURITY_REVIEW

        # Debugging keywords
        if any(kw in desc_lower for kw in ["debug", "bug", "error", "fix", "issue"]):
            return TaskType.DEBUGGING

        # Refactoring keywords
        if any(kw in desc_lower for kw in ["refactor", "clean", "improve", "optimize"]):
            return TaskType.REFACTORING

        # Testing keywords
        if any(kw in desc_lower for kw in ["test", "spec", "coverage", "unit"]):
            return TaskType.TESTING

        # Documentation keywords
        if any(kw in desc_lower for kw in ["document", "readme", "comment", "docstring"]):
            return TaskType.DOCUMENTATION

        # Formatting keywords
        if any(kw in desc_lower for kw in ["format", "lint", "style", "prettier"]):
            return TaskType.FORMATTING

        # Implementation is the default for code-related tasks
        if any(kw in desc_lower for kw in ["implement", "add", "create", "build", "feature"]):
            return TaskType.IMPLEMENTATION

        return TaskType.UNKNOWN

    def get_model_recommendation(self, effort: EffortLevel) -> str:
        """
        Get recommended model based on effort level.

        Args:
            effort: The effort level.

        Returns:
            Model identifier string.
        """
        model_map = {
            EffortLevel.HIGH: "claude-opus-4-5-20250514",
            EffortLevel.MEDIUM: "claude-sonnet-4-5-20250514",
            EffortLevel.LOW: "claude-haiku-4-5-20250514",
        }
        return model_map.get(effort, "claude-opus-4-5-20250514")

    def to_api_params(self, effort: EffortLevel) -> Dict[str, Any]:
        """
        Convert effort level to API parameters.

        Args:
            effort: The effort level.

        Returns:
            Dictionary of API parameters.
        """
        return {
            "model": self.get_model_recommendation(effort),
            "thinking": {
                "type": "enabled",
                "budget_tokens": self._get_budget_tokens(effort),
            },
        }

    def _get_budget_tokens(self, effort: EffortLevel) -> int:
        """Get thinking budget tokens for effort level."""
        budget_map = {
            EffortLevel.HIGH: 32000,
            EffortLevel.MEDIUM: 16000,
            EffortLevel.LOW: 8000,
        }
        return budget_map.get(effort, 32000)


def get_effort_for_description(description: str) -> EffortLevel:
    """
    Convenience function to get effort level from description.

    Args:
        description: Task description.

    Returns:
        Appropriate EffortLevel.
    """
    controller = EffortController()
    task = Task(description=description)
    return controller.get_effort_for_task(task)
