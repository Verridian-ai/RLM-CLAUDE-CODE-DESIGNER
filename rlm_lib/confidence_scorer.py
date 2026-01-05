"""
Confidence Scoring System for the Quality Gate.

Calculates confidence scores based on validation results and context quality.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

from .validators.base import ValidationResult, ValidationSeverity


class ConfidenceLevel(Enum):
    """Confidence levels for decision making."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class ConfidenceScore:
    """
    Represents a confidence score for a decision or action.

    Combines multiple validation results and context factors
    into a single confidence measure.
    """
    overall: float  # 0.0 to 1.0
    level: ConfidenceLevel
    components: Dict[str, float] = field(default_factory=dict)
    factors: Dict[str, Any] = field(default_factory=dict)
    validation_results: List[ValidationResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def can_proceed(self) -> bool:
        """Check if confidence is high enough to proceed."""
        return self.overall >= 0.6  # HIGH or VERY_HIGH

    @property
    def needs_review(self) -> bool:
        """Check if human review is recommended."""
        return 0.4 <= self.overall < 0.6  # MEDIUM

    @property
    def should_block(self) -> bool:
        """Check if action should be blocked."""
        return self.overall < 0.4  # LOW or VERY_LOW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "level": self.level.value,
            "can_proceed": self.can_proceed,
            "needs_review": self.needs_review,
            "should_block": self.should_block,
            "components": self.components,
            "factors": self.factors,
            "validation_count": len(self.validation_results),
            "timestamp": self.timestamp.isoformat(),
        }


class ConfidenceScorer:
    """
    Calculates confidence scores from validation results and context.

    Uses a weighted combination of:
    - Validation scores
    - Context completeness
    - Historical success rate
    - Complexity factors
    """

    # Default weights for different score components
    DEFAULT_WEIGHTS = {
        "requirements": 0.25,
        "architecture": 0.20,
        "design": 0.20,
        "security": 0.25,
        "context_completeness": 0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        minimum_threshold: float = 0.4,
    ):
        """
        Initialize the confidence scorer.

        Args:
            weights: Custom weights for score components.
            minimum_threshold: Minimum score to allow proceeding.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.minimum_threshold = minimum_threshold
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def calculate(
        self,
        validation_results: List[ValidationResult],
        context: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceScore:
        """
        Calculate a confidence score from validation results.

        Args:
            validation_results: List of validation results to consider.
            context: Additional context for scoring.

        Returns:
            ConfidenceScore with overall score and components.
        """
        components: Dict[str, float] = {}

        # Map validation results to components
        validator_map = {
            "RequirementsValidator": "requirements",
            "ArchitectureValidator": "architecture",
            "DesignValidator": "design",
            "SecurityValidator": "security",
        }

        for result in validation_results:
            component = validator_map.get(result.validator_name, result.validator_name)
            components[component] = result.score

        # Add context completeness if available
        if context:
            components["context_completeness"] = self._calculate_context_completeness(context)

        # Calculate weighted overall score
        overall = 0.0
        total_weight = 0.0

        for component, comp_score in components.items():
            weight = self.weights.get(component, 0.1)
            overall += comp_score * weight
            total_weight += weight

        if total_weight > 0:
            overall = overall / total_weight

        # Determine confidence level
        level = self._score_to_level(overall)

        return ConfidenceScore(
            overall=overall,
            level=level,
            components=components,
            factors=self._extract_factors(validation_results),
            validation_results=validation_results,
        )

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert a numeric score to a confidence level."""
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _calculate_context_completeness(self, context: Dict[str, Any]) -> float:
        """Calculate how complete the context is."""
        expected_keys = [
            "requirements", "design_spec", "affected_files",
            "dependencies", "test_coverage", "related_issues",
        ]

        present = sum(1 for key in expected_keys if context.get(key))
        return present / len(expected_keys)

    def _extract_factors(
        self, validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """Extract key factors from validation results."""
        factors = {
            "total_issues": 0,
            "critical_issues": 0,
            "error_issues": 0,
            "warning_issues": 0,
            "validators_passed": 0,
            "validators_failed": 0,
        }

        for result in validation_results:
            factors["total_issues"] += len(result.issues)
            factors["critical_issues"] += result.critical_count
            factors["error_issues"] += result.error_count
            factors["warning_issues"] += result.warning_count

            if result.passed:
                factors["validators_passed"] += 1
            else:
                factors["validators_failed"] += 1

        return factors

    def can_proceed(
        self,
        validation_results: List[ValidationResult],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Quick check if confidence is high enough to proceed.

        Args:
            validation_results: Validation results to check.
            context: Additional context.

        Returns:
            True if confidence is above threshold.
        """
        score = self.calculate(validation_results, context)
        return score.can_proceed

    def get_blocking_issues(
        self, validation_results: List[ValidationResult]
    ) -> List[Dict[str, Any]]:
        """
        Get list of issues that are blocking progress.

        Returns:
            List of critical and error issues.
        """
        blocking = []

        for result in validation_results:
            for issue in result.issues:
                if issue.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR):
                    blocking.append({
                        "validator": result.validator_name,
                        "code": issue.code,
                        "message": issue.message,
                        "severity": issue.severity.value,
                        "suggestion": issue.suggestion,
                    })

        return blocking
