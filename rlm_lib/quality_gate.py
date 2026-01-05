"""
Quality Gate System for Enterprise RLM-CLAUDE.

Orchestrates validation and enforces quality thresholds before allowing
actions to proceed. This is the primary enforcement point for ensuring
code quality and design compliance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Type

from .validators.base import BaseValidator, ValidationResult, ValidationSeverity
from .validators.requirements import RequirementsValidator
from .validators.architecture import ArchitectureValidator
from .validators.design import DesignValidator
from .validators.security import SecurityValidator
from .confidence_scorer import ConfidenceScorer, ConfidenceScore, ConfidenceLevel


class GateDecision(Enum):
    """Decision from the quality gate."""
    PASS = "pass"           # All checks passed, proceed
    REVIEW = "review"       # Needs human review
    BLOCK = "block"         # Blocked due to issues
    SKIP = "skip"           # Gate was skipped (disabled)


@dataclass
class GateResult:
    """Result from running the quality gate."""
    decision: GateDecision
    confidence: ConfidenceScore
    validation_results: List[ValidationResult] = field(default_factory=list)
    blocking_issues: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if gate passed."""
        return self.decision == GateDecision.PASS

    @property
    def needs_review(self) -> bool:
        """Check if human review is required."""
        return self.decision == GateDecision.REVIEW

    @property
    def blocked(self) -> bool:
        """Check if gate blocked the action."""
        return self.decision == GateDecision.BLOCK

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "passed": self.passed,
            "needs_review": self.needs_review,
            "blocked": self.blocked,
            "confidence": self.confidence.to_dict(),
            "validation_results": [r.to_dict() for r in self.validation_results],
            "blocking_issues": self.blocking_issues,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class QualityGate:
    """
    Quality Gate that orchestrates validation and enforces thresholds.

    The gate runs multiple validators against a context and combines
    their results into a single decision (PASS/REVIEW/BLOCK).
    """

    def __init__(
        self,
        enabled: bool = True,
        pass_threshold: float = 0.8,
        review_threshold: float = 0.6,
        validators: Optional[List[BaseValidator]] = None,
        scorer: Optional[ConfidenceScorer] = None,
    ):
        """
        Initialize the quality gate.

        Args:
            enabled: Whether the gate is active.
            pass_threshold: Minimum score to pass without review.
            review_threshold: Minimum score to allow with review.
            validators: List of validators to run.
            scorer: Confidence scorer for combining results.
        """
        self.enabled = enabled
        self.pass_threshold = pass_threshold
        self.review_threshold = review_threshold
        self.scorer = scorer or ConfidenceScorer()

        # Initialize default validators if none provided
        if validators is None:
            self.validators = [
                RequirementsValidator(),
                ArchitectureValidator(),
                DesignValidator(),
                SecurityValidator(),
            ]
        else:
            self.validators = validators

    def check(self, context: Dict[str, Any]) -> GateResult:
        """
        Run all validators and make a gate decision.

        Args:
            context: Context dictionary containing validation targets.

        Returns:
            GateResult with decision and details.
        """
        import time
        start_time = time.time()

        if not self.enabled:
            return GateResult(
                decision=GateDecision.SKIP,
                confidence=ConfidenceScore(
                    overall=1.0,
                    level=ConfidenceLevel.VERY_HIGH,
                ),
                metadata={"reason": "Gate disabled"},
            )

        # Run all validators
        validation_results: List[ValidationResult] = []
        for validator in self.validators:
            if validator.enabled:
                result = validator.validate(context)
                validation_results.append(result)

        # Calculate confidence score
        confidence = self.scorer.calculate(validation_results, context)

        # Get blocking issues
        blocking_issues = self.scorer.get_blocking_issues(validation_results)

        # Make decision based on confidence
        decision = self._make_decision(confidence, blocking_issues)

        duration_ms = (time.time() - start_time) * 1000

        return GateResult(
            decision=decision,
            confidence=confidence,
            validation_results=validation_results,
            blocking_issues=blocking_issues,
            duration_ms=duration_ms,
            metadata={"validators_run": len(validation_results)},
        )

    def _make_decision(
        self,
        confidence: ConfidenceScore,
        blocking_issues: List[Dict[str, Any]],
    ) -> GateDecision:
        """
        Make a gate decision based on confidence and issues.

        Args:
            confidence: The calculated confidence score.
            blocking_issues: List of critical/error issues.

        Returns:
            GateDecision (PASS/REVIEW/BLOCK).
        """
        # Any critical security issue is an automatic block
        has_critical = any(
            issue["severity"] == "critical" for issue in blocking_issues
        )
        if has_critical:
            return GateDecision.BLOCK

        # Check against thresholds
        if confidence.overall >= self.pass_threshold:
            return GateDecision.PASS
        elif confidence.overall >= self.review_threshold:
            return GateDecision.REVIEW
        else:
            return GateDecision.BLOCK

    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator to the gate."""
        self.validators.append(validator)

    def remove_validator(self, name: str) -> bool:
        """Remove a validator by name."""
        for i, v in enumerate(self.validators):
            if v.name == name:
                self.validators.pop(i)
                return True
        return False

    def enable_validator(self, name: str) -> bool:
        """Enable a validator by name."""
        for v in self.validators:
            if v.name == name:
                v.enabled = True
                return True
        return False

    def disable_validator(self, name: str) -> bool:
        """Disable a validator by name."""
        for v in self.validators:
            if v.name == name:
                v.enabled = False
                return True
        return False

    def can_proceed(self, context: Dict[str, Any]) -> bool:
        """
        Quick check if the gate would allow proceeding.

        Args:
            context: Context to validate.

        Returns:
            True if gate would pass or allow with review.
        """
        result = self.check(context)
        return result.decision in (GateDecision.PASS, GateDecision.REVIEW, GateDecision.SKIP)

    def get_status(self) -> Dict[str, Any]:
        """Get current gate status."""
        return {
            "enabled": self.enabled,
            "pass_threshold": self.pass_threshold,
            "review_threshold": self.review_threshold,
            "validators": [
                {"name": v.name, "enabled": v.enabled}
                for v in self.validators
            ],
        }


def create_quality_gate(
    strict: bool = False,
    security_only: bool = False,
) -> QualityGate:
    """
    Factory function to create a quality gate with common configurations.

    Args:
        strict: Use strict thresholds (higher requirements).
        security_only: Only run security validation.

    Returns:
        Configured QualityGate instance.
    """
    if strict:
        pass_threshold = 0.9
        review_threshold = 0.75
    else:
        pass_threshold = 0.8
        review_threshold = 0.6

    if security_only:
        validators = [SecurityValidator()]
    else:
        validators = None  # Use defaults

    return QualityGate(
        pass_threshold=pass_threshold,
        review_threshold=review_threshold,
        validators=validators,
    )
