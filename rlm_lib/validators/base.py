"""
Base Validator Classes for the Quality Gate System.

Provides abstract base classes and common types for all validators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Set


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue found during validation."""
    code: str
    message: str
    severity: ValidationSeverity
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None
    context: Optional[str] = None
    rule_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "file_path": str(self.file_path) if self.file_path else None,
            "line_number": self.line_number,
            "column": self.column,
            "suggestion": self.suggestion,
            "context": self.context,
            "rule_id": self.rule_id,
        }


@dataclass
class ValidationResult:
    """Result of a validation check."""
    validator_name: str
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
    
    @property
    def critical_count(self) -> int:
        """Count of critical-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validator_name": self.validator_name,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "score": self.score,
            "duration_ms": self.duration_ms,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseValidator(ABC):
    """Abstract base class for all validators."""
    
    def __init__(self, name: str, enabled: bool = True):
        """Initialize the validator."""
        self.name = name
        self.enabled = enabled
        self._rules: Dict[str, bool] = {}
    
    @abstractmethod
    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """
        Perform validation on the given context.
        
        Args:
            context: Dictionary containing validation context.
                     May include: files, code, requirements, design specs, etc.
        
        Returns:
            ValidationResult with issues found.
        """
        pass
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable a specific validation rule."""
        self._rules[rule_id] = True
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable a specific validation rule."""
        self._rules[rule_id] = False
    
    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a rule is enabled."""
        return self._rules.get(rule_id, True)
    
    def get_rules(self) -> List[str]:
        """Get list of all rule IDs."""
        return list(self._rules.keys())
    
    def _create_result(
        self,
        passed: bool,
        issues: List[ValidationIssue],
        score: float,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Helper to create a ValidationResult."""
        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            issues=issues,
            score=score,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
    
    def _issue(
        self,
        code: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
        **kwargs,
    ) -> ValidationIssue:
        """Helper to create a ValidationIssue."""
        return ValidationIssue(code=code, message=message, severity=severity, **kwargs)

