"""
Architectural Coherence Validator for Enterprise RLM-CLAUDE.

Ensures architectural consistency across the codebase:
- Pattern consistency checking
- Naming convention enforcement
- Dependency direction validation
- Layer boundary enforcement
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Pattern
from enum import Enum
import re
from pathlib import Path


class CoherenceIssueType(str, Enum):
    """Types of coherence issues."""
    PATTERN_VIOLATION = "pattern_violation"
    NAMING_VIOLATION = "naming_violation"
    DEPENDENCY_VIOLATION = "dependency_violation"
    LAYER_VIOLATION = "layer_violation"
    CONSISTENCY_VIOLATION = "consistency_violation"


class IssueSeverity(str, Enum):
    """Severity levels for coherence issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CoherenceIssue:
    """An architectural coherence issue."""
    type: str
    severity: str
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    pattern_expected: Optional[str] = None
    pattern_found: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class CoherenceReport:
    """Report from coherence validation."""
    passed: bool = True
    score: float = 1.0
    issues: List[CoherenceIssue] = field(default_factory=list)
    patterns_checked: int = 0
    patterns_passed: int = 0

    def add_issue(self, issue: CoherenceIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        if issue.severity in (IssueSeverity.CRITICAL.value, IssueSeverity.HIGH.value):
            self.passed = False
        # Recalculate score
        self._update_score()

    def _update_score(self) -> None:
        """Update score based on issues."""
        if not self.issues:
            self.score = 1.0
            return

        deductions = {
            IssueSeverity.CRITICAL.value: 0.3,
            IssueSeverity.HIGH.value: 0.15,
            IssueSeverity.MEDIUM.value: 0.05,
            IssueSeverity.LOW.value: 0.02,
            IssueSeverity.INFO.value: 0.0,
        }

        total_deduction = sum(deductions.get(i.severity, 0) for i in self.issues)
        self.score = max(0.0, 1.0 - total_deduction)


@dataclass
class NamingConvention:
    """A naming convention rule."""
    name: str
    pattern: str
    applies_to: str  # "class", "function", "variable", "file", "module"
    description: str
    severity: str = IssueSeverity.MEDIUM.value


@dataclass
class LayerDefinition:
    """Definition of an architectural layer."""
    name: str
    path_patterns: List[str]
    allowed_dependencies: List[str]
    forbidden_dependencies: List[str] = field(default_factory=list)


class PatternDetector:
    """Detects architectural patterns in code."""

    # Common design patterns
    PATTERNS = {
        "singleton": r"class\s+\w+.*:\s*(?:.*\n)*?\s*_instance\s*=\s*None",
        "factory": r"def\s+create_\w+|class\s+\w+Factory",
        "observer": r"def\s+(?:add|remove)_(?:observer|listener|subscriber)",
        "strategy": r"class\s+\w+Strategy|def\s+set_strategy",
        "decorator": r"@\w+\s*\n\s*def\s+\w+|def\s+\w+_decorator",
        "repository": r"class\s+\w+Repository",
        "service": r"class\s+\w+Service",
        "controller": r"class\s+\w+Controller",
    }

    def __init__(self):
        self.detected_patterns: Dict[str, List[str]] = {}

    def detect(self, code: str, file_path: Optional[str] = None) -> Dict[str, bool]:
        """Detect patterns in code."""
        results = {}
        for pattern_name, pattern in self.PATTERNS.items():
            if re.search(pattern, code, re.MULTILINE):
                results[pattern_name] = True
                if pattern_name not in self.detected_patterns:
                    self.detected_patterns[pattern_name] = []
                if file_path:
                    self.detected_patterns[pattern_name].append(file_path)
            else:
                results[pattern_name] = False
        return results

    def get_pattern_usage(self) -> Dict[str, int]:
        """Get count of each pattern usage."""
        return {k: len(v) for k, v in self.detected_patterns.items()}


class LayerEnforcer:
    """Enforces architectural layer boundaries."""

    def __init__(self, layers: Optional[List[LayerDefinition]] = None):
        self.layers = layers or self._default_layers()
        self._build_layer_map()

    def _default_layers(self) -> List[LayerDefinition]:
        """Default layer definitions."""
        return [
            LayerDefinition(
                name="presentation",
                path_patterns=["**/views/**", "**/controllers/**", "**/ui/**"],
                allowed_dependencies=["domain", "application"],
            ),
            LayerDefinition(
                name="application",
                path_patterns=["**/services/**", "**/usecases/**"],
                allowed_dependencies=["domain"],
            ),
            LayerDefinition(
                name="domain",
                path_patterns=["**/models/**", "**/entities/**", "**/domain/**"],
                allowed_dependencies=[],
            ),
            LayerDefinition(
                name="infrastructure",
                path_patterns=["**/repositories/**", "**/adapters/**", "**/db/**"],
                allowed_dependencies=["domain", "application"],
            ),
        ]

    def _build_layer_map(self) -> None:
        """Build mapping from path patterns to layers."""
        self.layer_map: Dict[str, LayerDefinition] = {}
        for layer in self.layers:
            for pattern in layer.path_patterns:
                self.layer_map[pattern] = layer

    def get_layer(self, file_path: str) -> Optional[LayerDefinition]:
        """Get the layer for a file path."""
        from fnmatch import fnmatch
        for pattern, layer in self.layer_map.items():
            if fnmatch(file_path, pattern):
                return layer
        return None

    def check_dependency(
        self,
        from_file: str,
        to_file: str,
    ) -> Optional[CoherenceIssue]:
        """Check if a dependency between files is allowed."""
        from_layer = self.get_layer(from_file)
        to_layer = self.get_layer(to_file)

        if not from_layer or not to_layer:
            return None  # Can't determine layers

        if from_layer.name == to_layer.name:
            return None  # Same layer is OK

        if to_layer.name in from_layer.forbidden_dependencies:
            return CoherenceIssue(
                type=CoherenceIssueType.LAYER_VIOLATION.value,
                severity=IssueSeverity.HIGH.value,
                message=f"Forbidden dependency: {from_layer.name} -> {to_layer.name}",
                file_path=from_file,
                suggestion=f"Layer '{from_layer.name}' should not depend on '{to_layer.name}'",
            )

        if to_layer.name not in from_layer.allowed_dependencies:
            return CoherenceIssue(
                type=CoherenceIssueType.LAYER_VIOLATION.value,
                severity=IssueSeverity.MEDIUM.value,
                message=f"Unexpected dependency: {from_layer.name} -> {to_layer.name}",
                file_path=from_file,
                suggestion=f"Consider if this dependency is architecturally correct",
            )

        return None


class ArchitecturalCoherenceValidator:
    """Main validator for architectural coherence."""

    def __init__(
        self,
        naming_conventions: Optional[List[NamingConvention]] = None,
        layers: Optional[List[LayerDefinition]] = None,
    ):
        self.naming_conventions = naming_conventions or self._default_conventions()
        self.pattern_detector = PatternDetector()
        self.layer_enforcer = LayerEnforcer(layers)

    def _default_conventions(self) -> List[NamingConvention]:
        """Default naming conventions."""
        return [
            NamingConvention(
                name="class_pascal_case",
                pattern=r"^[A-Z][a-zA-Z0-9]*$",
                applies_to="class",
                description="Classes should use PascalCase",
            ),
            NamingConvention(
                name="function_snake_case",
                pattern=r"^[a-z_][a-z0-9_]*$",
                applies_to="function",
                description="Functions should use snake_case",
            ),
            NamingConvention(
                name="constant_upper_case",
                pattern=r"^[A-Z][A-Z0-9_]*$",
                applies_to="constant",
                description="Constants should use UPPER_CASE",
            ),
            NamingConvention(
                name="private_underscore",
                pattern=r"^_[a-z_][a-z0-9_]*$",
                applies_to="private",
                description="Private members should start with underscore",
            ),
        ]

    def validate(
        self,
        code: str,
        file_path: Optional[str] = None,
    ) -> CoherenceReport:
        """Validate code for architectural coherence."""
        report = CoherenceReport()

        # Check naming conventions
        self._check_naming(code, report)

        # Detect patterns
        patterns = self.pattern_detector.detect(code, file_path)
        report.patterns_checked = len(patterns)
        report.patterns_passed = sum(1 for v in patterns.values() if v)

        # Check pattern consistency
        self._check_pattern_consistency(code, report)

        return report

    def _check_naming(self, code: str, report: CoherenceReport) -> None:
        """Check naming conventions."""
        # Extract class names
        class_pattern = r"class\s+(\w+)"
        for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            convention = next(
                (c for c in self.naming_conventions if c.applies_to == "class"),
                None
            )
            if convention and not re.match(convention.pattern, class_name):
                report.add_issue(CoherenceIssue(
                    type=CoherenceIssueType.NAMING_VIOLATION.value,
                    severity=convention.severity,
                    message=f"Class '{class_name}' violates naming convention",
                    pattern_expected=convention.pattern,
                    pattern_found=class_name,
                    suggestion=convention.description,
                ))

        # Extract function names
        func_pattern = r"def\s+(\w+)"
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            # Skip dunder methods
            if func_name.startswith("__") and func_name.endswith("__"):
                continue
            convention = next(
                (c for c in self.naming_conventions if c.applies_to == "function"),
                None
            )
            if convention and not re.match(convention.pattern, func_name):
                report.add_issue(CoherenceIssue(
                    type=CoherenceIssueType.NAMING_VIOLATION.value,
                    severity=convention.severity,
                    message=f"Function '{func_name}' violates naming convention",
                    pattern_expected=convention.pattern,
                    pattern_found=func_name,
                    suggestion=convention.description,
                ))

    def _check_pattern_consistency(self, code: str, report: CoherenceReport) -> None:
        """Check for pattern consistency issues."""
        # Check for mixed patterns that might indicate inconsistency
        has_factory = bool(re.search(r"class\s+\w+Factory|def\s+create_\w+", code))
        has_direct_init = bool(re.search(r"\w+\s*=\s*\w+\(\)", code))

        if has_factory and has_direct_init:
            report.add_issue(CoherenceIssue(
                type=CoherenceIssueType.CONSISTENCY_VIOLATION.value,
                severity=IssueSeverity.LOW.value,
                message="Mixed object creation patterns (factory and direct instantiation)",
                suggestion="Consider using factory pattern consistently",
            ))

    def check_dependencies(
        self,
        imports: Dict[str, List[str]],
    ) -> List[CoherenceIssue]:
        """Check dependency directions across files."""
        issues = []
        for from_file, imported_files in imports.items():
            for to_file in imported_files:
                issue = self.layer_enforcer.check_dependency(from_file, to_file)
                if issue:
                    issues.append(issue)
        return issues

    def get_pattern_summary(self) -> Dict[str, int]:
        """Get summary of detected patterns."""
        return self.pattern_detector.get_pattern_usage()


def create_coherence_validator(
    naming_conventions: Optional[List[NamingConvention]] = None,
    layers: Optional[List[LayerDefinition]] = None,
) -> ArchitecturalCoherenceValidator:
    """Factory function to create a coherence validator."""
    return ArchitecturalCoherenceValidator(
        naming_conventions=naming_conventions,
        layers=layers,
    )

