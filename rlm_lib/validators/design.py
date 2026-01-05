"""
Design Validator for the Quality Gate System.

Validates design patterns, code structure, and implementation quality.
"""

import re
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

from .base import BaseValidator, ValidationResult, ValidationIssue, ValidationSeverity


class DesignValidator(BaseValidator):
    """
    Validates design quality and patterns.

    Checks for:
    - Design pattern violations
    - Code complexity (cyclomatic complexity)
    - Naming conventions
    - Documentation completeness
    - SOLID principle violations
    """

    # Common naming patterns
    CLASS_NAME_PATTERN = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
    FUNCTION_NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
    CONSTANT_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

    # Anti-patterns to detect
    ANTI_PATTERNS = {
        "god_class": {"max_methods": 20, "max_attributes": 15},
        "long_method": {"max_lines": 50},
        "deep_nesting": {"max_depth": 4},
        "long_parameter_list": {"max_params": 5},
    }

    def __init__(self, enabled: bool = True):
        super().__init__("DesignValidator", enabled)
        self._rules = {
            "DES001": True,  # Check naming conventions
            "DES002": True,  # Check for anti-patterns
            "DES003": True,  # Check documentation
            "DES004": True,  # Check complexity
            "DES005": True,  # Check SOLID violations
        }

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate design quality from the context."""
        start_time = time.time()
        issues: List[ValidationIssue] = []

        classes = context.get("classes", [])
        functions = context.get("functions", [])
        code_files = context.get("code_files", {})

        # DES001: Check naming conventions
        if self.is_rule_enabled("DES001"):
            issues.extend(self._check_naming_conventions(classes, functions))

        # DES002: Check for anti-patterns
        if self.is_rule_enabled("DES002"):
            issues.extend(self._check_anti_patterns(classes, functions))

        # DES003: Check documentation
        if self.is_rule_enabled("DES003"):
            issues.extend(self._check_documentation(classes, functions))

        # DES004: Check complexity
        if self.is_rule_enabled("DES004"):
            issues.extend(self._check_complexity(functions))

        # DES005: Check SOLID violations
        if self.is_rule_enabled("DES005"):
            issues.extend(self._check_solid_violations(classes))

        # Calculate score
        total_items = len(classes) + len(functions)
        if total_items == 0:
            score = 1.0
        else:
            error_count = sum(1 for i in issues if i.severity in
                            (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL))
            warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
            score = max(0.0, 1.0 - (error_count * 0.1) - (warning_count * 0.02))

        passed = all(i.severity != ValidationSeverity.CRITICAL for i in issues)

        return self._create_result(
            passed=passed,
            issues=issues,
            score=score,
            duration_ms=self._elapsed(start_time),
            metadata={"classes_checked": len(classes), "functions_checked": len(functions)},
        )

    def _check_naming_conventions(
        self, classes: List[Dict], functions: List[Dict]
    ) -> List[ValidationIssue]:
        """Check naming conventions for classes and functions."""
        issues = []

        for cls in classes:
            name = cls.get("name", "")
            if name and not self.CLASS_NAME_PATTERN.match(name):
                issues.append(self._issue(
                    "DES001",
                    f"Class '{name}' does not follow PascalCase naming convention",
                    ValidationSeverity.WARNING,
                    suggestion="Use PascalCase for class names (e.g., MyClass)",
                    rule_id="DES001",
                ))

        for func in functions:
            name = func.get("name", "")
            if name and not name.startswith("_") and not self.FUNCTION_NAME_PATTERN.match(name):
                # Allow dunder methods
                if not (name.startswith("__") and name.endswith("__")):
                    issues.append(self._issue(
                        "DES001",
                        f"Function '{name}' does not follow snake_case naming convention",
                        ValidationSeverity.WARNING,
                        suggestion="Use snake_case for function names (e.g., my_function)",
                        rule_id="DES001",
                    ))

        return issues

    def _check_anti_patterns(
        self, classes: List[Dict], functions: List[Dict]
    ) -> List[ValidationIssue]:
        """Check for common anti-patterns."""
        issues = []

        for cls in classes:
            name = cls.get("name", "Unknown")
            method_count = cls.get("method_count", 0)
            attribute_count = cls.get("attribute_count", 0)

            # God class detection
            if (method_count > self.ANTI_PATTERNS["god_class"]["max_methods"] or
                attribute_count > self.ANTI_PATTERNS["god_class"]["max_attributes"]):
                issues.append(self._issue(
                    "DES002",
                    f"Class '{name}' may be a God Class (methods: {method_count}, attributes: {attribute_count})",
                    ValidationSeverity.WARNING,
                    suggestion="Consider splitting into smaller, focused classes",
                    rule_id="DES002",
                ))

        for func in functions:
            name = func.get("name", "Unknown")
            line_count = func.get("line_count", 0)
            param_count = func.get("parameter_count", 0)
            nesting_depth = func.get("nesting_depth", 0)

            # Long method detection
            if line_count > self.ANTI_PATTERNS["long_method"]["max_lines"]:
                issues.append(self._issue(
                    "DES002",
                    f"Function '{name}' is too long ({line_count} lines)",
                    ValidationSeverity.WARNING,
                    suggestion="Consider breaking into smaller functions",
                    rule_id="DES002",
                ))

            # Long parameter list
            if param_count > self.ANTI_PATTERNS["long_parameter_list"]["max_params"]:
                issues.append(self._issue(
                    "DES002",
                    f"Function '{name}' has too many parameters ({param_count})",
                    ValidationSeverity.WARNING,
                    suggestion="Consider using a parameter object or builder pattern",
                    rule_id="DES002",
                ))

            # Deep nesting
            if nesting_depth > self.ANTI_PATTERNS["deep_nesting"]["max_depth"]:
                issues.append(self._issue(
                    "DES002",
                    f"Function '{name}' has deep nesting ({nesting_depth} levels)",
                    ValidationSeverity.WARNING,
                    suggestion="Consider early returns or extracting nested logic",
                    rule_id="DES002",
                ))

        return issues

    def _check_documentation(
        self, classes: List[Dict], functions: List[Dict]
    ) -> List[ValidationIssue]:
        """Check for missing documentation."""
        issues = []

        for cls in classes:
            name = cls.get("name", "Unknown")
            has_docstring = cls.get("has_docstring", False)
            is_public = not name.startswith("_")

            if is_public and not has_docstring:
                issues.append(self._issue(
                    "DES003",
                    f"Public class '{name}' is missing documentation",
                    ValidationSeverity.INFO,
                    suggestion="Add a docstring explaining the class purpose",
                    rule_id="DES003",
                ))

        for func in functions:
            name = func.get("name", "Unknown")
            has_docstring = func.get("has_docstring", False)
            is_public = not name.startswith("_")

            if is_public and not has_docstring:
                issues.append(self._issue(
                    "DES003",
                    f"Public function '{name}' is missing documentation",
                    ValidationSeverity.INFO,
                    suggestion="Add a docstring explaining parameters and return value",
                    rule_id="DES003",
                ))

        return issues

    def _check_complexity(self, functions: List[Dict]) -> List[ValidationIssue]:
        """Check cyclomatic complexity."""
        issues = []

        for func in functions:
            name = func.get("name", "Unknown")
            complexity = func.get("cyclomatic_complexity", 0)

            if complexity > 10:
                severity = ValidationSeverity.ERROR if complexity > 20 else ValidationSeverity.WARNING
                issues.append(self._issue(
                    "DES004",
                    f"Function '{name}' has high cyclomatic complexity ({complexity})",
                    severity,
                    suggestion="Simplify by extracting logic or using polymorphism",
                    rule_id="DES004",
                ))

        return issues

    def _check_solid_violations(self, classes: List[Dict]) -> List[ValidationIssue]:
        """Check for SOLID principle violations."""
        issues = []

        for cls in classes:
            name = cls.get("name", "Unknown")

            # Single Responsibility: Too many responsibilities
            responsibility_count = cls.get("responsibility_count", 0)
            if responsibility_count > 1:
                issues.append(self._issue(
                    "DES005",
                    f"Class '{name}' may violate Single Responsibility Principle ({responsibility_count} responsibilities)",
                    ValidationSeverity.INFO,
                    suggestion="Each class should have only one reason to change",
                    rule_id="DES005",
                ))

            # Interface Segregation: Too many methods in an interface
            is_interface = cls.get("is_interface", False)
            method_count = cls.get("method_count", 0)
            if is_interface and method_count > 5:
                issues.append(self._issue(
                    "DES005",
                    f"Interface '{name}' has too many methods ({method_count}), violates ISP",
                    ValidationSeverity.WARNING,
                    suggestion="Consider splitting into smaller, focused interfaces",
                    rule_id="DES005",
                ))

            # Dependency Inversion: Concrete dependencies
            concrete_deps = cls.get("concrete_dependencies", 0)
            if concrete_deps > 3:
                issues.append(self._issue(
                    "DES005",
                    f"Class '{name}' depends on {concrete_deps} concrete classes, may violate DIP",
                    ValidationSeverity.INFO,
                    suggestion="Depend on abstractions, not concretions",
                    rule_id="DES005",
                ))

        return issues

    def _elapsed(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000
