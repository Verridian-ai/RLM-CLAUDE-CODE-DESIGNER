"""
Requirements Validator for the Quality Gate System.

Validates that requirements are complete, unambiguous, and implementable.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from .base import BaseValidator, ValidationResult, ValidationIssue, ValidationSeverity


@dataclass
class Requirement:
    """A single requirement to validate."""
    id: str
    text: str
    priority: str = "medium"
    source: Optional[str] = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class RequirementsValidator(BaseValidator):
    """
    Validates requirements for completeness and clarity.

    Checks for:
    - Ambiguous language
    - Missing acceptance criteria
    - Incomplete specifications
    - Circular dependencies
    - Testability
    """

    # Ambiguous words that often indicate unclear requirements
    AMBIGUOUS_WORDS = {
        "should", "could", "might", "may", "possibly", "probably",
        "appropriate", "reasonable", "adequate", "sufficient",
        "easy", "simple", "fast", "slow", "good", "bad",
        "user-friendly", "flexible", "robust", "scalable",
        "efficient", "effective", "intuitive", "seamless",
        "etc", "and so on", "and more", "various", "some",
    }

    # Words that indicate measurable requirements (good)
    MEASURABLE_WORDS = {
        "must", "shall", "will", "exactly", "at least", "at most",
        "within", "seconds", "milliseconds", "percent", "bytes",
        "greater than", "less than", "equal to", "between",
    }

    def __init__(self, enabled: bool = True):
        super().__init__("RequirementsValidator", enabled)
        self._rules = {
            "REQ001": True,  # Check for ambiguous language
            "REQ002": True,  # Check for measurability
            "REQ003": True,  # Check for completeness
            "REQ004": True,  # Check for testability
            "REQ005": True,  # Check for circular dependencies
        }

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate requirements from the context."""
        start_time = time.time()
        issues: List[ValidationIssue] = []

        requirements = context.get("requirements", [])
        if isinstance(requirements, str):
            requirements = self._parse_requirements_text(requirements)

        if not requirements:
            issues.append(self._issue(
                "REQ000",
                "No requirements provided for validation",
                ValidationSeverity.ERROR,
            ))
            return self._create_result(False, issues, 0.0, self._elapsed(start_time))

        # Validate each requirement
        for req in requirements:
            if isinstance(req, dict):
                req = Requirement(**req)
            elif isinstance(req, str):
                req = Requirement(id="auto", text=req)

            issues.extend(self._validate_requirement(req))

        # Check for circular dependencies
        if self.is_rule_enabled("REQ005"):
            issues.extend(self._check_circular_dependencies(requirements))

        # Calculate score
        total_checks = len(requirements) * 4  # 4 checks per requirement
        failed_checks = len([i for i in issues if i.severity in
                            (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)])
        score = max(0.0, 1.0 - (failed_checks / max(1, total_checks)))

        passed = all(i.severity != ValidationSeverity.CRITICAL for i in issues)

        return self._create_result(
            passed=passed,
            issues=issues,
            score=score,
            duration_ms=self._elapsed(start_time),
            metadata={"requirements_count": len(requirements)},
        )

    def _validate_requirement(self, req: Requirement) -> List[ValidationIssue]:
        """Validate a single requirement."""
        issues = []
        text_lower = req.text.lower()

        # REQ001: Check for ambiguous language
        if self.is_rule_enabled("REQ001"):
            found_ambiguous = [w for w in self.AMBIGUOUS_WORDS if w in text_lower]
            if found_ambiguous:
                issues.append(self._issue(
                    "REQ001",
                    f"Requirement '{req.id}' contains ambiguous language: {', '.join(found_ambiguous[:3])}",
                    ValidationSeverity.WARNING,
                    suggestion=f"Replace ambiguous terms with specific, measurable criteria",
                    rule_id="REQ001",
                ))

        # REQ002: Check for measurability
        if self.is_rule_enabled("REQ002"):
            has_measurable = any(w in text_lower for w in self.MEASURABLE_WORDS)
            if not has_measurable and len(req.text) > 20:
                issues.append(self._issue(
                    "REQ002",
                    f"Requirement '{req.id}' lacks measurable criteria",
                    ValidationSeverity.WARNING,
                    suggestion="Add specific metrics or acceptance criteria",
                    rule_id="REQ002",
                ))

        # REQ003: Check for completeness
        if self.is_rule_enabled("REQ003"):
            if len(req.text.strip()) < 10:
                issues.append(self._issue(
                    "REQ003",
                    f"Requirement '{req.id}' is too brief to be complete",
                    ValidationSeverity.ERROR,
                    rule_id="REQ003",
                ))

        # REQ004: Check for testability
        if self.is_rule_enabled("REQ004"):
            untestable_patterns = ["always", "never", "all cases", "every possible"]
            if any(p in text_lower for p in untestable_patterns):
                issues.append(self._issue(
                    "REQ004",
                    f"Requirement '{req.id}' may be difficult to test exhaustively",
                    ValidationSeverity.INFO,
                    suggestion="Consider defining specific test scenarios",
                    rule_id="REQ004",
                ))

        return issues

    def _check_circular_dependencies(self, requirements: List) -> List[ValidationIssue]:
        """Check for circular dependencies between requirements."""
        issues = []

        # Build dependency graph
        deps: Dict[str, Set[str]] = {}
        for req in requirements:
            if isinstance(req, dict):
                req_id = req.get("id", "unknown")
                req_deps = req.get("dependencies", [])
            elif hasattr(req, "id"):
                req_id = req.id
                req_deps = req.dependencies or []
            else:
                continue
            deps[req_id] = set(req_deps)

        # Detect cycles using DFS
        def has_cycle(node: str, visited: Set[str], path: Set[str]) -> Optional[List[str]]:
            visited.add(node)
            path.add(node)

            for dep in deps.get(node, []):
                if dep in path:
                    return [node, dep]
                if dep not in visited:
                    cycle = has_cycle(dep, visited, path)
                    if cycle:
                        return cycle

            path.remove(node)
            return None

        visited: Set[str] = set()
        for req_id in deps:
            if req_id not in visited:
                cycle = has_cycle(req_id, visited, set())
                if cycle:
                    issues.append(self._issue(
                        "REQ005",
                        f"Circular dependency detected: {' -> '.join(cycle)}",
                        ValidationSeverity.ERROR,
                        rule_id="REQ005",
                    ))

        return issues

    def _parse_requirements_text(self, text: str) -> List[Requirement]:
        """Parse requirements from plain text."""
        requirements = []
        lines = text.strip().split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try to extract ID from patterns like "REQ-001:" or "1."
            match = re.match(r"^(REQ[-_]?\d+|[0-9]+\.?)\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
            if match:
                req_id = match.group(1).rstrip(".")
                text = match.group(2)
            else:
                req_id = f"REQ{i+1:03d}"
                text = line

            requirements.append(Requirement(id=req_id, text=text))

        return requirements

    def _elapsed(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000
