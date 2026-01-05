"""
Architecture Validator for the Quality Gate System.

Validates architectural patterns, dependency management, and structural integrity.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

from .base import BaseValidator, ValidationResult, ValidationIssue, ValidationSeverity


class ArchitectureValidator(BaseValidator):
    """
    Validates architectural quality of the codebase.

    Checks for:
    - Circular dependencies between modules
    - Layer violations (e.g., UI calling database directly)
    - Missing abstractions
    - Improper coupling
    - Module cohesion
    """

    # Default layer hierarchy (lower layers should not depend on higher layers)
    DEFAULT_LAYERS = {
        "presentation": 0,  # UI, CLI, API endpoints
        "application": 1,   # Use cases, orchestration
        "domain": 2,        # Business logic, entities
        "infrastructure": 3, # Database, external services
    }

    # Patterns that suggest specific layers
    LAYER_PATTERNS = {
        "presentation": ["view", "controller", "handler", "route", "endpoint", "ui", "cli"],
        "application": ["service", "usecase", "use_case", "orchestrator", "workflow"],
        "domain": ["entity", "model", "domain", "aggregate", "value_object"],
        "infrastructure": ["repository", "adapter", "gateway", "client", "database", "db"],
    }

    def __init__(self, enabled: bool = True, layers: Optional[Dict[str, int]] = None):
        super().__init__("ArchitectureValidator", enabled)
        self.layers = layers or self.DEFAULT_LAYERS
        self._rules = {
            "ARCH001": True,  # Check for circular dependencies
            "ARCH002": True,  # Check for layer violations
            "ARCH003": True,  # Check for missing abstractions
            "ARCH004": True,  # Check for god classes/modules
            "ARCH005": True,  # Check for feature envy
        }

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate architecture from the context."""
        start_time = time.time()
        issues: List[ValidationIssue] = []

        modules = context.get("modules", {})
        dependencies = context.get("dependencies", {})
        knowledge_graph = context.get("knowledge_graph")

        # If we have a knowledge graph, extract dependencies from it
        if knowledge_graph and hasattr(knowledge_graph, "get_all_edges"):
            dependencies = self._extract_dependencies_from_graph(knowledge_graph)

        if not dependencies and not modules:
            issues.append(self._issue(
                "ARCH000",
                "No module information provided for architecture validation",
                ValidationSeverity.WARNING,
            ))
            return self._create_result(True, issues, 1.0, self._elapsed(start_time))

        # ARCH001: Check for circular dependencies
        if self.is_rule_enabled("ARCH001"):
            issues.extend(self._check_circular_dependencies(dependencies))

        # ARCH002: Check for layer violations
        if self.is_rule_enabled("ARCH002"):
            issues.extend(self._check_layer_violations(dependencies))

        # ARCH003: Check for missing abstractions
        if self.is_rule_enabled("ARCH003"):
            issues.extend(self._check_missing_abstractions(modules, dependencies))

        # ARCH004: Check for god classes
        if self.is_rule_enabled("ARCH004"):
            issues.extend(self._check_god_modules(modules))

        # Calculate score
        critical_count = sum(1 for i in issues if i.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)

        score = max(0.0, 1.0 - (critical_count * 0.3) - (error_count * 0.1) - (warning_count * 0.02))
        passed = critical_count == 0

        return self._create_result(
            passed=passed,
            issues=issues,
            score=score,
            duration_ms=self._elapsed(start_time),
            metadata={"modules_analyzed": len(modules), "dependencies_analyzed": len(dependencies)},
        )

    def _extract_dependencies_from_graph(self, graph) -> Dict[str, Set[str]]:
        """Extract module dependencies from a knowledge graph."""
        dependencies: Dict[str, Set[str]] = {}
        try:
            for edge in graph.get_all_edges():
                if edge.edge_type.value == "imports":
                    source = edge.source_id
                    target = edge.target_id
                    if source not in dependencies:
                        dependencies[source] = set()
                    dependencies[source].add(target)
        except Exception:
            pass
        return dependencies

    def _check_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[ValidationIssue]:
        """Detect circular dependencies using DFS."""
        issues = []

        def find_cycle(node: str, visited: Set[str], path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            path.append(node)

            for dep in dependencies.get(node, []):
                if dep in path:
                    cycle_start = path.index(dep)
                    return path[cycle_start:] + [dep]
                if dep not in visited:
                    cycle = find_cycle(dep, visited, path)
                    if cycle:
                        return cycle

            path.pop()
            return None

        visited: Set[str] = set()
        for module in dependencies:
            if module not in visited:
                cycle = find_cycle(module, visited, [])
                if cycle:
                    issues.append(self._issue(
                        "ARCH001",
                        f"Circular dependency detected: {' -> '.join(cycle)}",
                        ValidationSeverity.ERROR,
                        suggestion="Break the cycle by introducing an abstraction or inverting dependencies",
                        rule_id="ARCH001",
                    ))

        return issues

    def _infer_layer(self, module_name: str) -> Optional[str]:
        """Infer the architectural layer from module name."""
        module_lower = module_name.lower()
        for layer, patterns in self.LAYER_PATTERNS.items():
            if any(p in module_lower for p in patterns):
                return layer
        return None

    def _check_layer_violations(self, dependencies: Dict[str, Set[str]]) -> List[ValidationIssue]:
        """Check for layer violations (lower layers depending on higher layers)."""
        issues = []

        for module, deps in dependencies.items():
            source_layer = self._infer_layer(module)
            if not source_layer:
                continue

            source_level = self.layers.get(source_layer, -1)

            for dep in deps:
                target_layer = self._infer_layer(dep)
                if not target_layer:
                    continue

                target_level = self.layers.get(target_layer, -1)

                # Lower level modules should not depend on higher level modules
                if source_level > target_level:
                    issues.append(self._issue(
                        "ARCH002",
                        f"Layer violation: {module} ({source_layer}) depends on {dep} ({target_layer})",
                        ValidationSeverity.WARNING,
                        suggestion=f"Inject {dep} as a dependency or use dependency inversion",
                        rule_id="ARCH002",
                    ))

        return issues

    def _check_missing_abstractions(
        self, modules: Dict[str, Any], dependencies: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """Check for missing abstractions (concrete dependencies that should be abstract)."""
        issues = []

        # Count how many modules depend on each module
        dependency_count: Dict[str, int] = {}
        for deps in dependencies.values():
            for dep in deps:
                dependency_count[dep] = dependency_count.get(dep, 0) + 1

        # Modules with many dependents should probably be abstractions
        for module, count in dependency_count.items():
            if count >= 5:  # Threshold for "many" dependents
                module_info = modules.get(module, {})
                is_abstract = module_info.get("is_abstract", False)
                has_interface = module_info.get("has_interface", False)

                if not is_abstract and not has_interface:
                    issues.append(self._issue(
                        "ARCH003",
                        f"Module '{module}' has {count} dependents but no abstraction",
                        ValidationSeverity.INFO,
                        suggestion="Consider extracting an interface for better decoupling",
                        rule_id="ARCH003",
                    ))

        return issues

    def _check_god_modules(self, modules: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for 'god' modules that do too much."""
        issues = []

        for module_name, module_info in modules.items():
            if not isinstance(module_info, dict):
                continue

            # Check by various metrics
            function_count = module_info.get("function_count", 0)
            class_count = module_info.get("class_count", 0)
            line_count = module_info.get("line_count", 0)
            responsibility_count = module_info.get("responsibility_count", 0)

            is_god_module = (
                function_count > 30 or
                class_count > 10 or
                line_count > 1000 or
                responsibility_count > 5
            )

            if is_god_module:
                issues.append(self._issue(
                    "ARCH004",
                    f"Module '{module_name}' appears to violate Single Responsibility Principle",
                    ValidationSeverity.WARNING,
                    suggestion="Consider splitting into smaller, focused modules",
                    rule_id="ARCH004",
                    context=f"Functions: {function_count}, Classes: {class_count}, Lines: {line_count}",
                ))

        return issues

    def _elapsed(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000
