"""
Review Orchestrator for Enterprise RLM-CLAUDE.

Coordinates multi-agent code review with:
- Parallel agent execution
- Specialized review agents (code, security, types, tests)
- Confidence calculation and aggregation
- Fail-fast on critical issues
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ReviewAgentType(str, Enum):
    """Types of review agents."""
    CODE_REVIEW = "code_review"
    SECURITY_AUDIT = "security_audit"
    TYPE_VALIDATION = "type_validation"
    TEST_COVERAGE = "test_coverage"
    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"


class IssueSeverity(str, Enum):
    """Severity levels for review issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ReviewIssue:
    """An issue found during review."""
    agent_type: str
    severity: str
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class AgentReviewResult:
    """Result from a single review agent."""
    agent_type: str
    passed: bool
    confidence: float
    issues: List[ReviewIssue] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class OrchestratedReviewResult:
    """Aggregated result from all review agents."""
    passed: bool
    overall_confidence: float
    agent_results: Dict[str, AgentReviewResult] = field(default_factory=dict)
    critical_issues: List[ReviewIssue] = field(default_factory=list)
    all_issues: List[ReviewIssue] = field(default_factory=list)
    total_execution_time: float = 0.0
    fail_fast_triggered: bool = False


class ReviewAgent:
    """Base class for review agents."""

    def __init__(self, agent_type: ReviewAgentType):
        self.agent_type = agent_type
        self.name = agent_type.value

    def review(self, code: str, context: Optional[Dict] = None) -> AgentReviewResult:
        """Perform review. Override in subclasses."""
        return AgentReviewResult(
            agent_type=self.agent_type.value,
            passed=True,
            confidence=1.0,
        )


class CodeReviewAgent(ReviewAgent):
    """Agent for general code review."""

    def __init__(self):
        super().__init__(ReviewAgentType.CODE_REVIEW)

    def review(self, code: str, context: Optional[Dict] = None) -> AgentReviewResult:
        """Review code for quality issues."""
        start_time = time.time()
        issues = []

        # Check for common code smells
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 120:
                issues.append(ReviewIssue(
                    agent_type=self.name,
                    severity=IssueSeverity.LOW.value,
                    message=f"Line exceeds 120 characters ({len(line)} chars)",
                    line_number=i,
                ))

            # TODO comments
            if 'TODO' in line or 'FIXME' in line:
                issues.append(ReviewIssue(
                    agent_type=self.name,
                    severity=IssueSeverity.INFO.value,
                    message="TODO/FIXME comment found",
                    line_number=i,
                    code_snippet=line.strip(),
                ))

        # Calculate confidence based on issues
        critical_count = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL.value)
        high_count = sum(1 for i in issues if i.severity == IssueSeverity.HIGH.value)

        confidence = 1.0 - (critical_count * 0.3) - (high_count * 0.1)
        confidence = max(0.0, min(1.0, confidence))

        return AgentReviewResult(
            agent_type=self.name,
            passed=critical_count == 0,
            confidence=confidence,
            issues=issues,
            execution_time=time.time() - start_time,
        )


class SecurityAuditAgent(ReviewAgent):
    """Agent for security vulnerability detection."""

    def __init__(self):
        super().__init__(ReviewAgentType.SECURITY_AUDIT)
        self.patterns = [
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password detected", IssueSeverity.CRITICAL),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key detected", IssueSeverity.CRITICAL),
            (r"eval\s*\(", "Use of eval() is dangerous", IssueSeverity.HIGH),
            (r"exec\s*\(", "Use of exec() is dangerous", IssueSeverity.HIGH),
            (r"subprocess\.call\s*\([^)]*shell\s*=\s*True", "Shell injection risk", IssueSeverity.HIGH),
        ]

    def review(self, code: str, context: Optional[Dict] = None) -> AgentReviewResult:
        """Review code for security vulnerabilities."""
        import re
        start_time = time.time()
        issues = []

        for pattern, message, severity in self.patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                issues.append(ReviewIssue(
                    agent_type=self.name,
                    severity=severity.value,
                    message=message,
                    code_snippet=match.group(0)[:50],
                ))

        critical_count = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL.value)
        confidence = 1.0 - (critical_count * 0.4)
        confidence = max(0.0, min(1.0, confidence))

        return AgentReviewResult(
            agent_type=self.name,
            passed=critical_count == 0,
            confidence=confidence,
            issues=issues,
            execution_time=time.time() - start_time,
        )


class TypeValidationAgent(ReviewAgent):
    """Agent for type checking validation."""

    def __init__(self):
        super().__init__(ReviewAgentType.TYPE_VALIDATION)

    def review(self, code: str, context: Optional[Dict] = None) -> AgentReviewResult:
        """Check for type annotation issues."""
        import re
        start_time = time.time()
        issues = []

        # Check for functions without type hints
        func_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*:'
        for match in re.finditer(func_pattern, code):
            func_def = match.group(0)
            if '->' not in func_def:
                issues.append(ReviewIssue(
                    agent_type=self.name,
                    severity=IssueSeverity.MEDIUM.value,
                    message=f"Function '{match.group(1)}' missing return type annotation",
                    code_snippet=func_def,
                ))

        confidence = 1.0 - (len(issues) * 0.05)
        confidence = max(0.0, min(1.0, confidence))

        return AgentReviewResult(
            agent_type=self.name,
            passed=True,  # Type issues are warnings, not failures
            confidence=confidence,
            issues=issues,
            execution_time=time.time() - start_time,
        )


class TestCoverageAgent(ReviewAgent):
    """Agent for test coverage analysis."""

    def __init__(self):
        super().__init__(ReviewAgentType.TEST_COVERAGE)

    def review(self, code: str, context: Optional[Dict] = None) -> AgentReviewResult:
        """Analyze test coverage indicators."""
        import re
        start_time = time.time()
        issues = []

        # Check for test functions
        test_count = len(re.findall(r'def\s+test_\w+', code))
        func_count = len(re.findall(r'def\s+\w+', code))

        if func_count > 0 and test_count == 0:
            issues.append(ReviewIssue(
                agent_type=self.name,
                severity=IssueSeverity.MEDIUM.value,
                message="No test functions found in code",
            ))

        # Estimate coverage based on test ratio
        coverage_ratio = test_count / max(func_count, 1)
        confidence = min(1.0, coverage_ratio * 2)  # 50% test ratio = 100% confidence

        return AgentReviewResult(
            agent_type=self.name,
            passed=True,
            confidence=confidence,
            issues=issues,
            execution_time=time.time() - start_time,
        )


class ReviewOrchestrator:
    """Orchestrates multi-agent code review."""

    def __init__(
        self,
        agents: Optional[List[ReviewAgent]] = None,
        max_workers: int = 4,
        fail_fast: bool = True,
    ):
        self.agents = agents or self._default_agents()
        self.max_workers = max_workers
        self.fail_fast = fail_fast

    def _default_agents(self) -> List[ReviewAgent]:
        """Create default set of review agents."""
        return [
            CodeReviewAgent(),
            SecurityAuditAgent(),
            TypeValidationAgent(),
            TestCoverageAgent(),
        ]

    def add_agent(self, agent: ReviewAgent) -> None:
        """Add a review agent."""
        self.agents.append(agent)

    def review(
        self,
        code: str,
        context: Optional[Dict] = None,
        parallel: bool = True,
    ) -> OrchestratedReviewResult:
        """
        Run all review agents and aggregate results.

        Args:
            code: The code to review
            context: Optional context for agents
            parallel: Whether to run agents in parallel

        Returns:
            OrchestratedReviewResult with aggregated findings
        """
        start_time = time.time()
        result = OrchestratedReviewResult(passed=True, overall_confidence=1.0)

        if parallel:
            self._run_parallel(code, context, result)
        else:
            self._run_sequential(code, context, result)

        # Calculate overall metrics
        result.total_execution_time = time.time() - start_time
        self._calculate_overall_confidence(result)

        return result

    def _run_parallel(
        self,
        code: str,
        context: Optional[Dict],
        result: OrchestratedReviewResult,
    ) -> None:
        """Run agents in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(agent.review, code, context): agent
                for agent in self.agents
            }

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    agent_result = future.result()
                    self._process_agent_result(agent_result, result)

                    # Fail fast on critical issues
                    if self.fail_fast and result.critical_issues:
                        result.fail_fast_triggered = True
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                except Exception as e:
                    result.agent_results[agent.name] = AgentReviewResult(
                        agent_type=agent.name,
                        passed=False,
                        confidence=0.0,
                        error=str(e),
                    )

    def _run_sequential(
        self,
        code: str,
        context: Optional[Dict],
        result: OrchestratedReviewResult,
    ) -> None:
        """Run agents sequentially."""
        for agent in self.agents:
            try:
                agent_result = agent.review(code, context)
                self._process_agent_result(agent_result, result)

                if self.fail_fast and result.critical_issues:
                    result.fail_fast_triggered = True
                    break
            except Exception as e:
                result.agent_results[agent.name] = AgentReviewResult(
                    agent_type=agent.name,
                    passed=False,
                    confidence=0.0,
                    error=str(e),
                )

    def _process_agent_result(
        self,
        agent_result: AgentReviewResult,
        result: OrchestratedReviewResult,
    ) -> None:
        """Process a single agent's result."""
        result.agent_results[agent_result.agent_type] = agent_result

        if not agent_result.passed:
            result.passed = False

        for issue in agent_result.issues:
            result.all_issues.append(issue)
            if issue.severity == IssueSeverity.CRITICAL.value:
                result.critical_issues.append(issue)

    def _calculate_overall_confidence(self, result: OrchestratedReviewResult) -> None:
        """Calculate overall confidence from agent results."""
        if not result.agent_results:
            result.overall_confidence = 0.0
            return

        # Weighted average based on agent importance
        weights = {
            ReviewAgentType.SECURITY_AUDIT.value: 2.0,
            ReviewAgentType.CODE_REVIEW.value: 1.5,
            ReviewAgentType.TYPE_VALIDATION.value: 1.0,
            ReviewAgentType.TEST_COVERAGE.value: 1.0,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for agent_type, agent_result in result.agent_results.items():
            weight = weights.get(agent_type, 1.0)
            weighted_sum += agent_result.confidence * weight
            total_weight += weight

        result.overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0


def create_review_orchestrator(
    agents: Optional[List[ReviewAgent]] = None,
    max_workers: int = 4,
    fail_fast: bool = True,
) -> ReviewOrchestrator:
    """Factory function to create a review orchestrator."""
    return ReviewOrchestrator(agents=agents, max_workers=max_workers, fail_fast=fail_fast)

