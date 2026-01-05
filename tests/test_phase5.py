"""
Unit tests for Phase 5: Mode Manager Enhancement.
Tests Mode Manager v2, Review Orchestrator, and Coherence Validator.
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add .claude to path for mode_manager_v2
sys.path.insert(0, str(Path(__file__).parent.parent / ".claude"))

from mode_manager_v2 import (
    EnterpriseModeManager,
    AgentMode,
    TransitionResult,
    ModeTransition,
    ModeConfig,
    ModeTransitionRules,
    MODE_CONFIGS,
    create_enterprise_mode_manager,
)
from rlm_lib.review_orchestrator import (
    ReviewOrchestrator,
    ReviewAgent,
    CodeReviewAgent,
    SecurityAuditAgent,
    TypeValidationAgent,
    TestCoverageAgent,
    ReviewAgentType,
    ReviewIssue,
    IssueSeverity,
    AgentReviewResult,
    OrchestratedReviewResult,
    create_review_orchestrator,
)
from rlm_lib.coherence_validator import (
    ArchitecturalCoherenceValidator,
    PatternDetector,
    LayerEnforcer,
    LayerDefinition,
    NamingConvention,
    CoherenceIssue,
    CoherenceReport,
    CoherenceIssueType,
    create_coherence_validator,
)


class TestModeManagerV2:
    """Tests for Enterprise Mode Manager v2."""

    def test_mode_configs_exist(self):
        """Test that all modes have configurations."""
        for mode in AgentMode:
            assert mode in MODE_CONFIGS

    def test_review_and_commit_modes_exist(self):
        """Test that new REVIEW and COMMIT modes exist."""
        assert AgentMode.REVIEW.value == "review"
        assert AgentMode.COMMIT.value == "commit"

    def test_commit_requires_approval(self):
        """Test that COMMIT mode requires human approval."""
        config = MODE_CONFIGS[AgentMode.COMMIT]
        assert config.requires_human_approval
        assert config.requires_quality_gate

    def test_transition_rules_valid(self):
        """Test transition rules validation."""
        rules = ModeTransitionRules()

        # Valid transition
        valid, reason = rules.is_valid_transition(AgentMode.EXECUTION, AgentMode.REVIEW)
        assert valid

        # Invalid transition (PLANNING can't go directly to COMMIT)
        valid, reason = rules.is_valid_transition(AgentMode.PLANNING, AgentMode.COMMIT)
        assert not valid

    def test_factory_function(self):
        """Test factory function."""
        manager = create_enterprise_mode_manager()
        assert isinstance(manager, EnterpriseModeManager)


class TestReviewOrchestrator:
    """Tests for Review Orchestrator."""

    def test_orchestrator_creation(self):
        """Test creating orchestrator."""
        orchestrator = ReviewOrchestrator()
        assert len(orchestrator.agents) == 4  # Default agents

    def test_default_agents(self):
        """Test default agents are created."""
        orchestrator = ReviewOrchestrator()
        agent_types = [a.agent_type for a in orchestrator.agents]

        assert ReviewAgentType.CODE_REVIEW in agent_types
        assert ReviewAgentType.SECURITY_AUDIT in agent_types
        assert ReviewAgentType.TYPE_VALIDATION in agent_types
        assert ReviewAgentType.TEST_COVERAGE in agent_types

    def test_code_review_agent(self):
        """Test code review agent."""
        agent = CodeReviewAgent()
        code = "x = 1\n" + "y" * 150  # Long line
        result = agent.review(code)

        assert isinstance(result, AgentReviewResult)
        assert result.agent_type == ReviewAgentType.CODE_REVIEW.value

    def test_security_audit_detects_hardcoded_password(self):
        """Test security audit detects hardcoded passwords."""
        agent = SecurityAuditAgent()
        code = 'password = "secret123"'
        result = agent.review(code)

        assert not result.passed
        assert any(i.severity == IssueSeverity.CRITICAL.value for i in result.issues)

    def test_orchestrated_review(self):
        """Test full orchestrated review."""
        orchestrator = ReviewOrchestrator()
        code = '''
def hello():
    print("Hello, World!")
'''
        result = orchestrator.review(code, parallel=False)

        assert isinstance(result, OrchestratedReviewResult)
        assert len(result.agent_results) > 0

    def test_parallel_review(self):
        """Test parallel review execution."""
        orchestrator = ReviewOrchestrator()
        code = "x = 1"
        result = orchestrator.review(code, parallel=True)

        assert isinstance(result, OrchestratedReviewResult)

    def test_fail_fast_on_critical(self):
        """Test fail-fast on critical issues."""
        orchestrator = ReviewOrchestrator(fail_fast=True)
        code = 'api_key = "sk-secret123"'  # Critical security issue
        result = orchestrator.review(code, parallel=False)

        assert result.fail_fast_triggered or len(result.critical_issues) > 0

    def test_factory_function(self):
        """Test factory function."""
        orchestrator = create_review_orchestrator()
        assert isinstance(orchestrator, ReviewOrchestrator)


class TestCoherenceValidator:
    """Tests for Architectural Coherence Validator."""

    def test_validator_creation(self):
        """Test creating validator."""
        validator = ArchitecturalCoherenceValidator()
        assert validator is not None
        assert len(validator.naming_conventions) > 0

    def test_pattern_detector(self):
        """Test pattern detection."""
        detector = PatternDetector()
        code = '''
class UserFactory:
    def create_user(self, name):
        return User(name)
'''
        patterns = detector.detect(code)
        assert patterns.get("factory", False)

    def test_naming_convention_check(self):
        """Test naming convention checking."""
        validator = ArchitecturalCoherenceValidator()
        code = '''
class myClass:  # Should be MyClass
    def MyMethod(self):  # Should be my_method
        pass
'''
        report = validator.validate(code)

        assert isinstance(report, CoherenceReport)
        assert len(report.issues) > 0

    def test_valid_naming_passes(self):
        """Test valid naming passes."""
        validator = ArchitecturalCoherenceValidator()
        code = '''
class MyClass:
    def my_method(self):
        pass
'''
        report = validator.validate(code)

        # Should have no naming violations
        naming_issues = [i for i in report.issues
                        if i.type == CoherenceIssueType.NAMING_VIOLATION.value]
        assert len(naming_issues) == 0

    def test_layer_enforcer(self):
        """Test layer enforcement."""
        enforcer = LayerEnforcer()
        assert len(enforcer.layers) > 0

    def test_layer_definition(self):
        """Test layer definition."""
        layer = LayerDefinition(
            name="test",
            path_patterns=["**/test/**"],
            allowed_dependencies=["domain"],
        )
        assert layer.name == "test"

    def test_coherence_report_scoring(self):
        """Test coherence report scoring."""
        report = CoherenceReport()
        assert report.score == 1.0
        assert report.passed

        # Add a critical issue
        report.add_issue(CoherenceIssue(
            type=CoherenceIssueType.PATTERN_VIOLATION.value,
            severity="critical",
            message="Test issue",
        ))

        assert report.score < 1.0
        assert not report.passed

    def test_pattern_consistency_check(self):
        """Test pattern consistency checking."""
        validator = ArchitecturalCoherenceValidator()
        code = '''
class UserFactory:
    def create_user(self):
        return User()

# But also direct instantiation
user = User()
'''
        report = validator.validate(code)

        # Should detect mixed patterns
        consistency_issues = [i for i in report.issues
                             if i.type == CoherenceIssueType.CONSISTENCY_VIOLATION.value]
        assert len(consistency_issues) > 0

    def test_factory_function(self):
        """Test factory function."""
        validator = create_coherence_validator()
        assert isinstance(validator, ArchitecturalCoherenceValidator)

    def test_custom_naming_conventions(self):
        """Test custom naming conventions."""
        conventions = [
            NamingConvention(
                name="custom_class",
                pattern=r"^I[A-Z][a-zA-Z0-9]*$",  # Interface prefix
                applies_to="class",
                description="Classes should start with I",
            ),
        ]
        validator = ArchitecturalCoherenceValidator(naming_conventions=conventions)

        code = "class MyClass: pass"  # Doesn't start with I
        report = validator.validate(code)

        assert len(report.issues) > 0


class TestIntegration:
    """Integration tests for Phase 5 components."""

    def test_review_and_coherence_together(self):
        """Test using review orchestrator and coherence validator together."""
        orchestrator = ReviewOrchestrator()
        validator = ArchitecturalCoherenceValidator()

        code = '''
class myBadClass:
    password = "secret123"

    def BadMethod(self):
        eval("print('hello')")
'''

        review_result = orchestrator.review(code, parallel=False)
        coherence_result = validator.validate(code)

        # Both should find issues
        assert len(review_result.all_issues) > 0
        assert len(coherence_result.issues) > 0

    def test_mode_manager_with_quality_gate(self):
        """Test mode manager quality gate integration."""
        manager = create_enterprise_mode_manager()

        # Get current mode (may have been changed by previous tests)
        current = manager.get_current_mode()
        assert current in AgentMode

        # Test that we can get mode config
        config = manager.get_mode_config(current)
        assert config is not None

        # Test transition history tracking
        history = manager.get_transition_history()
        assert isinstance(history, list)

