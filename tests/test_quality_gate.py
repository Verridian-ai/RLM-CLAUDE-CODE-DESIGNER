"""
Unit tests for the Quality Gate and Confidence Scorer modules.
"""

import pytest
from rlm_lib.validators import ValidationResult, ValidationSeverity, ValidationIssue
from rlm_lib.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceScore,
    ConfidenceLevel,
)
from rlm_lib.quality_gate import (
    QualityGate,
    GateResult,
    GateDecision,
    create_quality_gate,
)


class TestConfidenceScore:
    """Tests for ConfidenceScore dataclass."""
    
    def test_can_proceed_high_confidence(self):
        """Test can_proceed with high confidence."""
        score = ConfidenceScore(overall=0.8, level=ConfidenceLevel.HIGH)
        assert score.can_proceed
        assert not score.needs_review
        assert not score.should_block
    
    def test_needs_review_medium_confidence(self):
        """Test needs_review with medium confidence."""
        score = ConfidenceScore(overall=0.5, level=ConfidenceLevel.MEDIUM)
        assert not score.can_proceed
        assert score.needs_review
        assert not score.should_block
    
    def test_should_block_low_confidence(self):
        """Test should_block with low confidence."""
        score = ConfidenceScore(overall=0.2, level=ConfidenceLevel.LOW)
        assert not score.can_proceed
        assert not score.needs_review
        assert score.should_block
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = ConfidenceScore(overall=0.75, level=ConfidenceLevel.HIGH)
        d = score.to_dict()
        assert d["overall"] == 0.75
        assert d["level"] == "high"


class TestConfidenceScorer:
    """Tests for ConfidenceScorer."""
    
    @pytest.fixture
    def scorer(self):
        return ConfidenceScorer()
    
    def test_calculate_with_no_results(self, scorer):
        """Test calculation with no validation results."""
        score = scorer.calculate([])
        assert score.overall == 0.0
        assert score.level == ConfidenceLevel.VERY_LOW
    
    def test_calculate_with_passing_results(self, scorer):
        """Test calculation with passing validation results."""
        results = [
            ValidationResult(validator_name="RequirementsValidator", passed=True, score=0.9),
            ValidationResult(validator_name="SecurityValidator", passed=True, score=0.95),
        ]
        score = scorer.calculate(results)
        assert score.overall > 0.8
        assert score.level in (ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH)
    
    def test_calculate_with_mixed_results(self, scorer):
        """Test calculation with mixed validation results."""
        results = [
            ValidationResult(validator_name="RequirementsValidator", passed=True, score=0.9),
            ValidationResult(validator_name="SecurityValidator", passed=False, score=0.3),
        ]
        score = scorer.calculate(results)
        assert 0.4 < score.overall < 0.8


class TestQualityGate:
    """Tests for QualityGate."""
    
    @pytest.fixture
    def gate(self):
        return QualityGate()
    
    def test_check_empty_context(self, gate):
        """Test checking an empty context."""
        result = gate.check({})
        assert isinstance(result, GateResult)
        assert result.decision in GateDecision
    
    def test_check_disabled_gate(self):
        """Test disabled gate returns SKIP."""
        gate = QualityGate(enabled=False)
        result = gate.check({})
        assert result.decision == GateDecision.SKIP
        assert result.passed is False  # SKIP is not PASS
    
    def test_check_with_good_context(self, gate):
        """Test checking a well-formed context."""
        context = {
            "requirements": [
                {"id": "REQ001", "text": "The system must handle 1000 requests per second"},
            ],
            "code": "def safe_function():\n    return 42",
            "classes": [],
            "functions": [],
        }
        result = gate.check(context)
        assert isinstance(result.confidence, ConfidenceScore)
        assert result.duration_ms > 0
    
    def test_add_and_remove_validator(self, gate):
        """Test adding and removing validators."""
        initial_count = len(gate.validators)
        
        from rlm_lib.validators import RequirementsValidator
        new_validator = RequirementsValidator()
        new_validator.name = "CustomValidator"
        
        gate.add_validator(new_validator)
        assert len(gate.validators) == initial_count + 1
        
        removed = gate.remove_validator("CustomValidator")
        assert removed
        assert len(gate.validators) == initial_count
    
    def test_enable_disable_validator(self, gate):
        """Test enabling and disabling validators."""
        gate.disable_validator("SecurityValidator")
        
        for v in gate.validators:
            if v.name == "SecurityValidator":
                assert not v.enabled
                break
        
        gate.enable_validator("SecurityValidator")
        for v in gate.validators:
            if v.name == "SecurityValidator":
                assert v.enabled
                break
    
    def test_get_status(self, gate):
        """Test getting gate status."""
        status = gate.get_status()
        assert "enabled" in status
        assert "validators" in status
        assert len(status["validators"]) > 0


class TestCreateQualityGate:
    """Tests for the create_quality_gate factory function."""
    
    def test_create_default_gate(self):
        """Test creating default gate."""
        gate = create_quality_gate()
        assert gate.pass_threshold == 0.8
        assert len(gate.validators) == 4
    
    def test_create_strict_gate(self):
        """Test creating strict gate."""
        gate = create_quality_gate(strict=True)
        assert gate.pass_threshold == 0.9
        assert gate.review_threshold == 0.75
    
    def test_create_security_only_gate(self):
        """Test creating security-only gate."""
        gate = create_quality_gate(security_only=True)
        assert len(gate.validators) == 1
        assert gate.validators[0].name == "SecurityValidator"

