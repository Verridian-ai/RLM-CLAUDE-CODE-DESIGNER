"""
Unit tests for the Validators module.
"""

import pytest
from rlm_lib.validators import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    RequirementsValidator,
    ArchitectureValidator,
    DesignValidator,
    SecurityValidator,
)


class TestRequirementsValidator:
    """Tests for RequirementsValidator."""
    
    @pytest.fixture
    def validator(self):
        return RequirementsValidator()
    
    def test_validates_empty_requirements(self, validator):
        """Test validation with no requirements."""
        result = validator.validate({})
        assert not result.passed
        assert len(result.issues) > 0
    
    def test_validates_good_requirements(self, validator):
        """Test validation with good requirements."""
        context = {
            "requirements": [
                {"id": "REQ001", "text": "The system must respond within 200 milliseconds for all API calls"},
                {"id": "REQ002", "text": "The system shall support at least 1000 concurrent users"},
            ]
        }
        result = validator.validate(context)
        assert result.passed
        assert result.score > 0.5
    
    def test_detects_ambiguous_language(self, validator):
        """Test detection of ambiguous words."""
        context = {
            "requirements": [
                {"id": "REQ001", "text": "The system should be user-friendly and fast"},
            ]
        }
        result = validator.validate(context)
        # Should have warnings for ambiguous words
        ambiguous_issues = [i for i in result.issues if i.code == "REQ001"]
        assert len(ambiguous_issues) > 0
    
    def test_detects_brief_requirements(self, validator):
        """Test detection of too-brief requirements."""
        context = {
            "requirements": [
                {"id": "REQ001", "text": "Be fast"},
            ]
        }
        result = validator.validate(context)
        brief_issues = [i for i in result.issues if i.code == "REQ003"]
        assert len(brief_issues) > 0


class TestArchitectureValidator:
    """Tests for ArchitectureValidator."""
    
    @pytest.fixture
    def validator(self):
        return ArchitectureValidator()
    
    def test_validates_empty_context(self, validator):
        """Test validation with empty context."""
        result = validator.validate({})
        assert result.passed  # No issues when no data
    
    def test_detects_circular_dependencies(self, validator):
        """Test detection of circular dependencies."""
        context = {
            "dependencies": {
                "module_a": {"module_b"},
                "module_b": {"module_c"},
                "module_c": {"module_a"},  # Circular!
            }
        }
        result = validator.validate(context)
        circular_issues = [i for i in result.issues if i.code == "ARCH001"]
        assert len(circular_issues) > 0
    
    def test_detects_god_modules(self, validator):
        """Test detection of god modules."""
        context = {
            "modules": {
                "god_module": {
                    "function_count": 50,
                    "class_count": 15,
                    "line_count": 2000,
                }
            },
            "dependencies": {},
        }
        result = validator.validate(context)
        god_issues = [i for i in result.issues if i.code == "ARCH004"]
        assert len(god_issues) > 0


class TestDesignValidator:
    """Tests for DesignValidator."""
    
    @pytest.fixture
    def validator(self):
        return DesignValidator()
    
    def test_validates_empty_context(self, validator):
        """Test validation with empty context."""
        result = validator.validate({})
        assert result.passed
        assert result.score == 1.0
    
    def test_detects_naming_violations(self, validator):
        """Test detection of naming convention violations."""
        context = {
            "classes": [{"name": "bad_class_name"}],  # Should be PascalCase
            "functions": [{"name": "BadFunction"}],  # Should be snake_case
        }
        result = validator.validate(context)
        naming_issues = [i for i in result.issues if i.code == "DES001"]
        assert len(naming_issues) > 0
    
    def test_detects_long_methods(self, validator):
        """Test detection of long methods."""
        context = {
            "classes": [],
            "functions": [{"name": "long_function", "line_count": 100}],
        }
        result = validator.validate(context)
        long_issues = [i for i in result.issues if i.code == "DES002"]
        assert len(long_issues) > 0


class TestSecurityValidator:
    """Tests for SecurityValidator."""
    
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    def test_validates_clean_code(self, validator):
        """Test validation of secure code."""
        context = {
            "code": "def hello():\n    return 'Hello, World!'"
        }
        result = validator.validate(context)
        assert result.passed
    
    def test_detects_hardcoded_secrets(self, validator):
        """Test detection of hardcoded secrets."""
        context = {
            "code": "password = 'supersecret123'"
        }
        result = validator.validate(context)
        secret_issues = [i for i in result.issues if i.code == "SEC001"]
        assert len(secret_issues) > 0
        assert not result.passed  # Critical issue should fail

