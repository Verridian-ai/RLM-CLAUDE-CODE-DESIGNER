"""
Unit tests for the Effort Control module.
"""

import pytest
from rlm_lib.effort_control import (
    EffortController,
    EffortLevel,
    Task,
    TaskType,
    get_effort_for_description,
)


class TestEffortLevel:
    """Tests for EffortLevel enum."""
    
    def test_effort_levels_exist(self):
        """Test all effort levels are defined."""
        assert EffortLevel.LOW == "low"
        assert EffortLevel.MEDIUM == "medium"
        assert EffortLevel.HIGH == "high"


class TestTask:
    """Tests for Task dataclass."""
    
    def test_task_creation(self):
        """Test creating a task."""
        task = Task(description="Fix a bug")
        assert task.description == "Fix a bug"
        assert task.task_type is None
        assert not task.affects_critical_path
    
    def test_task_with_type(self):
        """Test creating a task with type."""
        task = Task(
            description="Design new API",
            task_type=TaskType.ARCHITECTURE,
            affects_critical_path=True,
        )
        assert task.task_type == TaskType.ARCHITECTURE
        assert task.affects_critical_path


class TestEffortController:
    """Tests for EffortController."""
    
    @pytest.fixture
    def controller(self):
        return EffortController()
    
    def test_default_effort_is_high(self, controller):
        """Test default effort level."""
        assert controller.default_effort == EffortLevel.HIGH
    
    def test_architecture_task_gets_high_effort(self, controller):
        """Test architecture tasks get HIGH effort."""
        task = Task(
            description="Design system architecture",
            task_type=TaskType.ARCHITECTURE,
        )
        effort = controller.get_effort_for_task(task)
        assert effort == EffortLevel.HIGH
    
    def test_formatting_task_gets_medium_in_enterprise(self, controller):
        """Test formatting tasks get MEDIUM in enterprise mode."""
        task = Task(
            description="Format code",
            task_type=TaskType.FORMATTING,
        )
        effort = controller.get_effort_for_task(task)
        # Enterprise mode upgrades LOW to MEDIUM
        assert effort == EffortLevel.MEDIUM
    
    def test_critical_path_upgrades_to_high(self, controller):
        """Test critical path tasks get HIGH effort."""
        task = Task(
            description="Simple fix",
            task_type=TaskType.SIMPLE_FIX,
            affects_critical_path=True,
        )
        effort = controller.get_effort_for_task(task)
        assert effort == EffortLevel.HIGH
    
    def test_high_complexity_upgrades_to_high(self, controller):
        """Test high complexity tasks get HIGH effort."""
        task = Task(
            description="Implement feature",
            task_type=TaskType.IMPLEMENTATION,
            complexity_score=0.8,
        )
        effort = controller.get_effort_for_task(task)
        assert effort == EffortLevel.HIGH
    
    def test_many_files_upgrades_to_high(self, controller):
        """Test tasks affecting many files get HIGH effort."""
        task = Task(
            description="Refactor module",
            task_type=TaskType.REFACTORING,
            file_count=15,
        )
        effort = controller.get_effort_for_task(task)
        assert effort == EffortLevel.HIGH
    
    def test_classify_from_description(self, controller):
        """Test classification from description."""
        task = Task(description="Design the database schema for authentication")
        effort = controller.get_effort_for_task(task)
        assert effort == EffortLevel.HIGH  # Contains "design", "database", "authentication"
    
    def test_classify_task_type(self, controller):
        """Test task type classification."""
        assert controller.classify_task("Design new architecture") == TaskType.ARCHITECTURE
        assert controller.classify_task("Fix the security vulnerability") == TaskType.SECURITY_REVIEW
        assert controller.classify_task("Debug the login issue") == TaskType.DEBUGGING
        assert controller.classify_task("Add new feature") == TaskType.IMPLEMENTATION
    
    def test_get_model_recommendation(self, controller):
        """Test model recommendations."""
        assert "opus" in controller.get_model_recommendation(EffortLevel.HIGH)
        assert "sonnet" in controller.get_model_recommendation(EffortLevel.MEDIUM)
        assert "haiku" in controller.get_model_recommendation(EffortLevel.LOW)
    
    def test_to_api_params(self, controller):
        """Test API parameter generation."""
        params = controller.to_api_params(EffortLevel.HIGH)
        assert "model" in params
        assert "thinking" in params
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 32000


class TestGetEffortForDescription:
    """Tests for the convenience function."""
    
    def test_high_effort_description(self):
        """Test high effort from description."""
        effort = get_effort_for_description("Design the system architecture")
        assert effort == EffortLevel.HIGH
    
    def test_low_effort_description(self):
        """Test low effort from description."""
        effort = get_effort_for_description("Fix a simple typo in comment")
        # Enterprise mode upgrades LOW to MEDIUM
        assert effort == EffortLevel.MEDIUM

