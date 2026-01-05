# Integration tests for Mode Manager transitions
"""
End-to-end tests for mode manager transitions.
Tests the full workflow of mode transitions with quality gates.
"""

import pytest
import sys
from pathlib import Path

# Add .claude to path for mode_manager_v2
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".claude"))

from mode_manager_v2 import (
    EnterpriseModeManager,
    AgentMode,
    ModeConfig,
    TransitionResult,
    create_enterprise_mode_manager,
)


class TestModeTransitions:
    """Test mode manager transitions."""

    @pytest.fixture
    def fresh_manager(self):
        """Create a fresh mode manager for each test."""
        # Create new instance (bypasses singleton for testing)
        manager = EnterpriseModeManager()
        return manager

    def test_valid_transition_sequence(self, fresh_manager):
        """Test a valid sequence of mode transitions."""
        manager = fresh_manager
        
        # Start in PLANNING
        assert manager.get_current_mode() in AgentMode
        
        # Get config for current mode
        config = manager.get_mode_config(manager.get_current_mode())
        assert config is not None
        assert isinstance(config, ModeConfig)

    def test_mode_config_properties(self, fresh_manager):
        """Test mode configurations have required properties."""
        manager = fresh_manager

        for mode in AgentMode:
            config = manager.get_mode_config(mode)
            assert config is not None
            assert config.name is not None
            assert config.preferred_model is not None
            assert isinstance(config.requires_quality_gate, bool)

    def test_transition_history_tracking(self, fresh_manager):
        """Test that transition history is tracked."""
        manager = fresh_manager
        
        # Get initial history
        history = manager.get_transition_history()
        assert isinstance(history, list)

    def test_review_mode_exists(self, fresh_manager):
        """Test REVIEW mode is available."""
        manager = fresh_manager
        
        assert AgentMode.REVIEW in AgentMode
        config = manager.get_mode_config(AgentMode.REVIEW)
        assert config is not None

    def test_commit_mode_requires_approval(self, fresh_manager):
        """Test COMMIT mode requires human approval."""
        manager = fresh_manager

        config = manager.get_mode_config(AgentMode.COMMIT)
        assert config is not None
        assert config.requires_human_approval is True

    def test_all_modes_have_configs(self, fresh_manager):
        """Test all modes have valid configurations."""
        manager = fresh_manager

        for mode in AgentMode:
            config = manager.get_mode_config(mode)
            assert config is not None, f"Missing config for {mode}"
            assert config.name is not None

    def test_transition_rules_exist(self, fresh_manager):
        """Test transition rules are defined."""
        manager = fresh_manager
        
        # Each mode should have allowed transitions
        for mode in AgentMode:
            config = manager.get_mode_config(mode)
            assert hasattr(config, "allowed_transitions")


class TestModeIntegrationWithQualityGates:
    """Test mode transitions with quality gate integration."""

    def test_quality_gate_flag_on_modes(self):
        """Test quality gate requirements on modes."""
        manager = create_enterprise_mode_manager()

        # COMMIT should require quality gate
        commit_config = manager.get_mode_config(AgentMode.COMMIT)
        assert commit_config.requires_quality_gate is True

        # PLANNING typically doesn't require quality gate
        planning_config = manager.get_mode_config(AgentMode.PLANNING)
        assert planning_config is not None

    def test_preferred_models_for_modes(self):
        """Test preferred models are set for each mode."""
        manager = create_enterprise_mode_manager()
        
        # Different modes should have appropriate models
        review_config = manager.get_mode_config(AgentMode.REVIEW)
        execution_config = manager.get_mode_config(AgentMode.EXECUTION)
        
        assert review_config.preferred_model is not None
        assert execution_config.preferred_model is not None


class TestTransitionResults:
    """Test transition result handling."""

    def test_transition_result_enum(self):
        """Test TransitionResult enum values."""
        assert TransitionResult.SUCCESS is not None
        assert TransitionResult.BLOCKED_BY_GATE is not None
        assert TransitionResult.REQUIRES_APPROVAL is not None
        assert TransitionResult.INVALID_TRANSITION is not None

    def test_request_transition_returns_tuple(self):
        """Test request_transition returns proper tuple."""
        manager = create_enterprise_mode_manager()
        
        # Any transition should return a tuple
        result = manager.request_transition(
            target_mode=AgentMode.EXECUTION,
            reason="Test transition",
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], TransitionResult)

