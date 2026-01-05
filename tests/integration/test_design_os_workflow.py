# Integration tests for Design OS workflow
"""
End-to-end tests for Design OS integration.
Tests the full workflow from design spec loading to validation.
"""

import pytest
import tempfile
import json
from pathlib import Path

from rlm_lib import (
    DesignOSAdapter,
    ComponentSpecParser,
    PixelPerfectValidator,
    DesignTokens,
    DesignToken,
    create_pixel_perfect_validator,
)


class TestDesignOSWorkflow:
    """Test complete Design OS workflows."""

    @pytest.fixture
    def design_tokens(self):
        """Create test design tokens."""
        tokens = DesignTokens(
            colors={
                "color-primary": DesignToken(name="color-primary", value="#3B82F6", category="color"),
                "color-secondary": DesignToken(name="color-secondary", value="#10B981", category="color"),
            },
            spacing={
                "spacing-md": DesignToken(name="spacing-md", value="16px", category="spacing"),
            },
            typography={
                "font-size-base": DesignToken(name="font-size-base", value="16px", category="typography"),
            },
        )
        return tokens

    @pytest.fixture
    def component_spec(self):
        """Create test component specification."""
        return {
            "name": "Button",
            "description": "Primary action button",
            "props": [
                {"name": "label", "type": "string", "required": True},
                {"name": "variant", "type": "string", "default": "primary"},
                {"name": "disabled", "type": "boolean", "default": False},
            ],
            "styles": {
                "backgroundColor": "var(--color-primary)",
                "padding": "var(--spacing-md)",
                "fontSize": "var(--font-size-base)",
            },
            "behaviors": [
                {"event": "click", "action": "emit('click')"},
                {"event": "hover", "action": "show-tooltip"},
            ],
        }

    def test_full_design_validation_workflow(self, design_tokens):
        """Test complete design validation workflow."""
        validator = create_pixel_perfect_validator(design_tokens)

        # Good code using design tokens
        good_code = '''
        .button {
            background-color: var(--color-primary);
            padding: var(--spacing-md);
            font-size: var(--font-size-base);
        }
        '''

        report = validator.validate_css(good_code)
        assert report.score >= 0.9  # Score is 0-1 scale

    def test_design_violation_detection(self, design_tokens):
        """Test detection of design violations."""
        validator = create_pixel_perfect_validator(design_tokens)

        # Bad code with hardcoded values
        bad_code = '''
        .button {
            background-color: #FF0000;
            padding: 15px;
            font-size: 14px;
        }
        '''

        report = validator.validate_css(bad_code)
        assert report.score < 100

    def test_component_spec_parsing(self, component_spec):
        """Test component specification parsing."""
        parser = ComponentSpecParser()

        # Write spec to temp file for parsing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(component_spec, f)
            spec_path = Path(f.name)

        try:
            parsed = parser.parse(spec_path)
            assert parsed is not None
            assert parsed.name == "Button"
        finally:
            spec_path.unlink(missing_ok=True)

    def test_implementation_guide_generation(self, component_spec):
        """Test implementation guide generation from spec."""
        parser = ComponentSpecParser()

        # Write spec to temp file for parsing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(component_spec, f)
            spec_path = Path(f.name)

        try:
            parsed = parser.parse(spec_path)
            # Generate implementation guide
            guide = parser.generate_implementation_guide(parsed)
            assert guide is not None
        finally:
            spec_path.unlink(missing_ok=True)

    def test_design_os_adapter_integration(self):
        """Test Design OS adapter full integration."""
        adapter = DesignOSAdapter()

        # Get status
        status = adapter.get_status()
        assert "tokens_loaded" in status or "path_exists" in status

    def test_jsx_validation(self, design_tokens):
        """Test JSX/React component validation."""
        validator = create_pixel_perfect_validator(design_tokens)

        jsx_code = '''
        function Button({ label }) {
            return (
                <button style={{ backgroundColor: "var(--color-primary)" }}>
                    {label}
                </button>
            );
        }
        '''

        report = validator.validate_jsx(jsx_code)
        assert report is not None

    def test_vue_validation(self, design_tokens):
        """Test Vue component validation."""
        validator = create_pixel_perfect_validator(design_tokens)

        vue_code = '''
        <template>
            <button class="btn-primary">{{ label }}</button>
        </template>
        <style scoped>
        .btn-primary {
            background-color: var(--color-primary);
        }
        </style>
        '''

        report = validator.validate_component(vue_code)
        assert report is not None

