"""
Unit tests for Design OS Integration (Phase 4).
Tests Design OS Adapter, Component Spec Parser, and Pixel-Perfect Validator.
"""

import pytest
import json
import tempfile
from pathlib import Path

from rlm_lib.design_os_adapter import (
    DesignOSAdapter,
    DesignToken,
    DesignTokens,
    DesignTokenValidator,
    SpecificationLoader,
    ProductVision,
    ComponentSpec,
)
from rlm_lib.component_spec_parser import (
    ComponentSpecParser,
    ImplementationGuide,
)
from rlm_lib.pixel_perfect_validator import (
    PixelPerfectValidator,
    PixelPerfectReport,
    ValidationIssue,
    create_pixel_perfect_validator,
)


class TestDesignToken:
    """Tests for DesignToken dataclass."""

    def test_token_creation(self):
        """Test creating a design token."""
        token = DesignToken(
            name="color-primary",
            value="#3B82F6",
            category="colors",
        )
        assert token.name == "color-primary"
        assert token.value == "#3B82F6"

    def test_to_css_var(self):
        """Test CSS variable conversion."""
        token = DesignToken(name="spacing-md", value="16px", category="spacing")
        assert token.to_css_var() == "var(--spacing-md)"


class TestDesignTokens:
    """Tests for DesignTokens collection."""

    @pytest.fixture
    def tokens(self):
        tokens = DesignTokens()
        tokens.colors["primary"] = DesignToken("primary", "#3B82F6", "colors")
        tokens.spacing["md"] = DesignToken("md", "16px", "spacing")
        return tokens

    def test_get_token(self, tokens):
        """Test getting a token by name."""
        token = tokens.get_token("primary")
        assert token is not None
        assert token.value == "#3B82F6"

    def test_get_token_not_found(self, tokens):
        """Test getting non-existent token."""
        assert tokens.get_token("nonexistent") is None

    def test_all_tokens(self, tokens):
        """Test getting all tokens."""
        all_tokens = tokens.all_tokens()
        assert len(all_tokens) == 2

    def test_to_css_variables(self, tokens):
        """Test CSS variables generation."""
        css = tokens.to_css_variables()
        assert ":root {" in css
        assert "--primary: #3B82F6;" in css


class TestDesignTokenValidator:
    """Tests for DesignTokenValidator."""

    def test_validate_code_finds_hardcoded_colors(self):
        """Test detecting hardcoded colors."""
        validator = DesignTokenValidator()
        code = "color: #FF0000; background: #00FF00;"
        issues = validator.validate_code(code)
        assert len(issues) == 2
        assert all(i["type"] == "hardcoded_color" for i in issues)

    def test_validate_code_finds_hardcoded_spacing(self):
        """Test detecting hardcoded spacing."""
        validator = DesignTokenValidator()
        code = "padding: 16px; margin: 8px;"
        issues = validator.validate_code(code)
        assert len(issues) == 2
        assert all(i["type"] == "hardcoded_spacing" for i in issues)

    def test_load_tokens_from_file(self):
        """Test loading tokens from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "colors": {"primary": "#3B82F6"},
                "spacing": {"md": "16px"},
            }, f)
            f.flush()

            validator = DesignTokenValidator()
            tokens = validator.load(Path(f.name))

            assert tokens.get_token("primary") is not None
            assert tokens.get_token("md") is not None


class TestSpecificationLoader:
    """Tests for SpecificationLoader."""

    def test_load_vision_from_markdown(self):
        """Test loading product vision from markdown."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# My Product

## Description
A great product for users.

## Key Features
- Feature 1
- Feature 2

## Design Principles
- Clean design
- User-friendly
""")
            f.flush()

            loader = SpecificationLoader()
            vision = loader.load_vision(Path(f.name))

            assert vision.name == "My Product"
            assert "great product" in vision.description


class TestComponentSpecParser:
    """Tests for ComponentSpecParser."""

    def test_parser_creation(self):
        """Test creating parser."""
        parser = ComponentSpecParser()
        assert parser.get_parsed_specs() == {}

    def test_parse_json_spec(self):
        """Test parsing JSON spec file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "name": "Button",
                "description": "A button component",
                "props": [
                    {"name": "variant", "type": "string", "required": True},
                    {"name": "disabled", "type": "boolean", "default": False},
                ],
                "styles": {
                    "base": {"padding": "var(--spacing-md)"},
                    "variants": {"primary": {"background": "var(--color-primary)"}},
                },
                "behaviors": [
                    {"name": "click", "trigger": "onClick", "action": "emit click event"},
                ],
                "accessibility": {
                    "role": "button",
                    "aria_labels": ["Button label"],
                },
            }, f)
            f.flush()

            parser = ComponentSpecParser()
            spec = parser.parse(Path(f.name))

            assert spec.name == "Button"
            assert len(spec.props) == 2
            assert spec.props[0].name == "variant"
            assert spec.a11y_requirements.role == "button"

    def test_generate_implementation_guide(self):
        """Test generating implementation guide."""
        spec = ComponentSpec(
            name="Button",
            description="A button",
            props=[],
            behaviors=[],
        )

        parser = ComponentSpecParser()
        guide = parser.generate_implementation_guide(spec, framework="react")

        assert guide.component_name == "Button"
        assert guide.framework == "react"
        assert "Button" in guide.skeleton_code

    def test_generate_vue_skeleton(self):
        """Test generating Vue skeleton."""
        spec = ComponentSpec(name="Card", description="A card")

        parser = ComponentSpecParser()
        guide = parser.generate_implementation_guide(spec, framework="vue")

        assert "<template>" in guide.skeleton_code
        assert "card" in guide.skeleton_code


class TestPixelPerfectValidator:
    """Tests for PixelPerfectValidator."""

    def test_validator_creation(self):
        """Test creating validator."""
        validator = PixelPerfectValidator()
        assert validator.tolerance["spacing"] == 0

    def test_validate_clean_code(self):
        """Test validating code with design tokens."""
        validator = PixelPerfectValidator()
        code = """
        .button {
            padding: var(--spacing-md);
            color: var(--color-primary);
        }
        """
        report = validator.validate_component(code)
        # Should have no errors (only possible unknown token warnings)
        errors = [i for i in report.issues if i.severity == "error"]
        assert len(errors) == 0

    def test_validate_hardcoded_colors(self):
        """Test detecting hardcoded colors."""
        validator = PixelPerfectValidator()
        code = "color: #FF0000;"
        report = validator.validate_component(code)

        color_issues = [i for i in report.issues if i.type == "hardcoded_color"]
        assert len(color_issues) > 0

    def test_validate_hardcoded_spacing(self):
        """Test detecting hardcoded spacing."""
        validator = PixelPerfectValidator()
        code = "padding: 16px; margin: 8rem;"
        report = validator.validate_component(code)

        spacing_issues = [i for i in report.issues if i.type == "hardcoded_spacing"]
        assert len(spacing_issues) == 2

    def test_detect_generic_patterns(self):
        """Test detecting generic patterns."""
        validator = PixelPerfectValidator()
        code = "border-radius: 4px;"
        report = validator.validate_component(code)

        generic_issues = [i for i in report.issues if i.type == "generic_pattern"]
        assert len(generic_issues) > 0

    def test_detect_framework_defaults(self):
        """Test detecting framework defaults."""
        validator = PixelPerfectValidator()
        code = '<button class="btn-primary">Click</button>'
        report = validator.validate_component(code)

        framework_issues = [i for i in report.issues if i.type == "framework_default"]
        assert len(framework_issues) > 0

    def test_score_calculation(self):
        """Test score calculation."""
        validator = PixelPerfectValidator()

        # Clean code should have high score
        clean_code = "padding: var(--spacing-md);"
        clean_report = validator.validate_component(clean_code)

        # Dirty code should have lower score
        dirty_code = "padding: 16px; color: #FF0000; margin: 8px;"
        dirty_report = validator.validate_component(dirty_code)

        assert clean_report.score > dirty_report.score

    def test_validate_jsx(self):
        """Test validating JSX code."""
        validator = PixelPerfectValidator()
        jsx = '<div style={{padding: "16px", color: "#FF0000"}}>Hello</div>'
        report = validator.validate_jsx(jsx)

        assert len(report.issues) > 0

    def test_factory_function(self):
        """Test factory function."""
        validator = create_pixel_perfect_validator()
        assert isinstance(validator, PixelPerfectValidator)


class TestPixelPerfectReport:
    """Tests for PixelPerfectReport."""

    def test_report_creation(self):
        """Test creating report."""
        report = PixelPerfectReport(passed=True)
        assert report.passed
        assert report.score == 1.0

    def test_add_issue(self):
        """Test adding issues."""
        report = PixelPerfectReport(passed=True)
        report.add_issue(ValidationIssue(
            type="test",
            severity="error",
            message="Test error",
        ))

        assert len(report.issues) == 1
        assert not report.passed
        assert report.score < 1.0


class TestDesignOSAdapter:
    """Tests for DesignOSAdapter."""

    def test_adapter_creation(self):
        """Test creating adapter."""
        adapter = DesignOSAdapter()
        assert adapter.design_os_path == Path("design-os")

    def test_get_status(self):
        """Test getting adapter status."""
        adapter = DesignOSAdapter()
        status = adapter.get_status()
        assert "design_os_path" in status
        assert "path_exists" in status

    def test_validate_implementation(self):
        """Test validating implementation code."""
        adapter = DesignOSAdapter()
        code = "color: #FF0000; padding: 16px;"
        issues = adapter.validate_implementation(code)
        assert len(issues) > 0

