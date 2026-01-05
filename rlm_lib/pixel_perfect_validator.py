"""
Pixel-Perfect Validator for RLM-CLAUDE.

Validates UI implementations against design specifications to ensure
zero deviation from design intent. Checks design token usage, spacing,
colors, typography, and detects generic/default patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import re

from .design_os_adapter import DesignTokens, ComponentSpec


@dataclass
class ValidationIssue:
    """A single validation issue found in the code."""
    type: str
    severity: str  # "error", "warning", "info"
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    suggestion: str = ""
    code_snippet: str = ""


@dataclass
class PixelPerfectReport:
    """Report from pixel-perfect validation."""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 1.0
    summary: str = ""

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        self._recalculate_score()

    def _recalculate_score(self) -> None:
        """Recalculate score based on issues."""
        if not self.issues:
            self.score = 1.0
            self.passed = True
            return

        error_count = sum(1 for i in self.issues if i.severity == "error")
        warning_count = sum(1 for i in self.issues if i.severity == "warning")

        # Errors reduce score significantly, warnings less so
        self.score = max(0.0, 1.0 - (error_count * 0.2) - (warning_count * 0.05))
        self.passed = error_count == 0


class PixelPerfectValidator:
    """
    Validates UI implementations against design specifications.
    Ensures zero deviation from design intent.
    """

    # Tolerance settings for different properties
    DEFAULT_TOLERANCE = {
        "spacing": 0,      # No tolerance for spacing
        "color": 0,        # Exact hex match required
        "typography": 0,   # Exact font specs required
        "border_radius": 2, # 2px tolerance
    }

    # Generic patterns to detect and flag
    GENERIC_PATTERNS = [
        (r"border-radius:\s*4px", "Generic border-radius (4px) detected - use design token"),
        (r"border-radius:\s*0\.25rem", "Generic border-radius (0.25rem) detected - use design token"),
        (r"padding:\s*\d+px(?!\s*var)", "Hardcoded padding - use spacing token"),
        (r"margin:\s*\d+px(?!\s*var)", "Hardcoded margin - use spacing token"),
        (r"#[0-9a-fA-F]{3,8}(?![0-9a-fA-F])", "Hardcoded color - use color token"),
        (r"font-size:\s*\d+px(?!\s*var)", "Hardcoded font-size - use typography token"),
        (r"font-family:\s*['\"]?(?:Arial|Helvetica|sans-serif)", "Generic font-family detected"),
        (r"box-shadow:\s*0\s+\d+px\s+\d+px\s+rgba", "Hardcoded box-shadow - use shadow token"),
    ]

    # Bootstrap/framework default patterns (matches both CSS selectors and class attributes)
    FRAMEWORK_DEFAULTS = [
        (r"(?:\.|\bclass=['\"].*?)btn-primary", "Bootstrap default button class detected"),
        (r"(?:\.|\bclass=['\"].*?)btn-secondary", "Bootstrap default button class detected"),
        (r"(?:\.|\bclass=['\"].*?)card(?!\w)", "Generic card class - ensure custom styling"),
        (r"(?:\.|\bclass=['\"].*?)form-control", "Bootstrap form-control - ensure custom styling"),
        (r"(?:\.|\bclass=['\"].*?)container(?!\w)", "Generic container class - verify custom styling"),
    ]

    def __init__(
        self,
        design_tokens: Optional[DesignTokens] = None,
        tolerance: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the validator.

        Args:
            design_tokens: Design tokens to validate against.
            tolerance: Custom tolerance settings.
        """
        self.design_tokens = design_tokens or DesignTokens()
        self.tolerance = tolerance or self.DEFAULT_TOLERANCE.copy()

    def validate_component(
        self,
        implementation: str,
        spec: Optional[ComponentSpec] = None,
    ) -> PixelPerfectReport:
        """
        Comprehensive validation of component implementation.

        Args:
            implementation: Source code to validate.
            spec: Optional component specification.

        Returns:
            Validation report with issues and score.
        """
        report = PixelPerfectReport(passed=True)

        # Validate design token usage
        self._validate_tokens(implementation, report)

        # Validate spacing
        self._validate_spacing(implementation, report)

        # Validate colors
        self._validate_colors(implementation, report)

        # Validate typography
        self._validate_typography(implementation, report)

        # Detect generic patterns
        self._detect_generic_patterns(implementation, report)

        # Detect framework defaults
        self._detect_framework_defaults(implementation, report)

        # Generate summary
        report.summary = self._generate_summary(report)

        return report

    def _validate_tokens(self, code: str, report: PixelPerfectReport) -> None:
        """Validate that design tokens are used properly."""
        # Check for CSS custom property usage
        var_pattern = r'var\(--([a-zA-Z0-9-]+)\)'
        used_tokens = set(re.findall(var_pattern, code))

        # Verify used tokens exist in design system
        for token_name in used_tokens:
            if self.design_tokens.get_token(token_name) is None:
                report.add_issue(ValidationIssue(
                    type="unknown_token",
                    severity="warning",
                    message=f"Unknown design token: --{token_name}",
                    suggestion="Verify token exists in design system",
                ))

    def _validate_spacing(self, code: str, report: PixelPerfectReport) -> None:
        """Validate spacing values use design tokens."""
        spacing_pattern = r'(padding|margin|gap):\s*(\d+)(px|rem|em)'

        for match in re.finditer(spacing_pattern, code):
            prop = match.group(1)
            value = match.group(2) + match.group(3)

            report.add_issue(ValidationIssue(
                type="hardcoded_spacing",
                severity="error",
                message=f"Hardcoded {prop}: {value}",
                suggestion=f"Use var(--spacing-*) instead of {value}",
                code_snippet=match.group(0),
            ))

    def _validate_colors(self, code: str, report: PixelPerfectReport) -> None:
        """Validate color values use design tokens."""
        # Match hex colors - simple pattern without lookbehind
        hex_pattern = r'#([0-9a-fA-F]{3,8})\b'

        for match in re.finditer(hex_pattern, code):
            hex_value = "#" + match.group(1)

            # Check if this hex is inside a var() - skip if so
            start = match.start()
            prefix = code[max(0, start - 20):start]
            if 'var(--' in prefix:
                continue

            report.add_issue(ValidationIssue(
                type="hardcoded_color",
                severity="error",
                message=f"Hardcoded color: {hex_value}",
                suggestion="Use var(--color-*) instead",
                code_snippet=hex_value,
            ))

        # Match rgb/rgba colors (but not inside var())
        rgb_pattern = r'rgba?\([^)]+\)'
        for match in re.finditer(rgb_pattern, code):
            # Skip if inside var()
            start = match.start()
            prefix = code[max(0, start - 20):start]
            if 'var(--' in prefix:
                continue

            report.add_issue(ValidationIssue(
                type="hardcoded_color",
                severity="error",
                message=f"Hardcoded color: {match.group(0)}",
                suggestion="Use var(--color-*) instead",
                code_snippet=match.group(0),
            ))

    def _validate_typography(self, code: str, report: PixelPerfectReport) -> None:
        """Validate typography values use design tokens."""
        font_size_pattern = r'font-size:\s*(\d+)(px|rem|em)'

        for match in re.finditer(font_size_pattern, code):
            value = match.group(1) + match.group(2)

            report.add_issue(ValidationIssue(
                type="hardcoded_typography",
                severity="warning",
                message=f"Hardcoded font-size: {value}",
                suggestion="Use var(--font-size-*) instead",
                code_snippet=match.group(0),
            ))

        # Check for hardcoded font-weight
        font_weight_pattern = r'font-weight:\s*(\d+|bold|normal)'
        for match in re.finditer(font_weight_pattern, code):
            report.add_issue(ValidationIssue(
                type="hardcoded_typography",
                severity="info",
                message=f"Consider using typography token for font-weight: {match.group(1)}",
                suggestion="Use var(--font-weight-*) if available",
                code_snippet=match.group(0),
            ))

    def _detect_generic_patterns(self, code: str, report: PixelPerfectReport) -> None:
        """Detect usage of generic/default styles."""
        for pattern, message in self.GENERIC_PATTERNS:
            for match in re.finditer(pattern, code):
                report.add_issue(ValidationIssue(
                    type="generic_pattern",
                    severity="warning",
                    message=message,
                    code_snippet=match.group(0),
                ))

    def _detect_framework_defaults(self, code: str, report: PixelPerfectReport) -> None:
        """Detect usage of framework default classes."""
        for pattern, message in self.FRAMEWORK_DEFAULTS:
            for match in re.finditer(pattern, code):
                report.add_issue(ValidationIssue(
                    type="framework_default",
                    severity="info",
                    message=message,
                    suggestion="Ensure custom styling is applied",
                    code_snippet=match.group(0),
                ))

    def _generate_summary(self, report: PixelPerfectReport) -> str:
        """Generate a human-readable summary."""
        if report.passed:
            return f"✅ Validation passed with score {report.score:.2f}"

        error_count = sum(1 for i in report.issues if i.severity == "error")
        warning_count = sum(1 for i in report.issues if i.severity == "warning")
        info_count = sum(1 for i in report.issues if i.severity == "info")

        return (
            f"❌ Validation failed with score {report.score:.2f}\n"
            f"   Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}"
        )

    def validate_css(self, css_code: str) -> PixelPerfectReport:
        """Validate CSS code specifically."""
        return self.validate_component(css_code)

    def validate_jsx(self, jsx_code: str) -> PixelPerfectReport:
        """Validate JSX/TSX code specifically."""
        # Extract inline styles from JSX
        style_pattern = r'style=\{\{([^}]+)\}\}'
        inline_styles = " ".join(re.findall(style_pattern, jsx_code))

        # Validate both the JSX and extracted styles
        report = self.validate_component(jsx_code)

        if inline_styles:
            style_report = self.validate_component(inline_styles)
            for issue in style_report.issues:
                issue.message = f"[Inline style] {issue.message}"
                report.add_issue(issue)

        return report

    def get_score(self, code: str) -> float:
        """Get just the validation score for code."""
        report = self.validate_component(code)
        return report.score


def create_pixel_perfect_validator(
    design_tokens: Optional[DesignTokens] = None,
) -> PixelPerfectValidator:
    """Factory function to create a PixelPerfectValidator."""
    return PixelPerfectValidator(design_tokens=design_tokens)

