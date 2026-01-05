"""
Design OS Adapter for RLM-CLAUDE.

Integrates Design OS specifications into RLM-CLAUDE to ensure
pixel-perfect UI implementations. Loads product vision, component
specs, and design tokens from external design system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import re


@dataclass
class DesignToken:
    """A single design token (color, spacing, typography, etc.)."""
    name: str
    value: str
    category: str
    description: str = ""

    def to_css_var(self) -> str:
        """Convert to CSS variable format."""
        return f"var(--{self.name})"


@dataclass
class DesignTokens:
    """Collection of design tokens from Design OS."""
    colors: Dict[str, DesignToken] = field(default_factory=dict)
    spacing: Dict[str, DesignToken] = field(default_factory=dict)
    typography: Dict[str, DesignToken] = field(default_factory=dict)
    borders: Dict[str, DesignToken] = field(default_factory=dict)
    shadows: Dict[str, DesignToken] = field(default_factory=dict)

    def get_token(self, name: str) -> Optional[DesignToken]:
        """Get a token by name from any category."""
        for category in [self.colors, self.spacing, self.typography, self.borders, self.shadows]:
            if name in category:
                return category[name]
        return None

    def all_tokens(self) -> List[DesignToken]:
        """Get all tokens as a flat list."""
        tokens = []
        for category in [self.colors, self.spacing, self.typography, self.borders, self.shadows]:
            tokens.extend(category.values())
        return tokens

    def to_css_variables(self) -> str:
        """Generate CSS custom properties from tokens."""
        lines = [":root {"]
        for token in self.all_tokens():
            lines.append(f"  --{token.name}: {token.value};")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class ProductVision:
    """Product vision from Design OS."""
    name: str
    description: str
    target_audience: str = ""
    key_features: List[str] = field(default_factory=list)
    design_principles: List[str] = field(default_factory=list)
    brand_voice: str = ""


@dataclass
class PropDefinition:
    """Definition of a component prop."""
    name: str
    type: str
    required: bool = False
    default: Optional[Any] = None
    description: str = ""


@dataclass
class StyleDefinition:
    """Style definitions for a component."""
    base: Dict[str, str] = field(default_factory=dict)
    variants: Dict[str, Dict[str, str]] = field(default_factory=dict)
    states: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class BehaviorDefinition:
    """Behavior definition for a component."""
    name: str
    trigger: str
    action: str
    description: str = ""


@dataclass
class A11yRequirements:
    """Accessibility requirements for a component."""
    role: str = ""
    aria_labels: List[str] = field(default_factory=list)
    keyboard_nav: List[str] = field(default_factory=list)
    focus_management: str = ""
    screen_reader_text: str = ""


@dataclass
class ResponsiveRule:
    """Responsive design rule."""
    breakpoint: str
    changes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ComponentVariant:
    """A variant of a component."""
    name: str
    props: Dict[str, Any] = field(default_factory=dict)
    styles: Dict[str, str] = field(default_factory=dict)


@dataclass
class ComponentSpec:
    """Parsed component specification from Design OS."""
    name: str
    description: str
    props: List[PropDefinition] = field(default_factory=list)
    styles: StyleDefinition = field(default_factory=StyleDefinition)
    behaviors: List[BehaviorDefinition] = field(default_factory=list)
    a11y_requirements: A11yRequirements = field(default_factory=A11yRequirements)
    variants: List[ComponentVariant] = field(default_factory=list)
    responsive_rules: List[ResponsiveRule] = field(default_factory=list)
    design_tokens_used: List[str] = field(default_factory=list)


class DesignTokenValidator:
    """Validates design token usage in code."""

    def __init__(self, tokens: Optional[DesignTokens] = None):
        """Initialize with design tokens."""
        self.tokens = tokens or DesignTokens()

    def load(self, tokens_file: Path) -> DesignTokens:
        """Load design tokens from a JSON file."""
        if not tokens_file.exists():
            return DesignTokens()

        data = json.loads(tokens_file.read_text())
        tokens = DesignTokens()

        for category, items in data.items():
            if hasattr(tokens, category) and isinstance(items, dict):
                category_dict = getattr(tokens, category)
                for name, value in items.items():
                    if isinstance(value, dict):
                        category_dict[name] = DesignToken(
                            name=name,
                            value=value.get("value", ""),
                            category=category,
                            description=value.get("description", ""),
                        )
                    else:
                        category_dict[name] = DesignToken(
                            name=name,
                            value=str(value),
                            category=category,
                        )

        self.tokens = tokens
        return tokens

    def validate_code(self, code: str) -> List[Dict[str, Any]]:
        """
        Validate that code uses design tokens instead of raw values.

        Args:
            code: Source code to validate.

        Returns:
            List of validation issues.
        """
        issues = []

        # Check for hardcoded colors
        hex_pattern = r'#[0-9a-fA-F]{3,8}'
        for match in re.finditer(hex_pattern, code):
            issues.append({
                "type": "hardcoded_color",
                "value": match.group(),
                "message": f"Use design token instead of hardcoded color: {match.group()}",
                "suggestion": "Replace with var(--color-*)",
            })

        # Check for hardcoded pixel values in common properties
        px_pattern = r'(padding|margin|gap|width|height):\s*(\d+)px'
        for match in re.finditer(px_pattern, code):
            issues.append({
                "type": "hardcoded_spacing",
                "property": match.group(1),
                "value": match.group(2) + "px",
                "message": f"Use spacing token for {match.group(1)}",
                "suggestion": "Replace with var(--spacing-*)",
            })

        return issues


class SpecificationLoader:
    """Loads specifications from Design OS files."""

    def load_vision(self, vision_file: Path) -> ProductVision:
        """
        Load product vision from a markdown file.

        Args:
            vision_file: Path to vision.md file.

        Returns:
            Parsed ProductVision.
        """
        if not vision_file.exists():
            return ProductVision(name="Unknown", description="No vision file found")

        content = vision_file.read_text()

        # Parse markdown sections
        name = self._extract_heading(content, 1) or "Unknown"
        description = self._extract_section(content, "Description") or ""

        return ProductVision(
            name=name,
            description=description,
            target_audience=self._extract_section(content, "Target Audience") or "",
            key_features=self._extract_list(content, "Key Features"),
            design_principles=self._extract_list(content, "Design Principles"),
            brand_voice=self._extract_section(content, "Brand Voice") or "",
        )

    def _extract_heading(self, content: str, level: int) -> Optional[str]:
        """Extract first heading of given level."""
        pattern = rf'^{"#" * level}\s+(.+)$'
        match = re.search(pattern, content, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract content under a section heading."""
        pattern = rf'##\s+{section_name}\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_list(self, content: str, section_name: str) -> List[str]:
        """Extract list items from a section."""
        section = self._extract_section(content, section_name)
        if not section:
            return []

        items = []
        for line in section.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                items.append(line[2:].strip())
        return items


class DesignOSAdapter:
    """
    Integrates Design OS specifications into RLM-CLAUDE.
    Ensures pixel-perfect UI implementations.
    """

    def __init__(self, design_os_path: Optional[Path] = None):
        """
        Initialize the Design OS adapter.

        Args:
            design_os_path: Path to Design OS directory.
        """
        self.design_os_path = design_os_path or Path("design-os")
        self.spec_loader = SpecificationLoader()
        self.token_validator = DesignTokenValidator()
        self._design_tokens: Optional[DesignTokens] = None
        self._product_vision: Optional[ProductVision] = None
        self._component_specs: Dict[str, ComponentSpec] = {}

    def load_product_vision(self) -> ProductVision:
        """Load and parse product vision from Design OS."""
        if self._product_vision is None:
            vision_file = self.design_os_path / "vision.md"
            self._product_vision = self.spec_loader.load_vision(vision_file)
        return self._product_vision

    def load_design_tokens(self) -> DesignTokens:
        """Load design tokens (colors, spacing, typography)."""
        if self._design_tokens is None:
            tokens_file = self.design_os_path / "tokens.json"
            self._design_tokens = self.token_validator.load(tokens_file)
        return self._design_tokens

    def validate_implementation(self, code: str) -> List[Dict[str, Any]]:
        """
        Validate that implementation uses design tokens.

        Args:
            code: Source code to validate.

        Returns:
            List of validation issues.
        """
        self.load_design_tokens()
        return self.token_validator.validate_code(code)

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "design_os_path": str(self.design_os_path),
            "path_exists": self.design_os_path.exists(),
            "vision_loaded": self._product_vision is not None,
            "tokens_loaded": self._design_tokens is not None,
            "component_specs_count": len(self._component_specs),
        }

