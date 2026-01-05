"""
Component Specification Parser for RLM-CLAUDE.

Parses Design OS component specifications and generates
implementation guides for pixel-perfect UI development.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

# Try to import yaml, fall back to json-based parsing
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    import json

from .design_os_adapter import (
    ComponentSpec,
    PropDefinition,
    StyleDefinition,
    BehaviorDefinition,
    A11yRequirements,
    ComponentVariant,
    ResponsiveRule,
)


@dataclass
class ImplementationGuide:
    """Generated implementation guide for a component."""
    component_name: str
    framework: str = "react"
    props_interface: str = ""
    styles: str = ""
    test_cases: List[str] = field(default_factory=list)
    skeleton_code: str = ""
    accessibility_notes: List[str] = field(default_factory=list)


class ComponentSpecParser:
    """
    Parses Design OS component specifications.
    Outputs structured specs for implementation guidance.
    """

    def __init__(self):
        """Initialize the parser."""
        self._parsed_specs: Dict[str, ComponentSpec] = {}

    def parse(self, spec_file: Path) -> ComponentSpec:
        """
        Parse a component specification file.

        Args:
            spec_file: Path to YAML/JSON spec file.

        Returns:
            Parsed ComponentSpec.
        """
        if not spec_file.exists():
            raise FileNotFoundError(f"Spec file not found: {spec_file}")

        content = spec_file.read_text()

        if HAS_YAML and spec_file.suffix in ('.yaml', '.yml'):
            raw = yaml.safe_load(content)
        else:
            raw = self._parse_json_or_simple(content)

        spec = ComponentSpec(
            name=raw.get("name", spec_file.stem),
            description=raw.get("description", ""),
            props=self._parse_props(raw.get("props", [])),
            styles=self._parse_styles(raw.get("styles", {})),
            behaviors=self._parse_behaviors(raw.get("behaviors", [])),
            a11y_requirements=self._parse_a11y(raw.get("accessibility", {})),
            variants=self._parse_variants(raw.get("variants", [])),
            responsive_rules=self._parse_responsive(raw.get("responsive", [])),
            design_tokens_used=raw.get("tokens", []),
        )

        self._parsed_specs[spec.name] = spec
        return spec

    def _parse_json_or_simple(self, content: str) -> Dict[str, Any]:
        """Parse JSON or simple key-value format."""
        try:
            import json
            return json.loads(content)
        except Exception:
            # Simple key-value parsing fallback
            return {"name": "unknown", "description": content}

    def _parse_props(self, props_data: List[Dict]) -> List[PropDefinition]:
        """Parse prop definitions."""
        props = []
        for p in props_data:
            if isinstance(p, dict):
                props.append(PropDefinition(
                    name=p.get("name", ""),
                    type=p.get("type", "any"),
                    required=p.get("required", False),
                    default=p.get("default"),
                    description=p.get("description", ""),
                ))
        return props

    def _parse_styles(self, styles_data: Dict) -> StyleDefinition:
        """Parse style definitions."""
        return StyleDefinition(
            base=styles_data.get("base", {}),
            variants=styles_data.get("variants", {}),
            states=styles_data.get("states", {}),
        )

    def _parse_behaviors(self, behaviors_data: List[Dict]) -> List[BehaviorDefinition]:
        """Parse behavior definitions."""
        behaviors = []
        for b in behaviors_data:
            if isinstance(b, dict):
                behaviors.append(BehaviorDefinition(
                    name=b.get("name", ""),
                    trigger=b.get("trigger", ""),
                    action=b.get("action", ""),
                    description=b.get("description", ""),
                ))
        return behaviors

    def _parse_a11y(self, a11y_data: Dict) -> A11yRequirements:
        """Parse accessibility requirements."""
        return A11yRequirements(
            role=a11y_data.get("role", ""),
            aria_labels=a11y_data.get("aria_labels", []),
            keyboard_nav=a11y_data.get("keyboard_nav", []),
            focus_management=a11y_data.get("focus_management", ""),
            screen_reader_text=a11y_data.get("screen_reader_text", ""),
        )

    def _parse_variants(self, variants_data: List[Dict]) -> List[ComponentVariant]:
        """Parse component variants."""
        variants = []
        for v in variants_data:
            if isinstance(v, dict):
                variants.append(ComponentVariant(
                    name=v.get("name", ""),
                    props=v.get("props", {}),
                    styles=v.get("styles", {}),
                ))
        return variants

    def _parse_responsive(self, responsive_data: List[Dict]) -> List[ResponsiveRule]:
        """Parse responsive rules."""
        rules = []
        for r in responsive_data:
            if isinstance(r, dict):
                rules.append(ResponsiveRule(
                    breakpoint=r.get("breakpoint", ""),
                    changes=r.get("changes", {}),
                ))
        return rules

    def generate_implementation_guide(
        self,
        spec: ComponentSpec,
        framework: str = "react",
    ) -> ImplementationGuide:
        """
        Generate detailed implementation guide from spec.
        Includes: skeleton code, style definitions, test cases.

        Args:
            spec: Parsed component specification.
            framework: Target framework (react, vue, etc.).

        Returns:
            Implementation guide with code templates.
        """
        guide = ImplementationGuide(
            component_name=spec.name,
            framework=framework,
        )

        # Generate TypeScript interface for props
        guide.props_interface = self._generate_props_interface(spec.props)

        # Generate CSS/styled-components from design tokens
        guide.styles = self._generate_styles(spec.styles, spec.design_tokens_used)

        # Generate test cases from behaviors
        guide.test_cases = self._generate_test_cases(spec.behaviors)

        # Generate skeleton code
        guide.skeleton_code = self._generate_skeleton(spec, framework)

        # Add accessibility notes
        guide.accessibility_notes = self._generate_a11y_notes(spec.a11y_requirements)

        return guide

    def _generate_props_interface(self, props: List[PropDefinition]) -> str:
        """Generate TypeScript interface for props."""
        lines = ["interface Props {"]
        for prop in props:
            optional = "" if prop.required else "?"
            lines.append(f"  {prop.name}{optional}: {prop.type};")
        lines.append("}")
        return "\n".join(lines)

    def _generate_styles(
        self,
        styles: StyleDefinition,
        tokens: List[str],
    ) -> str:
        """Generate CSS styles from style definition."""
        lines = []

        # Base styles
        if styles.base:
            lines.append(".component {")
            for prop, value in styles.base.items():
                lines.append(f"  {prop}: {value};")
            lines.append("}")

        # Variant styles
        for variant_name, variant_styles in styles.variants.items():
            lines.append(f"\n.component--{variant_name} {{")
            for prop, value in variant_styles.items():
                lines.append(f"  {prop}: {value};")
            lines.append("}")

        # State styles
        for state_name, state_styles in styles.states.items():
            lines.append(f"\n.component:{state_name} {{")
            for prop, value in state_styles.items():
                lines.append(f"  {prop}: {value};")
            lines.append("}")

        return "\n".join(lines)

    def _generate_test_cases(self, behaviors: List[BehaviorDefinition]) -> List[str]:
        """Generate test case descriptions from behaviors."""
        test_cases = []
        for behavior in behaviors:
            test_cases.append(
                f"it('should {behavior.action} when {behavior.trigger}')"
            )
        return test_cases

    def _generate_skeleton(self, spec: ComponentSpec, framework: str) -> str:
        """Generate skeleton component code."""
        if framework == "react":
            return self._generate_react_skeleton(spec)
        elif framework == "vue":
            return self._generate_vue_skeleton(spec)
        return ""

    def _generate_react_skeleton(self, spec: ComponentSpec) -> str:
        """Generate React component skeleton."""
        props_list = ", ".join(p.name for p in spec.props)

        return f'''import React from 'react';

{self._generate_props_interface(spec.props)}

export const {spec.name}: React.FC<Props> = ({{ {props_list} }}) => {{
  return (
    <div className="{spec.name.lower()}">
      {{/* TODO: Implement {spec.name} */}}
    </div>
  );
}};
'''

    def _generate_vue_skeleton(self, spec: ComponentSpec) -> str:
        """Generate Vue component skeleton."""
        return f'''<template>
  <div class="{spec.name.lower()}">
    <!-- TODO: Implement {spec.name} -->
  </div>
</template>

<script setup lang="ts">
// Props
defineProps<{{
  // TODO: Add props
}}>();
</script>
'''

    def _generate_a11y_notes(self, a11y: A11yRequirements) -> List[str]:
        """Generate accessibility implementation notes."""
        notes = []
        if a11y.role:
            notes.append(f"Use role='{a11y.role}'")
        for label in a11y.aria_labels:
            notes.append(f"Add aria-label: {label}")
        for nav in a11y.keyboard_nav:
            notes.append(f"Keyboard: {nav}")
        if a11y.focus_management:
            notes.append(f"Focus: {a11y.focus_management}")
        return notes

    def get_parsed_specs(self) -> Dict[str, ComponentSpec]:
        """Get all parsed specifications."""
        return self._parsed_specs.copy()

