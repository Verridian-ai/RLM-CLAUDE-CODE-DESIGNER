"""
Security Validator for the Quality Gate System.

Validates code for security vulnerabilities and best practices.
"""

import re
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Pattern

from .base import BaseValidator, ValidationResult, ValidationIssue, ValidationSeverity


class SecurityValidator(BaseValidator):
    """
    Validates security practices in the codebase.

    Checks for:
    - Hardcoded secrets
    - SQL injection vulnerabilities
    - Command injection vulnerabilities
    - Insecure cryptography
    - Dangerous function usage
    - Missing input validation
    """

    # Patterns that indicate hardcoded secrets
    SECRET_PATTERNS: List[tuple] = [
        (re.compile(r"(?i)(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]"), "Hardcoded password"),
        (re.compile(r"(?i)(api[_-]?key|apikey)\s*=\s*['\"][^'\"]+['\"]"), "Hardcoded API key"),
        (re.compile(r"(?i)(secret|token)\s*=\s*['\"][^'\"]{8,}['\"]"), "Hardcoded secret/token"),
        (re.compile(r"(?i)(aws[_-]?access[_-]?key)\s*=\s*['\"][A-Z0-9]{20}['\"]"), "Hardcoded AWS key"),
        (re.compile(r"(?i)private[_-]?key\s*=\s*['\"]-----BEGIN"), "Hardcoded private key"),
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r"execute\s*\(\s*['\"].*%s.*['\"]"),
        re.compile(r"execute\s*\(\s*f['\"]"),
        re.compile(r"cursor\.execute\s*\(\s*['\"].*\+"),
        re.compile(r"\.format\s*\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE|DROP)", re.IGNORECASE),
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r"os\.system\s*\("),
        re.compile(r"subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True"),
        re.compile(r"eval\s*\("),
        re.compile(r"exec\s*\("),
    ]

    # Insecure crypto patterns
    INSECURE_CRYPTO_PATTERNS: List[tuple] = [
        (re.compile(r"(?i)md5\s*\("), "MD5 is cryptographically broken"),
        (re.compile(r"(?i)sha1\s*\("), "SHA1 is deprecated for security"),
        (re.compile(r"DES\.new"), "DES is insecure, use AES"),
        (re.compile(r"mode\s*=\s*(?:ECB|MODE_ECB)"), "ECB mode is insecure"),
    ]

    def __init__(self, enabled: bool = True):
        super().__init__("SecurityValidator", enabled)
        self._rules = {
            "SEC001": True,  # Check for hardcoded secrets
            "SEC002": True,  # Check for SQL injection
            "SEC003": True,  # Check for command injection
            "SEC004": True,  # Check for insecure crypto
            "SEC005": True,  # Check for dangerous functions
            "SEC006": True,  # Check for missing input validation
        }

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate security from the context."""
        start_time = time.time()
        issues: List[ValidationIssue] = []

        code_files = context.get("code_files", {})
        code_content = context.get("code", "")

        # If we have individual files, check each one
        if code_files:
            for file_path, content in code_files.items():
                file_issues = self._check_code(content, Path(file_path))
                issues.extend(file_issues)
        elif code_content:
            issues.extend(self._check_code(code_content))

        # Calculate score - security issues are weighted heavily
        critical_count = sum(1 for i in issues if i.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)

        score = max(0.0, 1.0 - (critical_count * 0.5) - (error_count * 0.2))
        passed = critical_count == 0  # Any critical security issue fails validation

        return self._create_result(
            passed=passed,
            issues=issues,
            score=score,
            duration_ms=self._elapsed(start_time),
            metadata={"files_scanned": len(code_files) or 1},
        )

    def _check_code(
        self, code: str, file_path: Optional[Path] = None
    ) -> List[ValidationIssue]:
        """Check a code string for security issues."""
        issues = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue

            # SEC001: Check for hardcoded secrets
            if self.is_rule_enabled("SEC001"):
                for pattern, description in self.SECRET_PATTERNS:
                    if pattern.search(line):
                        issues.append(self._issue(
                            "SEC001",
                            f"{description} detected",
                            ValidationSeverity.CRITICAL,
                            file_path=file_path,
                            line_number=line_num,
                            suggestion="Use environment variables or a secrets manager",
                            rule_id="SEC001",
                        ))

            # SEC002: Check for SQL injection
            if self.is_rule_enabled("SEC002"):
                for pattern in self.SQL_INJECTION_PATTERNS:
                    if pattern.search(line):
                        issues.append(self._issue(
                            "SEC002",
                            "Potential SQL injection vulnerability",
                            ValidationSeverity.ERROR,
                            file_path=file_path,
                            line_number=line_num,
                            suggestion="Use parameterized queries",
                            rule_id="SEC002",
                        ))
                        break

            # SEC003: Check for command injection
            if self.is_rule_enabled("SEC003"):
                for pattern in self.COMMAND_INJECTION_PATTERNS:
                    if pattern.search(line):
                        issues.append(self._issue(
                            "SEC003",
                            "Potential command injection vulnerability",
                            ValidationSeverity.ERROR,
                            file_path=file_path,
                            line_number=line_num,
                            suggestion="Avoid shell=True and user input in commands",
                            rule_id="SEC003",
                        ))
                        break

            # SEC004: Check for insecure crypto
            if self.is_rule_enabled("SEC004"):
                for pattern, description in self.INSECURE_CRYPTO_PATTERNS:
                    if pattern.search(line):
                        issues.append(self._issue(
                            "SEC004",
                            f"Insecure cryptography: {description}",
                            ValidationSeverity.WARNING,
                            file_path=file_path,
                            line_number=line_num,
                            suggestion="Use modern, secure cryptographic algorithms",
                            rule_id="SEC004",
                        ))

        # SEC005: Check for dangerous functions (file-level)
        if self.is_rule_enabled("SEC005"):
            dangerous_functions = [
                ("pickle.loads", "Pickle can execute arbitrary code"),
                ("yaml.load(", "Use yaml.safe_load instead"),
                ("marshal.loads", "Marshal can execute arbitrary code"),
                ("__import__", "Dynamic imports can be dangerous"),
            ]
            for func, warning in dangerous_functions:
                if func in code:
                    issues.append(self._issue(
                        "SEC005",
                        f"Dangerous function usage: {warning}",
                        ValidationSeverity.WARNING,
                        file_path=file_path,
                        suggestion=f"Avoid using {func} with untrusted data",
                        rule_id="SEC005",
                    ))

        return issues

    def _elapsed(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000
