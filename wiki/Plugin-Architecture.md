# Plugin Architecture

## 7. Validators (`validators/`)

**Purpose**: Modular validation system for different aspects of code quality.

**Available Validators**:

- `RequirementsValidator`
- `ArchitectureValidator`
- `DesignValidator`
- `SecurityValidator`

**Extending**:

Create a class inheriting from `BaseValidator`:

```python
from rlm_lib.validators import BaseValidator, ValidationResult

class CustomValidator(BaseValidator):
    def validate(self, content: str, context: dict) -> ValidationResult:
        # Check logic here
        return ValidationResult(passed=True, issues=[])
```
