# Contributing to RLM-CLAUDE-CODE-DESIGNER

Thank you for your interest in contributing! We welcome contributions from the community to help make this the best enterprise RLM architecture available.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/RLM-CLAUDE-CODE-DESIGNER.git
   cd RLM-CLAUDE-CODE-DESIGNER
   ```

3. **Install dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

4. **Create a branch** for your feature or fix:

   ```bash
   git checkout -b feature/amazing-feature
   ```

## Development Workflow

- We use **pytest** for testing. Ensure all tests pass before submitting a PR.

  ```bash
  python -m pytest
  ```

- Use **Black** for code formatting and **isort** for import sorting.
- Add type hints and check with **mypy**.

## Pull Request Process

1. Ensure your code follows the project's style and patterns.
2. Update documentation (Wiki/README) if you are changing functionality.
3. Add tests for any new features or bug fixes.
4. Ensure the test suite passes locally.
5. Push your branch and open a Pull Request against the `main` branch.
6. Provide a clear description of the changes and the problem they solve.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the GitHub tracker. Provide as much detail as possible, including reproduction steps and environment details.

## License

By contributing, you agree that your contributions will be licensed under the MIT License used by this project.
