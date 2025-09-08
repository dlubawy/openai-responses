# Repository Guidelines

## Project Structure & Module Organization

```
/workspace/openai-responses
├── src/          # Core logic
├── docs/         # Markdown guides
└── .github/      # GitHub Actions, issue templates
```

- **`src/`** contains the main source code.
- **`tests/`** mirrors the package layout; each module has a corresponding `test_*.py`.
- **`docs/`** holds documentation and contributor guides.
- **`scripts/`** includes automation tools.

## Coding Style & Naming Conventions

- **Indentation**: 4 spaces, no tabs.
- **Python**: PEP‑8; format with `ruff`.
- **File names**: snake_case (`my_module.py`).
- **Class names**: CamelCase (`MyObject`).
- **Constants**: UPPER_SNAKE_CASE.
- **Functions**: lower_snake_case.
- **Docstrings**: Google style, brief description + parameters.

## Commit & Pull Request Guidelines

- **Commit messages**: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
- **PR description**: Summarize changes, link related issues (`#123`), and include screenshots if UI changes.
- **Checklist**:
  - [ ] Tests added/updated
  - [ ] Documentation updated
  - [ ] Code reviewed by at least one other contributor

______________________________________________________________________

Follow these guidelines to keep the codebase clean, testable, and easy to contribute to.
