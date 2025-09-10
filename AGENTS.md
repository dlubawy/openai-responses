# Repository Guidelines

## Project Structure & Module Organization

```
/workspace/openai-responses
├── src/                 # Core implementation package (`openai_responses`)
├── tests/               # Unit test suite (pytest discovery, mirrors package layout)
├── pyproject.toml       # Dependency specification for `uv`
├── uv.lock              # Resolved dependency lockfile
├── flake.nix            # Optional Nix flake for reproducible builds
└── README.md            # Project overview and contribution guide
```

- **`src/`** contains the full source tree. All public code resides under the
  `openai_responses` package.
- **`tests/`** parallels the package layout; each module typically has a
  corresponding `test_*.py` file. Tests are executed with `pytest` via `uv`.
- There is no dedicated `docs/` or `.github/` directory in this minimal
  example, but those sections can be added if documentation or CI configuration
  becomes required in the future.

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

## Development & Testing

The repository uses **uv** as the Python dependency manager and the test runner.

### Installing Dependencies

Before running any code or tests you must sync the project dependencies. This ensures
all required packages from `pyproject.toml` and optional development extras are
available. The recommended command is:

```bash
uv sync
```

If you prefer to limit the installation to the runtime environment only, you can use:

```bash
uv sync --no-dev
```

The `uv sync` step installs both the runtime dependencies and the optional
`dev` dependencies (pytest, ruff, mypy, etc.) that are needed for running the tests.

### Running Tests

All unit tests are located under the `tests/` directory and follow the standard
`pytest` discovery pattern (`test_*.py`). To execute the test suite run:

```bash
uv run pytest
```

Running `uv run pytest` performs the following automatically:

1. Activates the virtual environment created by `uv`.
1. Executes the `pytest` command using the installed test dependencies.

If you need to run tests for a specific module or with additional options you can
pass the usual `pytest` flags. For example, to run only a single test file:

```bash
uv run pytest tests/test_foo.py
```

Or to increase verbosity:

```bash
uv run pytest -vv
```

### Running Linter and Formatter

Project code is formatted with `ruff`. To run the formatter locally:

```bash
uv run ruff format
```

To check for lint violations without making changes, run:

```bash
uv run ruff check
```

These tools help maintain consistency with the project's style guide.
