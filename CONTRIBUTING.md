# Contributing to palette-auth

Thank you for your interest in contributing to `palette-auth`! We welcome contributions from the community.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/a1b2d56/palette-auth.git
   cd palette-auth
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install the package in editable mode with development & AI dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e ".[dev,ai]"
   ```

## Running Tests

Run the full pytest suite:
```bash
pytest tests/ -v
```

To run only the core tests (without PyTorch):
```bash
pytest tests/test_core.py tests/test_crypto.py tests/test_blocks.py tests/test_embed.py tests/test_cli.py -v
```

## Code Quality Standards

We use `ruff` for linting and code formatting, and `mypy` for static type verification:

```bash
# Check code style & lints
ruff check .

# Auto-format code
ruff format .

# Type checking
mypy palette_auth/
```

## Pull Request Guidelines

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Commit your changes with clear, imperative commit messages (e.g., `feat: add adaptive quantization support`).
3. Ensure all existing and newly added unit tests pass.
4. Push your branch to GitHub and submit a Pull Request targeting `main`.
