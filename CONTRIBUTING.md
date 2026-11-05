# Contributing to StumpVision AI

Thank you for your interest in contributing to StumpVision AI!

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/Stumpeye-project.git
   cd Stumpeye-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Run the test suite:
   ```bash
   pytest -q
   ```

## Pull Request Guidelines

- Ensure all 104 unit & integration tests pass before opening a PR.
- Follow PEP 8 style standards.
- Write meaningful commit messages following the Conventional Commits specification (`feat:`, `fix:`, `docs:`, `test:`).

