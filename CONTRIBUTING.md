# Contributing to MCTS-Agent

Thank you for your interest in contributing to **MCTS-Agent**! We welcome bug reports, feature suggestions, documentation enhancements, and code contributions.

---

## Code of Conduct

Please be respectful, collaborative, and constructive when engaging in this project. Treat fellow contributors with kindness.

---

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/mcts-agent.git
cd mcts-agent
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

---

## Running Tests

Tests run using Python's standard `unittest` framework or `pytest` with mock primitives enabled by default (`USE_MOCK_PRIMITIVES=true`):

```bash
# Run via unittest
python3 -m unittest discover tests

# Or run via pytest
pytest -v
```

All test cases in `tests/` execute hermetically and require no external API keys.

---

## Running the Agent Locally

### Mock Mode (Zero API Keys)
Verify that the full search and execution loop functions:
```bash
python3 main.py --mock
```

### Pure MCTS Tree Search
To run tree search with deep iterations:
```bash
python3 run_pure_mcts.py 10
```

### Live Mode
Set your environment variables in `.env`:
```bash
TYPESAFE_API_KEY=your_typesafe_api_key
```
And ensure your Gemini environment (or `agy` CLI) is available:
```bash
python3 main.py
```

---

## Pull Request Guidelines

1. **Create a branch**: Use a descriptive branch name (e.g. `feature/parallel-eval` or `fix/puct-fpu-calc`).
2. **Follow style guidelines**:
   - Write clean, type-annotated Python (`typing` / Python 3.10+ annotations).
   - Maintain docstrings for modules, classes, and public functions.
3. **Add tests**: Any new feature or bug fix must include tests in `tests/`.
4. **Ensure clean CI**: Verify that all tests pass locally before opening a pull request.
5. **Write clear commit messages**: Describe the *what* and *why* of the change.

---

## Questions & Discussions

Feel free to open an Issue on GitHub for any architecture questions, bug reports, or feature proposals.
