# Discriminative MCTS Agent

[![CI](https://github.com/luizh/mcts-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/luizh/mcts-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeSafe Jev Primitives](https://img.shields.io/badge/TypeSafe-Jev%20System%20One-purple.svg)](https://docs.typesafe.ai)

An autonomous reasoning and execution agent powered by **Discriminative Monte Carlo Tree Search (MCTS)**. The system couples **TypeSafe Jev System One Primitives** (`Noul`, `Choice`, `Score`) for lightning-fast, structured discriminative evaluations with **Gemini 3.8 Flash** for action generation in a closed-loop execution environment.

Includes an interactive **D3.js Tree Visualizer** to inspect and replay search rollouts, branch pruning, and value backpropagation frame-by-frame.

---

## Key Highlights

- **Fast Discriminative Pruning & Scoring**: Traditional MCTS with LLMs suffers from sluggish text generation and unpredictable parsing. This agent uses TypeSafe Jev System One models to evaluate validity, calculate priors, and score states directly in code with zero token parsing.
- **Single-Call Batched Pruning (`Noul`)**: Evaluates all candidate actions in parallel in a single `system_one()` call. No per-action network roundtrips, no context rot.
- **Dynamic Prime Branching (`Choice`)**: Dynamically samples expansion widths from prime numbers ($2, 3, 5, 7, 11, 13$) based on state uncertainty and goal complexity.
- **Closed-Loop Execution & Grounding**: Follows a strict cycle of Plan & Choose -> Execute -> Review -> Adapt -> Assess. Only the immediate winning action is executed; git diffs and terminal observations are folded back into context for subsequent decisions.
- **Interactive D3 Visualizer**: Load generated JSON search logs directly in `visualizer.html` to replay search trees, inspect PUCT scores, view batched Noul gates, and analyze state trajectories.

---

## Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Goal & Initial State   │
                          └────────────┬─────────────┘
                                       │
                ┌──────────────────────▼──────────────────────┐
                │ 1. PLAN & CHOOSE (Discriminative MCTS)       │
                │    ├── Selection: PUCT + First-Play Urgency │
                │    ├── Expansion: Gemini proposes actions   │
                │    ├── Prune: Batched Noul gate             │
                │    ├── Priors: Choice policy distribution   │
                │    ├── Value: Score rubric evaluation       │
                │    └── Backpropagation: Value sum & visits  │
                └──────────────────────┬──────────────────────┘
                                       │ Best Immediate Action
                ┌──────────────────────▼──────────────────────┐
                │ 2. EXECUTE                                  │
                │    Execute ONLY the selected immediate step │
                │    in the designated workspace.             │
                └──────────────────────┬──────────────────────┘
                                       │ Returncode, stdout, stderr
                ┌──────────────────────▼──────────────────────┐
                │ 3. REVIEW                                   │
                │    Inspect environment changes, git status, │
                │    new files, and terminal outcomes.        │
                └──────────────────────┬──────────────────────┘
                                       │ Observation
                ┌──────────────────────▼──────────────────────┐
                │ 4. ADAPT                                    │
                │    Ground real-world execution observation  │
                │    into updated state description.          │
                └──────────────────────┬──────────────────────┘
                                       │ Grounded State
                ┌──────────────────────▼──────────────────────┐
                │ 5. ASSESS (Early Stopping)                  │
                │    Noul evaluates: Is the goal complete?    │
                │    - Yes (conf >= 0.85) -> Finish!          │
                │    - No  (conf < 0.85)  -> Next MCTS step   │
                └─────────────────────────────────────────────┘
```

---

## TypeSafe Jev System One Primitives

Large language models (LLMs) are System Two text-generators. When software needs fast, deterministic judgments, coercing an LLM into producing structured JSON and parsing strings introduces latency, format failures, and context rot.

**TypeSafe Jev** is a **System One** model built for instant, structured judgments consumed directly in software:

| Primitive | Role in MCTS | Mechanism & Benefit |
| :--- | :--- | :--- |
| **`Noul`** | **Batched Action Pruning** & **Early Stopping** | Evaluates boolean propositions ($0.0 - 1.0$). Evaluates *all* candidate actions in a single atomic API call without per-action roundtrips. Prunes invalid actions before tree insertion (threshold $\ge 0.8$). Also gates early task termination (threshold $\ge 0.85$). |
| **`Choice`** | **Action Priors** & **Dynamic Branching** | Returns normalized probability distributions across discrete candidates. Assigns initial policy prior probabilities $P(s, a)$ used in the PUCT formula. Also selects optimal branching factor from primes ($2, 3, 5, 7, 11, 13$). |
| **`Score`** | **State Value Heuristic** | Evaluates states on a continuous 1–10 rubric. Replaces costly, random Monte Carlo rollouts with a fast discriminative heuristic value estimate $V(s)$. |

### PUCT Formula with First-Play Urgency (FPU)

Nodes are selected during tree traversal using the predictor upper confidence bound applied to trees (PUCT):

$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

- $Q(s, a)$ is normalized to $[0, 1]$ (divided by max score $10.0$) to maintain scale balance with the exploration term.
- **First-Play Urgency (FPU)**: Unvisited child nodes inherit their parent's estimated value rather than defaulting to 0, preventing starvation of unvisited siblings when one child discovers a high score early.

---

## Visualizer Replay Guide

The repository includes a standalone web visualizer [`visualizer.html`](visualizer.html) built with D3.js.

### How to Run the Visualizer

1. **Via the CLI tool**:
   ```bash
   mcts-agent visualize --port 8000
   ```
   This automatically serves the repository root over HTTP and launches `http://127.0.0.1:8000/visualizer.html` in your default browser.

2. **Or open directly**:
   ```bash
   # Option A: Open directly in your browser
   open visualizer.html     # macOS
   xdg-open visualizer.html # Linux

   # Option B: Run via standard http.server
   python3 -m http.server 8000
   # Then open http://localhost:8000/visualizer.html
   ```

2. **Load Search Logs**:
   - Click the **"📂 Load JSON Log"** button in the top navigation bar.
   - Select any run log from the `logs/` directory (e.g. `logs/mcts_20260915_235315_step1.json`).

3. **Explore the Search Tree**:
   - **Playback Controls**: Use Play ($\blacktriangleright$), Pause ($\mathbf{||}$), Step Forward ($\mathbf{>|}$), and Step Backward ($|\mathbf{<}$) to watch the tree expand iteration-by-iteration.
   - **Timeline Scrubber**: Drag the slider across search iterations.
   - **Node Details**: Click any node in the SVG canvas to view its accumulated visits, average score, prior probability, simulated state, and child branches in the right-hand inspection drawer.
   - **Pruning Inspector**: Review candidate actions filtered or kept by the batched Noul gate with confidence scores.

---

## Quickstart

### Prerequisites

- Python $\ge$ 3.10
- Git
- [TypeSafe API Key](https://typesafe.ai)
- Antigravity CLI (`agy`) or Gemini API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/luizh/mcts-agent.git
cd mcts-agent

# 2. Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode with CLI entry point
pip install -e .
```

### Environment Configuration

Create a `.env` file in the project root:

```ini
TYPESAFE_API_KEY=your_typesafe_api_key_here
# Optional model overrides:
AGY_MODEL=gemini-3.8-flash-medium
```

---

## Usage

The CLI provides a modern, Rich-stylized terminal experience with colored panels, search rollout progress, and step tables, along with full headless / machine-parseable support for AI agents.

### 1. Run Closed-Loop MCTS

Execute search and execution with a custom goal and workspace:

```bash
mcts-agent run --goal "Build a REST API" --workspace ./my-project --max-steps 5
```

### 2. Live Demo Run

Run against the built-in demo goal:

```bash
mcts-agent demo
```

### 3. Hermetic Mock Mode (Zero Keys Required)

Run the full closed-loop search loop with mock primitives (no network calls or API keys needed):

```bash
mcts-agent demo --mock
# or
mcts-agent run --mock --goal "Create a hello world app"
```

### 4. Interactive Human-in-the-Loop Mode

Interactively inspect MCTS action recommendations and approve, skip, or manually redirect states:

```bash
mcts-agent interactive
```

### 5. Launch the Visualizer Server

Start a local HTTP server serving the D3 visualizer:

```bash
mcts-agent visualize --port 8000
```

### 6. Backward Compatibility

You can continue invoking the CLI via `main.py` directly:

```bash
python main.py --mock --max-steps 3
```

---

## AI-Friendly & Headless Features

For automated pipelines, orchestrators, and AI agents, `mcts-agent` provides clean stdout modes:

### Machine-Parseable JSON Output (`--json`)
Emits a clean, valid JSON summary to `stdout` with zero ANSI escape codes, progress banners, or decorative styling:

```bash
mcts-agent run --mock --goal "Refactor user authentication" --json > result.json
```

Or pipe directly into tools like `jq`:
```bash
mcts-agent run --mock --json | jq '.steps[] | {step, action, score, visits}'
```

Example JSON response:
```json
{
  "goal": "Refactor user authentication",
  "completed": true,
  "completion_confidence": 0.92,
  "total_steps_run": 2,
  "steps": [
    {
      "step": 1,
      "action": "Analyze existing authentication handlers in auth/login.py",
      "score": 8.75,
      "visits": 10,
      "mcts_log": "logs/mcts_20260917_214342_step1.json",
      "execution": {
        "success": true,
        "stdout": "...",
        "returncode": 0
      }
    }
  ],
  "summary_file": "logs/run_20260917_214342_summary.json"
}
```

### Quiet and Headless Flags
- `--quiet` / `-q`: Suppresses informational banners and progress spinners while retaining essential results.
- `--no-color`: Disables ANSI coloring and sets `NO_COLOR=1` for clean plaintext logging.

---

## CLI Options Reference

### Commands
- `mcts-agent run [OPTIONS]`: Run closed-loop MCTS search with custom goal, state, and execution options.
- `mcts-agent demo [OPTIONS]`: Run pre-configured task management demo.
- `mcts-agent interactive [OPTIONS]`: Step-by-step human-in-the-loop search with manual approval.
- `mcts-agent visualize [OPTIONS]`: Launch local HTTP server for `visualizer.html`.

### Common Options (`run` & `demo`)

| Option | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--goal` | `-g` | *Demo goal* | Custom task goal string |
| `--state` | `-s` | *Demo state* | Initial state / context description |
| `--workspace` | `-w` | `.` | Target directory for workspace actions |
| `--max-steps` | | `5` | Maximum outer loop execution steps |
| `--iterations` | `-n` | `10` | MCTS search iterations per reasoning step |
| `--sim-depth` | | *Dynamic* | Fixed lookahead simulation depth (primes 2, 3, 5) |
| `--actions` | `-a` | *Dynamic* | Fixed candidate actions per node (primes $\le 13$) |
| `--mock` | | `False` | Run hermetically with mock primitives (no API keys required) |
| `--no-early-stop` | | `False` | Disable Noul-based completion early stopping |
| `--no-execute` | | `False` | Plan only; skip executing actions in the workspace |
| `--json` | | `False` | Output clean machine-parseable JSON summary to stdout |
| `--quiet` | `-q` | `False` | Suppress informational messages and progress banners |
| `--no-color` | | `False` | Disable ANSI color output |

### `visualize` Options

| Option | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--port` | `-p` | `8000` | Port for local HTTP visualizer server |
| `--host` | | `127.0.0.1` | Host interface to bind server to |
| `--browser` / `--no-browser` | | `True` | Automatically open default web browser |

---

## Project Structure

```
mcts-agent/
├── agent/
│   ├── __init__.py
│   ├── logger.py         # Structured JSON event emission for visualizer replay
│   ├── mcts.py           # Selection, Expansion, Simulation, Backpropagation loop
│   ├── node.py           # Search tree Node data structure & PUCT calculations
│   └── primitives.py     # TypeSafe Jev primitives (Noul, Choice, Score) wrappers
├── logs/                 # Search event logs & run summaries (replayable in visualizer)
├── prompts/
│   └── rubrics.txt       # Evaluation rubrics and reference criteria
├── tests/
│   ├── __init__.py
│   └── test_mcts.py      # Unit tests for primitives, PUCT, and search loop
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI matrix testing
├── .gitignore            # Clean git exclusion rules
├── CONTRIBUTING.md       # Contribution guide & development setup
├── LICENSE               # MIT License
├── pyproject.toml        # PEP 517/621 packaging metadata
├── requirements.txt      # Python dependencies
├── main.py               # Main CLI entrypoint for closed-loop execution
├── run_pure_mcts.py      # Standalone script for deep pure MCTS exploration
├── run_test.py           # Integration benchmark runner
└── visualizer.html       # Standalone D3.js interactive search tree visualizer
```

---

## Testing

Run unit tests across all components using Python's standard `unittest` or `pytest`:

```bash
# Using unittest
python3 -m unittest discover tests -v

# Or using pytest
pytest -v
```

All test cases execute hermetically in mock mode and verify:
- Node initialization, PUCT computation, and First-Play Urgency
- Batched validity gating (`Noul`)
- Prior probability distribution generation (`Choice`)
- Continuous state scoring (`Score`)
- MCTSLogger event emission and JSON serialization
- Multi-step closed-loop search execution

---

## License

This project is licensed under the [MIT License](LICENSE).
