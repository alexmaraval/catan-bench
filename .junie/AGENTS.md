# Development Information for Catan Bench

This document provides project-specific information for developers working on `catan-bench`.

## Build and Configuration

The project uses `uv` for dependency management.

### Prerequisites
- Python 3.11 or later.
- `uv` installed.

### Setup
Install dependencies and sync the environment:
```bash
uv sync
```

### Environment Variables
LLM players require API keys. You can provide them in a `.env` file at the project root or in parent directories (the runner searches up to 4 levels).
Common keys:
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `ANTHROPIC_API_KEY`

## Running the Benchmark

You can run the benchmark using the `catan_bench` module:

```bash
uv run python -m catan_bench --game configs/game.toml --players configs/openai-players.toml
```

### CLI Arguments
- `--game`: Path to the game TOML configuration (engine, victory points, etc.).
- `--players`: Path to the players TOML configuration (LLM models, API bases, types).
- `--run-dir`: Base directory for storing run artifacts (logs, histories, traces).
- `--resume-run`: Resume an interrupted game.
- `--debug`: Enable interactive debugging (pauses for each player decision).
- `--analyze`: Print a summary analysis after the game finishes.

## Testing Information

### Running Tests
Tests are located in the `tests/` directory and use `pytest` (or `unittest`).

Run all tests:
```bash
uv run pytest
```

Run a specific test file:
```bash
uv run pytest tests/test_config_and_runner.py
```

### Adding New Tests
Follow the existing pattern of using `unittest.TestCase` or `pytest` functions. 

Example of a simple smoke test (`tests/test_simple_smoke.py`):
```python
import unittest
from pathlib import Path
from catan_bench.config import load_game_config

class SmokeTest(unittest.TestCase):
    def test_load_basic_config(self) -> None:
        """Verify that the basic game.toml config can be loaded."""
        config_path = Path("configs/game.toml")
        config = load_game_config(config_path)
        assert config.engine == "catanatron"
```

## Additional Development Information

### Code Style
- Use `ruff` for linting and formatting. Configuration is in `pyproject.toml`.
- Python 3.11+ type annotations are required.
- Standard project layout: source in `src/catan_bench`, tests in `tests/`.

### Key Components
- **Orchestrator (`orchestrator.py`)**: Manages the game loop, player interactions, and state reporting.
- **Engine (`engine.py`)**: Interfaces with the Catan game engine (currently `catanatron`).
- **LLM (`llm.py`, `players.py`)**: Handles LLM API calls and prompt generation.
- **Templates (`templates/`)**: Jinja2 templates for constructing LLM prompts.

### Debugging Runs
Game artifacts are stored in `runs/`. Each run includes:
- `metadata.json`: Configuration and run details.
- `checkpoint.json`: Snapshot of the game state for resuming.
- `public_history.jsonl`: Trace of game events.
- `action_trace.jsonl`: Detailed log of player decisions and prompts.
