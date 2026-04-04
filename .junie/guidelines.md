# Project Guidelines: catan-bench

## Coding Conventions

- **Python Version**: Target Python 3.11+.
- **Type Hinting**: Mandatory for all new functions and classes. Use `from __future__ import annotations` to support postponed evaluation of annotations.
- **Style & Formatting**: Follow existing patterns. The project uses `ruff` for linting and formatting.
- **Naming**: 
  - Classes: PascalCase (e.g., `GameOrchestrator`, `LLMPlayer`).
  - Functions & Variables: snake_case (e.g., `choose_action`, `game_seed`).
  - Private methods/functions: Start with a single underscore (e.g., `_resolve_run_dir`).
- **Data Models**: Use `dataclasses` (often with `slots=True`) or simple classes with `to_dict` and `from_dict` methods for serialization (found in `schemas.py`).
- **Documentation**: Use triple-quoted docstrings for classes and public methods when they provide non-obvious context.

## Code Organization and Package Structure

The project follows a standard `src` layout:

- `src/catan_bench/`: Main package.
  - `orchestrator.py`: Contains `GameOrchestrator`, which manages the game loop, player turns, and state persistence.
  - `players.py`: Implementation of different player types (`LLMPlayer`, `ScriptedPlayer`, `RandomLegalPlayer`).
  - `engine.py`: Adapter layer for the underlying Catan engine (e.g., `catanatron`).
  - `schemas.py`: Centralized definitions for data structures (Actions, Events, Observations, etc.).
  - `prompting.py` & `prompts.py`: Logic for building LLM prompts and managing templates.
  - `templates/`: Jinja2 templates for LLM interactions.
  - `storage.py`: Utilities for reading/writing game artifacts (JSON, logs).
  - `llm.py`: Client abstractions for LLM providers.
- `tests/`: Contains test suites mimicking the package structure (e.g., `test_players.py`, `test_orchestrator.py`).
- `configs/`: Directory for game and benchmark configurations (TOML files).

## Unit and Integration Testing Approach

- **Framework**: Primarily uses `unittest`. `pytest` is used as the test runner.
- **Test Organization**: Tests are located in the `tests/` directory and prefixed with `test_`.
- **Mocking & Fakes**: 
  - Use fake LLM clients (like `FakeLLMClient` or `RawCompletionClient` in `tests/test_players.py`) to simulate model responses without making actual API calls.
  - Test prompt rendering by capturing the payload passed to the renderer (e.g., `CapturingRenderer`).
- **Granularity**:
  - **Unit Tests**: Focus on specific components like schema serialization, prompt rendering, or player decision logic.
  - **Integration Tests**: Verify the interaction between the `GameOrchestrator`, `Player` implementations, and the `EngineAdapter` (e.g., `test_config_and_runner.py`).
- **Verification**: Ensure that tests cover both successful paths and edge cases (e.g., invalid JSON from an LLM, illegal moves, retries).
