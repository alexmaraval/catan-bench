# catan-bench v0.3.0

`catan-bench` is a benchmark harness for running multi-player Settlers of Catan games with baseline bots and LLM players on top of a live `catanatron` engine.

Version `0.3.0` is a simplification pass. The benchmark now revolves around:

- one shared public history stream for the whole table,
- one two-slot `PlayerMemory` per player with `long_term` and `short_term`,
- one explicit active-turn loop with `start_turn`, `choose_action`, and `end_turn`,
- compact prompt-facing state derived from `catanatron`,
- prompt templates stored as package-local Jinja files,
- a single cursor-based dashboard that shows the board, public log, and every player panel at once.

See [plan.md](plan.md) for the refactor plan and [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md) for the longer research notes and benchmark background.

## Why this project exists

Settlers of Catan is a useful benchmark for LLM agents because it mixes long-horizon planning, hidden information, multi-player interaction, negotiation, and partial observability. The goal here is to measure decision-making under controlled observability, not who built the heaviest agent scaffolding.

## Core model

The runtime is organized around a small set of boundaries:

- `EngineAdapter`
  - exposes the current decision point,
  - exposes compact public state,
  - exposes player-private state,
  - applies validated actions to the live engine.
- `GameOrchestrator`
  - auto-rolls at the start of the active turn,
  - starts the player's short-term turn plan,
  - loops on action choice until the player stops or the turn is interrupted,
  - routes setup, robber, discard, and trade responses through a reactive path,
  - rewrites long-term memory at the end of the turn,
  - persists public events, public state snapshots, memory snapshots, and prompt traces.
- `Player`
  - `start_turn(observation) -> short_term`
  - `choose_action(observation) -> {action, short_term}`
  - `end_turn(observation) -> long_term`
  - `respond_reactive(observation) -> action`
  - optional trade-chat helpers for open / reply / owner decision.
- `Storage`
  - `public_history.jsonl`
  - `public_state_trace.jsonl`
  - `players/<id>/memory.json`
  - `players/<id>/memory_trace.jsonl`
  - `players/<id>/prompt_trace.jsonl`

The turn model is intentionally simple:

- `long_term` survives across turns and is overwritten once at turn end.
- `short_term` is written at turn start, can be revised after each action, and is cleared when the turn ends.
- Reactive off-turn decisions can read `long_term`, but they do not persist `short_term`.
- Public history is the only shared transcript. There is no private history stream anymore.

## What ships in 0.3.0

- a live `CatanatronEngineAdapter` for the current GitHub version of `catanatron`,
- baseline `RandomLegalPlayer`, `FirstLegalPlayer`, and `ScriptedPlayer` implementations,
- an `LLMPlayer` backed by an OpenAI-compatible chat client,
- compact observation builders for turn-start, action, turn-end, reactive, and trade-chat stages,
- Jinja prompt templates under `src/catan_bench/templates/`,
- append-only storage for public events, public state snapshots, memory snapshots, and prompt traces,
- a Streamlit dashboard with one global history cursor and concurrent player panels,
- a simple public replay exporter built from the public event log.

The adapter targets the current GitHub version of `catanatron`, not the older PyPI release, because the GitHub version includes the domestic trade flow needed for this benchmark.

## Installation

Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the trade-capable `catanatron` engine from GitHub:

```bash
pip install git+https://github.com/bcollazo/catanatron.git
```

If you use `uv`, the repo already includes a lockfile:

```bash
uv sync --group dev --group dashboard
```

## Quick start

1. Put your API key in the environment or in a nearby `.env` file:

```bash
export OPENAI_API_KEY=<your_api_key>
```

An example file lives at [`.env.example`](.env.example). The CLI looks for `.env` starting from the players config directory and walks up to four parent directories.

2. Run a benchmark game:

```bash
python -m catan_bench --game configs/game.toml --players configs/openai-players.toml
```

You can also override the run path from the CLI with one flag:

```bash
python -m catan_bench --game configs/game.toml --players configs/openai-players.toml --run-dir runs/0.3.0/dev/
```

3. The configured `run_dir` is treated as a base directory. Each execution creates a timestamped child directory such as:

```text
runs/0.3.0/dev/<game-id>-<timestamp>-<token>/
```

The resolved run directory is written into both `metadata.json` and `result.json`.

If `--run-dir` points to an existing run directory, the CLI resumes that run in place instead of creating a new child directory:

```bash
python -m catan_bench --game configs/game.toml --players configs/openai-players.toml --run-dir runs/0.3.0/dev/<run-name>
```

4. Export a simple public replay page for a completed run:

```bash
python -m catan_bench.replay runs/0.3.0/dev/<run-name>
```

This writes `replay.html` inside that run directory.

5. Open the live dashboard against the same run:

```bash
streamlit run dashboard.py -- --run-dir runs/0.3.0/dev/<run-name>
```

The dashboard reads the live artifacts directly, so it can stay open while a game is still running.

## Config files

The repository includes sample config files under [`configs/`](configs/):

- [`configs/game.toml`](configs/game.toml)
- [`configs/openai-players.toml`](configs/openai-players.toml)
- [`configs/groq-players.toml`](configs/groq-players.toml)
- [`configs/local-players.toml`](configs/local-players.toml)
- [`configs/mixed-players.toml`](configs/mixed-players.toml)

Game config example:

```toml
[game]
engine = "catanatron"
seed = 12
discard_limit = 7
vps_to_win = 10
history_window = 100
trading_chat_enabled = true
trading_chat_max_failed_attempts_per_turn = 5
trading_chat_max_rooms_per_turn = 5
trading_chat_max_rounds_per_attempt = 3
run_dir = "runs/0.3.0/dev/"
```

Player config example:

```toml
[[players]]
id = "RED"
type = "llm"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
temperature = 0.2
prompt_history_limit = 12
```

Supported player types:

- `random`
- `first_legal`
- `llm`

`llm` players can point to any OpenAI-compatible chat endpoint by setting `api_base` and `api_key_env`. The bundled sample configs show OpenAI, Groq, Together, and local Ollama-style endpoints.

## Run artifacts

Each run directory contains a small set of canonical artifacts:

- `metadata.json`
- `public_history.jsonl`
- `public_state_trace.jsonl`
- `players/<id>/memory.json`
- `players/<id>/memory_trace.jsonl`
- `players/<id>/prompt_trace.jsonl`
- `result.json`

These files are keyed around the shared `history_index`, which lets the dashboard move a single cursor through the public timeline while showing each player's latest memory and prompt activity at that point.

## Dashboard and replay

The dashboard is designed for monitoring and debugging live runs:

- one global history cursor,
- the selected public board and trade state,
- the public log around the cursor,
- one panel per player shown concurrently,
- prompt trace inspection for the selected history window.

The replay exporter is intentionally simpler. It turns `public_history.jsonl` into a lightweight static HTML timeline for quick inspection and sharing.

## Prompt templates

Prompt text lives in package-local Jinja templates under [`src/catan_bench/templates/`](src/catan_bench/templates/). The current template set covers:

- turn start,
- action choice,
- turn end,
- reactive decisions,
- trade chat open / reply / owner decision,
- shared rules and response-contract partials.

This keeps prompt editing out of Python source and makes it easier to iterate on stage-specific instructions.

## Development

Run the test suite with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The `catanatron` adapter tests are skipped automatically when `catanatron` is not installed in the environment.

Useful entry points:

- `python -m catan_bench`
- `python -m catan_bench.replay`
- `streamlit run dashboard.py -- --run-dir <run-dir>`

Useful source files:

- [`src/catan_bench/orchestrator.py`](src/catan_bench/orchestrator.py)
- [`src/catan_bench/players.py`](src/catan_bench/players.py)
- [`src/catan_bench/observations.py`](src/catan_bench/observations.py)
- [`src/catan_bench/storage.py`](src/catan_bench/storage.py)
- [`src/catan_bench/prompting.py`](src/catan_bench/prompting.py)
- [`src/catan_bench/dashboard.py`](src/catan_bench/dashboard.py)

## License

This repository is licensed under the terms in [LICENSE](LICENSE).

If this project integrates tightly with `catanatron`, its GPL-3.0 license implications should be reviewed before distribution.
