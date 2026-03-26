# catan-bench

`catan-bench` is a benchmark harness for evaluating LLM agents in multi-player Settlers of Catan.

It runs real games on top of a live [`catanatron`](https://github.com/bcollazo/catanatron) engine, stores structured artifacts for every run, and ships with a dashboard for replay, debugging, and post-game analysis.

## Why Catan?

Catan is a useful benchmark for language agents because it combines:

- long-horizon planning
- negotiation and trade
- hidden information
- shifting incentives between opponents
- tactical reactivity under partial observability

The goal of this project is to study decision-making quality under controlled observability, not to build the heaviest agent scaffolding possible.

## What the project includes

- a live adapter around the current GitHub version of `catanatron`
- baseline bot players plus OpenAI-compatible LLM players
- compact prompt-facing observations for opening, turn start, action choice, reactive decisions, trade chat, and turn end
- append-only run artifacts for public history, state snapshots, memories, and prompt traces
- a Streamlit dashboard for live monitoring and replay
- post-game analysis for trade behavior, strategy evolution, victory progression, and other player metrics

## Installation

Create an environment and install the project:

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

To enable local pre-commit hooks:

```bash
uv run pre-commit install
```

## Quick start

1. Put your provider API key in the environment or in a nearby `.env` file:

```bash
export OPENAI_API_KEY=<your_api_key>
```

An example file lives at [`.env.example`](.env.example).

2. Run a game:

```bash
python -m catan_bench --game configs/game.toml --players configs/openai-players.toml
```

3. Open the dashboard for a run:

```bash
streamlit run dashboard.py -- --run-dir runs
```

4. Export a static replay page:

```bash
python -m catan_bench.replay runs/<run-name>
```

5. Generate post-game analysis:

```bash
python -m catan_bench.analysis runs
```

6. Generate cross-run ELO rankings and rubric scores:

```bash
python -m catan_bench.benchmark runs
```

## Configuration

The repository includes sample configs under [`configs/`](configs/):

- [`configs/game.toml`](configs/game.toml)
- [`configs/openai-players.toml`](configs/openai-players.toml)
- [`configs/openrouter-players.toml`](configs/openrouter-players.toml)
- [`configs/groq-players.toml`](configs/groq-players.toml)
- [`configs/togetherai-players.toml`](configs/togetherai-players.toml)
- [`configs/local-players.toml`](configs/local-players.toml)
- [`configs/mixed-players.toml`](configs/mixed-players.toml)

Minimal game config:

```toml
[game]
engine = "catanatron"
vps_to_win = 10
history_window = 100
prompt_history_limit = 12
public_chat_enabled = true
trading_chat_enabled = true
run_dir = "runs/"
run_tags = ["0.4.0", "dev"]
```

New runs are created directly under `runs/`, with tags prefixed into the run directory name. For example:

```text
runs/0.4.0-dev-<game-id>-<timestamp>-<token>/
```

Minimal player config:

```toml
[[players]]
id = "RED"
type = "llm"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
temperature = 0.2
```

Supported player types:

- `llm`
- `random`
- `first_legal`

LLM players use an OpenAI-compatible chat interface, so the same harness can target different providers by changing `api_base`, `model`, and `api_key_env`.

## Run artifacts

Each run directory contains a compact set of artifacts such as:

- `metadata.json`
- `public_history.jsonl`
- `public_state_trace.jsonl`
- `players/<id>/memory.json`
- `players/<id>/memory_trace.jsonl`
- `players/<id>/prompt_trace.jsonl`
- `result.json`
- optionally `analysis.json`

These artifacts are designed to support replay, debugging, and cross-run analysis without needing access to hidden engine internals after the run completes.

## Dashboard

The dashboard is meant for both live monitoring and post-hoc inspection. It includes:

- a board view and public state cursor
- player summaries and turn timelines
- prompt and memory inspection
- replay tooling
- post-game analysis views for strategy, trading, roads, army, and market structure

## Development

Run the main checks with:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest tests
```

Useful entry points:

- `python -m catan_bench`
- `python -m catan_bench.analysis <run-dir>`
- `python -m catan_bench.benchmark <run-dir-or-runs-base>`
- `python -m catan_bench.replay <run-dir>`
- `python -m catan_bench.cleanup_runs runs`
- `streamlit run dashboard.py -- --run-dir <run-dir>`

For more background, see [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md) and [plan.md](plan.md).

## License

This repository is licensed under the terms in [LICENSE](LICENSE).

`catan-bench` integrates with `catanatron`; if you plan to distribute derived work, review the license implications of that dependency as well.
