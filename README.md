# catan-bench

`catan-bench` runs multiplayer Settlers of Catan benchmarks for OpenAI-compatible LLM players on top of [`catanatron`](https://github.com/bcollazo/catanatron).

It includes player-scoped observations, structured run artifacts, replay and analysis commands, and a Streamlit dashboard.

## Install

```bash
uv sync --group dev --group dashboard
pip install git+https://github.com/bcollazo/catanatron.git
```

## Run

Set your API key:

```bash
export OPENAI_API_KEY=<your_api_key>
```

An example file lives at [`.env.example`](.env.example).

Run a game:

```bash
python -m catan_bench --game configs/game.toml --players configs/openai-players.toml
```

Open the dashboard:

```bash
streamlit run dashboard.py -- --run-dir runs
```

Useful follow-up commands:

```bash
python -m catan_bench.replay runs/<run-dir>
python -m catan_bench.analysis runs
python -m catan_bench.benchmark runs
```

## Config

The repo keeps two example configs:

- [`configs/game.toml`](configs/game.toml)
- [`configs/openai-players.toml`](configs/openai-players.toml)

To target another OpenAI-compatible provider or a local server, copy `configs/openai-players.toml` and change `api_base`, `api_key_env`, and `model`.

## Development

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest tests
```

## Roadmap

- Strengthen baselines and tournament workflows.
- Add VLM-backed players so models can see the board.
- Extend the benchmark with communication and continual-learning tracks.

More design notes live in [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md).

## License

This project is licensed under [LICENSE](LICENSE). `catanatron` has its own license terms, so review those as well before redistribution.
