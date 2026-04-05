# catan-bench

[![CI](https://github.com/alexmaraval/catan-bench/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alexmaraval/catan-bench/actions/workflows/ci.yml)

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

## Elo Snapshot

Static snapshot from `runs/elo-games/1.0.0`, computed on April 5, 2026 from the same run artifacts used by the dashboard. This schedule contains 6 completed games across 8 models, with each model appearing in 3 games. `ELO` is sequential and order-sensitive; `BT` is an order-invariant Bradley-Terry batch fit from the same aggregate pairwise outcomes.

| Rank | Model | ELO | BT | Games | Wins | Win Rate | Avg VP | Pairwise W-L-D |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `deepseek/deepseek-v3.2` | 1592 | 1647 | 3 | 2 | 66.67% | 8.7 | 8-1-0 |
| 2 | `nvidia/nemotron-3-super-120b-a12b:free` | 1564 | 1578 | 3 | 1 | 33.33% | 7.7 | 6-2-1 |
| 3 | `x-ai/grok-4.1-fast` | 1509 | 1506 | 3 | 1 | 33.33% | 6.0 | 4-4-1 |
| 4 | `minimax/minimax-m2.7` | 1492 | 1485 | 3 | 0 | 0.00% | 7.3 | 4-4-1 |
| 5 | `z-ai/glm-5-turbo` | 1489 | 1478 | 3 | 1 | 33.33% | 7.0 | 4-5-0 |
| 6 | `xiaomi/mimo-v2-flash` | 1476 | 1471 | 3 | 0 | 0.00% | 6.0 | 3-5-1 |
| 7 | `arcee-ai/trinity-large-preview:free` | 1462 | 1462 | 3 | 1 | 33.33% | 7.0 | 3-5-1 |
| 8 | `qwen/qwen3-235b-a22b-2507` | 1418 | 1373 | 3 | 0 | 0.00% | 5.7 | 1-7-1 |

Game winners:

- `game-01` (seed 12): `arcee-ai/trinity-large-preview:free`
- `game-02` (seed 24): `deepseek/deepseek-v3.2`
- `game-03` (seed 42): `nvidia/nemotron-3-super-120b-a12b:free`
- `game-04` (seed 123): `z-ai/glm-5-turbo`
- `game-05` (seed 777): `x-ai/grok-4.1-fast`
- `game-06` (seed 42069): `deepseek/deepseek-v3.2`

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

## License

This project is licensed under [LICENSE](LICENSE). `catanatron` has its own license terms, so review those as well before redistribution.
