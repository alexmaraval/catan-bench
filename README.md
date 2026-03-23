# catan-bench v0.2.0

`catan-bench` is an early harness for benchmarking LLM players in multi-player Settlers of Catan.

Version `0.2.0` adds LLM player support on top of the benchmark core:

- a game orchestrator that owns full engine state,
- player-scoped observations with public and private information,
- append-only public/private logs,
- per-player private memory writes,
- rule/instruction bundles suitable for LLM prompting,
- action validation before the engine applies a move,
- a live `catanatron` engine adapter,
- an `LLMPlayer` backed by an OpenAI-compatible chat client,
- prompt trace recording for every LLM decision,
- static HTML replay export from completed run artifacts.

The adapter targets the current GitHub version of `catanatron`, not the older PyPI release, because the GitHub version includes the domestic trade flow needed for this benchmark.

The previous long-form research summary and roadmap were moved to [docs/PROJECT_NOTES.md](/Users/alexandremaraval/Documents/Projects/catan-bench/docs/PROJECT_NOTES.md).

## Why this project exists

Settlers of Catan is a good benchmark candidate for LLM agents because it combines:

- long-horizon planning,
- hidden information,
- multi-player interaction,
- negotiation and trading,
- partial observability.

The design goal is to measure **decision-making under controlled observability**, not who built the heaviest agent scaffolding.

## Architecture

The harness is centered on four boundaries:

1. `EngineAdapter`
   - exposes the current decision point,
   - exposes public state,
   - exposes player-private state,
   - applies validated actions.

2. `GameOrchestrator`
   - identifies the next acting player,
   - builds the observation,
   - calls the player adapter,
   - validates the action,
   - persists logs and memory writes.

3. `Player`
   - receives an `Observation`,
   - receives game rules, public history, private history, and decision prompt context,
   - returns `{action, private_reasoning, private_memory_write}`.

4. `Storage`
   - `public_history.jsonl`
   - `players/<id>/private_history.jsonl`
   - `players/<id>/memory.jsonl`
   - `players/<id>/prompt_trace.jsonl`

Each player's `private_history.jsonl` now also captures their own prior decisions, including the chosen action and their fuller private reasoning for that turn. The long-term `memory.jsonl` remains a distilled summary store rather than a dump of verbose reasoning.
For LLM-backed players, `prompt_trace.jsonl` records the exact prompt messages and JSON response for each decision, including any repair attempt after an illegal move.

The harness is modeled around **decision points**, not just turns, so it can support trade responses, robber choices, initial placements, and other non-standard turn interactions.

## What is implemented in v0.2.0

- Core dataclasses for actions, events, observations, decisions, memory entries, and game results.
- A protocol-based engine interface.
- An observation builder that combines engine state with stored history.
- A prompt/rules bundle that explains the game to LLM-backed players.
- Append-only event and memory stores with optional JSONL persistence.
- A game orchestrator with legal-action checking.
- A small scripted player adapter for tests and demos.
- An `LLMPlayer` plus an OpenAI-compatible chat client adapter.
- A live `CatanatronEngineAdapter` with public/private state extraction.
- Support for dynamic domestic trade offers via adapter-side validation of `OFFER_TRADE`.
- Semantic public event logs for replay-friendly game interactions.
- A static replay exporter that renders a public `replay.html` plus per-player personal replay pages.
- Post-game trade metadata in completed results.
- Unit tests covering both the mock harness flow and a real `catanatron` trade flow.

## Repository layout

- [src/catan_bench/schemas.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/schemas.py)
- [src/catan_bench/engine.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/engine.py)
- [src/catan_bench/observations.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/observations.py)
- [src/catan_bench/orchestrator.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/orchestrator.py)
- [src/catan_bench/storage.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/storage.py)
- [src/catan_bench/players.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/players.py)
- [src/catan_bench/config.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/config.py)
- [src/catan_bench/runner.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/runner.py)
- [src/catan_bench/catanatron_adapter.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/catanatron_adapter.py)
- [src/catan_bench/replay.py](/Users/alexandremaraval/Documents/Projects/catan-bench/src/catan_bench/replay.py)
- [configs/game.toml](/Users/alexandremaraval/Documents/Projects/catan-bench/configs/game.toml)
- [configs/players.toml](/Users/alexandremaraval/Documents/Projects/catan-bench/configs/players.toml)
- [tests/test_catanatron_adapter.py](/Users/alexandremaraval/Documents/Projects/catan-bench/tests/test_catanatron_adapter.py)
- [tests/test_config_and_runner.py](/Users/alexandremaraval/Documents/Projects/catan-bench/tests/test_config_and_runner.py)
- [tests/test_orchestrator.py](/Users/alexandremaraval/Documents/Projects/catan-bench/tests/test_orchestrator.py)
- [tests/test_replay.py](/Users/alexandremaraval/Documents/Projects/catan-bench/tests/test_replay.py)

## Quick start

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

Run a benchmark game from the provided TOML configs:

```bash
./.venv/bin/python -m catan_bench --game configs/game.toml --players configs/players.toml
```

The CLI auto-loads a `.env` file by searching from the players config directory upward for up to four parent directories, so you can keep credentials in a project-local `.env`:

```bash
OPENAI_API_KEY=<your_api_key>
```

An example file is included at [`.env.example`](/Users/alexandremaraval/Documents/Projects/catan-bench/.env.example). Keep the real `.env` untracked.

This writes run artifacts to the `run_dir` defined in [configs/game.toml](/Users/alexandremaraval/Documents/Projects/catan-bench/configs/game.toml), which currently points to `runs/quickstart`.

Generate a replay page from that run directory:

```bash
./.venv/bin/python -m catan_bench.replay runs/quickstart
```

This creates:

- `runs/quickstart/replay.html`
- `runs/quickstart/players/RED/replay.html`
- `runs/quickstart/players/BLUE/replay.html`
- `runs/quickstart/players/ORANGE/replay.html`
- `runs/quickstart/players/WHITE/replay.html`

Open `runs/quickstart/replay.html` in a browser to inspect the shared public transcript as a long chat-style timeline with:

- one color-coded bubble per player,
- special bubbles for major game events like trades and dice rolls,
- turn/phase/decision metadata on each entry,
- expandable raw event payloads for debugging.

That public page now links to each player's personal replay page.

Each player-specific page combines:

- the shared public transcript,
- that player's `private_history.jsonl`,
- that player's `memory.jsonl`.

The personal replay pages also include simple filters so you can isolate:

- `Public`
- `Private`
- `Memory`

The replay exporter reads:

- `metadata.json`
- `public_history.jsonl`
- `result.json`
- `players/<id>/private_history.jsonl`
- `players/<id>/memory.jsonl`

Run the current test suite:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Minimal example:

```python
from catan_bench import CatanatronEngineAdapter, GameOrchestrator

adapter = CatanatronEngineAdapter(seed=1)
decision = adapter.current_decision()
print(decision.phase, decision.acting_player_id)
print(decision.legal_actions[:3])
```

Minimal replay export example:

```python
from catan_bench import export_player_replay_html, export_replay_html

public_path = export_replay_html("runs/quickstart")
red_private_path = export_player_replay_html("runs/quickstart", "RED")
print(public_path)
print(red_private_path)
```

## Config files

The repo now includes TOML config files for both the engine and the player table.

Game config example:

```toml
[game]
engine = "catanatron"
seed = 7
discard_limit = 7
vps_to_win = 6
history_window = 40
run_dir = "runs/quickstart"
```

Player config example:

```toml
[[players]]
id = "RED"
type = "llm"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
temperature = 0.2
```

Supported player types right now:

- `random`
- `first_legal`
- `llm`

For an OpenAI-compatible endpoint, set the API key in the configured environment variable and optionally override `api_base`.
For the sample config in [configs/players.toml](/Users/alexandremaraval/Documents/Projects/catan-bench/configs/players.toml), that means either exporting:

```bash
export OPENAI_API_KEY=<your_api_key>
```

or placing the same key in a nearby `.env` file.

The built-in LLM player sends:

- a compact rules summary,
- the public game state,
- player-private state,
- recent public history,
- recent private history,
- prior private memory,
- the legal action set.

It expects strict JSON back with:

- `action_index`
- `action`
- `private_reasoning`
- `private_memory_write`

## Status

What works now:

- harness orchestration,
- player-scoped observation construction,
- public/private log persistence,
- private memory persistence,
- LLM-ready rules and history context in observations,
- action validation,
- OpenAI-compatible LLM players,
- a live `catanatron` adapter,
- semantic public event logs for replay/export,
- static HTML public and per-player replay generation from completed runs,
- real domestic trade progression through `OFFER_TRADE`, `ACCEPT_TRADE`, and `REJECT_TRADE`,
- trade-like benchmark flow in tests,
- post-game trade metadata in result payloads.

What is next:

- improve observation redaction and presentation for LLM prompting,
- improve replay polish and summarization beyond the current transcript-style views,
- add richer evaluation metrics,
- add tournament runners and seat-rotation evaluation.
- support richer structured negotiation interfaces on top of domestic trade.
- harden packaging so the trade-capable `catanatron` dependency is easier to install.

## Design principles

- The engine owns the full game state.
- Players only see information they are allowed to know.
- Public history is shared.
- Private history and memory are player-scoped.
- Memory is authored by the player but persisted by the harness.
- The benchmark should not depend on hidden chain-of-thought.

## License

This repository is licensed under the terms in [LICENSE](/Users/alexandremaraval/Documents/Projects/catan-bench/LICENSE).

If this project integrates tightly with `catanatron`, its GPL-3.0 license implications should be reviewed before distribution.
