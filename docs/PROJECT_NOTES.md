# Catan LLM Benchmark: Thoughts and Roadmap

## What the paper does

Paper: *Agents of Change: Self-Evolving LLM Agents for Strategic Planning*  
arXiv: 2506.04651  
Submitted: June 5, 2025  
Revised: October 13, 2025

The paper argues that prompt-only LLM agents are weak at long-horizon strategy games because they keep re-parsing a large evolving state every turn. Their solution is **HexMachina**, a continual-learning multi-agent system that:

1. learns the environment interface,
2. writes adapter code,
3. writes and rewrites a compiled player policy,
4. evaluates that policy through repeated simulation.

So the LLM is not mainly acting as a live per-turn player. It is acting more like a strategy engineer that keeps refining code artifacts.

The environment they use is **Catanatron**, an open-source Python implementation of Settlers of Catan. Their strongest baseline is Catanatron's built-in **AlphaBeta** bot.

## Main findings from the paper

- HexMachina beats prompt-centric baselines by moving work out of the prompt and into reusable code artifacts.
- The "discovery first, improvement second" split matters a lot. Trying to learn the API and strategy at the same time performs much worse.
- Leaner agent orchestration worked better than a larger cast of specialist agents.
- In their reported comparison table, the best HexMachina setup reaches **54.1% win rate** and **8.2 +/- 0.1 victory points**, versus **51.0%** and **7.8 +/- 0.2** for AlphaBeta.
- Their prompt-centric LLM baseline is much weaker: **16.4% win rate** and **5.2 +/- 1.2 victory points**.

## Why the paper is useful for your project

The paper is useful as:

- evidence that Catan is a good long-horizon benchmark,
- evidence that raw per-turn prompting struggles,
- evidence that Catanatron is fast enough for large-scale evaluation,
- a warning that interface design matters as much as the model.

## Why it is not the benchmark template I would copy

For your goal, I would treat the paper as **motivation**, not as the design to reproduce.

Main reasons:

- Their experiments are run in **controlled 2-player games**, even though classic Catan is fundamentally a **3-4 player** game.
- Their core contribution is a **self-evolving artifact-centric system**, not a clean benchmark for simple LLM players.
- Their comparison is heavily shaped by their own scaffolding, memory system, and code-writing loop.
- Their prompt-only baseline appears expensive and small-scale, so it is useful directionally but not enough as the final word on simple LLM play.
- Their evaluation metrics are fairly coarse: mostly win rate and final victory points.

In short: they are answering "can LLM systems self-improve into strong Catan code?"  
You want to answer something closer to "how good are relatively simple LLM players, with controlled memory and observability, in real multi-player Catan?"

That is a better benchmark question in my opinion.

## My recommendation on benchmark philosophy

I would build a **stripped-down benchmark** around these rules:

- The engine owns the true game state.
- Each player receives only the information that player is allowed to know.
- All players share a structured **public history**.
- Each player has its own **private game memory** for what it has seen and inferred.
- If you want a private reasoning channel, store it as a **player-private scratchpad or memory write**, not as a benchmark requirement that depends on hidden chain-of-thought availability.
- Memory should reset between games by default.
- Do not let agents directly edit code or tools during the main benchmark.

This keeps the benchmark centered on **decision-making**, not on who built the best agent scaffolding.

## Recommended state model

I would explicitly separate four layers:

### 1. Referee state

This is the full internal engine state. Only the benchmark harness can see it.

Contents:

- full board state,
- all private hands,
- development cards,
- random seed state,
- legality checks,
- full action outcomes.

### 2. Public history

This is shared with all players and becomes the canonical game transcript.

Examples:

- initial board and seating,
- dice rolls,
- builds and upgrades,
- robber moves,
- visible steals,
- public trade offers and acceptances,
- longest road / largest army changes,
- visible victory point totals,
- turn boundaries,
- game end.

### 3. Private history per player

This is the player-scoped observable history.

Examples:

- own hand changes,
- own development cards,
- private outcomes only that player can know exactly,
- player-authored memory notes,
- private beliefs or summaries if you want to support inference-tracking.

### 4. Private reasoning log per player

If you want this, keep it fully private and non-authoritative.

Important design note:

- For cross-provider comparability, the benchmark should not require raw hidden reasoning.
- A better contract is: the player may optionally emit a `memory_write` or `private_note` after each turn, and only that note is persisted.

That gives you a "private history" without tying the benchmark to inaccessible internal traces.

## Recommended player API

I would not expose raw `catanatron.Game` to LLM players.

Instead, the harness should give each player a structured observation like:

```json
{
  "game_id": "uuid",
  "player_id": "RED",
  "turn_index": 17,
  "phase": "main_turn",
  "public_state": {
    "board": "...",
    "robber": "...",
    "visible_scores": "...",
    "longest_road": "...",
    "largest_army": "...",
    "public_trade_state": "..."
  },
  "private_state": {
    "resources": "...",
    "dev_cards": "...",
    "ports": "...",
    "available_buildings": "..."
  },
  "recent_public_events": ["..."],
  "recent_private_events": ["..."],
  "legal_actions": ["canonical action objects here"],
  "memory": "player-private memory summary"
}
```

And the player returns something like:

```json
{
  "action": "...canonical action...",
  "memory_write": "optional private update",
  "summary": "optional short rationale"
}
```

That is enough for a benchmark, easy to log, and much cleaner than giving the model direct engine access.

## Recommended orchestrator boundary

The first implementation step should be a **game harness / orchestrator**, not an MCP server.

The harness should:

- own the full engine state,
- build player-scoped observations,
- distribute only partial information to each player,
- trigger interactions at every decision point,
- persist the public log and each player's private history,
- let each player return an `action` plus optional `memory_write`,
- validate and apply the action back into the engine.

One important implementation detail:

- players should be responsible for **authoring** their own memory,
- the harness should be responsible for **storing** that memory.

That gives you reproducibility and keeps the benchmark logic outside the model adapter.

I would also model the orchestration loop around **decision points**, not only turns. In Catan, the next acting player is not always simply "the player whose turn it is". You also need to handle:

- initial settlement placement,
- initial road placement,
- robber discards,
- robber movement and steal selection,
- trade offers,
- trade responses,
- trade confirmations.

## My review of Catanatron as the game engine

## Verdict

**Yes, I think Catanatron is a good choice for the game engine, with one major condition: wrap it behind a benchmark interface that enforces partial observability.**

## Why I like it

- It is a real Python game engine, not just a UI.
- It is designed for automated simulation at scale.
- It supports custom players with a simple `decide(game, playable_actions)` API.
- It exposes step-wise gameplay through `play_tick()`, which is ideal for benchmark orchestration.
- It supports copying game state, which is useful for heuristic or search baselines.
- It supports seeding and reproducible simulation.
- It already has hooks, JSON output, CSV/Parquet export, and a Gymnasium interface.
- It supports multi-player games, not only 1v1.

## What I would be careful about

### 1. Information leakage

This is the biggest issue for your benchmark.

Catanatron's custom bot API gives a player the **complete game state**. The docs also expose helpers for reading a player's hand and dev cards from state. That is convenient for bot research, but it is wrong for a benchmark where players should only know public info plus their own private info.

So:

- use Catanatron as the engine,
- do **not** expose raw engine state to LLM players,
- add an observation wrapper that redacts hidden information.

### 2. Baseline strength is not yet your benchmark strength

The repo's strongest documented bot is AlphaBeta, and the paper also uses AlphaBeta. But both the docs and the paper lean heavily on **1v1 or 2-player** evaluation. That is not enough to certify strength for full 4-player benchmark play.

So I would still use AlphaBeta-style bots as baselines, but I would re-evaluate them in your actual 4-player setting.

### 3. License

The GitHub repo is marked **GPL-3.0**. That is probably fine for research, but it matters if you want to distribute your benchmark framework in a way that links tightly to the engine. You should check the legal implications for the way you plan to publish the project.

### 4. Trading / negotiation design

Catan includes negotiation, which is part of what makes it interesting. But free-form natural-language bargaining makes benchmarking messy very quickly.

My recommendation:

- v1: support **structured trade actions** only,
- v2: add optional natural-language negotiation,
- v3: study communication as a separate benchmark track.

That will keep the first version much cleaner.

### 5. Trading support is present, but it is structured

This matters a lot for your benchmark idea.

Catanatron does appear to implement an actual trading flow, not just bank exchange. In the source, `ActionType` includes:

- `MARITIME_TRADE`,
- `OFFER_TRADE`,
- `ACCEPT_TRADE`,
- `REJECT_TRADE`,
- `CONFIRM_TRADE`,
- `CANCEL_TRADE`.

It also defines trade-related prompts:

- `DECIDE_TRADE`,
- `DECIDE_ACCEPTEES`.

So my read is:

- **yes**, the simulator includes a trading phase / trade interaction model,
- **yes**, it supports player-to-player structured trades,
- **no**, it is not natively a free-form natural-language negotiation environment.

That is actually a good fit for v1 of this benchmark. You can benchmark strategic trade decisions first, and only later decide whether to add open-ended language negotiation.

## Proposed benchmark architecture

### Core decision loop

1. The orchestrator reads the full `catanatron` state.
2. It identifies the current decision point and the acting player or players.
3. It builds a public observation and a player-private observation.
4. It attaches that player's prior private memory.
5. The player returns:
   - a canonical `action`,
   - an optional `memory_write`.
6. The orchestrator validates the action, applies it in the engine, persists logs and memory, and advances to the next decision point.

### Core components

1. **Engine adapter**
   - wraps Catanatron,
   - steps the game,
   - converts engine state to benchmark observations,
   - validates actions.

2. **Referee / harness**
   - controls decision-point order,
   - manages seeds,
   - stores one public history stream plus snapshot artifacts,
   - calls model adapters,
   - persists player-authored memory,
   - handles retries / invalid outputs / timeouts.

3. **Player adapters**
   - one adapter per provider/model,
   - translate structured observation to model input,
   - parse action output,
   - optionally emit `memory_write`.

4. **Logs**
   - `public_history.jsonl`
   - `public_state_trace.jsonl`
   - `players/<id>/memory_trace.jsonl`
   - `players/<id>/prompt_trace.jsonl`

5. **Evaluation runner**
   - runs fixed-seed suites,
   - rotates seating positions,
   - computes win rate, average rank, VP, latency, tokens, cost, invalid action rate.

## Current implementation in this repo

The initial harness scaffold is now in place.

- `src/catan_bench/schemas.py`
  - core dataclasses for `Action`, `DecisionPoint`, turn/reactive observations, public events, memory snapshots, and results
- `src/catan_bench/engine.py`
  - the minimal `EngineAdapter` protocol the orchestrator expects from a game engine
- `src/catan_bench/observations.py`
  - the `ObservationBuilder` that constructs compact player-scoped observations from engine state plus the shared public log and player memory
- `src/catan_bench/storage.py`
  - append-only public event logs, public state snapshots, per-player memory snapshots, and prompt trace storage
- `src/catan_bench/orchestrator.py`
  - the `GameOrchestrator` that runs the simplified turn loop, validates actions, applies them to the engine, and persists the new artifacts
- `src/catan_bench/players.py`
  - the `Player` protocol plus a small `ScriptedPlayer` for tests and demos
- `tests/test_orchestrator.py`
  - a mock trade-focused engine test that verifies:
    - public event routing,
    - private event routing,
    - player-authored memory persistence,
    - invalid action rejection before engine application

The key point is that the harness is already shaped around **decision points**, not just turns, and around **public history + private history + private memory** as separate stores.

## Immediate next build step

The next implementation step should be a real `catanatron` adapter that maps:

- the engine's current prompt to a benchmark `DecisionPoint`,
- full game state to benchmark `public_state`,
- player-visible state to benchmark `private_state`,
- engine actions to benchmark `Action`,
- engine transitions to public and private `Event` records.

Once that adapter exists, the current orchestrator should already be able to run real games.

## Metrics I would report

At minimum:

- win rate,
- average finishing rank,
- average final victory points,
- average game length,
- invalid action rate,
- timeout / parse failure rate,
- average tokens per turn,
- average latency per turn,
- estimated cost per game.

For richer analysis:

- road/settlement/city timing,
- trade acceptance rate,
- robber effectiveness,
- resource discard behavior,
- calibration of `memory_write` usefulness,
- seat bias sensitivity.

## Benchmark tracks I would define

### Track A: Pure decision benchmark

- 4-player standard base game,
- structured actions only,
- no natural-language chat,
- private memory allowed,
- no cross-game learning.

This should be the main benchmark.

### Track B: Communication benchmark

- same game,
- structured actions plus bounded natural-language trading/chat,
- separate leaderboard.

### Track C: Continual learning benchmark

- agents may preserve memory across games in a match series,
- still no code writing during evaluation,
- separate from the main benchmark.

This is where the paper's style becomes more relevant.

## Roadmap

## Phase 0: Finalize benchmark scope

- Lock the first version to **4-player base Catan**.
- Decide whether v1 includes player-to-player trades.
- Decide whether the benchmark stores only `memory_write` notes or full private reasoning traces.
- Define a canonical action schema and observation schema.

## Phase 1: Engine integration

- Add Catanatron as the engine dependency.
- Build a thin adapter around game creation, stepping, trade flow, and replay export.
- Verify deterministic reproduction with fixed seeds.
- Confirm seat rotation and board randomization are easy to control.

## Phase 2: Harness scaffolding

- Implement the game orchestrator around decision points, not only turns.
- Route trade offers and responses through the orchestrator.
- Define how players emit `memory_write`.
- Persist public history plus player-private logs.

## Phase 3: Observability wrapper

- Implement public-state extraction.
- Implement player-private observation extraction.
- Redact all hidden opponent information.
- Add tests proving no player can access opponent hidden cards through the benchmark API.

This is the most important engineering step.

## Phase 4: Logging and memory

- Create public and private event logs.
- Add player-scoped `memory_write` persistence owned by the harness.
- Define replay files that can reconstruct a game exactly.
- Make all logs easy to inspect offline.

## Phase 5: Baselines

- Random baseline.
- Heuristic baseline using Catanatron players where possible.
- AlphaBeta-based baseline adapted to your 4-player setting.
- One or two simple LLM baselines:
  - no memory,
  - public + private memory.

## Phase 6: Evaluation protocol

- Fixed development seed set.
- Held-out test seed set.
- Rotate seating positions for every model.
- Run enough games to get stable confidence intervals.
- Report both aggregate and per-seat results.

## Phase 7: Benchmark package

- CLI to run tournaments.
- Replay viewer support.
- Model adapter interface for OpenAI / Anthropic / local models.
- Standard output artifacts for leaderboard submission.

## Phase 8: Extensions

- Communication track.
- Continual-learning track.
- Hidden-belief modeling track.
- VLM-backed player track so models can see the board directly.
- Human-vs-model and mixed-table evaluation.

## My bottom-line recommendation

I would build this project around:

- **Catanatron as the engine**
- **a new observability and logging harness on top**
- **simple per-game LLM players with private memory**
- **no artifact-writing or cross-game code evolution in the main benchmark**

That gives you a benchmark that is:

- closer to actual multi-player Catan,
- fairer across models,
- easier to reproduce,
- cleaner to interpret than the paper's heavier system.

If you want, the paper's HexMachina-style setup can still become a **separate advanced track**, but I would not make it the default benchmark.

## Sources

- Paper abstract and version info: https://arxiv.org/abs/2506.04651
- Paper PDF: https://arxiv.org/pdf/2506.04651
- Catanatron GitHub repo: https://github.com/bcollazo/catanatron
- Catanatron docs: https://docs.catanatron.com/
- Catanatron API docs: https://catanatron.readthedocs.io/en/latest/catanatron.html
