# 13-model-group Elo schedule

This directory contains a pairwise-complete 13-model Elo schedule for four-player Catan games.

## Process

This set uses 13 games for 13 models, which is the clean four-player case where a true balanced design exists:

- 13 models total
- 13 games total
- 4 players per game
- each model appears in 4 games
- each model gets `RED`, `BLUE`, `ORANGE`, and `WHITE` exactly once
- every pair of models meets exactly once

The schedule is built from a cyclic 4-player block design, so each game is just a rotation of the same offset pattern.

## Models

- `nvidia/nemotron-3-super-120b-a12b:free`
- `z-ai/glm-5`
- `xiaomi/mimo-v2-flash`
- `arcee-ai/trinity-large-thinking`
- `mistralai/mistral-large-2512`
- `qwen/qwen3.6-plus:free`
- `deepseek/deepseek-v3.2`
- `minimax/minimax-m2.7`
- `moonshotai/kimi-k2.5`
- `prime-intellect/intellect-3`
- `openai/gpt-oss-120b`
- `stepfun/step-3.5-flash`
- `google/gemma-4-31b-it`

## Notes

- Shared runtime settings live in `game-common.toml`.
- Per-game player assignments live in `game-01.toml` through `game-13.toml`.
- `minimax/minimax-m2.7` uses `reasoning_effort = "minimal"` because these configs do not disable reasoning for that model.

Run a game with:

```bash
python -m catan_bench --game configs/elo-games/13-model-group/game-common.toml --players configs/elo-games/13-model-group/game-01.toml
```
