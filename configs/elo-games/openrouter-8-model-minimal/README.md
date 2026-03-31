# OpenRouter 8-model Elo schedule

This directory contains the 6 player configs for the minimum practical 8-model Elo schedule.

Model mapping:

- `M1` = `nvidia/nemotron-3-super-120b-a12b:free`
- `M2` = `z-ai/glm-4.5-air:free`
- `M3` = `xiaomi/mimo-v2-flash`
- `M4` = `openai/gpt-oss-120b`
- `M5` = `x-ai/grok-4.1-fast`
- `M6` = `qwen/qwen3-235b-a22b-thinking-2507`
- `M7` = `deepseek/deepseek-v3.2`
- `M8` = `minimax/minimax-m2.7`

Game schedule:

- `game-01.toml`: `M1 M2 M3 M4`
- `game-02.toml`: `M1 M3 M5 M7`
- `game-03.toml`: `M1 M4 M6 M8`
- `game-04.toml`: `M2 M4 M5 M7`
- `game-05.toml`: `M2 M5 M6 M8`
- `game-06.toml`: `M3 M6 M7 M8`

The seat assignments were chosen so each model appears in 3 distinct colors across its 3 games, and each color is used exactly 6 times overall.

Run a game with:

```bash
python -m catan_bench --game configs/game.toml --players configs/elo-games/openrouter-8-model-minimal/game-01.toml
```
