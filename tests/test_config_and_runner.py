from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catan_bench import GameResult
from catan_bench.config import load_game_config, load_player_configs
from catan_bench.reporter import DebugTerminalReporter
from catan_bench.runner import (
    _find_dotenv,
    _load_resume_game_id,
    build_players,
    main,
    run_from_config_files,
)


class _StubEngine:
    game_id = "mock-game"
    player_ids = ("RED", "BLUE")


class ConfigAndRunnerTests(unittest.TestCase):
    def test_load_example_configs(self) -> None:
        game_config = load_game_config("configs/game.toml")
        player_configs = load_player_configs("configs/openai-players.toml")

        self.assertEqual(game_config.engine, "catanatron")
        self.assertEqual(game_config.seed, 12)
        self.assertTrue(game_config.trading_chat_enabled)
        self.assertEqual(game_config.trading_chat_max_rooms_per_turn, 5)
        self.assertEqual(len(player_configs), 4)
        self.assertEqual(player_configs[0].id, "RED")

    def test_load_llm_player_config_without_prompt_memory_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            players_toml = Path(tmpdir) / "players.toml"
            players_toml.write_text(
                (
                    "[[players]]\n"
                    "id = \"RED\"\n"
                    "type = \"llm\"\n"
                    "model = \"gpt-4o-mini\"\n"
                    "prompt_history_limit = 10\n\n"
                    "[[players]]\n"
                    "id = \"BLUE\"\n"
                    "type = \"random\"\n\n"
                ),
                encoding="utf-8",
            )

            configs = load_player_configs(players_toml)

            self.assertEqual(configs[0].model, "gpt-4o-mini")
            self.assertEqual(configs[0].prompt_history_limit, 10)

    def test_build_players_from_config(self) -> None:
        player_configs = load_player_configs("configs/openai-players.toml")
        players = build_players(player_configs)
        self.assertEqual(set(players.keys()), {"RED", "BLUE", "ORANGE", "WHITE"})

    def test_runner_executes_game_from_toml_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            run_dir = Path(tmpdir) / "run"

            game_toml.write_text(
                (
                    "[game]\n"
                    "engine = \"catanatron\"\n"
                    "seed = 3\n"
                    f"run_dir = \"{run_dir}\"\n"
                ),
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[[players]]\n"
                    "id = \"RED\"\n"
                    "type = \"random\"\n\n"
                    "[[players]]\n"
                    "id = \"BLUE\"\n"
                    "type = \"random\"\n"
                ),
                encoding="utf-8",
            )

            with patch("catan_bench.runner.build_engine", return_value=_StubEngine()):
                with patch("catan_bench.runner.GameOrchestrator.run") as run_mock:
                    run_mock.return_value = GameResult(
                        game_id="mock-game",
                        winner_ids=("RED",),
                        total_decisions=3,
                        public_event_count=3,
                        memory_writes=0,
                        metadata={},
                    )
                    result = run_from_config_files(
                        game_config_path=game_toml,
                        players_config_path=players_toml,
                    )

            self.assertEqual(result.winner_ids, ("RED",))
            run_mock.assert_called_once()

    def test_runner_uses_debug_reporter_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            game_toml.write_text("[game]\nengine = \"catanatron\"\n", encoding="utf-8")
            players_toml.write_text(
                (
                    "[[players]]\n"
                    "id = \"RED\"\n"
                    "type = \"random\"\n\n"
                    "[[players]]\n"
                    "id = \"BLUE\"\n"
                    "type = \"random\"\n"
                ),
                encoding="utf-8",
            )

            with patch("catan_bench.runner.build_engine", return_value=_StubEngine()):
                with patch("catan_bench.runner.GameOrchestrator") as orchestrator_cls:
                    orchestrator_cls.return_value.run.return_value = GameResult(
                        game_id="mock-game",
                        winner_ids=("RED",),
                        total_decisions=1,
                        public_event_count=0,
                        memory_writes=0,
                        metadata={},
                    )
                    run_from_config_files(
                        game_config_path=game_toml,
                        players_config_path=players_toml,
                        debug=True,
                    )

            reporter = orchestrator_cls.call_args.kwargs["reporter"]
            self.assertIsInstance(reporter, DebugTerminalReporter)

    def test_main_parses_debug_flag(self) -> None:
        with patch("catan_bench.runner.run_from_config_files") as run_mock:
            run_mock.return_value = GameResult(
                game_id="mock-game",
                winner_ids=("RED",),
                total_decisions=1,
                public_event_count=0,
                memory_writes=0,
                metadata={},
            )
            exit_code = main(["--game", "game.toml", "--players", "players.toml", "--debug"])

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with(
            game_config_path="game.toml",
            players_config_path="players.toml",
            resume_run_dir=None,
            debug=True,
            debug_from_setup=False,
            debug_trade=False,
        )

    def test_main_parses_resume_flag(self) -> None:
        with patch("catan_bench.runner.run_from_config_files") as run_mock:
            run_mock.return_value = GameResult(
                game_id="mock-game",
                winner_ids=("RED",),
                total_decisions=1,
                public_event_count=0,
                memory_writes=0,
                metadata={},
            )
            exit_code = main(
                [
                    "--game",
                    "game.toml",
                    "--players",
                    "players.toml",
                    "--resume-run",
                    "runs/existing",
                ]
            )

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with(
            game_config_path="game.toml",
            players_config_path="players.toml",
            resume_run_dir="runs/existing",
            debug=False,
            debug_from_setup=False,
            debug_trade=False,
        )

    def test_load_resume_game_id_reads_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "metadata.json").write_text(
                '{"game_id": "saved-game-id"}\n',
                encoding="utf-8",
            )

            self.assertEqual(_load_resume_game_id(run_dir), "saved-game-id")

    def test_find_dotenv_walks_up_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_file = root / ".env"
            nested = root / "a" / "b" / "c"
            nested.mkdir(parents=True)
            env_file.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")

            found = _find_dotenv(nested)

            self.assertEqual(found, env_file)


if __name__ == "__main__":
    unittest.main()
