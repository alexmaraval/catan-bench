from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from catan_bench import GameResult
from catan_bench.config import load_game_config, load_player_configs
from catan_bench.runner import _find_dotenv, build_players, run_from_config_files


class ConfigAndRunnerTests(unittest.TestCase):
    def test_load_example_configs(self) -> None:
        game_config = load_game_config("configs/game.toml")
        player_configs = load_player_configs("configs/openai-players.toml")

        self.assertEqual(game_config.engine, "catanatron")
        self.assertEqual(game_config.seed, 12)
        self.assertEqual(len(player_configs), 4)
        self.assertEqual(player_configs[0].id, "RED")

    def test_build_players_from_config(self) -> None:
        player_configs = load_player_configs("configs/openai-players.toml")
        players = build_players(player_configs)

        self.assertEqual(set(players.keys()), {"RED", "BLUE", "ORANGE", "WHITE"})

    def test_runner_executes_game_from_toml_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "openai-players.toml"
            run_dir = Path(tmpdir) / "run"

            game_toml.write_text(
                (
                    "[game]\n"
                    "engine = \"catanatron\"\n"
                    "seed = 3\n"
                    "discard_limit = 7\n"
                    "vps_to_win = 6\n"
                    f"run_dir = \"{run_dir}\"\n"
                ),
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[[players]]\n"
                    "id = \"RED\"\n"
                    "type = \"random\"\n"
                    "seed = 1\n\n"
                    "[[players]]\n"
                    "id = \"BLUE\"\n"
                    "type = \"random\"\n"
                    "seed = 2\n\n"
                    "[[players]]\n"
                    "id = \"ORANGE\"\n"
                    "type = \"random\"\n"
                    "seed = 3\n\n"
                    "[[players]]\n"
                    "id = \"WHITE\"\n"
                    "type = \"random\"\n"
                    "seed = 4\n"
                ),
                encoding="utf-8",
            )

            with patch("catan_bench.runner.build_engine", return_value=object()):
                with patch("catan_bench.runner.GameOrchestrator.run") as run_mock:
                    run_mock.return_value = GameResult(
                        game_id="mock-game",
                        winner_ids=("RED",),
                        total_decisions=10,
                        public_event_count=8,
                        private_event_count=4,
                        memory_writes=0,
                        metadata={},
                    )
                    result = run_from_config_files(
                        game_config_path=game_toml,
                        players_config_path=players_toml,
                    )

            self.assertEqual(result.winner_ids, ("RED",))
            run_mock.assert_called_once()
            self.assertIsInstance(load_game_config(game_toml).run_dir, Path)

    def test_load_llm_player_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            players_toml = Path(tmpdir) / "openai-players.toml"
            players_toml.write_text(
                (
                    "[[players]]\n"
                    "id = \"RED\"\n"
                    "type = \"llm\"\n"
                    "model = \"gpt-4o-mini\"\n\n"
                    "[[players]]\n"
                    "id = \"BLUE\"\n"
                    "type = \"random\"\n\n"
                ),
                encoding="utf-8",
            )

            configs = load_player_configs(players_toml)

            self.assertEqual(configs[0].type, "llm")
            self.assertEqual(configs[0].model, "gpt-4o-mini")
            self.assertEqual(configs[0].api_key_env, "OPENAI_API_KEY")

    def test_runner_loads_dotenv_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()
            game_toml = config_dir / "game.toml"
            players_toml = config_dir / "openai-players.toml"
            env_file = config_dir / ".env"

            game_toml.write_text(
                "[game]\nengine = \"catanatron\"\n",
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[[players]]\n"
                    "id = \"RED\"\n"
                    "type = \"llm\"\n"
                    "model = \"gpt-4o-mini\"\n\n"
                    "[[players]]\n"
                    "id = \"BLUE\"\n"
                    "type = \"random\"\n"
                ),
                encoding="utf-8",
            )
            env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                with patch("catan_bench.runner.build_engine", return_value=object()):
                    with patch("catan_bench.runner.GameOrchestrator.run") as run_mock:
                        run_mock.return_value = GameResult(
                            game_id="mock-game",
                            winner_ids=("RED",),
                            total_decisions=1,
                            public_event_count=0,
                            private_event_count=0,
                            memory_writes=0,
                            metadata={},
                        )
                        run_from_config_files(
                            game_config_path=game_toml,
                            players_config_path=players_toml,
                        )

                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-dotenv")

    def test_runner_does_not_override_existing_env_with_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()
            game_toml = config_dir / "game.toml"
            players_toml = config_dir / "openai-players.toml"
            env_file = config_dir / ".env"

            game_toml.write_text(
                "[game]\nengine = \"catanatron\"\n",
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[[players]]\n"
                    "id = \"RED\"\n"
                    "type = \"llm\"\n"
                    "model = \"gpt-4o-mini\"\n\n"
                    "[[players]]\n"
                    "id = \"BLUE\"\n"
                    "type = \"random\"\n"
                ),
                encoding="utf-8",
            )
            env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

            with patch.dict("os.environ", {"OPENAI_API_KEY": "from-shell"}, clear=True):
                with patch("catan_bench.runner.build_engine", return_value=object()):
                    with patch("catan_bench.runner.GameOrchestrator.run") as run_mock:
                        run_mock.return_value = GameResult(
                            game_id="mock-game",
                            winner_ids=("RED",),
                            total_decisions=1,
                            public_event_count=0,
                            private_event_count=0,
                            memory_writes=0,
                            metadata={},
                        )
                        run_from_config_files(
                            game_config_path=game_toml,
                            players_config_path=players_toml,
                        )

                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-shell")

    def test_find_dotenv_stops_after_config_local_ancestor_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            distant_env_dir = root / "workspace"
            distant_env_dir.mkdir()
            (distant_env_dir / ".env").write_text(
                "OPENAI_API_KEY=too-far-away\n",
                encoding="utf-8",
            )

            nested = distant_env_dir / "a" / "b" / "c" / "d" / "e"
            nested.mkdir(parents=True)

            self.assertIsNone(_find_dotenv(nested, max_depth=3))
            self.assertEqual(_find_dotenv(nested, max_depth=5), distant_env_dir / ".env")


if __name__ == "__main__":
    unittest.main()
