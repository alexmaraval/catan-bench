from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from catan_bench import GameResult
from catan_bench.config import load_game_config, load_player_configs
from catan_bench.runner import build_players, run_from_config_files


class ConfigAndRunnerTests(unittest.TestCase):
    def test_load_example_configs(self) -> None:
        game_config = load_game_config("configs/game.toml")
        player_configs = load_player_configs("configs/players.toml")

        self.assertEqual(game_config.engine, "catanatron")
        self.assertEqual(game_config.seed, 12)
        self.assertEqual(len(player_configs), 4)
        self.assertEqual(player_configs[0].id, "RED")

    def test_build_players_from_config(self) -> None:
        player_configs = load_player_configs("configs/players.toml")
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


if __name__ == "__main__":
    unittest.main()
