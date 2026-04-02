from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catan_bench import GameResult
from catan_bench.config import (
    load_game_config,
    load_game_config_overrides,
    load_player_configs,
)
from catan_bench.orchestrator import _resolve_run_dir
from catan_bench.prompts import CATAN_RULES_SUMMARY
from catan_bench.reporter import DebugTerminalReporter
from catan_bench.runner import (
    _AUTO_RESUME_REQUEST,
    _find_dotenv,
    _is_existing_run_directory,
    _load_resume_game_id,
    _resolve_requested_run_dir,
    build_players,
    main,
    run_from_config_files,
)


class _StubEngine:
    game_id = "mock-game"
    player_ids = ("RED", "BLUE")


def _write_resume_artifacts(
    run_dir: Path,
    *,
    game_id: str = "saved-game-id",
    metadata_contents: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if metadata_contents is None:
        metadata_contents = f'{{"game_id": "{game_id}"}}\n'
    (run_dir / "metadata.json").write_text(metadata_contents, encoding="utf-8")
    for file_name, contents in (
        ("checkpoint.json", "{}\n"),
        ("public_history.jsonl", ""),
        ("public_state_trace.jsonl", ""),
        ("action_trace.jsonl", ""),
    ):
        (run_dir / file_name).write_text(contents, encoding="utf-8")


class ConfigAndRunnerTests(unittest.TestCase):
    def test_load_example_configs(self) -> None:
        game_config = load_game_config("configs/game.toml")
        player_configs = load_player_configs("configs/openai-players.toml")

        self.assertEqual(game_config.engine, "catanatron")
        self.assertIsNone(game_config.seed)
        self.assertEqual(game_config.version, "1.1.1")
        self.assertEqual(game_config.prompt_history_limit, 30)
        self.assertEqual(game_config.run_dir, Path("runs/test"))
        self.assertEqual(game_config.run_tags, ("1.1.1",))
        self.assertIn(
            f"The first player to reach {game_config.vps_to_win} victory points wins.",
            CATAN_RULES_SUMMARY,
        )
        self.assertTrue(game_config.public_chat_enabled)
        self.assertEqual(game_config.public_chat_message_chars, 500)
        self.assertEqual(game_config.public_chat_history_limit, 15)
        self.assertTrue(game_config.trading_chat_enabled)
        self.assertEqual(game_config.trading_chat_max_rooms_per_turn, 5)
        self.assertEqual(len(player_configs), 4)
        self.assertEqual(player_configs[0].id, "RED")
        self.assertTrue(
            all(
                config.api_base == "https://api.openai.com/v1"
                for config in player_configs
            )
        )

    def test_load_game_config_with_prompt_history_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            game_toml.write_text(
                ('[game]\nengine = "catanatron"\nprompt_history_limit = 10\n'),
                encoding="utf-8",
            )

            config = load_game_config(game_toml)

            self.assertEqual(config.prompt_history_limit, 10)

    def test_load_game_config_with_run_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            game_toml.write_text(
                (
                    '[game]\nengine = "catanatron"\nrun_dir = "runs/"\n'
                    'run_tags = ["1.1.1", "experiment-a"]\n'
                ),
                encoding="utf-8",
            )

            config = load_game_config(game_toml)

            self.assertEqual(config.run_dir, Path("runs"))
            self.assertEqual(config.run_tags, ("1.1.1", "experiment-a"))

    def test_load_game_config_overrides_from_players_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            game_toml.write_text(
                (
                    "[game]\n"
                    'engine = "catanatron"\n'
                    "seed = 12\n"
                    'run_dir = "runs/elo"\n'
                    "prompt_history_limit = 15\n"
                ),
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[game]\nseed = 777\n\n"
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "random"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
                ),
                encoding="utf-8",
            )

            base_config = load_game_config(game_toml)
            merged = load_game_config_overrides(players_toml, base_config)

            self.assertEqual(base_config.seed, 12)
            self.assertEqual(merged.seed, 777)
            self.assertEqual(merged.run_dir, Path("runs/elo"))
            self.assertEqual(merged.prompt_history_limit, 15)

    def test_load_player_config_rejects_prompt_history_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            players_toml = Path(tmpdir) / "players.toml"
            players_toml.write_text(
                (
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "llm"\n'
                    'model = "gpt-4o-mini"\n'
                    "prompt_history_limit = 10\n\n"
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n\n'
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "game config"):
                load_player_configs(players_toml)

    def test_load_player_config_with_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            players_toml = Path(tmpdir) / "players.toml"
            players_toml.write_text(
                (
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "llm"\n'
                    'model = "openai/gpt-oss-120b"\n'
                    'reasoning_effort = "low"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
                ),
                encoding="utf-8",
            )

            config = load_player_configs(players_toml)

            self.assertEqual(config[0].reasoning_effort, "low")
            self.assertIsNone(config[0].reasoning_enabled)

    def test_load_player_config_rejects_reasoning_enabled_and_effort_together(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            players_toml = Path(tmpdir) / "players.toml"
            players_toml.write_text(
                (
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "llm"\n'
                    'model = "openai/gpt-oss-120b"\n'
                    "reasoning_enabled = false\n"
                    'reasoning_effort = "low"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "either `reasoning_enabled` or `reasoning_effort`"
            ):
                load_player_configs(players_toml)

    def test_build_players_from_config(self) -> None:
        game_config = load_game_config("configs/game.toml")
        player_configs = load_player_configs("configs/openai-players.toml")
        players = build_players(player_configs, game_config)
        self.assertEqual(set(players.keys()), {"RED", "BLUE", "ORANGE", "WHITE"})
        self.assertEqual(
            players["RED"].prompt_history_limit, game_config.prompt_history_limit
        )

    def test_runner_executes_game_from_toml_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            run_dir = Path(tmpdir) / "run"

            game_toml.write_text(
                (f'[game]\nengine = "catanatron"\nseed = 3\nrun_dir = "{run_dir}"\n'),
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "random"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
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

    def test_runner_passes_run_tags_to_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            game_toml.write_text(
                (
                    "[game]\n"
                    'engine = "catanatron"\n'
                    "vps_to_win = 5\n"
                    'run_dir = "runs/"\n'
                    'run_tags = ["1.1.1", "dev"]\n'
                ),
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "random"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
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
                    )

            self.assertEqual(
                orchestrator_cls.call_args.kwargs["run_tags"], ("1.1.1", "dev")
            )
            self.assertEqual(orchestrator_cls.call_args.kwargs["run_label"], "players")
            self.assertIsNone(orchestrator_cls.call_args.kwargs["game_seed"])
            self.assertIn(
                "The first player to reach 5 victory points wins.",
                orchestrator_cls.call_args.kwargs["observation_builder"].game_rules,
            )

    def test_runner_uses_game_seed_override_from_players_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            game_toml.write_text(
                (
                    "[game]\n"
                    'engine = "catanatron"\n'
                    "seed = 12\n"
                    'run_dir = "runs/"\n'
                ),
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[game]\nseed = 777\n\n"
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "random"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
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
                    )

            self.assertEqual(orchestrator_cls.call_args.kwargs["game_seed"], 777)

    def test_resolve_run_dir_places_logs_under_version_directory(self) -> None:
        resolved = _resolve_run_dir(
            Path("runs"),
            game_id="mock-game",
            run_version="1.1.1",
            run_tags=("1.1.1", "dev"),
            run_label="mixed-players",
        )

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.parent, Path("runs") / "1.1.1")
        self.assertRegex(
            resolved.name,
            r"^tags-dev-mixed-players-mock-game-\d{8}T\d{6}Z-[0-9a-f]{8}$",
        )

    def test_resolve_run_dir_includes_seed_when_configured(self) -> None:
        resolved = _resolve_run_dir(
            Path("runs"),
            game_id="mock-game",
            run_version="1.1.1",
            run_tags=("1.1.1", "dev"),
            run_label="mixed-players",
            game_seed=12,
        )

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.parent, Path("runs") / "1.1.1")
        self.assertRegex(
            resolved.name,
            r"^tags-dev-mixed-players-seed-12-mock-game-\d{8}T\d{6}Z-[0-9a-f]{8}$",
        )

    def test_runner_uses_debug_reporter_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            game_toml.write_text('[game]\nengine = "catanatron"\n', encoding="utf-8")
            players_toml.write_text(
                (
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "random"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
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

    def test_runner_existing_run_dir_argument_resumes_and_ignores_config_run_dir(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            resume_run_dir = Path(tmpdir) / "existing-run"
            _write_resume_artifacts(resume_run_dir)
            game_toml.write_text(
                (
                    "[game]\n"
                    'engine = "catanatron"\n'
                    f'run_dir = "{Path(tmpdir) / "new-run-base"}"\n'
                ),
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "random"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
                ),
                encoding="utf-8",
            )

            with patch(
                "catan_bench.runner.build_engine", return_value=_StubEngine()
            ) as build_engine_mock:
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
                        run_dir=resume_run_dir,
                    )

            self.assertEqual(
                build_engine_mock.call_args.kwargs["game_id"], "saved-game-id"
            )
            self.assertIsNone(orchestrator_cls.call_args.kwargs["run_dir"])
            self.assertEqual(
                orchestrator_cls.call_args.kwargs["resume_run_dir"],
                resume_run_dir,
            )

    def test_runner_cli_run_dir_overrides_config_base_for_new_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            game_toml = Path(tmpdir) / "game.toml"
            players_toml = Path(tmpdir) / "players.toml"
            cli_run_dir = Path(tmpdir) / "cli-runs"
            game_toml.write_text(
                (
                    "[game]\n"
                    'engine = "catanatron"\n'
                    f'run_dir = "{Path(tmpdir) / "config-runs"}"\n'
                ),
                encoding="utf-8",
            )
            players_toml.write_text(
                (
                    "[[players]]\n"
                    'id = "RED"\n'
                    'type = "random"\n\n'
                    "[[players]]\n"
                    'id = "BLUE"\n'
                    'type = "random"\n'
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
                        run_dir=cli_run_dir,
                    )

            self.assertEqual(orchestrator_cls.call_args.kwargs["run_dir"], cli_run_dir)
            self.assertIsNone(orchestrator_cls.call_args.kwargs["resume_run_dir"])

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
            exit_code = main(
                ["--game", "game.toml", "--players", "players.toml", "--debug"]
            )

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with(
            game_config_path="game.toml",
            players_config_path="players.toml",
            run_dir=None,
            debug=True,
            debug_from_setup=False,
            debug_trade=False,
        )

    def test_main_parses_run_dir_flag(self) -> None:
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
                    "--run-dir",
                    "runs/existing",
                ]
            )

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with(
            game_config_path="game.toml",
            players_config_path="players.toml",
            run_dir="runs/existing",
            debug=False,
            debug_from_setup=False,
            debug_trade=False,
        )

    def test_main_backcompat_resume_run_alias_maps_to_run_dir(self) -> None:
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
            run_dir="runs/existing",
            debug=False,
            debug_from_setup=False,
            debug_trade=False,
        )

    def test_main_resume_run_without_path_requests_auto_detection(self) -> None:
        with patch("catan_bench.runner.run_from_config_files") as run_mock:
            run_mock.return_value = GameResult(
                game_id="mock-game",
                winner_ids=("RED",),
                total_decisions=1,
                public_event_count=0,
                memory_writes=0,
                metadata={},
            )
            exit_code = main(["--game", "game.toml", "--players", "players.toml", "--resume-run"])

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with(
            game_config_path="game.toml",
            players_config_path="players.toml",
            run_dir=_AUTO_RESUME_REQUEST,
            debug=False,
            debug_from_setup=False,
            debug_trade=False,
        )

    def test_resolve_requested_run_dir_uses_config_when_cli_missing(self) -> None:
        configured = Path("runs/default")

        run_dir, resume_run_dir = _resolve_requested_run_dir(
            requested_run_dir=None,
            configured_run_dir=configured,
        )

        self.assertEqual(run_dir, configured)
        self.assertIsNone(resume_run_dir)

    def test_resolve_requested_run_dir_detects_existing_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            _write_resume_artifacts(run_path, game_id="saved")

            run_dir, resume_run_dir = _resolve_requested_run_dir(
                requested_run_dir=run_path,
                configured_run_dir=Path("runs/default"),
            )

        self.assertIsNone(run_dir)
        self.assertEqual(resume_run_dir, run_path)

    def test_resolve_requested_run_dir_auto_detects_unique_matching_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_run_dir = Path(tmpdir) / "runs"
            players_toml = Path(tmpdir) / "players.toml"
            players_toml.write_text("[[players]]\nid = \"RED\"\ntype = \"random\"\n", encoding="utf-8")

            matching_run = base_run_dir / "1.1.1" / "tags-players"
            _write_resume_artifacts(
                matching_run,
                metadata_contents=(
                    "{\n"
                    '  "game_id": "saved-game-id",\n'
                    f'  "players_config_path": "{players_toml.resolve()}",\n'
                    '  "run_version": "1.1.1",\n'
                    '  "run_label": "players",\n'
                    '  "game_seed": 777,\n'
                    '  "run_tags": ["1.1.1"]\n'
                    "}\n"
                ),
            )
            _write_resume_artifacts(
                base_run_dir / "1.1.0" / "tags-other",
                metadata_contents='{"game_id":"saved-game-id","players_config_path":"/tmp/other.toml"}\n',
            )

            run_dir, resume_run_dir = _resolve_requested_run_dir(
                requested_run_dir=_AUTO_RESUME_REQUEST,
                configured_run_dir=base_run_dir,
                players_config_path=players_toml.resolve(),
                run_label="players",
                game_seed=777,
                run_version="1.1.1",
                run_tags=("1.1.1",),
            )

        self.assertIsNone(run_dir)
        self.assertEqual(resume_run_dir, matching_run)

    def test_resolve_requested_run_dir_auto_resume_rejects_multiple_matches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_run_dir = Path(tmpdir) / "runs"
            players_toml = Path(tmpdir) / "players.toml"
            players_toml.write_text("[[players]]\nid = \"RED\"\ntype = \"random\"\n", encoding="utf-8")

            metadata_contents = (
                "{\n"
                '  "game_id": "saved-game-id",\n'
                f'  "players_config_path": "{players_toml.resolve()}",\n'
                '  "run_version": "1.1.1",\n'
                '  "run_label": "players",\n'
                '  "game_seed": 777,\n'
                '  "run_tags": ["1.1.1"]\n'
                "}\n"
            )
            _write_resume_artifacts(
                base_run_dir / "1.1.1" / "tags-match-a",
                metadata_contents=metadata_contents,
            )
            _write_resume_artifacts(
                base_run_dir / "1.1.1" / "tags-match-b",
                metadata_contents=metadata_contents,
            )

            with self.assertRaisesRegex(ValueError, "Multiple matching run directories"):
                _resolve_requested_run_dir(
                    requested_run_dir=_AUTO_RESUME_REQUEST,
                    configured_run_dir=base_run_dir,
                    players_config_path=players_toml.resolve(),
                    run_label="players",
                    game_seed=777,
                    run_version="1.1.1",
                    run_tags=("1.1.1",),
                )

    def test_resolve_requested_run_dir_auto_resume_rejects_when_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_run_dir = Path(tmpdir) / "runs"
            players_toml = Path(tmpdir) / "players.toml"
            players_toml.write_text("[[players]]\nid = \"RED\"\ntype = \"random\"\n", encoding="utf-8")
            _write_resume_artifacts(
                base_run_dir / "1.1.0" / "tags-other",
                metadata_contents='{"game_id":"saved-game-id","players_config_path":"/tmp/other.toml"}\n',
            )

            with self.assertRaisesRegex(ValueError, "No matching run directory found"):
                _resolve_requested_run_dir(
                    requested_run_dir=_AUTO_RESUME_REQUEST,
                    configured_run_dir=base_run_dir,
                    players_config_path=players_toml.resolve(),
                    run_label="players",
                    game_seed=777,
                    run_version="1.1.1",
                    run_tags=("1.1.1",),
                )

    def test_is_existing_run_directory_checks_artifact_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.assertFalse(_is_existing_run_directory(path))
            (path / "checkpoint.json").write_text("{}\n", encoding="utf-8")
            self.assertFalse(_is_existing_run_directory(path))
            _write_resume_artifacts(path)
            self.assertTrue(_is_existing_run_directory(path))

    def test_resolve_requested_run_dir_rejects_incomplete_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            (run_path / "metadata.json").write_text(
                '{"game_id":"saved"}\n', encoding="utf-8"
            )
            (run_path / "checkpoint.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required resume files"):
                _resolve_requested_run_dir(
                    requested_run_dir=run_path,
                    configured_run_dir=Path("runs/default"),
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
