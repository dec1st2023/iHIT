import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_main_loads_root_config_and_runs_workflow(self):
        import main

        config = object()
        with mock.patch("main._require_runtime_dependencies"), mock.patch("main.load_config", return_value=config) as load_config, mock.patch("main.workflow.run") as run:
            code = main.main()

        load_config.assert_called_once_with(ROOT / "config.yaml", repo_root=ROOT)
        run.assert_called_once_with(config)
        self.assertEqual(code, 0)

    def test_runtime_dependency_error_uses_generic_reproduction_command(self):
        import main

        with mock.patch("main.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requirements.txt"):
                main._require_runtime_dependencies()


if __name__ == "__main__":
    unittest.main()
