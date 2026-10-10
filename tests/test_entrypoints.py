import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from experiment import require_current_method
from rouge.config import load_config
import run_experiments
from tests.workspace import TemporaryDirectory

HERE = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_default_and_three_ablations_use_the_current_controller(self):
        config = load_config(HERE / 'configs/rogueevo.json')
        for variant in ('full', 'no_composition', 'no_potion', 'uniform_route'):
            with self.subTest(variant=variant):
                row = copy.deepcopy(config)
                row['complement']['campaign_ablation'] = variant
                require_current_method(row)
        config['complement']['campaign_guided'] = False
        with self.assertRaises(ValueError):
            require_current_method(config)

    def test_dry_run_does_not_create_outputs_or_start_processes(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'preview'
            text = io.StringIO()
            with mock.patch.object(run_experiments.subprocess, 'run') as start, contextlib.redirect_stdout(text):
                run_experiments.main(['--output', str(output), '--dry-run', '--variants', 'full', 'base',
                                      '--seeds', '1', '2', '--panels', 'dev80', 'test199'])
            start.assert_not_called()
            self.assertFalse(output.exists())
            plan = json.loads(text.getvalue())
            self.assertEqual([row['seed'] for row in plan], [1, 2, 0])
            self.assertEqual([row['evaluation_seed'] for row in plan], [0, 0, 0])
            self.assertEqual([len(row['commands']) for row in plan], [3, 3, 2])

    def test_search_seed_and_report_seed_are_separate(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'experiment'
            with mock.patch.object(run_experiments.subprocess, 'run') as start:
                run_experiments.main(['--output', str(output), '--seeds', '2', '--budget', '750000',
                                      '--steps', '2', '--variants', 'no_potion'])
            folder = output / 'no_potion_s2_b750000'
            search = json.loads((folder / 'search_config.json').read_text(encoding='utf-8'))
            report = json.loads((folder / 'report_config.json').read_text(encoding='utf-8'))
            self.assertEqual((search['seed'], report['seed']), (2, 0))
            self.assertEqual((search['execution']['max_steps'], report['execution']['max_steps']), (4, 2))
            self.assertEqual(search['budget']['development'], 750000)
            self.assertEqual(search['complement']['campaign_ablation'], 'no_potion')
            self.assertEqual(start.call_count, 2)
