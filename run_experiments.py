from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from experiment import provenance, require_current_method
from rouge.config import load_config, resolve_path
from rouge.core import write_json
from rouge.deck import starter_build

HERE = Path(__file__).resolve().parent
VARIANTS = ('full', 'no_composition', 'no_potion', 'uniform_route', 'base')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run RogueEvo, its three ablations, or Base with fixed report panels.')
    parser.add_argument('--config', type=Path, default=HERE / 'configs/rogueevo.json')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['full'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--budget', type=int, choices=(750000, 1500000), default=1500000)
    parser.add_argument('--panels', nargs='+', choices=('dev80', 'test199'), default=['dev80'])
    parser.add_argument('--steps', type=int, choices=(2, 4), default=4,
                        help='Frozen-report steps only; search execution is unchanged')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if any(seed < 0 for seed in args.seeds):
        parser.error('Seeds must be nonnegative')
    for name in ('variants', 'seeds', 'panels'):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f'Duplicate {name}')
    config = load_config(args.config)
    require_current_method(config)
    config['dataset_manifest'] = resolve_path(config['dataset_manifest'])
    config['model']['path'] = str(args.model_path.expanduser().resolve()) if args.model_path else resolve_path(config['model']['path'])
    config['budget']['development'] = args.budget
    root = args.output.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('Use a new output directory; frozen reports can be resumed with evaluate.py')
    plan = []
    for variant in args.variants:
        seeds = [0] if variant == 'base' else args.seeds
        for seed in seeds:
            folder = root / f'{variant}_s{seed}_b{args.budget}'
            row = copy.deepcopy(config)
            row['seed'] = seed
            row['complement']['campaign_ablation'] = 'full' if variant == 'base' else variant
            report_config = copy.deepcopy(row)
            report_config['seed'] = 0
            report_config['execution']['max_steps'] = args.steps
            commands = []
            if variant == 'base':
                build, origin = folder / 'base_build.json', folder / 'base_origin.json'
                development = ['--development-evidence', str(folder / 'base_development.json')]
            else:
                commands.append([sys.executable, str(HERE / 'experiment.py'), '--mode', 'search',
                                 '--config', str(folder / 'search_config.json'), '--out-dir', str(folder / 'search'),
                                 '--label', f'RogueEvo/{variant}/seed{seed}'])
                build, origin = folder / 'search/best_build.json', folder / 'search/provenance.json'
                development = ['--development-source', str(folder / 'search/final_info.json')]
            for panel in args.panels:
                commands.append([sys.executable, str(HERE / 'evaluate.py'),
                    '--config', str(folder / 'report_config.json'), '--build', str(build),
                    '--origin', str(origin), *development, '--out-dir', str(folder / panel), '--panel', panel,
                    '--label', f'{variant}/seed{seed}/steps{args.steps}'])
            plan.append({'variant': variant, 'seed': seed, 'search_budget': 0 if variant == 'base' else args.budget,
                         'evaluation_seed': 0, 'folder': str(folder), 'commands': commands})
            if not args.dry_run:
                write_json(folder / 'search_config.json', row)
                write_json(folder / 'report_config.json', report_config)
                if variant == 'base':
                    write_json(build, starter_build().to_dict())
                    write_json(origin, {**provenance(args), 'execution_id': starter_build().execution_id})
                    write_json(folder / 'base_development.json', {'source': 'none', 'method': 'base',
                        'spent': 0, 'limit': 0, 'tool_price': config['budget']['tool_price']})
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    write_json(root / 'experiment_plan.json', plan)
    for row in plan:
        for command in row['commands']:
            subprocess.run(command, check=True, cwd=HERE)
    return 0


if __name__ == '__main__':
    sys.exit(main())
