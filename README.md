# RogueEvo

Budgeted agent-harness evolution with a frozen language model. RogueEvo revises and combines reusable instruction cards, explores input-observation capabilities, and selects a complete build for deployment. The actor and evolver share the same model weights.

This repository contains the current method, the SpreadsheetBench executor, frozen-build evaluation, and selected experiment drivers.

## Method

The executable harness has four instruction slots: **inspect**, **plan**, **edit**, and **verify**. A candidate replaces one slot or combines revisions in different slots. Search maintains a training frontier and a separate deployment champion.

1. Select a parent and training evidence. Failure coverage, measured route credit, and seeded exploration determine which opportunities are tried.
2. Propose complete slot revisions. Retain reusable cards in a bounded pool, including cards that may become useful with a partner.
3. Screen individual and joint builds. The optional capability branch supplies temporary input facts, then constructs a consumer revision and a deployable observer when supported by the evidence.
4. Admit training progress using task success and training-only cell progress. Preserve a separate selection budget for choosing the deployment champion.
5. Evaluate the frozen champion on a reporting panel. Temporary support is disabled at deployment. A retained observer recomputes facts from each new input.

The roguelike inspiration is the construction of a stronger build through varied choices and complementary components. In code, temporary “potions” are input-only column, numeric, or layout observations. Pair interactions are measured through budgeted four-arm evaluations of the same parent and tasks. Randomness comes from seeded proposal sampling and search choices.

## Code map

| File | Role |
| --- | --- |
| `rouge/campaign_search.py` | Current search controller, coverage-aware routing, card combinations, and interaction audits |
| `rouge/campaign_evolution.py` | Training evidence selection and current proposal contracts |
| `rouge/frontier_search.py` | Training frontier, admission, and champion selection |
| `rouge/expedition_search.py` | Completion opportunities, budget reserves, and paired selection |
| `rouge/response_search.py`, `demand_search.py`, `complement.py` | Capability response, consumer dependencies, and shared search operations |
| `rouge/evolution.py`, `items.py`, `deck.py` | Proposals, typed build representation, and the starting harness |
| `rouge/support.py`, `demands.py` | Temporary observations and permanent observer contracts |
| `rouge/execution.py`, `sandbox.py` | Spreadsheet agent execution and isolated tools |
| `rouge/model.py`, `batching.py` | Shared frozen model and bounded inference batches |
| `rouge/core.py`, `config.py` | Budget accounting, events, and configuration validation |
| `prepare_data.py`, `validate_data.py` | Data preparation and scorer compatibility audit |
| `experiment.py` | Environment checks and latest-method search |
| `evaluate.py`, `report_budget.py` | Frozen evaluation, resumable reports, and durable cost accounting |
| `run_experiments.py` | Full method, Base, budget settings, and three ablations |

The search modules form an inheritance chain used by the current controller. All listed modules are runtime dependencies.

## Setup

Use Python 3.11 or newer, NVIDIA GPUs, and Docker on Linux. The default configuration uses local Qwen3-14B weights, FP16, an 8,192-token context, greedy actor decoding, and inference batches of up to four requests. `device_map=balanced` uses the GPUs visible to the process.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-model.txt
docker build -t rouge-spreadsheet:0.2 .
```

Install a PyTorch build compatible with the host CUDA driver. Place the model checkpoint in `models/Qwen3-14B`, set `model.path` in `configs/rogueevo.json`, or pass `--model-path` to the search or experiment driver. Model weights are loaded locally.

## Data

Find the public **SpreadsheetBench Verified 400** release and obtain `spreadsheetbench_verified_400.tar.gz`. Obtain `evaluation/evaluation.py` from the official project at the revision below. Save the two files locally, for example in `downloads/`. Search terms and required filenames are also listed in `THIRD_PARTY.md`.

```bash
python prepare_data.py --archive downloads/spreadsheetbench_verified_400.tar.gz \
  --scorer downloads/evaluation.py --output data --seed 2026
python validate_data.py --manifest data/manifest.json \
  --compatible-manifest data/compatible_manifest.json
```

The loader reads the supplied files and verifies the official evaluator against commit `49b73a94775fb489063f60ca1865e3a650079a79` and SHA-256 `4ae77cee8df01d1f34684fceab972810d696886533d33be2e89373de6b4d3de3`. It records the data archive checksum. The compatibility audit compares reference workbooks with themselves and excludes incompatible tasks before model execution. With the experiment data, this yields 392 tasks: 97 training, 96 validation, and 199 test tasks.

| Use | Fixed task panel |
| --- | --- |
| Search evidence | First 24 training tasks |
| Champion selection | First 8 validation tasks |
| `dev80` report | Validation tasks at indices `[8:88]` |
| `test199` report | All 199 test tasks |

`dev80` and `test199` are disjoint. Search never executes test tasks. These are task-level custom splits. Dataset files, model weights, and generated experiment outputs are excluded from this repository.

Check the environment before a model experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiment.py --mode doctor --deep
```

## Search and frozen evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiment.py --mode search \
  --config configs/rogueevo.json --seed 0 --out-dir runs/full_s0/search

CUDA_VISIBLE_DEVICES=0,1 python evaluate.py \
  --config configs/rogueevo.json \
  --build runs/full_s0/search/best_build.json \
  --origin runs/full_s0/search/provenance.json \
  --development-source runs/full_s0/search/final_info.json \
  --panel dev80 --out-dir runs/full_s0/dev80
```

Use `--panel test199` and a new output directory for the full test panel. The default report configuration has evaluation seed 0. Search seeds and evaluation seeds are distinct controls; `run_experiments.py` fixes evaluation seed 0 for every search seed.

The search budget is **prompt tokens + generated tokens + 256 × tool calls**. The default development cap is 1,500,000 units. Search also has a 12,480-second wall-time cap. Actual usage and the stop reason are recorded. Report evaluation uses a separate 6,000,000-unit cap.

Search requires a fresh output directory. To resume an interrupted frozen report, repeat its command with `--resume`. The report verifies the build, configuration, task panel, source identity, and persisted budget before reusing completed tasks. A complete report has `valid_report=true`; incomplete reports retain pending tasks and do not report a final score.

## Selected experiments

Preview commands without creating outputs or starting inference:

```bash
python run_experiments.py --output runs/preview --dry-run
```

Run the full method for three search seeds, reporting both panels:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_experiments.py \
  --output runs/main --variants full --seeds 0 1 2 \
  --budget 1500000 --panels dev80 test199
```

Run the lower search budget:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_experiments.py \
  --output runs/budget075 --variants full --seeds 0 \
  --budget 750000 --panels dev80 test199
```

Run three ablations and the unchanged starting harness:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_experiments.py \
  --output runs/ablations --variants no_composition no_potion uniform_route base \
  --seeds 0 --budget 1500000 --panels dev80
```

| Variant | Change |
| --- | --- |
| `full` | Complete RogueEvo |
| `no_composition` | Evaluate revisions individually without joint composition |
| `no_potion` | Disable the temporary capability branch and producer completion |
| `uniform_route` | Use uniform alternatives for feedback-guided search routing |
| `base` | Freeze the starting harness without search |

The driver runs selected jobs sequentially on the visible GPUs and stops if a command fails. Base is evaluated once with evaluation seed 0. Search runs retain the same actor, scorer, data panels, and execution limits across variants.

For inference-budget analysis, reuse the **same frozen build** with `evaluate.py` and two report configuration files whose `execution.max_steps` values are 2 and 4. Keep all other fields, including evaluation seed, fixed, and use separate output directories. Report success count, mean charged cost per task, and charged cost per successful task from the complete reports. The charged cost is the accounting unit defined above.

## Outputs and checks

Search writes `config.json`, `provenance.json`, `events.jsonl`, `best_build.json`, and `final_info.json`, together with method-specific records and execution traces. Frozen evaluation writes `report_plan.json`, `report_budget.json`, per-task records, and `final_info.json`. Preserve these files together to retain the link between a frozen build, its development cost, and its report.

```bash
python -m unittest discover -s tests -t .
```

The CPU tests cover complementary cards, temporary support boundaries, evidence selection, model preflight, batched inference, training credit, and interrupted-report recovery. Five scorer integration tests additionally require the official evaluator obtained during data preparation and are skipped when it is absent. Synthetic outcomes test implementation behavior; model experiments use the commands above.

See `THIRD_PARTY.md` for external dependency retrieval instructions.
