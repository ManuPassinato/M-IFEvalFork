# Experiment artifacts

This directory centralizes generated responses and evaluation outputs. Benchmark
input datasets remain under `data/`.

## Current campaigns

- `data_open_1`, `data_open_2`, `data_open_3`: Portuguese open-model responses for the three runs.
- `eval_open_1`, `eval_open_2`, `eval_open_3`: evaluations corresponding to those runs.
- `data_pten`, `data_pten_2`, `data_pten_3`: open-model responses for translated IFEval.
- `eval_pten`, `eval_pten_2`, `eval_pten_3`: translated-IFEval evaluations.
- `data_open_all`, `eval_open_all`: multilingual open-model artifacts.
- `data_close`, `evaluations_close`: API-model artifacts.
- `data_paper`, `evaluations_paper`: legacy paper-specific artifacts pending provenance review.
- `data_closepten`, `eval_closepten`: legacy translated API-model artifacts pending deduplication.
- `legacy_data`, `evaluations`: earlier M-IFEval responses and evaluations.

`filter_audit.json` records every artifact filtered to the canonical set of 535
Portuguese prompts. Run `python scripts/filter_experiment_prompts.py --dry-run`
to check whether any oversized native-Portuguese artifacts remain.

## Calculate scores

Use the unified calculator for any experiment directory. Supply the canonical
dataset when the result must be restricted to Portuguese-IFEval's 535 prompts:

```bash
python calc_eval.py --evaluations-dir experiments/eval_open_3 --canonical-data data/pt_input_data.jsonl
```

For experiments with a different dataset, omit `--canonical-data`:

```bash
python calc_eval.py --evaluations-dir experiments/eval_pten_3
```

## Run evaluations

Evaluate Portuguese responses against the canonical 535-prompt dataset:

```bash
python run_eval_only.py --responses-dir experiments/data_open_3 --evaluations-dir experiments/eval_open_3 --languages pt
```

Evaluate the 541-prompt translated Portuguese benchmark:

```bash
python run_eval_only.py --responses-dir experiments/data_close --evaluations-dir experiments/evaluations_close --languages pten
```

Use `--dry-run` to validate prompt coverage before writing evaluation outputs.
The legacy `pt_en_*` files in `data_pten*` require their matching historical
input dataset; supply it explicitly with `--pten-input-data` before evaluating
them.

## Generate and run a benchmark

`universal_inference.py` uses explicit dataset names and never mixes its input
and output folders. `pt` resolves to `data/pt_input_data.jsonl` (535 prompts)
and `pten` to `experiments/data_close/pten_input_data.jsonl` (541 prompts).

```bash
python universal_inference.py --model_name organization/model --datasets pt pten --output-dir experiments/generated_responses
```

To generate one or more models and then evaluate their responses in the same
layout, use the runner:

```bash
python benchmark_runner.py --datasets pt pten --models organization/model --responses-dir experiments/generated_responses --evaluations-dir experiments/generated_evaluations
```

Both commands accept `--dry-run`. Dependency installation is opt-in in the
runner through `--install-dependencies`.
