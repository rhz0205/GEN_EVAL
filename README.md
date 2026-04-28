# worldbench

`worldbench` is an offline evaluation framework for generated autonomous-driving videos. It is the clean rebuild of `GEN_EVAL` with a flat `src/` source layout and a script-first offline workflow.

Current status: clean rebuild structure is in place. Dataset loading, reference preparation scaffolding, evaluation orchestration, result summarization, and visualization entrypoints are present. OpenSeeD inference remains adapter-reserved unless a local implementation is provided.

Canonical workflow:

`data list -> dataset normalization -> reference data generation -> modules evaluation -> summarization -> visualization`

Directory structure:

- `configs/`
- `data/`
- `outputs/`
- `pretrained_models/`
- `scripts/`
- `src/dataset/`
- `src/modules/`
- `src/reference/`
- `src/models/`
- `src/visualization/`
- `src/third_party/`

Canonical config files:

- `configs/dataset.yaml`
- `configs/metrics.yaml`
- `configs/reference.yaml`
- `configs/run.yaml`

Canonical dataset names:

- `cosmos`
- `geely`
- `real`
- `sample`

Naming rules:

- Data file: `data/{dataset_name}_{data_count}_{timestamp}.json`
- Output directory: `outputs/{dataset_name}/{data_count}_{timestamp}/`

Main scripts:

- `scripts/random_select.py`
- `scripts/prepare_references.py`
- `scripts/run_eval.py`
- `scripts/summarize_results.py`
- `scripts/visualize_results.py`

Basic commands:

- `python scripts/random_select.py --help`
- `python scripts/prepare_references.py --help`
- `python scripts/run_eval.py --help`
- `python scripts/summarize_results.py --help`
- `python scripts/visualize_results.py --help`

Notes:

- `semantic_consistency.py` reads prepared semantic masks and does not run OpenSeeD inference directly.
- Module implementations live under `src/modules/` and should not depend on a separate `metrics` package.
- Third-party source code belongs under `src/third_party/`.
- Model weights belong under `pretrained_models/`.
