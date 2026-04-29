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

Dataset and reference protocol:

- A valid sample must contain a non-empty `sample_id`.
- A valid sample must contain `metadata.camera_videos` as a non-empty object.
- `metadata.camera_videos` must contain all expected 6 views defined by the dataset config.
- Each expected camera view path must be non-empty. Path existence checks are optional and controlled by the caller.
- `scripts/random_select.py` writes `results/data_inspection.json` first and stops early if dataset inspection reports a structural error.
- `scripts/random_select.py` accepts `--dataset-config` so dataset inspection and selection can use the same dataset config path as evaluation.
- `GenEval.evaluate()` only evaluates `load_valid_samples()` and no longer evaluates samples that fail dataset inspection.
- When reference preparation writes `results/enriched_data.json`, it must preserve the dataset entry protocol used by `src/dataset/`.
- Reference generators may append prepared fields under `metadata`, such as `semantic_masks`, `semantic_num_classes`, and `semantic_ignore_label`.
- Reference generators must not overwrite protected dataset-entry fields such as `metadata.camera_videos`.
- If `enriched_data.json` breaks the dataset protocol, evaluation now fails at dataset inspection before module execution.

Notes:

- `semantic_consistency.py` reads prepared semantic masks and does not run OpenSeeD inference directly.
- Module implementations live under `src/modules/` and should not depend on a separate `metrics` package.
- Third-party source code belongs under `src/third_party/`.
- Model weights belong under `pretrained_models/`.
